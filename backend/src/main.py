import uuid
import base64
import io
import asyncio
import re
import pprint
from PIL import Image
from typing import List, Optional, Dict, Any, Union, cast
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form

from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from groq.types.chat import ChatCompletionMessageParam

import backend.db.models as models
from backend.db.database import get_db, init_db

from backend.src.schemas import (
    ConversationResponse,
    ConversationDetailResponse,
    CreateConversationRequest,
    TitleUpdateRequest,
)
from backend.src.services import groq_stream, generate_conversation_title

# --- Lifespan Setup for Async DB Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="LLM Chat Application (Async & Modular)", lifespan=lifespan)

# --- Routes: Chat Management ---

@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: CreateConversationRequest, db: AsyncSession = Depends(get_db)):
    new_conv = models.Conversation(id=str(uuid.uuid4()), user_email=request.user_email)
    db.add(new_conv)
    await db.commit()
    await db.refresh(new_conv)
    return new_conv

@app.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(user_email: str, skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(models.Conversation)
        .where(models.Conversation.user_email == user_email)
        .order_by(models.Conversation.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return result.scalars().all()

@app.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def read_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(models.Conversation).options(selectinload(models.Conversation.messages)).where(models.Conversation.id == conversation_id)
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv

@app.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation_title(conversation_id: str, request: TitleUpdateRequest, db: AsyncSession = Depends(get_db)):
    stmt = select(models.Conversation).where(models.Conversation.id == conversation_id)
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.title = request.title # type: ignore
    await db.commit()
    await db.refresh(conv)
    return conv

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(models.Conversation).where(models.Conversation.id == conversation_id)
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conv)
    await db.commit()
    return {"message": "Conversation deleted successfully"}

# --- Routes: Chat Functionality ---

def compress_image_sync(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
        
    img.thumbnail((800, 800))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=75)
    return buffer.getvalue()

@app.post("/conversations/{conversation_id}/chat")
async def chat(
    conversation_id: str,
    content: str = Form(...),
    image: Optional[UploadFile] = File(None),
    model_choice: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    # 1. Fetch the conversation and its messages
    stmt = select(models.Conversation).options(selectinload(models.Conversation.messages)).where(models.Conversation.id == conversation_id)
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 2. Prepare the separate payloads for the Database and the LLM
    full_content = content

    llm_content_payload: Union[str, List[Dict[str, Any]]] 
    
    # 3. Handle Image Upload & Compression
    if image:
        image_bytes = await image.read()
        
        compressed_bytes = await asyncio.to_thread(compress_image_sync, image_bytes)

        base64_encoded = base64.b64encode(compressed_bytes).decode("utf-8")
        mime_type = "image/jpeg"
        
        full_content += f"\n\n![{image.filename}](data:{mime_type};base64,{base64_encoded})"
        
        secret_instruction = (
            "\n\n[SYSTEM NOTE: The user just attached an image. After fulfilling their main request, "
            "you MUST append a section labeled '🖼️ Image Memory:' where you briefly but as accurately as possible "
            "describe all key details, objects, colors, and text in the image. You will need this "
            "description to answer follow-up questions because the image pixels will be deleted "
            "after this turn.]"
        )
        
        llm_content_payload = [
            {"type": "text", "text": content + secret_instruction},
            {"type": "image_url", "image_url": { "url": f"data:{mime_type};base64,{base64_encoded}"}}
        ]
        
    else:
        llm_content_payload = content

    # 4. Calculate sequencing and save the User Message
    message_count = len(conv.messages)
    user_seq = message_count + 1
    assistant_seq = user_seq + 1
    turn_number = (message_count // 2) + 1

    user_msg = models.Message(
        conversation_id=conversation_id,
        role="user",
        content=full_content,
        status="completed",
        sequence_number=user_seq,
        turn_number=turn_number,
    )
    db.add(user_msg)
    
    conv.updated_at = datetime.now(timezone.utc) # type: ignore
    await db.commit()

    # 5. Reconstruct history for the LLM context window
    sticky_image_memory = ""
    sorted_messages = sorted(conv.messages, key=lambda m: m.sequence_number)
    
    for msg in sorted_messages:
        if msg.sequence_number < user_seq and msg.role == "assistant":
            if "🖼️ Image Memory:" in (msg.content or ""):
                # Extract everything after the marker
                parts = msg.content.split("🖼️ Image Memory:")
                sticky_image_memory = "🖼️ Image Memory:" + parts[-1]
    
    system_instruction = (
        "You are a highly capable AI. "
        "The chat UI strips old images from the history to save tokens. "
        "If the user asks a follow-up question about an image, use the text descriptions "
        "from your past responses to answer. NEVER refuse by saying 'there is no image provided' "
        "or 'I cannot see the image'.\n\n"
    )
    
    if sticky_image_memory:
        system_instruction += (
            f"The user previously uploaded an image. Use the following details "
            f"to answer any follow-up questions about it:\n\n{sticky_image_memory}"
        )

    history = [{"role": "system", "content": system_instruction}]
    
    for msg in sorted(conv.messages, key=lambda m: m.sequence_number):
        if msg.sequence_number < user_seq:
            # Strip out massive Base64 strings from history to save tokens
            clean_content = re.sub(
                r'!\[.*?\]\(data:.*?;base64,.*?\)', 
                '[Image from previous turn]', 
                msg.content or "",
                flags=re.DOTALL
            )
            history.append({"role": msg.role, "content": clean_content})
    
    current_message: Dict[str, Any] = {
        "role": "user", 
        "content": llm_content_payload
    }
    history.append(current_message)
    
    print("\n" + "="*50)
    print("FINAL HISTORY PAYLOAD GOING TO GROQ:")
    pprint.pprint(history, width=120)
    print("="*50 + "\n")

    # 6. Stream the assistant's response
    async def stream_generator():
        llm_response = ""
        llm_meta: dict[str, object] = {}
        
        async for chunk in groq_stream(cast(list[ChatCompletionMessageParam], history), model=model_choice):
            if isinstance(chunk, dict):
                llm_meta = chunk
            else:
                llm_response += chunk
                yield chunk

        assistant_msg = models.Message(
            conversation_id=conversation_id,
            role="assistant",
            content=llm_response.strip(),
            status="completed",
            sequence_number=assistant_seq,
            turn_number=turn_number,
            model_choice=llm_meta.get("model"),
            temperature=llm_meta.get("temperature"),
            max_output_tokens=llm_meta.get("max_tokens"),
        )
        db.add(assistant_msg)

        conv.updated_at = datetime.now(timezone.utc) # type: ignore

        if user_seq == 1:
            generated_title = await generate_conversation_title(content, llm_response)
            conv.title = generated_title # type: ignore

        await db.commit()

    return StreamingResponse(stream_generator(), media_type="text/plain")