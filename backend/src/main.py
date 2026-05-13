import uuid
import base64
import io
import re
from PIL import Image
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form

load_dotenv()
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

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

    conv.title = request.title
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

@app.post("/conversations/{conversation_id}/chat")
async def chat(
    conversation_id: str,
    content: str = Form(...),
    model_choice: str = Form("meta-llama/llama-4-scout-17b-16e-instruct"),
    image: Optional[UploadFile] = File(None),
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
    llm_content_payload = [{"type": "text", "text": content}]

    # 3. Handle Image Upload & Compression
    if image:
        image_bytes = await image.read()
        
        # Open the image using Pillow
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert transparent PNGs to RGB so they can be saved as JPEG
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        # Resize the image so the longest side is a maximum of 800 pixels
        img.thumbnail((800, 800))
        
        # Save the compressed image back to a bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75) # 75 quality is a good balance of size/clarity
        compressed_bytes = buffer.getvalue()

        # Base64 encode the COMPRESSED image, not the raw original
        base64_encoded = base64.b64encode(compressed_bytes).decode("utf-8")
        mime_type = "image/jpeg" # We forced it to JPEG above
        
        # Append the Markdown to the DB content so Streamlit renders it in history
        full_content += f"\n\n![{image.filename}](data:{mime_type};base64,{base64_encoded})"
        
        # Append the actual Base64 data to the LLM's payload in the correct Vision format
        llm_content_payload.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_encoded}"
            }
        })

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
    
    conv.updated_at = datetime.now(timezone.utc)
    await db.commit()

    # 5. Reconstruct history for the LLM context window
    history = []
    for msg in sorted(conv.messages, key=lambda m: m.sequence_number):
        if msg.sequence_number < user_seq:
            # Strip out massive Base64 strings from history to save tokens
            clean_content = re.sub(
                r'!\[[^\]]*\]\(data:image/[^;]+;base64,[^\)]+\)', 
                '[Image from previous turn]', 
                msg.content or ""
            )
            history.append({"role": msg.role, "content": clean_content})
    
    history.append({"role": "user", "content": llm_content_payload})

    # 6. Stream the assistant's response
    async def stream_generator():
        llm_response = ""
        llm_meta: dict[str, object] = {}
        
        async for chunk in groq_stream(history, model=model_choice):
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

        conv.updated_at = datetime.now(timezone.utc)

        if user_seq == 1:
            generated_title = await generate_conversation_title(content, llm_response)
            conv.title = generated_title

        await db.commit()

    return StreamingResponse(stream_generator(), media_type="text/plain")