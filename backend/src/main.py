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

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

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
        
    img.thumbnail((400, 400))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=75)
    return buffer.getvalue()

@app.post("/conversations/{conversation_id}/chat")
async def chat(
    conversation_id: str,
    content: str = Form(...),
    model_choice: str = Form(...),
    branch_id: int = Form(0),
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    # 1. Fetch the conversation and its messages
    stmt = select(models.Conversation).options(selectinload(models.Conversation.messages)).where(models.Conversation.id == conversation_id)
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # 2. Prepare the payloads and Handle Image Upload (LLM-Integration)
    full_content = content
    llm_content_payload: Union[str, List[Dict[str, Any]]] 

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

    # 3. Calculate sequencing based on active branch (HEAD)
    if branch_id == 0:
        message_count = len([m for m in conv.messages if m.branch_id == 0])
        user_seq = message_count + 1
        assistant_seq = user_seq + 1
        turn_number = (message_count // 2) + 1
        fork_start_seq = None
        
        raw_history_messages = sorted(
            [m for m in conv.messages if m.branch_id == 0 and m.sequence_number < user_seq],
            key=lambda m: m.sequence_number,
        )
    else:
        branch_messages = sorted(
            [m for m in conv.messages if m.branch_id == branch_id],
            key=lambda m: m.sequence_number,
        )
        if not branch_messages:
            raise HTTPException(status_code=404, detail="Branch not found")

        fork_start_seq = branch_messages[0].fork_start_seq
        branch_msg_count = len(branch_messages)
        user_seq = branch_msg_count + 1
        assistant_seq = user_seq + 1
        turn_number = (branch_msg_count // 2) + 1

        trunk_before = sorted(
            [m for m in conv.messages if m.branch_id == 0 and m.sequence_number < fork_start_seq],
            key=lambda m: m.sequence_number,
        )
        raw_history_messages = trunk_before + branch_messages

    # 4. Save the User Message
    user_msg = models.Message(
        conversation_id=conversation_id,
        role="user",
        content=full_content,
        status="completed",
        sequence_number=user_seq,
        turn_number=turn_number,
        branch_id=branch_id,
        fork_start_seq=fork_start_seq,
    )
    db.add(user_msg)
    
    conv.updated_at = datetime.now(timezone.utc).replace(tzinfo=None) # type: ignore
    await db.commit()

    # 5. Reconstruct history with Regex Stripping and Sticky Memory (Merged Logic)
    sticky_image_memory = ""
    
    for msg in raw_history_messages:
        if msg.role == "assistant" and "🖼️ Image Memory:" in (msg.content or ""):
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

    history: List[Dict[str, Any]] = [{"role": "system", "content": system_instruction}]
    
    for msg in raw_history_messages:
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
            branch_id=branch_id,
            fork_start_seq=fork_start_seq,
            model_choice=llm_meta.get("model"),
            temperature=llm_meta.get("temperature"),
            max_output_tokens=llm_meta.get("max_tokens"),
        )
        db.add(assistant_msg)

        conv.updated_at = datetime.now(timezone.utc).replace(tzinfo=None) # type: ignore

        if branch_id == 0 and user_seq == 1:
            generated_title = str(await generate_conversation_title(content, llm_response))
            conv.title = generated_title # type: ignore

        await db.commit()

    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.post("/conversations/{conversation_id}/messages/{message_id}/fork")
async def fork_message(
    conversation_id: str,
    message_id: str,
    content: str = Form(...),
    model_choice: str = Form("llama-3.3-70b-versatile"), 
    image: Optional[UploadFile] = File(None), # Upgraded to accept images
    db: AsyncSession = Depends(get_db),
):
    stmt = select(models.Conversation).options(selectinload(models.Conversation.messages)).where(models.Conversation.id == conversation_id)
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    target = next((m for m in conv.messages if m.id == message_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Message not found")
    if target.role != "user":
        raise HTTPException(status_code=422, detail="Can only fork user messages")
    if target.branch_id != 0:
        raise HTTPException(status_code=422, detail="Can only fork messages on the main branch")

    fork_start_seq = target.sequence_number
    new_branch_id = max(m.branch_id for m in conv.messages) + 1

    # --- 1. Image Retention & Payload Logic ---
    full_content = content
    llm_content_payload: Union[str, List[Dict[str, Any]]] = content

    if image:
        # If the user uploads a brand new image during the edit
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
        # If no new image, check if the TARGET message being edited originally had an image!
        original_image_match = re.search(r'!\[(.*?)\]\((data:image/[^;]+;base64,[^\)]+)\)', target.content or "")
        if original_image_match:
            filename = original_image_match.group(1)
            image_data_url = original_image_match.group(2)
            
            # Carry the old image over to the new branch's database record
            full_content += f"\n\n![{filename}]({image_data_url})"
            
            secret_instruction = (
                "\n\n[SYSTEM NOTE: The user just attached an image. After fulfilling their main request, "
                "you MUST append a section labeled '🖼️ Image Memory:' where you briefly but as accurately as possible "
                "describe all key details, objects, colors, and text in the image. You will need this "
                "description to answer follow-up questions because the image pixels will be deleted "
                "after this turn.]"
            )
            
            # Reconstruct the multimodal payload for Groq
            llm_content_payload = [
                {"type": "text", "text": content + secret_instruction},
                {"type": "image_url", "image_url": { "url": image_data_url }}
            ]

    # --- 2. Reconstruct History & Sticky Memory ---
    trunk_before = sorted(
        [m for m in conv.messages if m.branch_id == 0 and m.sequence_number < fork_start_seq],
        key=lambda m: m.sequence_number,
    )

    sticky_image_memory = ""
    for msg in trunk_before:
        if msg.role == "assistant" and "🖼️ Image Memory:" in (msg.content or ""):
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

    history: List[Dict[str, Any]] = [{"role": "system", "content": system_instruction}]

    for m in trunk_before:
        clean_content = re.sub(
            r'!\[.*?\]\(data:.*?;base64,.*?\)', 
            '[Image from previous turn]', 
            m.content or "",
            flags=re.DOTALL
        )
        history.append({"role": m.role, "content": clean_content})
        
    history.append({"role": "user", "content": llm_content_payload})

    # --- 3. Save and Stream ---
    fork_user_msg = models.Message(
        conversation_id=conversation_id,
        role="user",
        content=full_content,
        status="completed",
        sequence_number=1,
        turn_number=1,
        branch_id=new_branch_id,
        fork_start_seq=fork_start_seq,
    )
    db.add(fork_user_msg)
    await db.commit()

    async def stream_generator():
        llm_response = ""
        llm_meta: dict[str, object] = {}
        
        async for chunk in groq_stream(cast(list[ChatCompletionMessageParam], history), model=model_choice):
            if isinstance(chunk, dict):
                llm_meta = chunk
            else:
                llm_response += chunk
                yield chunk

        fork_asst_msg = models.Message(
            conversation_id=conversation_id,
            role="assistant",
            content=llm_response.strip(),
            status="completed",
            sequence_number=2,
            turn_number=1,
            branch_id=new_branch_id,
            fork_start_seq=fork_start_seq,
            model_choice=llm_meta.get("model"),
            temperature=llm_meta.get("temperature"),
            max_output_tokens=llm_meta.get("max_tokens"),
        )
        db.add(fork_asst_msg)
        await db.commit()

    headers = {"X-Branch-Id": str(new_branch_id)}
    return StreamingResponse(stream_generator(), media_type="text/plain", headers=headers)