import uuid
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

import db.models as models
from db.database import get_db, init_db

from src.schemas import (
    ConversationResponse,
    ConversationDetailResponse,
    CreateConversationRequest,
    TitleUpdateRequest,
)
from src.services import groq_stream, generate_conversation_title

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
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    stmt = select(models.Conversation).options(selectinload(models.Conversation.messages)).where(models.Conversation.id == conversation_id)
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    image_info = ""
    if image:
        image_info = f"\n[User uploaded an image: {image.filename}]"

    final_content = content + image_info
    message_count = len(conv.messages)
    user_seq = message_count + 1
    assistant_seq = user_seq + 1
    turn_number = (message_count // 2) + 1

    user_msg = models.Message(
        conversation_id=conversation_id,
        role="user",
        content=final_content,
        status="completed",
        sequence_number=user_seq,
        turn_number=turn_number,
    )
    db.add(user_msg)
    await db.commit()

    history = [
        {"role": msg.role, "content": msg.content}
        for msg in sorted(conv.messages, key=lambda m: m.sequence_number)
        if msg.sequence_number < user_seq
    ]
    history.append({"role": "user", "content": final_content})

    async def stream_generator():
        llm_response = ""
        llm_meta: dict[str, object] = {}
        async for chunk in groq_stream(history):
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

        if user_seq == 1:
            generated_title = await generate_conversation_title(final_content, llm_response)
            conv.title = generated_title

        await db.commit()

    return StreamingResponse(stream_generator(), media_type="text/plain")
