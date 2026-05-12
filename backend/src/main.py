import uuid
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

import models
from database import get_db, init_db

from src.schemas import (
    ConversationResponse,
    ConversationDetailResponse,
    TitleUpdateRequest
)
from src.services import mock_llm_stream, generate_conversation_title

# --- Lifespan Setup for Async DB Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="LLM Chat Application (Async & Modular)", lifespan=lifespan)

# --- Routes: Chat Management ---

@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(db: AsyncSession = Depends(get_db)):
    new_conv = models.Conversation(id=str(uuid.uuid4()))
    db.add(new_conv)
    await db.commit()
    await db.refresh(new_conv)
    return new_conv

@app.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    stmt = select(models.Conversation).order_by(models.Conversation.updated_at.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return result.scalars().all()

@app.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def read_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)):
    # selectinload fetches the relationship eagerly, required for async sessions
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
    
    user_msg = models.Message(
        conversation_id=conversation_id,
        role="user",
        content=final_content,
        sequence_number=user_seq
    )
    db.add(user_msg)
    await db.commit()

    async def stream_generator():
        llm_response = ""
        # 1. Yield chunks asynchronously
        async for chunk in mock_llm_stream(final_content):
            llm_response += chunk
            yield chunk
        
        # 2. Save the assistant's message after stream completes
        assistant_seq = user_seq + 1
        assistant_msg = models.Message(
            conversation_id=conversation_id,
            role="assistant",
            content=llm_response.strip(),
            sequence_number=assistant_seq
        )
        db.add(assistant_msg)
        
        # 3. Generate and update title if first message
        if user_seq == 1:
            generated_title = generate_conversation_title(final_content, llm_response)
            conv.title = generated_title
            
        await db.commit()

    return StreamingResponse(stream_generator(), media_type="text/plain")