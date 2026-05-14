import uuid
from typing import List, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form

load_dotenv()
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
    branch_id: int = Form(0),
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

    if branch_id == 0:
        # Main thread — unchanged logic
        message_count = len([m for m in conv.messages if m.branch_id == 0])
        user_seq = message_count + 1
        assistant_seq = user_seq + 1
        turn_number = (message_count // 2) + 1
        fork_start_seq = None

        history = [
            {"role": msg.role, "content": msg.content}
            for msg in sorted(
                (m for m in conv.messages if m.branch_id == 0),
                key=lambda m: m.sequence_number,
            )
        ]
    else:
        # Fork branch — hybrid history
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
        history = [{"role": m.role, "content": m.content} for m in trunk_before]
        history += [{"role": m.role, "content": m.content} for m in branch_messages]

    history.append({"role": "user", "content": final_content})

    user_msg = models.Message(
        conversation_id=conversation_id,
        role="user",
        content=final_content,
        status="completed",
        sequence_number=user_seq,
        turn_number=turn_number,
        branch_id=branch_id,
        fork_start_seq=fork_start_seq,
    )
    db.add(user_msg)
    await db.commit()

    async def stream_generator():
        llm_response = ""
        llm_meta: dict[str, object] = {}
        async for chunk in groq_stream(history):  # type: ignore[arg-type]
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

        if branch_id == 0 and user_seq == 1:
            generated_title = await generate_conversation_title(final_content, llm_response)
            conv.title = generated_title

        await db.commit()

    return StreamingResponse(stream_generator(), media_type="text/plain")


@app.post("/conversations/{conversation_id}/messages/{message_id}/fork")
async def fork_message(
    conversation_id: str,
    message_id: str,
    content: str = Form(...),
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

    # LLM context: branch-0 messages strictly before the fork point
    trunk_before = sorted(
        [m for m in conv.messages if m.branch_id == 0 and m.sequence_number < fork_start_seq],
        key=lambda m: m.sequence_number,
    )
    history = [{"role": m.role, "content": m.content} for m in trunk_before]
    history.append({"role": "user", "content": content})

    fork_user_msg = models.Message(
        conversation_id=conversation_id,
        role="user",
        content=content,
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
        async for chunk in groq_stream(history):  # type: ignore[arg-type]
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
