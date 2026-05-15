from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class MessageSchema(BaseModel):
    id: str
    role: str
    content: Optional[str]
    sequence_number: int
    turn_number: Optional[int] = None
    branch_id: int = 0
    fork_start_seq: Optional[int] = None
    created_at: datetime
    model_choice: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    class Config:
        from_attributes = True

class ConversationBase(BaseModel):
    title: Optional[str] = None

class CreateConversationRequest(BaseModel):
    user_email: str

class ConversationResponse(ConversationBase):
    id: str
    user_email: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ConversationDetailResponse(ConversationResponse):
    messages: List[MessageSchema] = []

class TitleUpdateRequest(BaseModel):
    title: str