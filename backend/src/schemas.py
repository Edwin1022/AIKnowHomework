from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class MessageSchema(BaseModel):
    id: str
    role: str
    content: Optional[str]
    sequence_number: int
    created_at: datetime

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