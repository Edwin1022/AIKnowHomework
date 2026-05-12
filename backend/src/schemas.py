from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class MessageSchema(BaseModel):
    id: str
    role: str
    content: str
    sequence_number: int
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationBase(BaseModel):
    title: Optional[str] = None

class ConversationResponse(ConversationBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ConversationDetailResponse(ConversationResponse):
    messages: List[MessageSchema] = []

class TitleUpdateRequest(BaseModel):
    title: str