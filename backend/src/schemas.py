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


class MessageUsageSchema(BaseModel):
    message_id: str
    completion_status: str
    abort_reason: Optional[str] = None
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    model: str
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationUsageSummary(BaseModel):
    conversation_id: str
    total_messages: int
    completed: int
    aborted: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float