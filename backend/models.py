import uuid
import enum

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # One conversation has many messages
    # Deleting a conversation automatically deletes all its messages
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.sequence_number",
    )

    def __repr__(self):
        return f"<Conversation id={self.id!r} title={self.title!r}>"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(
        String,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String, nullable=False)  # "user" or "assistant"
    user_email = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sequence_number = Column(Integer, nullable=False)  # global order within conversation
    turn_number = Column(Integer, nullable=True)       # groups user+assistant pair
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    temperature = Column(Float, nullable=True)  # type: ignore[var-annotated]
    seed = Column(Integer, nullable=True)
    max_output_token = Column(Integer, nullable=True)
    model_choice = Column(String, nullable=True)
    setting = Column(String, nullable=True)

    # Safeguard: no two messages in the same conversation can share a position
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "sequence_number", name="uq_conversation_sequence"
        ),
        CheckConstraint("role IN ('user', 'assistant')", name="ck_message_role"),
    )

    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return (
            f"<Message id={self.id!r} "
            f"conversation_id={self.conversation_id!r} "
            f"role={self.role!r} "
            f"seq={self.sequence_number!r}>"
        )
