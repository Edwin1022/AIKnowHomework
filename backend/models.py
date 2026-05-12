import uuid

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
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
    user_email = Column(String, nullable=False)

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


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(
        String,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String, nullable=False)
    content = Column(Text, nullable=True)  # NULL when status=pending
    status = Column(String, nullable=False, default="completed")
    sequence_number = Column(Integer, nullable=False)
    turn_number = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Only populated on assistant rows
    temperature = Column(Float, nullable=True)  # type: ignore[var-annotated]
    seed = Column(Integer, nullable=True)
    max_output_tokens = Column(Integer, nullable=True)
    model_choice = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "sequence_number", name="uq_conversation_sequence"
        ),
        CheckConstraint("role IN ('user', 'assistant')", name="ck_message_role"),
        CheckConstraint(
            "status IN ('pending', 'completed', 'failed')", name="ck_message_status"
        ),
        CheckConstraint(
            "(role = 'user' AND model_choice IS NULL AND temperature IS NULL"
            " AND seed IS NULL AND max_output_tokens IS NULL)"
            " OR (role = 'assistant')",
            name="ck_params_only_on_assistant",
        ),
        Index("ix_message_conversation_id", "conversation_id"),
        Index("ix_message_turn_number", "conversation_id", "turn_number"),
    )

    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return (
            f"<Message id={self.id!r} "
            f"conversation_id={self.conversation_id!r} "
            f"role={self.role!r} "
            f"seq={self.sequence_number!r}>"
        )
