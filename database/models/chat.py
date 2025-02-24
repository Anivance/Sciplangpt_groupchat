from sqlalchemy import Column, String, DateTime, JSON, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

from .base import Base

class ChatRoom(Base):
    __tablename__ = 'chat_rooms'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100))
    room_metadata = Column(JSON)  # 改名為 room_metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    messages = relationship("ChatMessage", back_populates="chat_room", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChatRoom {self.name}>"

class ChatMessage(Base):
    __tablename__ = 'chat_messages'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    room_id = Column(UUID(as_uuid=True), ForeignKey('chat_rooms.id'))
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'))
    content = Column(Text)
    message_type = Column(String(50))
    message_metadata = Column(JSON)  # 改名為 message_metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chat_room = relationship("ChatRoom", back_populates="messages")
    agent = relationship("Agent", back_populates="chat_messages")

    def __repr__(self):
        return f"<ChatMessage {self.id}>"