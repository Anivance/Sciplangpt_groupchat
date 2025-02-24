from sqlalchemy import Column, String, DateTime, JSON, Text, Table, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

from .base import Base

# Agent 和 KnowledgeFile 的關聯表
agent_knowledge_files = Table(
    'agent_knowledge_files',
    Base.metadata,
    Column('agent_id', UUID(as_uuid=True), ForeignKey('agents.id', ondelete='CASCADE')),
    Column('knowledge_file_id', UUID(as_uuid=True), ForeignKey('knowledge_files.id', ondelete='CASCADE')),
)

class KnowledgeFile(Base):
    __tablename__ = 'knowledge_files'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    content = Column(Text)
    file_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        """轉換為字典格式"""
        return {
            "id": str(self.id),
            "filename": self.filename,
            "file_type": self.file_type,
            "created_at": self.created_at.isoformat()
        }

class Agent(Base):
    __tablename__ = 'agents'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    type = Column(String(50))
    base_prompt = Column(Text)
    template_name = Column(String(100))
    parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 關聯
    knowledge_files = relationship(
        "KnowledgeFile",
        secondary=agent_knowledge_files,
        backref="agents"
    )
    embeddings = relationship("Embedding", back_populates="agent", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="agent", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        """轉換為字典格式"""
        result = {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "base_prompt": self.base_prompt,
            "template_name": self.template_name,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        
        # 如果 knowledge_files 已加載，則包含它們
        if 'knowledge_files' in self.__dict__:
            result["knowledge_files"] = [
                {
                    "id": str(kf.id),
                    "filename": kf.filename,
                    "file_type": kf.file_type,
                    "created_at": kf.created_at.isoformat() if kf.created_at else None
                } for kf in self.knowledge_files
            ]
        else:
            result["knowledge_files"] = []
            
        return result