from sqlalchemy import Column, DateTime, JSON, ForeignKey, LargeBinary
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

from .base import Base

class Embedding(Base):
    __tablename__ = 'embeddings'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'))
    vector_data = Column(LargeBinary)  # 存儲向量的二進制數據
    embedding_metadata = Column(JSON)  # 改名為 embedding_metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    agent = relationship("Agent", back_populates="embeddings")

    def __repr__(self):
        return f"<Embedding {self.id}>"