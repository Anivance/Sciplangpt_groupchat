"""Database package initialization."""

from .models.base import Base  # 先導入 Base
from .models.agent import Agent, KnowledgeFile  # 再導入具體模型
from .models.chat import ChatRoom, ChatMessage
from .models.embedding import Embedding
from .setup import DatabaseManager
from .services.postgres_service import PostgresService
from .services.redis_service import RedisService
from .services.faiss_service import FAISSService

__all__ = [
    'Base',
    'Agent',
    'KnowledgeFile',
    'ChatRoom',
    'ChatMessage',
    'Embedding',
    'DatabaseManager',
    'PostgresService',
    'RedisService',
    'FAISSService',
]

# 全局數據庫管理器實例
_db_manager = None

async def get_db() -> DatabaseManager:
    """獲取全局數據庫管理器實例."""
    global _db_manager
    if _db_manager is None:
        from config import settings
        _db_manager = DatabaseManager(
            database_url=settings.POSTGRES_URL,
            redis_url=settings.REDIS_URL,
            faiss_dimension=settings.VECTOR_DIMENSION,
            faiss_index_path=settings.FAISS_INDEX_PATH
        )
        await _db_manager.initialize()
    return _db_manager

async def close_db():
    """關閉數據庫連接."""
    global _db_manager
    if _db_manager is not None:
        await _db_manager.cleanup()
        _db_manager = None