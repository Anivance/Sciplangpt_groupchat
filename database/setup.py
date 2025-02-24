from typing import Optional
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from .services.postgres_service import PostgresService
from .services.redis_service import RedisService
from .services.faiss_service import FAISSService
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(
        self,
        database_url: str,  # 改用 database_url 而不是 postgres_url
        redis_url: str,
        faiss_dimension: int,
        faiss_index_path: Optional[str] = None
    ):
        self.database_url = database_url
        self.redis_url = redis_url
        self.faiss_dimension = faiss_dimension
        self.faiss_index_path = faiss_index_path
        
        # 初始化 SQLAlchemy 引擎
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            future=True
        )
        
        # 創建會話工廠
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # 初始化服務
        self.postgres = PostgresService(self.async_session)
        self.redis = RedisService(self.redis_url)
        self.faiss = FAISSService(
            dimension=self.faiss_dimension,
            index_path=self.faiss_index_path
        )

    async def initialize(self):
        """初始化數據庫和服務"""
        try:
            # 創建數據庫表
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            # 初始化 Redis 連接
            await self.redis.initialize()
            
            # 初始化 FAISS 索引
            await self.faiss.initialize()
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database manager: {str(e)}")
            raise

    async def cleanup(self):
        """清理資源"""
        try:
            # 關閉 SQLAlchemy 引擎
            await self.engine.dispose()
            
            # 清理 Redis 連接
            await self.redis.cleanup()
            
            # 保存 FAISS 索引
            if self.faiss_index_path:
                await self.faiss.save_index()
            
            logger.info("Database manager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    async def get_session(self) -> AsyncSession:
        """獲取數據庫會話"""
        return self.async_session()