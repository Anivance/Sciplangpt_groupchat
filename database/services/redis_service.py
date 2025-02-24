from redis.asyncio import Redis
import json
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RedisService:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.default_ttl = 3600  # 1 小時
        self._redis = None
        
    async def initialize(self):
        """初始化 Redis 連接"""
        try:
            self._redis = Redis.from_url(self.redis_url, decode_responses=True)
            # 測試連接
            await self._redis.ping()
            logger.info("Redis connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise
        
    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """設置快取"""
        try:
            serialized = json.dumps(value)
            await self._redis.setex(key, ttl or self.default_ttl, serialized)
        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {e}")
            raise
        
    async def get_cache(self, key: str) -> Optional[Any]:
        """獲取快取"""
        try:
            data = await self._redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting cache for key {key}: {e}")
            return None
        
    async def delete_cache(self, key: str) -> None:
        """刪除快取"""
        try:
            await self._redis.delete(key)
        except Exception as e:
            logger.error(f"Error deleting cache for key {key}: {e}")
            raise
            
    async def cleanup(self):
        """清理資源"""
        try:
            if self._redis:
                await self._redis.aclose()
            logger.info("Redis connection closed successfully")
        except Exception as e:
            logger.error(f"Error during Redis cleanup: {e}")
            raise