import pytest
import numpy as np
import os
from pathlib import Path
import uuid
from datetime import datetime
from sqlalchemy import text
from database.setup import DatabaseManager
from database.models import Agent, Base

# 建立測試目錄
TEST_ROOT = Path(__file__).parent
TEST_DATA_DIR = TEST_ROOT / "test_data"
FAISS_INDEX_DIR = TEST_DATA_DIR / "faiss_indices"

@pytest.fixture(scope="session")
def test_config():
    """測試配置"""
    return {
        "POSTGRES_URL": "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db_test",
        "REDIS_URL": "redis://localhost:6379/1",
        "FAISS_INDEX_PATH": str(FAISS_INDEX_DIR),
        "VECTOR_DIMENSION": 1536
    }

@pytest.fixture(autouse=True)
async def setup_database(test_config):
    """設置測試資料庫"""
    manager = DatabaseManager(
        database_url=test_config["POSTGRES_URL"],
        redis_url=test_config["REDIS_URL"],
        faiss_dimension=test_config["VECTOR_DIMENSION"],
        faiss_index_path=test_config["FAISS_INDEX_PATH"]
    )
    await manager.initialize()
    
    # 清理所有表格
    async with manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    await manager.cleanup()

@pytest.fixture
async def db_manager(test_config):
    """資料庫管理器"""
    manager = DatabaseManager(
        database_url=test_config["POSTGRES_URL"],
        redis_url=test_config["REDIS_URL"],
        faiss_dimension=test_config["VECTOR_DIMENSION"],
        faiss_index_path=test_config["FAISS_INDEX_PATH"]
    )
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.cleanup()

def generate_unique_name():
    """生成唯一的測試名稱"""
    return f"test_agent_{uuid.uuid4().hex[:8]}"

@pytest.mark.asyncio
async def test_postgres_connection(db_manager):
    """測試 PostgreSQL 連接"""
    # 使用唯一名稱建立 agent
    test_name = generate_unique_name()
    agent_data = {
        "name": test_name,
        "description": "Test agent for database",
        "type": "dynamic",
        "base_prompt": "You are a test agent",
        "parameters": {"test_param": "value"}
    }
    
    # 測試建立 agent
    agent = await db_manager.postgres.create_agent(agent_data)
    assert isinstance(agent, Agent)
    assert agent.name == test_name
    assert agent.type == "dynamic"
    
    # 測試讀取 agent
    loaded_agent = await db_manager.postgres.get_agent_by_name(test_name)
    assert loaded_agent is not None
    assert loaded_agent.name == test_name
    assert loaded_agent.description == "Test agent for database"

@pytest.mark.asyncio
async def test_redis_connection(db_manager):
    """測試 Redis 連接"""
    test_key = f"test_key_{uuid.uuid4().hex[:8]}"
    test_data = {"name": "test", "value": 123}
    
    # 測試設置快取
    await db_manager.redis.set_cache(test_key, test_data)
    
    # 測試獲取快取
    cached_data = await db_manager.redis.get_cache(test_key)
    assert cached_data == test_data
    
    # 測試刪除快取
    await db_manager.redis.delete_cache(test_key)
    deleted_data = await db_manager.redis.get_cache(test_key)
    assert deleted_data is None

@pytest.mark.asyncio
async def test_faiss_service(db_manager):
    """測試 FAISS 服務"""
    # 建立測試向量
    vector_dimension = db_manager.faiss.dimension
    test_vectors = np.random.rand(5, vector_dimension).astype('float32')
    test_metadata = [{"id": str(i), "text": f"Test vector {i}"} for i in range(5)]
    
    # 測試添加向量
    await db_manager.faiss.add_vectors(test_vectors, test_metadata)
    
    # 測試搜索
    query_vector = test_vectors[0].reshape(1, -1)
    results = await db_manager.faiss.search(query_vector, k=3)
    
    assert len(results) == 3
    assert isinstance(results[0], dict)
    assert "score" in results[0]
    assert results[0]["score"] > 0.9  # 第一個結果應該是查詢向量本身