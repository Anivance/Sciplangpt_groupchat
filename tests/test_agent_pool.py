import pytest
from agents.agent_pool_manager import AgentPoolManager, AgentCreateBase
from database.setup import DatabaseManager
import os
import shutil
from pathlib import Path
import uuid
from datetime import datetime

@pytest.fixture
async def test_db_config():
    """資料庫測試配置"""
    return {
        "database_url": "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db_test",
        "redis_url": "redis://localhost:6379/1",
        "faiss_dimension": 1536,
        "faiss_index_path": "test_data/faiss_indices"
    }

@pytest.fixture
async def db_manager(test_db_config):
    """資料庫管理器 fixture"""
    try:
        manager = DatabaseManager(
            database_url=test_db_config["database_url"],
            redis_url=test_db_config["redis_url"],
            faiss_dimension=test_db_config["faiss_dimension"],
            faiss_index_path=test_db_config["faiss_index_path"]
        )
        await manager.initialize()
        yield manager
    finally:
        if manager:
            await manager.cleanup()

@pytest.fixture
async def agent_pool(db_manager):
    """代理池管理器 fixture"""
    test_dir = Path("test_agent_pool")
    try:
        # 確保測試目錄乾淨
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True)
        
        # 清理測試資料庫中的代理
        await db_manager.postgres.clear_all_agents()
        
        # 創建代理池並設置資料庫管理器
        pool = AgentPoolManager(base_dir=str(test_dir))
        pool.set_db_manager(db_manager)
        await pool.initialize()
        yield pool
    finally:
        if 'pool' in locals():
            await pool.cleanup()
        # 清理測試目錄
        if test_dir.exists():
            shutil.rmtree(test_dir)
        # 清理測試資料庫
        await db_manager.postgres.clear_all_agents()

async def test_create_agent(agent_pool):
    """Test agent creation"""
    # Create test file
    test_file = "test_knowledge.txt"
    with open(test_file, "w") as f:
        f.write("This is test knowledge content")
    
    try:
        # Create agent
        agent_data = AgentCreateBase(
            name="test_agent",
            description="Test agent",
            type="dynamic",
            parameters={"test_param": "value"}
        )
        
        agent = await agent_pool.create_agent(agent_data, [test_file])
        
        # Verify agent creation
        assert agent["name"] == "test_agent"
        assert agent["type"] == "dynamic"
        assert len(agent["knowledge_files"]) == 1
        
        # Verify file system
        agent_dir = os.path.join("test_agent_pool", "test_agent")
        assert os.path.exists(agent_dir)
        assert os.path.exists(os.path.join(agent_dir, "docs"))
        
        # Verify database
        db_agent = await agent_pool.get_agent("test_agent")
        assert db_agent is not None
        assert db_agent["name"] == "test_agent"
        
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)

async def test_agent_crud_operations(agent_pool):
    """Test CRUD operations for agents"""
    # 使用唯一名稱
    unique_name = f"crud_test_agent_{uuid.uuid4().hex[:8]}"
    
    agent_data = AgentCreateBase(
        name=unique_name,
        description="Test CRUD operations",
        type="dynamic"
    )
    
    agent = await agent_pool.create_agent(agent_data, [])
    
    # Read agent
    loaded_agent = await agent_pool.get_agent(unique_name)
    assert loaded_agent is not None
    assert loaded_agent["name"] == unique_name
    
    # Update agent
    updates = {
        "description": "Updated description"
    }
    updated_agent = await agent_pool.update_agent(unique_name, updates)
    assert updated_agent["description"] == "Updated description"
    
    # Delete agent
    success = await agent_pool.delete_agent(unique_name)
    assert success
    
    # Verify deletion
    deleted_agent = await agent_pool.get_agent(unique_name)
    assert deleted_agent is None


async def test_agent_instance(agent_pool):
    """Test agent instance creation and functionality"""
    # 使用唯一名稱
    unique_name = f"instance_test_agent_{uuid.uuid4().hex[:8]}"
    
    agent_data = AgentCreateBase(
        name=unique_name,
        description="Test agent instance",
        type="dynamic"
    )
    
    await agent_pool.create_agent(agent_data, [])
    
    # Get agent instance
    agent_instance = await agent_pool.get_agent_instance(unique_name)
    assert agent_instance is not None
    assert agent_instance.name == unique_name
    
    # Test agent functionality
    test_message = "Hello, agent!"
    response = await agent_instance.process_query(test_message)
    assert response is not None