import pytest
import os
import sys
from pathlib import Path
import shutil

# Add project root to Python path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Set test environment variables
os.environ["TESTING"] = "True"
os.environ["POSTGRES_DB"] = "agent_db_test"
os.environ["REDIS_DB"] = "1"

@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        "POSTGRES_URL": "postgresql://postgres:postgres@localhost:5432/agent_db_test",
        "REDIS_URL": "redis://localhost:6379/1",
        "FAISS_INDEX_PATH": str(ROOT_DIR / "tests" / "test_data" / "faiss_indices"),
        "VECTOR_DIMENSION": 1536
    }

@pytest.fixture(autouse=True)
async def cleanup(db_manager):
    yield
    # 清理文件系統
    if os.path.exists("test_agent_pool"):
        shutil.rmtree("test_agent_pool")
    # 清理資料庫
    async with db_manager.get_session() as session:
        await session.execute(delete(Agent))
        await session.commit()

def pytest_configure(config):
    """Create necessary test directories"""
    test_data_dir = ROOT_DIR / "tests" / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    (test_data_dir / "faiss_indices").mkdir(parents=True, exist_ok=True)

# Set default event loop policy for macOS
def pytest_configure_asyncio():
    """Configure asyncio for testing"""
    import asyncio
    import platform
    if platform.system() == "Darwin":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())