from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class DatabaseConfig(BaseSettings):
    # 数据库配置
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="agent_db")
    
    # Redis 配置
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    
    # FAISS 配置
    FAISS_INDEX_PATH: str = Field(default="data/faiss_indices")
    VECTOR_DIMENSION: int = Field(default=1536)
    
    # OpenAI 和其他配置（添加这些字段）
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    AUTOGEN_USE_DOCKER: Optional[str] = Field(default="0")
    
    @property
    def POSTGRES_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # 允许额外的环境变量

# 创建配置实例
db_config = DatabaseConfig()

# 测试环境配置
class TestDatabaseConfig(DatabaseConfig):
    POSTGRES_DB: str = Field(default="agent_db_test")
    REDIS_DB: int = Field(default=1)
    FAISS_INDEX_PATH: str = Field(default="test_data/faiss_indices")

# 测试配置实例
test_db_config = TestDatabaseConfig()