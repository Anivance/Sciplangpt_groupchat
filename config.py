from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os

class Settings(BaseSettings):
    # 應用配置
    DEBUG: bool = Field(default=True)
    HOST: str = Field(default="127.0.0.1")
    PORT: int = Field(default=8000)
    
    # 數據庫配置
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
    
    # API 鍵配置
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
        extra = "ignore"  # 允許額外的環境變量

# 測試環境配置
class TestSettings(Settings):
    # 使用測試數據庫
    POSTGRES_DB: str = Field(default="agent_db_test")
    REDIS_DB: int = Field(default=1)
    FAISS_INDEX_PATH: str = Field(default="test_data/faiss_indices")
    
    class Config:
        env_file = ".env.test"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

# 根據環境選擇配置
settings = TestSettings() if os.getenv("TESTING") == "True" else Settings()
test_settings = TestSettings()