from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from contextlib import asynccontextmanager
from routes.agent_routes import router as agent_router
from routes.chat_routes import router as chat_router
from database.setup import DatabaseManager
from config import Settings

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加載配置
settings = Settings()

# 創建數據庫管理器
db_manager = DatabaseManager(
    postgres_url=settings.POSTGRES_URL,
    redis_url=settings.REDIS_URL,
    faiss_dimension=settings.VECTOR_DIMENSION,
    faiss_index_path=settings.FAISS_INDEX_PATH
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程序生命週期管理"""
    try:
        # 啟動時初始化數據庫
        logger.info("Initializing database connections...")
        await db_manager.initialize()
        logger.info("Database initialization completed")
        
        yield
        
    finally:
        # 關閉時清理資源
        logger.info("Cleaning up resources...")
        await db_manager.cleanup()
        logger.info("Cleanup completed")

app = FastAPI(lifespan=lifespan)

# CORS 配置
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 確保存在上傳文件目錄
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 確保存在 FAISS 索引目錄
os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)

# 設置靜態文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 設置模板
templates = Jinja2Templates(directory="templates")

# 將數據庫管理器添加到應用狀態
async def get_db():
    return db_manager

app.dependency_overrides[get_db] = get_db

# 註冊路由器
app.include_router(agent_router)
app.include_router(chat_router)

@app.get("/")
async def home(request: Request):
    """首頁"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat")
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/knowledge-base")
async def knowledge_base(request: Request):
    # 從數據庫獲取統計信息
    try:
        stats = await db_manager.postgres.get_knowledge_base_stats()
        knowledge_chunks = await db_manager.postgres.get_knowledge_chunks()
        
        return templates.TemplateResponse(
            "knowledge_base.html", 
            {
                "request": request, 
                "stats": stats,
                "knowledge_base": knowledge_chunks
            }
        )
    except Exception as e:
        logger.error(f"Error getting knowledge base data: {e}")
        # fallback to default values if database query fails
        return templates.TemplateResponse(
            "knowledge_base.html", 
            {
                "request": request, 
                "stats": {
                    "file_count": 0,
                    "total_chunks": 0,
                    "total_size": "0MB"
                },
                "knowledge_base": {}
            }
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        ws_max_size=1024*1024,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*"
    )