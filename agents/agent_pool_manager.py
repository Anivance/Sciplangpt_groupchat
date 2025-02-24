from fastapi import HTTPException
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import create_embedding
from agents.base_agent import BaseAgent
from pydantic import BaseModel, Field
import numpy as np

# 導入數據庫服務
from database.setup import DatabaseManager
from database.config import db_config

logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    name: str
    description: str
    template_name: Optional[str] = None
    base_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="Base prompt for the agent"
    )
    type: Optional[str] = Field(default="dynamic", description="Type of the agent")
    query_templates: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class AgentCreateBase(BaseModel):
    name: str
    description: str
    template_name: Optional[str] = None
    type: Optional[str] = "dynamic"
    parameters: Dict[str, Any] = Field(default_factory=dict)

    def to_config(self) -> AgentConfig:
        """Convert creation request to full configuration"""
        base_prompts = {
            "dynamic": "You are a dynamic processing agent capable of handling various tasks.",
            "qa": "You are a specialized Q&A agent designed to provide accurate answers based on available knowledge.",
            "research": """You are a research analysis agent focused on deep analysis and knowledge synthesis. 
                         When your confidence is low, you can actively search for additional information to enhance your knowledge."""
        }
        
        default_parameters = {
            "dynamic": {},
            "qa": {},
            "research": {
                "confidence_threshold": 0.7,
                "max_research_papers": 5,
                "research_wait_time": 2.0
            }
        }
        
        merged_parameters = {
            **default_parameters.get(self.type, {}),
            **self.parameters
        }
        
        return AgentConfig(
            name=self.name,
            description=self.description,
            template_name=self.template_name,
            base_prompt=base_prompts.get(self.type, base_prompts["dynamic"]),
            type=self.type,
            parameters=merged_parameters
        )

class Agent(BaseModel):
    name: str
    description: str
    created_at: str
    type: Optional[str] = None

class AgentPoolManager:
    def __init__(self, base_dir: str = "agent_pool", db_manager: Optional[DatabaseManager] = None):
        """
        初始化 AgentPoolManager
        
        Args:
            base_dir: 代理池的基礎目錄
            db_manager: 可選的資料庫管理器實例。如果不提供，將在 initialize 時使用配置創建
        """
        self.base_dir = Path(base_dir)
        self.agents_file = self.base_dir / "agents.json"
        self.db = db_manager
        self._initialize_pool()

    def set_db_manager(self, db_manager: DatabaseManager) -> None:
        """設置數據庫管理器

        Args:
            db_manager (DatabaseManager): 數據庫管理器實例
        """
        self.db = db_manager

    async def initialize(self):
        """初始化資料庫和檔案系統"""
        try:
            # 如果沒有提供 db_manager，則使用配置創建一個
            if self.db is None:
                self.db = DatabaseManager(
                    database_url=db_config.POSTGRES_URL,
                    redis_url=db_config.REDIS_URL,
                    faiss_dimension=db_config.VECTOR_DIMENSION,
                    faiss_index_path=db_config.FAISS_INDEX_PATH
                )
            
            # 初始化資料庫
            await self.db.initialize()
            logger.info("Database initialized successfully")
            
            # 初始化檔案系統
            self._initialize_pool()
            logger.info("Agent pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agent pool: {str(e)}")
            raise
            
    def _initialize_pool(self) -> None:
        """初始化代理池目錄"""
        self.base_dir.mkdir(exist_ok=True)
        if not self.agents_file.exists():
            with open(self.agents_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    async def cleanup(self):
        """清理資源"""
        try:
            if self.db:
                await self.db.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    def __del__(self):
        """確保清理資源"""
        try:
            import asyncio
            if hasattr(self, 'db') and self.db is not None:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
                else:
                    loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error in cleanup during deletion: {str(e)}")

    async def ensure_embeddings(self, docs_dir: Union[str, Path]) -> bool:
        """Ensure embedding files exist and store in FAISS"""
        try:
            docs_dir = Path(docs_dir)
            embedding_dir = docs_dir / "embedding"
            if not embedding_dir.exists():
                logger.info(f"Generating embeddings for {docs_dir}...")
                embeddings = await create_embedding.process_files_async(str(docs_dir))
                
                # Store embeddings in FAISS
                if embeddings:
                    vectors = np.array([emb['embedding'] for emb in embeddings])
                    metadata = [emb['metadata'] for emb in embeddings]
                    await self.db.faiss.add_vectors(vectors, metadata)
            return True
        except Exception as e:
            logger.error(f"Error ensuring embeddings: {str(e)}")
            return False
    
    async def create_agent(self, agent_data: AgentCreateBase, knowledge_files: List[str]) -> Dict:
        """建立新的代理"""
        try:
            logger.info(f"Creating agent: {agent_data.name}")
            config = agent_data.to_config()
            
            # 檢查數據庫中是否已存在同名代理
            existing_agent = await self.db.postgres.get_agent_by_name(config.name)
            if existing_agent:
                logger.warning(f"Agent with name '{config.name}' already exists in database")
                raise HTTPException(status_code=400, detail=f"Agent '{config.name}' already exists")

            # 創建目錄結構
            agent_dir = self.base_dir / config.name
            if agent_dir.exists():
                logger.warning(f"Agent directory '{agent_dir}' already exists")
                raise HTTPException(status_code=400, detail=f"Agent '{config.name}' already exists")
            
            agent_dir.mkdir(parents=True)
            docs_dir = agent_dir / "docs"
            docs_dir.mkdir()
            
            # 複製知識文件並建立資料庫記錄
            knowledge_file_records = []
            for file_path in knowledge_files:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, str(docs_dir))
                    
                    # 儲存文件資訊到資料庫
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        knowledge_file = await self.db.postgres.create_knowledge_file({
                            'filename': os.path.basename(file_path),
                            'content': content,
                            'file_type': os.path.splitext(file_path)[1][1:]
                        })
                        knowledge_file_records.append(knowledge_file)
            
            # 生成嵌入向量
            if not await self.ensure_embeddings(docs_dir):
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to generate embeddings for agent {config.name}"
                )
            
            # 創建代理配置
            agent_config = {
                "name": config.name,
                "description": config.description,
                "created_at": datetime.now().isoformat(),
                "base_prompt": config.base_prompt,
                "type": config.type,
                "template_name": config.template_name,
                "parameters": config.parameters,
                "knowledge_files": [os.path.basename(f) for f in knowledge_files],
                "docs_dir": str(docs_dir)
            }
            
            # 儲存到檔案系統
            config_path = agent_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(agent_config, f, ensure_ascii=False, indent=2)
            
            # 準備僅保留資料庫允許的欄位
            db_agent_data = {
                "name": config.name,
                "description": config.description,
                "type": config.type,
                "base_prompt": config.base_prompt,
                "template_name": config.template_name,
                "parameters": config.parameters,
            }
            
            # 儲存到資料庫
            db_agent = await self.db.postgres.create_agent({
                **db_agent_data,
                'knowledge_files': knowledge_file_records
            })
            
            # 快取代理資料
            await self.db.redis.set_cache(
                f"agent:{db_agent.id}",
                agent_config,
                ttl=3600
            )
            
            # 更新代理列表
            agents = await self._read_agents_file()
            agents.append(Agent(
                name=config.name,
                description=config.description,
                created_at=agent_config["created_at"],
                type=config.type
            ).dict())
            
            await self._write_agents_file(agents)
            logger.info(f"Successfully created agent: {config.name}")
            return agent_config
            
        except HTTPException:
            # 清理已創建的目錄
            if agent_dir and agent_dir.exists():
                shutil.rmtree(agent_dir)
            raise
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            if 'agent_dir' in locals() and agent_dir and agent_dir.exists():
                shutil.rmtree(agent_dir)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        """取得代理配置"""
        try:
            # 先嘗試從快取取得
            cached_agent = await self.db.redis.get_cache(f"agent:{name}")
            if cached_agent:
                return cached_agent
            
            # 從資料庫取得
            db_agent = await self.db.postgres.get_agent_dict_by_name(name)
            if db_agent:
                await self.db.redis.set_cache(f"agent:{name}", db_agent)
                return db_agent
            
            # 從檔案系統取得
            logger.info(f"Getting agent configuration for: {name}")
            agent_dir = self.base_dir / name
            config_path = agent_dir / "config.json"
            
            if not config_path.exists():
                return None
                    
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                safe_config = {
                    "name": config.get("name", name),
                    "description": config.get("description", ""),
                    "type": config.get("type", "dynamic"),
                    "template_name": config.get("template_name"),
                    "base_prompt": config.get("base_prompt", ""),
                    "created_at": config.get("created_at", datetime.now().isoformat()),
                    "parameters": config.get("parameters", {}),
                    "docs_dir": str(config.get("docs_dir", str(agent_dir / "docs"))),
                    "knowledge_files": config.get("knowledge_files", []),
                    "query_templates": config.get("query_templates", [])
                }
                
                # 儲存到資料庫供未來使用
                db_agent_data = {k: v for k, v in safe_config.items() 
                                if k not in ['docs_dir', 'knowledge_files', 'query_templates']}
                created_agent = await self.db.postgres.create_agent(db_agent_data)
                
                # 更新快取
                await self.db.redis.set_cache(f"agent:{name}", safe_config)
                
                return safe_config
                
            except Exception as e:
                logger.error(f"Error reading config file for agent {name}: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting agent configuration for {name}: {str(e)}")
            return None

    async def get_agent_instance(self, name: str) -> Optional[BaseAgent]:
        """取得代理實例"""
        try:
            config = await self.get_agent(name)
            if not config:
                return None
            
            # 根據類型創建代理實例
            agent_class = None
            if config.get("type") == "question_based":
                from agents.question_based_agent import QuestionBasedResearchAgent
                agent_class = QuestionBasedResearchAgent
            elif config.get("type") == "research":
                from agents.research_agent import ResearchAgent
                agent_class = ResearchAgent
            else:
                from agents.dynamic_agent import DynamicAgent
                agent_class = DynamicAgent
            
            # 支援 docs_dir 為 None 的情況
            docs_dir = config.get("docs_dir")
            if not docs_dir and self.base_dir:
                docs_dir = str(self.base_dir / name / "docs")
            
            agent = agent_class(
                name=config["name"],
                base_prompt=config.get("base_prompt", ""),
                docs_dir=docs_dir,
                description=config.get("description", ""),
                parameters=config.get("parameters", {}),
                query_templates=config.get("query_templates", [])
            )
            
            # 確保嵌入向量可用
            if docs_dir:
                await self.ensure_embeddings(docs_dir)
            
            return agent
                
        except Exception as e:
            logger.error(f"Error getting agent instance: {str(e)}")
            return None
    
    async def list_agents(self) -> List[Dict]:
        """List all agents"""
        try:
            # Try database first
            db_agents = await self.db.postgres.list_agents()
            if db_agents:
                return [agent.to_dict() for agent in db_agents]
            
            # Fallback to file system
            agents = await self._read_agents_file()
            formatted_agents = []
            for agent in agents:
                if isinstance(agent, dict):
                    formatted_agents.append({
                        "name": agent.get("name", ""),
                        "description": agent.get("description", ""),
                        "created_at": agent.get("created_at", ""),
                        "type": agent.get("type", "")
                    })
                elif isinstance(agent, Agent):
                    formatted_agents.append({
                        "name": agent.name,
                        "description": agent.description,
                        "created_at": agent.created_at,
                        "type": agent.type
                    })
            
            # Store in database for future use
            for agent in formatted_agents:
                await self.db.postgres.create_agent(agent)
                
            return formatted_agents
            
        except Exception as e:
            logger.error(f"Error listing agents: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")
        
    async def delete_agent(self, name: str) -> bool:
        """Delete specified agent"""
        try:
            # Delete from database
            await self.db.postgres.delete_agent(name)
            
            # Delete from cache
            await self.db.redis.delete_cache(f"agent:{name}")
            
            # Delete from file system
            agent_dir = self.base_dir / name
            if agent_dir.exists():
                shutil.rmtree(agent_dir)
            
            # Update agents list
            agents = await self._read_agents_file()
            agents = [a for a in agents if a["name"] != name]
            await self._write_agents_file(agents)
            
            logger.info(f"Successfully deleted agent: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting agent: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")
    
    async def update_agent(self, name: str, updates: Dict) -> Optional[Dict]:
        """Update agent configuration"""
        try:
            # Update in database
            updated_agent = await self.db.postgres.update_agent(name, updates)
            
            if updated_agent:
                # 檢查返回類型並轉換為字典
                agent_dict = updated_agent.to_dict() if hasattr(updated_agent, 'to_dict') else updated_agent
                
                # Update cache
                await self.db.redis.delete_cache(f"agent:{name}")
                await self.db.redis.set_cache(
                    f"agent:{name}",
                    agent_dict
                )
            
                # Update file system
                agent_dir = self.base_dir / name
                config_path = agent_dir / "config.json"
            
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                
                    config.update(updates)
                
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)
                
                    if "name" in updates or "description" in updates:
                        agents = await self._read_agents_file()
                        for agent in agents:
                            if agent["name"] == name:
                                agent.update({
                                    "name": updates.get("name", name),
                                    "description": updates.get("description", agent["description"])
                                })
                        await self._write_agents_file(agents)
                
                    return config
            
                return agent_dict
            return None
            
        except Exception as e:
            logger.error(f"Error updating agent: {str(e)}")
            # 添加更詳細的錯誤信息
            error_msg = f"Error updating agent '{name}': {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    async def _read_agents_file(self) -> List[Dict]:
        """Read agents file with error handling"""
        try:
            if not self.agents_file.exists():
                return []
            with open(self.agents_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading agents file: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error reading agents file: {str(e)}"
            )

    async def _write_agents_file(self, agents: List[Dict]) -> None:
        """Write agents file with error handling"""
        try:
            with open(self.agents_file, 'w', encoding='utf-8') as f:
                json.dump(agents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error writing agents file: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error writing agents file: {str(e)}"
            )
            
    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.db.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    def __del__(self):
        """Ensure cleanup on deletion"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.cleanup())
            else:
                loop.run_until_complete(self.cleanup())
        except Exception as e:
            logger.error(f"Error in cleanup during deletion: {str(e)}")