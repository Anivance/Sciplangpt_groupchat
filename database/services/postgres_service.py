from typing import Dict, List, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, delete
import logging
from sqlalchemy.orm import selectinload, joinedload

from ..models import Base, Agent, ChatRoom, ChatMessage, Embedding, KnowledgeFile

logger = logging.getLogger(__name__)

class PostgresService:
    def __init__(self, session_maker):
        self.session_maker = session_maker

    def get_session(self) -> AsyncSession:
        """取得資料庫連線"""
        return self.session_maker()

    async def create_tables(self):
        """創建所有資料表"""
        async with self.get_session() as session:
            async with session.begin():
                await session.run_sync(Base.metadata.create_all)

    async def create_knowledge_file(self, file_data: Dict[str, Any]) -> KnowledgeFile:
        """創建知識文件記錄"""
        async with self.get_session() as session:
            async with session.begin():
                try:
                    knowledge_file = KnowledgeFile(**file_data)
                    session.add(knowledge_file)
                    await session.flush()
                    await session.refresh(knowledge_file)
                    return knowledge_file
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Error creating knowledge file: {e}")
                    raise

    async def create_agent(self, agent_data: Dict[str, Any]) -> Agent:
        """創建新的 Agent"""
        async with self.get_session() as session:
            async with session.begin():
                try:
                    agent = Agent(**agent_data)
                    session.add(agent)
                    await session.flush()
                    await session.refresh(agent, ['id', 'name', 'description', 'type', 'base_prompt', 'parameters'])
                    return agent
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Error creating agent: {e}")
                    raise

    async def get_knowledge_file(self, file_id: str) -> Optional[KnowledgeFile]:
        """根據 ID 取得知識文件"""
        async with self.get_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(KnowledgeFile).where(KnowledgeFile.id == file_id)
                )
                return result.scalar_one_or_none()

    async def list_knowledge_files(self, agent_id: Optional[str] = None) -> List[KnowledgeFile]:
        """列出知識文件"""
        async with self.get_session() as session:
            async with session.begin():
                query = select(KnowledgeFile)
                if agent_id:
                    query = query.join(Agent.knowledge_files).where(Agent.id == agent_id)
                result = await session.execute(query)
                return result.scalars().all()

    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """根據 ID 取得 Agent"""
        async with self.get_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(Agent).where(Agent.id == agent_id)
                )
                return result.scalar_one_or_none()

    async def get_agent_by_name(self, name: str) -> Optional[Agent]:
        """根據名稱取得 Agent，返回 Agent 模型對象"""
        async with self.get_session() as session:
            async with session.begin():
                try:
                    # 使用 selectinload 預加載 knowledge_files 關聯
                    result = await session.execute(
                        select(Agent)
                        .options(selectinload(Agent.knowledge_files))
                        .where(Agent.name == name)
                    )
                    return result.scalar_one_or_none()
                except Exception as e:
                    logger.error(f"Error getting agent by name: {e}")
                    raise

    async def get_agent_dict_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根據名稱取得 Agent，返回字典格式"""
        agent = await self.get_agent_by_name(name)
        if agent:
            return agent.to_dict()
        return None

    async def list_agents(self) -> List[Agent]:
        """列出所有 Agents"""
        async with self.get_session() as session:
            async with session.begin():
                result = await session.execute(select(Agent))
                return result.scalars().all()

    async def update_agent(self, name: str, updates: Dict[str, Any]) -> Optional[Agent]:
        """根據名稱更新代理"""
        async with self.get_session() as session:
            # 先查詢代理
            query = select(Agent).where(Agent.name == name)
            result = await session.execute(query)
            agent = result.scalar_one_or_none()
            
            if not agent:
                return None
            
            # 更新找到的代理屬性
            for key, value in updates.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            
            await session.commit()
            await session.refresh(agent)
            return agent

    async def delete_agent(self, name: str) -> bool:
        """根據名稱刪除代理"""
        async with self.get_session() as session:
            # 先查詢代理是否存在
            query = select(Agent).where(Agent.name == name)
            result = await session.execute(query)
            agent = result.scalar_one_or_none()
            
            if not agent:
                # 代理不存在，無需刪除
                return False
            
            # 刪除找到的代理
            await session.delete(agent)
            await session.commit()
            return True

    async def clear_all_agents(self):
        """清除所有代理（用於測試）"""
        try:
            async with self.get_session() as session:
                from sqlalchemy import delete
                from ..models import Agent
                
                # 刪除所有 Agent 記錄
                await session.execute(delete(Agent))
                await session.commit()
                
                # 還可以清理相關表
                # await session.execute(delete(Embedding))
                # await session.execute(delete(ChatMessage))
                # await session.commit()
        except Exception as e:
            logger.error(f"Error clearing agents: {str(e)}")

    async def create_chat_room(self, room_data: Dict[str, Any]) -> ChatRoom:
        """創建聊天室"""
        async with self.get_session() as session:
            async with session.begin():
                chat_room = ChatRoom(**room_data)
                session.add(chat_room)
                await session.flush()
                await session.refresh(chat_room)
                return chat_room

    async def get_chat_room(self, room_id: str) -> Optional[ChatRoom]:
        """獲取聊天室"""
        async with self.get_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(ChatRoom).where(ChatRoom.id == room_id)
                )
                return result.scalar_one_or_none()

    async def create_chat_message(self, message_data: Dict[str, Any]) -> ChatMessage:
        """創建聊天消息"""
        async with self.get_session() as session:
            async with session.begin():
                message = ChatMessage(**message_data)
                session.add(message)
                await session.flush()
                await session.refresh(message)
                return message

    async def get_chat_history(self, room_id: str) -> List[ChatMessage]:
        """獲取聊天歷史"""
        async with self.get_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(ChatMessage)
                    .where(ChatMessage.room_id == room_id)
                    .order_by(ChatMessage.created_at)
                )
                return result.scalars().all()

    async def save_embedding(self, embedding_data: Dict[str, Any]) -> Embedding:
        """保存 Embedding"""
        async with self.get_session() as session:
            async with session.begin():
                embedding = Embedding(**embedding_data)
                session.add(embedding)
                await session.flush()
                await session.refresh(embedding)
                return embedding

    async def get_agent_embeddings(self, agent_id: str) -> List[Embedding]:
        """獲取 Agent 的所有 Embeddings"""
        async with self.get_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(Embedding).where(Embedding.agent_id == agent_id)
                )
                return result.scalars().all()