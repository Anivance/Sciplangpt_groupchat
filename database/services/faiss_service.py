import faiss
import numpy as np
from typing import List, Dict, Any
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FAISSService:
    def __init__(self, dimension: int, index_path: str):
        self.dimension = dimension
        self.index_path = Path(index_path) / "faiss.index"
        self.index = None
        
    async def initialize(self):
        """初始化 FAISS 索引"""
        try:
            self.index = self._load_or_create_index()
            logger.info("FAISS index initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
            
    def _load_or_create_index(self) -> faiss.Index:
        """載入現有索引或創建新的索引"""
        try:
            if self.index_path.exists():
                return faiss.read_index(str(self.index_path))
        except Exception as e:
            logger.warning(f"Error loading existing index: {e}")
            
        # 創建新索引
        index = faiss.IndexFlatIP(self.dimension)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        return index
    
    async def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """添加向量到索引"""
        if not self.index:
            raise RuntimeError("FAISS index not initialized")
            
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量維度不匹配：預期 {self.dimension}，實際得到 {vectors.shape[1]}")
            
        # 正規化向量
        faiss.normalize_L2(vectors)
        
        # 添加到索引
        self.index.add(vectors)
        
        # 保存索引到文件
        await self.save_index()
        logger.info(f"Added {len(vectors)} vectors to index")
    
    async def search(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        if not self.index:
            raise RuntimeError("FAISS index not initialized")
            
        if query.shape[1] != self.dimension:
            raise ValueError(f"查詢向量維度不匹配：預期 {self.dimension}，實際得到 {query.shape[1]}")
            
        # 正規化查詢向量
        faiss.normalize_L2(query)
        
        # 搜索
        D, I = self.index.search(query, k)
        
        # 格式化結果
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx != -1:  # 有效結果
                results.append({
                    "id": int(idx),
                    "score": float(score)
                })
                
        return results
        
    async def save_index(self) -> None:
        """保存索引到文件"""
        if self.index:
            try:
                faiss.write_index(self.index, str(self.index_path))
                logger.info("FAISS index saved successfully")
            except Exception as e:
                logger.error(f"Error saving FAISS index: {e}")
                raise
        
    async def cleanup(self) -> None:
        """清理資源"""
        try:
            await self.save_index()
            if self.index_path.exists():
                os.remove(self.index_path)
                logger.info("FAISS index file removed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self.index = None