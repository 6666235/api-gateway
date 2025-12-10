"""
RAG 向量检索模块
支持：本地向量存储、Pinecone、Milvus
"""
import numpy as np
import hashlib
import json
import sqlite3
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import httpx
import asyncio

DB_PATH = "data.db"

@dataclass
class Document:
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict] = None
    score: float = 0.0

class EmbeddingProvider:
    """嵌入向量提供者基类"""
    async def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI 嵌入"""
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.dimension = 1536
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            return [self._simple_embed(t) for t in texts]
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts}
            )
            if response.status_code == 200:
                data = response.json()
                return [item["embedding"] for item in data["data"]]
        
        return [self._simple_embed(t) for t in texts]
    
    def _simple_embed(self, text: str) -> List[float]:
        """简单的本地嵌入（用于测试）"""
        np.random.seed(int(hashlib.md5(text.encode()).hexdigest()[:8], 16))
        return np.random.randn(self.dimension).tolist()

class LocalEmbedding(EmbeddingProvider):
    """本地嵌入（基于词频的简单实现）"""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # 基于文本哈希生成伪随机向量
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            vec = np.random.randn(self.dimension)
            # 归一化
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec.tolist())
        return embeddings

class VectorStore:
    """向量存储基类"""
    async def add(self, documents: List[Document]) -> bool:
        raise NotImplementedError
    
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        raise NotImplementedError
    
    async def delete(self, doc_ids: List[str]) -> bool:
        raise NotImplementedError

class SQLiteVectorStore(VectorStore):
    """SQLite 向量存储（适合小规模数据）"""
    def __init__(self, kb_id: int):
        self.kb_id = kb_id
    
    async def add(self, documents: List[Document]) -> bool:
        with sqlite3.connect(DB_PATH) as conn:
            for doc in documents:
                embedding_blob = np.array(doc.embedding, dtype=np.float32).tobytes()
                conn.execute("""
                    INSERT OR REPLACE INTO vector_documents 
                    (kb_id, chunk_id, content, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (self.kb_id, doc.id, doc.content, embedding_blob, 
                      json.dumps(doc.metadata or {})))
        return True
    
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT chunk_id, content, embedding, metadata 
                FROM vector_documents WHERE kb_id = ?
            """, (self.kb_id,)).fetchall()
        
        results = []
        for row in rows:
            if row["embedding"]:
                doc_vec = np.frombuffer(row["embedding"], dtype=np.float32)
                # 余弦相似度
                score = float(np.dot(query_vec, doc_vec) / 
                            (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8))
                results.append(Document(
                    id=row["chunk_id"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    score=score
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    async def delete(self, doc_ids: List[str]) -> bool:
        with sqlite3.connect(DB_PATH) as conn:
            for doc_id in doc_ids:
                conn.execute("DELETE FROM vector_documents WHERE chunk_id = ?", (doc_id,))
        return True

class PineconeVectorStore(VectorStore):
    """Pinecone 向量存储"""
    def __init__(self, api_key: str, index_name: str, environment: str = "us-east-1"):
        self.api_key = api_key
        self.index_name = index_name
        self.host = f"https://{index_name}-{environment}.svc.pinecone.io"
    
    async def add(self, documents: List[Document]) -> bool:
        vectors = [{
            "id": doc.id,
            "values": doc.embedding,
            "metadata": {"content": doc.content[:1000], **(doc.metadata or {})}
        } for doc in documents]
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.host}/vectors/upsert",
                headers={"Api-Key": self.api_key},
                json={"vectors": vectors}
            )
            return response.status_code == 200
    
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.host}/query",
                headers={"Api-Key": self.api_key},
                json={
                    "vector": query_embedding,
                    "topK": top_k,
                    "includeMetadata": True
                }
            )
            if response.status_code == 200:
                data = response.json()
                return [Document(
                    id=match["id"],
                    content=match.get("metadata", {}).get("content", ""),
                    metadata=match.get("metadata", {}),
                    score=match["score"]
                ) for match in data.get("matches", [])]
        return []
    
    async def delete(self, doc_ids: List[str]) -> bool:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.host}/vectors/delete",
                headers={"Api-Key": self.api_key},
                json={"ids": doc_ids}
            )
            return response.status_code == 200

class MilvusVectorStore(VectorStore):
    """Milvus 向量存储"""
    def __init__(self, host: str = "localhost", port: int = 19530, collection: str = "documents"):
        self.host = host
        self.port = port
        self.collection = collection
        self.api_url = f"http://{host}:{port}/v1"
    
    async def add(self, documents: List[Document]) -> bool:
        data = [{
            "id": doc.id,
            "vector": doc.embedding,
            "content": doc.content,
            "metadata": json.dumps(doc.metadata or {})
        } for doc in documents]
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.api_url}/entities/insert",
                json={"collectionName": self.collection, "data": data}
            )
            return response.status_code == 200
    
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.api_url}/search",
                json={
                    "collectionName": self.collection,
                    "vector": query_embedding,
                    "limit": top_k,
                    "outputFields": ["content", "metadata"]
                }
            )
            if response.status_code == 200:
                data = response.json()
                return [Document(
                    id=str(hit.get("id", "")),
                    content=hit.get("content", ""),
                    metadata=json.loads(hit.get("metadata", "{}")),
                    score=hit.get("distance", 0)
                ) for hit in data.get("data", [])]
        return []
    
    async def delete(self, doc_ids: List[str]) -> bool:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.api_url}/entities/delete",
                json={"collectionName": self.collection, "filter": f"id in {doc_ids}"}
            )
            return response.status_code == 200

class TextSplitter:
    """文本分块器"""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, text: str) -> List[str]:
        """按字符数分块，保持句子完整性"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                # 尝试在句子边界分割
                for sep in ['。', '！', '？', '.', '!', '?', '\n\n', '\n']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + 1
                        break
            
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        
        return [c for c in chunks if c]

class RAGEngine:
    """RAG 检索增强生成引擎"""
    def __init__(self, 
                 embedding_provider: EmbeddingProvider = None,
                 vector_store: VectorStore = None):
        self.embedding = embedding_provider or LocalEmbedding()
        self.vector_store = vector_store
        self.splitter = TextSplitter()
    
    async def index_document(self, content: str, doc_id: str = None, 
                            metadata: Dict = None) -> List[str]:
        """索引文档"""
        chunks = self.splitter.split(content)
        chunk_ids = []
        documents = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id or hashlib.md5(content.encode()).hexdigest()[:8]}_{i}"
            chunk_ids.append(chunk_id)
            documents.append(Document(
                id=chunk_id,
                content=chunk,
                metadata={**(metadata or {}), "chunk_index": i}
            ))
        
        # 批量生成嵌入
        embeddings = await self.embedding.embed([d.content for d in documents])
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        
        # 存储
        await self.vector_store.add(documents)
        return chunk_ids
    
    async def search(self, query: str, top_k: int = 5) -> List[Document]:
        """语义搜索"""
        query_embedding = (await self.embedding.embed([query]))[0]
        return await self.vector_store.search(query_embedding, top_k)
    
    async def query(self, question: str, top_k: int = 3) -> Tuple[str, List[Document]]:
        """RAG 查询：检索相关文档并生成上下文"""
        docs = await self.search(question, top_k)
        
        if not docs:
            return "", []
        
        context = "\n\n---\n\n".join([
            f"[来源 {i+1}] (相关度: {doc.score:.2f})\n{doc.content}"
            for i, doc in enumerate(docs)
        ])
        
        return context, docs

# 工厂函数
def create_rag_engine(kb_id: int, 
                      provider: str = "local",
                      config: Dict = None) -> RAGEngine:
    """创建 RAG 引擎"""
    config = config or {}
    
    # 选择嵌入提供者
    if provider == "openai":
        embedding = OpenAIEmbedding(config.get("api_key"))
    else:
        embedding = LocalEmbedding()
    
    # 选择向量存储
    store_type = config.get("vector_store", "sqlite")
    if store_type == "pinecone":
        vector_store = PineconeVectorStore(
            config.get("pinecone_api_key"),
            config.get("pinecone_index"),
            config.get("pinecone_env", "us-east-1")
        )
    elif store_type == "milvus":
        vector_store = MilvusVectorStore(
            config.get("milvus_host", "localhost"),
            config.get("milvus_port", 19530)
        )
    else:
        vector_store = SQLiteVectorStore(kb_id)
    
    return RAGEngine(embedding, vector_store)
