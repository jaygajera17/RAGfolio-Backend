import asyncio
import logging
from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.core.config import settings
from app.core.logger import get_logger
from app.rag.embedding import EmbeddingService, EMBEDDING_DIM

logger = get_logger(__name__)


class QdrantService:
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.embedding_svc = EmbeddingService()

        self.client = QdrantClient(
            url=settings.QDRANT_HOST,
            api_key=settings.QDRANT_API_KEY,
        )

        self._ensure_collection()

        # Used ONLY for read operations (similarity search)
        # aadd_documents has a confirmed bug: github.com/langchain-ai/langchain/issues/32195
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_svc.model,
            validate_collection_config=False,
        )

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: '{self.collection_name}'")
        else:
            logger.info(f"Collection already exists: '{self.collection_name}'")

    # ------------------------------------------------------------------ #
    #  Write — raw qdrant client bypasses LangChain bug                   #
    # ------------------------------------------------------------------ #

    async def add_documents(self, chunks: List[Document]) -> List[str]:
        """
        Embed chunks via EmbeddingService, then upsert directly via raw qdrant client.
        Payload format matches LangChain's expected keys so search still works:
            { "page_content": "...", "metadata": { ... } }
        """
        # 1. Embed all chunk texts
        texts = [chunk.page_content for chunk in chunks]
        vectors = await self.embedding_svc.embed_texts(texts)  # List[List[float]]

        # 2. Build PointStructs with LangChain-compatible payload
        uuids = [str(uuid4()) for _ in chunks]
        points = [
            PointStruct(
                id=uid,
                vector=vector,
                payload={
                    "page_content": chunk.page_content,   # LangChain CONTENT_KEY
                    "metadata": chunk.metadata,            # LangChain METADATA_KEY
                },
            )
            for uid, vector, chunk in zip(uuids, vectors, chunks)
        ]

        logger.info(f"Upserting {len(points)} points into '{self.collection_name}'...")

        # 3. Upsert — run sync client in thread so event loop stays free
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.collection_name,
            points=points,
        )

        logger.info(f"Successfully upserted {len(points)} points")
        return uuids

    def delete_by_source(self, source: str) -> None:
        """Delete all points matching a source file path."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(
                    key="metadata.source",   # nested key since metadata is a dict
                    match=MatchValue(value=source),
                )]
            ),
        )
        logger.info(f"Deleted existing points for source='{source}'")

    # ------------------------------------------------------------------ #
    #  Read — QdrantVectorStore works fine for search                     #
    # ------------------------------------------------------------------ #

    async def similarity_search(self, query: str, top_k: int = 5) -> List[dict]:
        results = await self.vector_store.asimilarity_search(query, k=top_k)
        return [{"text": doc.page_content, "metadata": doc.metadata} for doc in results]

    async def similarity_search_with_score(
        self, query: str, top_k: int = 5, score_threshold: float = 0.5
    ) -> List[dict]:
        results = await self.vector_store.asimilarity_search_with_score(query, k=top_k)
        return [
            {"text": doc.page_content, "score": round(score, 4), "metadata": doc.metadata}
            for doc, score in results
            if score >= score_threshold
        ]

    def as_retriever(self, top_k: int = 5):
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})

    # ------------------------------------------------------------------ #
    #  Collection info                                                     #
    # ------------------------------------------------------------------ #

    def get_collection_info(self) -> dict:
        info = self.client.get_collection(self.collection_name)
        return {
            "collection": self.collection_name,
            "points_count": info.points_count,
            "status": str(info.status),
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
        }

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: '{self.collection_name}'")