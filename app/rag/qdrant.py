import asyncio
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
        
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """
        Create keyword indexes on filterable payload fields.

        Why needed:
            Qdrant refuses Filter queries on un-indexed fields.
            PayloadSchemaType.KEYWORD = exact-match string index.
            PayloadSchemaType.INTEGER  = range/match index for numbers.
        """
        from qdrant_client.models import PayloadSchemaType
        indexes = [
            ("metadata.modality",      PayloadSchemaType.KEYWORD),
            ("metadata.source",        PayloadSchemaType.KEYWORD),
            ("metadata.region_type",   PayloadSchemaType.KEYWORD),
            ("metadata.page_num",      PayloadSchemaType.INTEGER),
        ]

        for field, schema_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema_type,
                )
                logger.info(f"Payload index ensured: '{field}'")
            except Exception as e:
                # Qdrant raises if index already exists in some versions
                logger.debug(f"Index '{field}' already exists or skipped: {e}")
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
                    "page_content": chunk.page_content,  
                    "metadata": chunk.metadata,    # includes modality="text"
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
        query_vector =  await self.embedding_svc.embed_query(query)
        
        response = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold
        )
        hits = response.points
        return [
            {
                "text":     hit.payload.get("page_content", ""),
                "score":    round(hit.score, 4),
                "metadata": hit.payload.get("metadata", {}),
                # metadata["modality"] tells you "text" or "image"
            }
            for hit in hits
        ]
    
    async def similarity_search_by_modality(
        self,
        query: str,
        modality: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[dict]:
        """
        Retrive only text chunks or only images matching the query,
        useful to display results in separate UI sections.
        """
        query_vector = await self.embedding_svc.embed_query(query)  
        
        response = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            query_filter=Filter(
                must=[FieldCondition(
                    key="metadata.modality",
                    match=MatchValue(value=modality),
                )]
            ),
        )
        hits = response.points
        return [
            {
                "text":     hit.payload.get("page_content", ""),
                "score":    round(hit.score, 4),
                "metadata": hit.payload.get("metadata", {}),
            }
            for hit in hits
        ]

    async def similarity_search_multimodal(
    self,
    query: str,
    text_k: int = 5,
    image_k: int = 3,
    text_threshold: float = 0.5,
    image_threshold: float = 0.3,
) -> dict:
        """
        Runs text and image searches IN PARALLEL with separate thresholds.
        Always returns both modalities — images never get crowded out by text.

        Use this instead of similarity_search_with_score for multimodal RAG.
        """
        query_vector = await self.embedding_svc.embed_query(query)

    # Both searches hit Qdrant simultaneously
        text_response, image_response = await asyncio.gather(
            asyncio.to_thread(
                self.client.query_points,
                collection_name=self.collection_name,
                query=query_vector,
                limit=text_k,
                score_threshold=text_threshold,
                with_payload=True,
                query_filter=Filter(must=[FieldCondition(
                 key="metadata.modality",
                    match=MatchValue(value="text"),
                )]),
            ),
            asyncio.to_thread(
                self.client.query_points,
                collection_name=self.collection_name,
                query=query_vector,
                limit=image_k,
                score_threshold=image_threshold,
                with_payload=True,
                query_filter=Filter(must=[FieldCondition(
                    key="metadata.modality",
                    match=MatchValue(value="image"),
                )]),
            ),
        )

        return {
            "text_results": [
                {
                    "text":     hit.payload.get("page_content", ""),
                    "score":    round(hit.score, 4),
                    "metadata": hit.payload.get("metadata", {}),
                }
                for hit in text_response.points
            ],
            "image_results": [
                {
                    "text":     "",
                    "score":    round(hit.score, 4),
                    "metadata": hit.payload.get("metadata", {}),
                }
                for hit in image_response.points
            ],
        }

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
        
    async def add_image_documents(self,image_docs: List[Document]) -> List[str]:
        """
        Embed image Documents and upsert into the SAME collection.
 
        Each Document must have in its metadata:
            base64_image : str
            mime_type    : str   (e.g. "image/png")
            modality     : "image"
        
        The base64 bytes are stored in the payload so they can be returned
        at retrieval time for display or as context to a vision LLM.
        
        TODO: base64 payloads can be large, store in S3 and save image_url
        """
        
        if not image_docs:
            logger.info("No image documents to add.")
            return []
        
        image_inputs = [
            {
                "base64_image": doc.metadata["base64_image"],
                "mime_type": doc.metadata["mime_type"],
            }
            for doc in image_docs
        ]
        vectors = await self.embedding_svc.embed_images(image_inputs) 
        uuids = [str(uuid4()) for _ in image_docs]
        points = [
            PointStruct(
                id=uid,
                vector=vector,
                payload={
                    "page_content": "",  # empty since embedding encodes the image bytes directly
                    "metadata": doc.metadata,   # includes base64_image, modality="image"
                },
            )
            for uid, vector, doc in zip(uuids, vectors, image_docs)
        ]
        logger.info(f"Upserting {len(points)} image points into '{self.collection_name}'...")
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Successfully upserted {len(points)} image points")
        return uuids
