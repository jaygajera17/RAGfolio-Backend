from typing import List
import asyncio

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings
from app.core.logger import get_logger

EMBEDDING_DIM = settings.EMBEDDING_DIM
MAX_CONCURRENT = 5
logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self):
        self.model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            api_key=settings.GOOGLE_API_KEY,
            output_dimensionality=EMBEDDING_DIM,
        )
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)


    async def embed_query(self, text: str):
        """Embed a single query string. Use for search/retrieval."""
        return await self.model.aembed_query(text)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings concurrently.

        Root cause note:
        - aembed_documents / batchEmbedContents treats the whole list
          as ONE multimodal document → always returns 1 vector regardless
          of input count.
        - aembed_query / embedContent embeds ONE text → always correct.
        - Fix: gather N concurrent aembed_query calls with a semaphore
          to avoid rate-limit errors.
        """
        async def _embed_one(text: str) -> List[float]:
            async with self._semaphore:
                return await self.model.aembed_query(text)

        vectors = await asyncio.gather(*[_embed_one(t) for t in texts])

        # Hard guard — zip would silently truncate if this ever drifts
        assert len(vectors) == len(texts), (
            f"Embedding mismatch: {len(vectors)} vectors for {len(texts)} texts"
        )
        logger.debug(f"Embedded {len(vectors)} texts (dim={EMBEDDING_DIM})")
        return list(vectors)

    async def embed_chunks(self, chunks: List[str]):
        """Embed a list of Document chunks. Returns list of dicts ready for Qdrant upsert:"""
        texts = [chunk.page_content for chunk in chunks]
        vectors = await self.model.embed_documents(texts)

        return [
            {
                "text": text,
                "vector": vector,
                "metadata": chunk.metadata,
            }
            for text, vector, chunk in zip(texts, vectors, chunks)
        ]
