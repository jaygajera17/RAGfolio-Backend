from typing import Dict, List
import asyncio
import base64

from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.logger import get_logger

EMBEDDING_DIM = settings.EMBEDDING_DIM
MAX_CONCURRENT = 1
logger = get_logger(__name__)

_GEMINI_EMBED_MODEL = "models/gemini-embedding-2"

def make_document_text(title: str, text: str) -> str:
    """Format a text chunk as a retrieval document."""
    t = title.strip() if title.strip() else "none"
    return f"title: {t} | text: {text}"


class EmbeddingService:
    def __init__(self):
        # LangChain wrapper for text embedding
        self.model = GoogleGenerativeAIEmbeddings(
            model=_GEMINI_EMBED_MODEL,
            api_key=settings.GOOGLE_API_KEY,
            output_dimensionality=EMBEDDING_DIM,
        )
        # Modern Google GenAI Client for multimodal image embedding
        self.genai_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
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
                # Direct timeout to slow down ingestion and avoid rate limit
                await asyncio.sleep(10.0)
                return await self.model.aembed_query(text)

        vectors = await asyncio.gather(*[_embed_one(t) for t in texts])

        # Hard guard — zip would silently truncate if this ever drifts
        assert len(vectors) == len(texts), (
            f"Embedding mismatch: {len(vectors)} vectors for {len(texts)} texts"
        )
        logger.debug(f"Embedded {len(vectors)} texts (dim={EMBEDDING_DIM})")
        return list(vectors)

    async def embed_chunks(self, chunks: List[str]):
        """Embed a list of Document chunks. Returns list of dicts ready for Qdrant upsert."""
        texts = [chunk.page_content for chunk in chunks]
        
        # Direct timeout to avoid rate limit error
        await asyncio.sleep(10.0)
        vectors = await self.model.embed_documents(texts)

        return [
            {
                "text": text,
                "vector": vector,
                "metadata": chunk.metadata,
            }
            for text, vector, chunk in zip(texts, vectors, chunks)
        ]

    async def embed_image(
        self,
        base64_image: str,
        mime_type: str = "image/png",
    ) -> List[float]:
        """
        Embed a single image using the modern Google GenAI SDK.

        Gemini Embedding 2.0 is multimodal: text and image vectors live in
        the same space, so a text query will surface relevant images at
        retrieval time with no extra steps.

        Parameters
        ----------
        base64_image : str
            Raw image bytes encoded as a base64 string (no data-URI prefix).
        mime_type : str
            MIME type, e.g. "image/png" or "image/jpeg".

        Returns
        -------
        List[float]
            Embedding vector of length EMBEDDING_DIM.
        """
        async with self._semaphore:
            # Direct timeout
            await asyncio.sleep(10.0)
            
            # Decode the base64 string to raw bytes
            raw_bytes = base64.b64decode(base64_image)

            # Construct the Part object with raw bytes
            part = types.Part.from_bytes(
                data=raw_bytes,
                mime_type=mime_type
            )

            # Run in worker thread since client call is synchronous
            response = await asyncio.to_thread(
                self.genai_client.models.embed_content,
                model=_GEMINI_EMBED_MODEL,
                contents=part,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIM,
                )
            )

            # Extract float values from the returned embedding
            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].values
            else:
                raise ValueError("No embeddings returned by Google GenAI API.")

    async def embed_images(
        self,
        image: List[Dict]
    ):
        """
        Embed a list of images concurrently
        """
        vectors = await asyncio.gather(*[
            self.embed_image(
                base64_image=img["base64_image"],
                mime_type=img["mime_type"]
            )
            for img in image
        ])
        return list(vectors)
