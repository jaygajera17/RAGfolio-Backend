from app.core.logger import get_logger
from pathlib import Path
from app.rag.extract import load_and_chunk_pdf
from app.rag.qdrant import QdrantService
from qdrant_client.models import List

logger = get_logger(__name__)

DEFAULT_COLLECTION = "documents"


async def injest_pdf():
    qdrant_svc = QdrantService(collection_name=DEFAULT_COLLECTION)
    chunks = await load_and_chunk_pdf(800, 200)

    if not chunks:
        raise ValueError(
            "No chunks were created from the PDF. Check the file and chunking parameters."
        )

    ids = await qdrant_svc.add_documents(chunks)

    info = qdrant_svc.get_collection_info()

    return {
        "message": f"Successfully ingested PDF into Qdrant collection '{DEFAULT_COLLECTION}'.",
        "ingested_chunks": len(ids),
        "collection_info": info,
    }


async def query_rag(query: str, top_k: int = 5, score_threshold: float = 0.5):

    qdrant_svc = QdrantService(collection_name=DEFAULT_COLLECTION)
    return await qdrant_svc.similarity_search_with_score(query, top_k, score_threshold)
