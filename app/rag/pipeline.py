from app.core.logger import get_logger
from pathlib import Path
from app.rag.extract import load_and_chunk_pdf
from app.rag.qdrant import QdrantService
from qdrant_client.models import List

logger = get_logger(__name__)

DEFAULT_COLLECTION = "documents"


async def injest_pdf():
    qdrant_svc = QdrantService(collection_name=DEFAULT_COLLECTION)
    text_chunk, image_docs = await load_and_chunk_pdf()

    if not text_chunk and not image_docs:
        raise ValueError(
            "No content extracted from the PDF"
        )

    text_ids = await qdrant_svc.add_documents(text_chunk)
    image_ids = await qdrant_svc.add_image_documents(image_docs)

    info = qdrant_svc.get_collection_info()

    return {
        "message": f"Successfully ingested PDF into Qdrant collection '{DEFAULT_COLLECTION}'.",
        "ingested_text_chunks": len(text_ids),
        "ingested_image_docs": len(image_ids),
        "collection_info": info,
        # "text_chunk": text_chunk, 
        # "image_docs": image_docs
    }


async def query_rag(query: str, top_k: int = 5, score_threshold: float = 0.5):
    qdrant_svc = QdrantService(collection_name=DEFAULT_COLLECTION)
    return await qdrant_svc.similarity_search_multimodal(
        query=query,
        text_k=top_k,
        image_k=3,
        text_threshold=score_threshold,  
        image_threshold=0.3,             
    )