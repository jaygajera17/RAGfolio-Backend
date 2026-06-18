from app.core.logger import get_logger
from pathlib import Path
from app.rag.extract import load_and_chunk_pdf
from app.rag.qdrant import QdrantService
from app.core.config import settings
from qdrant_client.models import List

logger = get_logger(__name__)


async def injest_pdf(pdf_path: str = None):
    collection = settings.DEFAULT_COLLECTION
    qdrant_svc = QdrantService(collection_name=collection)
    if not pdf_path:
        pdf_path = settings.PDF_PATH or "static/fund-factsheet-for-may-2026-51-97.pdf"
    text_chunk, table_docs, image_docs = await load_and_chunk_pdf(pdf_path)

    if not text_chunk and not table_docs and not image_docs:
        raise ValueError(
            "No content extracted from the PDF"
        )

    # Table docs have structured Markdown page_content → embed via text path
    text_ids = await qdrant_svc.add_documents(text_chunk + table_docs)
    image_ids = await qdrant_svc.add_image_documents(image_docs)

    info = qdrant_svc.get_collection_info()

    return {
        "message": f"Successfully ingested PDF into Qdrant collection '{collection}'.",
        "ingested_text_chunks": len(text_chunk),
        "ingested_table_docs": len(table_docs),
        "ingested_image_docs": len(image_ids),
        "collection_info": info,
        # "text_chunk": text_chunk,
        # "image_docs": image_docs
    }


async def query_rag(query: str, top_k: int = 10, score_threshold: float = 0.45):
    qdrant_svc = QdrantService(collection_name=settings.DEFAULT_COLLECTION)
    return await qdrant_svc.similarity_search_multimodal(
        query=query,
        text_k=top_k,
        image_k=3,
        text_threshold=score_threshold,  
        image_threshold=0.3,             
    )