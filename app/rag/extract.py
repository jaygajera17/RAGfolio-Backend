from pathlib import Path
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.logger import get_logger
import asyncio
from uuid import uuid4

from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger(__name__)


base_dir = Path(__file__).resolve().parents[2]
pdf_path = base_dir / "static" / "sample2.pdf"
if not pdf_path.is_file():
    raise FileNotFoundError(f"PDF not found at {pdf_path}")


async def load_and_chunk_pdf(chunk_size: int = 800, chunk_overlap: int = 200):
    """
    Load a PDF and split it into chunks.
    """

    loader = PyMuPDF4LLMLoader(
        file_path=str(pdf_path),
        mode="single",
    )
    docs = await asyncio.to_thread(loader.load)
    logger.info(f"Loaded {len(docs)} documents from PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["source_file"] = pdf_path.name

    logger.info(f"Split documents into {len(chunks)} chunks.")

    return chunks
