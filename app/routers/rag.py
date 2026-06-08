from fastapi import APIRouter
from app.rag.pipeline import injest_pdf
from app.schemas.rag_schema import SearchQuery,AskRequest
from app.rag.pipeline import query_rag
from app.rag.retrival import RetrivalService

router = APIRouter(tags=["rag"])

@router.post("/ingest", response_model=dict)
async def ingest_pdf():
    """Endpoint to ingest a PDF document into the RAG system."""
    result = await injest_pdf()
    return result

@router.post("/query", response_model=dict)
async def query_rag_pipeline(request:SearchQuery):
    """Endpoint to perform a similarity search query against the RAG system."""
    results = await query_rag(
        query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold
    )
    return {"query": request.query, "results": results}
    
@router.post("/ask")
async def ask_endpoint(request : AskRequest):
    try:
        retriver = RetrivalService()
        return await retriver.query_rag_with_answer(request.query)
    except Exception as e:
        return {"error": str(e)}
