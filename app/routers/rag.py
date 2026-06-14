from fastapi import APIRouter
from app.rag.pipeline import injest_pdf
from app.schemas.rag_schema import SearchQuery,AskRequest
from app.rag.pipeline import query_rag
from app.rag.retrival import RetrivalService
from auth0_fastapi.config import Auth0Config
from auth0_fastapi.auth.auth_client import AuthClient
from fastapi_plugin.fast_api_client import Auth0FastAPI
from auth0_fastapi.server.routes import router as auth_router, register_auth_routes
from app.core.config import settings

# Create router for custom authentication endpoints
router = APIRouter()

# Create Auth0Config using app settings
config = Auth0Config(
    domain=settings.AUTH0_DOMAIN,
    client_id=settings.AUTH0_CLIENT_ID,
    client_secret=settings.AUTH0_CLIENT_SECRET,
    app_base_url=settings.APP_BASE_URL,
    secret=settings.SESSION_SECRET,
)

# Instantiate the AuthClient
auth_client = AuthClient(config)

# Register default SDK endpoints (/auth/login, /auth/logout, /auth/callback)
register_auth_routes(auth_router, config)

auth0 = Auth0FastAPI(
    domain=settings.AUTH0_DOMAIN,
    audience=settings.AUTH0_AUDIENCE
)
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
