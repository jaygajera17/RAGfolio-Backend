from fastapi import APIRouter, Depends
from fastapi_plugin.fast_api_client import Auth0FastAPI
from app.core.config import settings
from app.routers import rag


auth0 = Auth0FastAPI(
    domain=settings.AUTH0_DOMAIN,
    audience=settings.AUTH0_AUDIENCE
)

api_router = APIRouter()
api_router.include_router(
    rag.router,
    prefix="/rag",
    tags=["rag"],
    dependencies=[Depends(auth0.require_auth())]
)
