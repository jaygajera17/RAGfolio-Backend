from fastapi import APIRouter,Depends
from auth0_fastapi.config import Auth0Config
from fastapi_plugin.fast_api_client import Auth0FastAPI
from app.core.config import settings
from app.routers import rag


# Create Auth0Config using app settings
config = Auth0Config(
    domain=settings.AUTH0_DOMAIN,
    client_id=settings.AUTH0_CLIENT_ID,
    client_secret=settings.AUTH0_CLIENT_SECRET,
    app_base_url=settings.APP_BASE_URL,
    secret=settings.SESSION_SECRET,
)

auth0 = Auth0FastAPI(
    domain=settings.AUTH0_DOMAIN,
    audience=settings.AUTH0_AUDIENCE
)


api_router = APIRouter()
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])

