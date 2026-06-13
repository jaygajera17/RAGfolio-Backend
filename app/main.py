from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.core.config import settings
from app.core.logger import get_logger
from app.db.database import connect_db, disconnect_db
from app.exceptions.handlers import register_exception_handlers
from app.middleware.request_logger import LoggingMiddleware
from app.routers.router import api_router
from auth0_fastapi.auth.auth_client import AuthClient

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    try:
        yield
    finally:
        await disconnect_db()



from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.is_dev else None,
    redoc_url="/redoc" if settings.is_dev else None,
    openapi_url="/openapi.json" if settings.is_dev else None,
    lifespan=lifespan,
)

app.state.auth_client = AuthClient 
app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET)
app.add_middleware(LoggingMiddleware)

if settings.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if settings.ALLOWED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

register_exception_handlers(app)
# test_embedding()

app.include_router(api_router, prefix=settings.API_V1_PREFIX)

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API!",
        "version": settings.APP_VERSION,
    }


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}

