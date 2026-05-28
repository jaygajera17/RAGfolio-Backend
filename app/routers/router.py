from fastapi import APIRouter

from app.routers import rag

api_router = APIRouter()
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])

