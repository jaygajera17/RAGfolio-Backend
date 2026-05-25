from fastapi import APIRouter, Depends

from app.dependencies.user_dependency import get_user_service
from app.schemas.user_schema import User
from app.services.user_service import UserService

router = APIRouter(tags=["users"])

@router.get("/", response_model=list[User])
async def get_users(
    service: UserService = Depends(get_user_service),
) -> list[User]:
    return await service.get_users()
