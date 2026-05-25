from app.schemas.user_schema import User


class UserService:
    async def get_users(self) -> list[User]:
        return [User(id=1, name="John Doe", email="john.doe@example.com")]