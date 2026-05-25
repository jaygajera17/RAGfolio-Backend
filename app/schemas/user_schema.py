from pydantic import EmailStr

from app.models.models import CustomBase


class User(CustomBase):
    id: int
    name: str
    email: EmailStr