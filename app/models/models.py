from pydantic import BaseModel, ConfigDict


class CustomBase(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )