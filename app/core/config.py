
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_parse_delimiter=",",
    )

    PROJECT_NAME: str = "FastAPI App"
    APP_VERSION: str = "0.1.0"
    ENV: str = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: list[str] = Field(default_factory=list)
    ALLOWED_HOSTS: list[str] = Field(default_factory=list)

    @property
    def is_dev(self) -> bool:
        return self.DEBUG or self.ENV.lower() == "development"


settings = Settings()