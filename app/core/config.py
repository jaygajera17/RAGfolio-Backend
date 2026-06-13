
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_parse_delimiter=",",
        extra="ignore",

    )

    PROJECT_NAME: str = "FastAPI App"
    APP_VERSION: str = "0.1.0"
    ENV: str = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: list[str] = Field(default_factory=list)
    ALLOWED_HOSTS: list[str] = Field(default_factory=list)
    GOOGLE_API_KEY:str
    QDRANT_API_KEY:str
    QDRANT_HOST:str
    EMBEDDING_DIM: int = 768

    # LangSmith tracing
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "default"

    # Auth0 settings
    AUTH0_DOMAIN: str
    AUTH0_CLIENT_ID: str
    AUTH0_CLIENT_SECRET: str
    SESSION_SECRET: str 
    APP_BASE_URL: str
    AUTH0_AUDIENCE: str

    @property
    def is_dev(self) -> bool:
        return self.DEBUG or self.ENV.lower() == "development"


settings = Settings()

# Push LangSmith vars into os.environ so the LangSmith SDK can detect them.
# pydantic-settings reads .env into Python objects but does NOT set os.environ,
# and LangSmith reads directly from os.environ at import time.
os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.LANGSMITH_TRACING)
os.environ.setdefault("LANGSMITH_TRACING", settings.LANGSMITH_TRACING)
os.environ.setdefault("LANGSMITH_ENDPOINT", settings.LANGSMITH_ENDPOINT)
os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)
if settings.LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
