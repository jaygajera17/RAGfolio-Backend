import logging
from app.core.config import settings

LOG_LEVEL = settings.LOG_LEVEL


logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)