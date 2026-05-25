import logging


logger = logging.getLogger(__name__)


async def connect_db() -> None:
    logger.info("Database connect skipped: no backend configured.")


async def disconnect_db() -> None:
    logger.info("Database disconnect skipped: no backend configured.")
