from typing import Any


def success_response(
    data: Any | None = None,
    message: str = "Success",
) -> dict[str, Any]:
    return {
        "success": True,
        "message": message,
        "data": data,
    }