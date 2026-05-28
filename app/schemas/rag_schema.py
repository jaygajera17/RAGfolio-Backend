from pydantic import BaseModel



class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.5