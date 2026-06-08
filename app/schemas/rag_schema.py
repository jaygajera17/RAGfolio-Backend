from pydantic import BaseModel,Field



class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.4

class AskRequest(BaseModel):
    query: str = Field(..., description="User question")