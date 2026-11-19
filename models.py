from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=2000)


class SourceOut(BaseModel):
    n: int
    title: str
    section: str
    source: str
    snippet: str
    score: float
    url: str = ""


class ChatResponse(BaseModel):
    response: str
    route: str  # crisis | out_of_scope | answered | not_found
    abstained: bool = False
    sources: list[SourceOut] = []
