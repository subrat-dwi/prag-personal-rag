from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,                          # ... means required, no default
        min_length=1,
        max_length=500,
        description="Natural language question about your documents",
        examples=["what is my professionalemail", "what is my CGPA"]
    )
    top_k: int = Field(
        default=5,
        ge=1,                         # ge = greater than or equal to
        le=20,                        # le = less than or equal to
        description="Number of document chunks to retrieve"
    )


class SourceReference(BaseModel):
    filename: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    # sources: list[SourceReference]
    # query: str                        # echo back the original query


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    ollama: str