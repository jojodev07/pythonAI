from pydantic import BaseModel, Field, field_validator

class QueryRequest(BaseModel):
    question : str
    top_k : int = Field(default=3, ge=1)

class QueryResponse(BaseModel):
    answer : str
    context : list[str]

