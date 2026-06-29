from pydantic import BaseModel,Field,field_validator
from typing import Annotated

class SMStext(BaseModel):
    text: Annotated[str, Field(...,min_length=1,max_length=500,description="Input text to check for Spam/Ham",example="Win a Free iphone 13")]

    @field_validator("text")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Text must not be empty or whitespace-only")
        return v
    