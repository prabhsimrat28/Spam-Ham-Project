from pydantic import BaseModel, Field
from typing import Dict

class PredictionResponse(BaseModel):
    predicted_result: str = Field(
        ...,
        description="Spam/Ham",
        example="Spam"
    )
    confidence: float = Field(
        ...,
        description="Model's confidence score for the predicted class (range: 0 to 1)",
        example=0.8432
    )
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution for Spam and Ham",
        example={"Spam": 0.81, "Ham": 0.19}
    )