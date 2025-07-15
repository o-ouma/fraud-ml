from pydantic import BaseModel
from typing import Dict, Any

class PredictionInput(BaseModel):
    features: Dict[str, Any]

class PredictionOutput(BaseModel):
    prediction: float
    probability: float
    anomaly_score: float
    is_potentially_fraudulent: int
