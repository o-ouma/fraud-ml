from fastapi import APIRouter, HTTPException
from app.model.schemas import PredictionInput, PredictionOutput
from app.services.fraud_detector import FraudDetectionService

router = APIRouter()
fraud_detector = FraudDetectionService()

@router.get("/")
def index():
    return {"message": "ML API is running!!"}

@router.post("/predict", response_model=PredictionOutput)
async def predict_fraud(input_data: PredictionInput):
    try:
        result = await fraud_detector.predict_fraud(input_data.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}