from app.model.ml_model import FraudGNNModel
from app.core.config import settings

class FraudDetectionService:
    def __init__(self):
        self.model = FraudGNNModel(settings.MODEL_PATH)

    async def predict_fraud(self, features):
        return self.model.predict(features)