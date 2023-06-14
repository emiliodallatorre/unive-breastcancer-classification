from fastapi import FastAPI

from api.predict_request_model import PredictRequestModel
from api.predict import predict

app = FastAPI()


@app.post("/predict")
def predict(data: PredictRequestModel):
    return predict(data)
