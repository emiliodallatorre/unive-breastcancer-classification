from fastapi import FastAPI

from api.predict_request_model import PredictRequestModel
from api.predict import predict as api_predict

app = FastAPI()


@app.post("/predict")
def predict(data: PredictRequestModel):
    return api_predict(data)
