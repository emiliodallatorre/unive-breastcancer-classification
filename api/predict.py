import joblib
from sklearn.ensemble import RandomForestClassifier

from api.predict_request_model import PredictRequestModel

model: RandomForestClassifier = joblib.load("../output/random_forest_model.joblib")


def predict(data: PredictRequestModel):
    classes: list[str] = model.classes_
    probabilities: list[float] = model.predict_proba(data.to_dataframe())[0]
    return {
        "diagnosis": classes[probabilities.index(max(probabilities))],
        "probabilities": {
            classes[0]: probabilities[0],
            classes[1]: probabilities[1]
        }
    }


predict(PredictRequestModel(radius=15, perimeter=140, area=200))
