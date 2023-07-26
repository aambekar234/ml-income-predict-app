from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import dvc.api
from ml.process_data import process_data
import logging
import pandas as pd

params = dvc.api.params_show()
artifacts_path = params['artifacts-path']

logging.config.fileConfig("log_config.ini")
logger = logging.getLogger()

app = FastAPI()

# Define a Pydantic model to handle the request payload


class PredictionRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


# Instantiate your model
model = joblib.load(os.path.join(artifacts_path, 'model.pkl'))
encoder = joblib.load(os.path.join(artifacts_path, 'category_encoder.joblib'))


@app.get("/")
def welcome():
    return "Welcome to Income predict model inference!!"


@app.post("/predict")
def predict(request: PredictionRequest):

    logger.info(f"Received post request {request}")
    # construct dataframe from request
    try:
        df = pd.DataFrame(
            {'age': [request.age], 'workclass': [request.workclass],
             'fnlgt': [request.fnlgt],
             'education': [request.education],
             'marital-status': [request.marital_status],
             'occupation': [request.occupation],
             'relationship': [request.relationship], 'race': [request.race],
             'sex': [request.sex], 'capital-gain': [request.capital_gain],
             'capital-loss': [request.capital_loss],
             'hours-per-week': [request.hours_per_week],
             'native-country': [request.native_country]})

        X, y = process_data(df, label=None, inference=True)

        # Perform the model inference
        prediction = model.predict(X)
        logger.info(f"Prediction is {prediction}")
        # Return the prediction
        return {f"prediction is {prediction[0]}"}

    except Exception as e:
        logger.error(str(e))


# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
