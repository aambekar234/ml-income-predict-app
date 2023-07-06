from preprocessing import preprocess_data
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import dvc.api
params = dvc.api.params_show()
artifacts_path = params['artifacts-path']

logging.config.fileConfig("log_config.ini")
logger = logging.getLogger()

app = FastAPI()

# Define a Pydantic model to handle the request payload


class PredictionRequest(BaseModel):
    data: str


# Instantiate your model
model = joblib.load(os.path.join(artifacts_path, 'model.pkl'))
encoder = joblib.load(os.path.join(artifacts_path, 'category_encoder.joblib'))


@app.get("/")
def welcome():
    return "Welcome to Income predict model inference!!"


@app.post("/predict")
def predict(request: PredictionRequest):
    # Preprocess the input data
    preprocessed_data = preprocess_data(request.data)

    # Perform the model inference
    prediction = model.predict(preprocessed_data)

    # Return the prediction
    return {"prediction": prediction}


# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
