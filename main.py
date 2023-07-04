from fastapi import FastAPI
from pydantic import BaseModel

# Import your model and any necessary preprocessing functions
from your_model_module import YourModel
from preprocessing import preprocess_data

app = FastAPI()

# Define a Pydantic model to handle the request payload


class PredictionRequest(BaseModel):
    data: str


# Instantiate your model
model = YourModel()

# Define the endpoint for the POST request


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
