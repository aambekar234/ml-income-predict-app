# test_app.py
import pytest
from fastapi.testclient import TestClient
from main import app

# Create a TestClient instance to make requests to the FastAPI app
client = TestClient(app)

@pytest.fixture
def get_payload():
    payload = {
    "age": 50,
    "workclass": "Private",
    "fnlgt": 83311,
    "education": 14,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 14084,
    "capital_loss": 0,
    "hours_per_week": 80,
    "native_country": "United-States"}
    return payload

@pytest.fixture
def get_payload_invalid():
    payload = {
    "age": 50,
    "workclass": "Private",
    "fnlgt": 83311,
    "education_num": 14}
    return payload

def test_greeting():
    response = client.get("/")
    print(response)
    assert response.status_code == 200
    assert response.json() == "Welcome to Income predict model inference!!"

def test_inference(get_payload):
    response = client.post("/predict", json=get_payload)
    print(response)
    assert response.status_code == 200

def test_inference_invalid_request(get_payload_invalid):
    response = client.post("/predict", json=get_payload_invalid)
    assert response.status_code == 422