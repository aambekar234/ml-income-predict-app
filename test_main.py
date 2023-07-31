# test_app.py
import pytest
from fastapi.testclient import TestClient
from main import app

# Create a TestClient instance to make requests to the FastAPI app
client = TestClient(app)


@pytest.fixture
def get_payload():
    return {
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


@pytest.fixture
def get_payload2():
    return {
        "age": 10,
        "workclass": "Private",
        "fnlgt": 83311,
        "education": 14,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 0,
        "native_country": "United-States"}


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
    assert response.status_code == 200
    assert response.json() == "Welcome to Income predict model inference!!"


def test_inference_1(get_payload):
    response = client.post("/predict", json=get_payload)
    assert response.status_code == 200
    assert response.json()[0] == "prediction is 1"


def test_inference_0(get_payload2):
    response = client.post("/predict", json=get_payload2)
    assert response.status_code == 200
    assert response.json()[0] == "prediction is 0"


def test_inference_invalid_request(get_payload_invalid):
    response = client.post("/predict", json=get_payload_invalid)
    assert response.status_code == 422
