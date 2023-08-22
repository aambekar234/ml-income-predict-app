# test_app.py
import pytest
from fastapi.testclient import TestClient
from main import app

# Create a TestClient instance to make requests to the FastAPI app
client = TestClient(app)


@pytest.fixture
def get_payload():
    """This is a fixture function to load the dummy positive
    payload for the test requests
    """
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
    """This is a fixture function to load the dummy negative
    payload for the test requests
    """
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
    """This is a fixture function to load the invalid
    payload for the test requests
    """
    payload = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 83311,
        "education_num": 14}
    return payload


def test_greeting():
    """Unit test for get call 
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to Income predict model inference!!"


def test_inference_1(get_payload):
    """Unit test for positive inference

    Args:
        get_payload : fixture function which loads the dummy payload
    """
    response = client.post("/predict", json=get_payload)
    assert response.status_code == 200
    assert response.json()[0] == "prediction is 1"


def test_inference_0(get_payload2):
    """Unit test for negative inference

    Args:
        get_payload : fixture function which loads the dummy payload
    """
    response = client.post("/predict", json=get_payload2)
    assert response.status_code == 200
    assert response.json()[0] == "prediction is 0"


def test_inference_invalid_request(get_payload_invalid):
    """Unit test for invalid inference request

    Args:
        get_payload : fixture function which loads the dummy payload
    """
    response = client.post("/predict", json=get_payload_invalid)
    assert response.status_code == 422
