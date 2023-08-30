'''
Author: Abhijeet Ambekar
Date: 06/26/2023
'''
import os
import numpy as np
import pytest
import process_data as prd
import train_model as trm
import evaluate as evl
import pandas as pd
import joblib
from sklearn.utils.validation import check_is_fitted
from process_data import process_data

categorical_features = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "education"
]

numerical_features = [
    "age",
    "fnlgt",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]


@pytest.fixture
def get_dataframe1():
    """This is a fixture function which return a pandas dataframe to test the
    encoder
    """
    data_row = {
        "age": [50],
        "workclass": ["Private"],
        "fnlgt": [83311],
        "education": ["Bachelors"],
        "marital-status": ["Married"],
        "occupation": ["Exec-managerial"],
        "relationship": ["Husband"],
        "race": ["White"],
        "sex": ["Male"],
        "capital-gain": [14084],
        "capital-loss": [0],
        "hours-per-week": [80],
        "native-country": ["United-States"]}

    return pd.DataFrame(data_row)


@pytest.fixture
def get_dataframe2():
    """This is a fixture function which return a pandas dataframe to test the
    encoder
    """
    data_row = {
        "age": [50],
        "workclass": ["Private"],
        "fnlgt": [83311],
        "education": ["Masters"],
        "marital-status": ["Married"],
        "occupation": ["Exec-managerial"],
        "relationship": ["Husband"],
        "race": ["White"],
        "sex": ["Male"],
        "capital-gain": [14084],
        "capital-loss": [0],
        "hours-per-week": [80],
        "native-country": ["United-States"]}

    return pd.DataFrame(data_row)


@pytest.fixture
def load_data():
    """This is a fixture function to load the data used by unit test cases
    """
    file_path = './data/census.csv'
    columns = ['salary']
    prd.main(file_path, columns)
    yield


@pytest.fixture
def train_model(load_data):
    """This is a fixture function which trains model and makes it available
    for the unit test case
    Args:
        load_data : fixture function to load the data
    """
    trm.main("LR")
    return joblib.load("./artifacts/model.pkl")


def test_process_data():
    """Unit test case for process data script

    Args:
        load_data: fixture function to load the data
    """
    artifacts_list = [
        './artifacts/data_train.joblib',
        './artifacts/labels_train.joblib',
        './artifacts/test.joblib',
        './artifacts/category_encoder.joblib']

    for artifact in artifacts_list:
        assert os.path.exists(artifact)


def test_is_fitted(train_model):
    """Unit test to cehk the model is fitted or not

    Args:
        train_model: ficture function which trains and returns the model

    Raises:
        AssertionError: Raises error if model is not fitted
    """
    try:
        check_is_fitted(train_model)
    except AssertionError:
        raise AssertionError


def test_train_model(train_model):
    """Unit test case for train model script

    Args:
        train_model: fixture function to train the model
    """
    check_is_fitted(train_model)
    artifacts_list = [
        "./artifacts/model.pkl",
        "./artifacts/metrics_train.json"]
    for artifact in artifacts_list:
        assert os.path.exists(artifact)


def test_evaluate(train_model):
    """Unit test case for evaluate script
    Args:
        train_model: fixture function to train the model
    """
    check_is_fitted(train_model)
    artifacts_list = ["./artifacts/metrics_evaluate.json"]
    evl.evaluate()
    for artifact in artifacts_list:
        assert os.path.exists(artifact)


def test_encoder(get_dataframe1, get_dataframe2):
    """Unit test to test the categorical encoder
    Args:
        get_dataframe (_type_): _description_
    """
    X1, y2 = process_data(get_dataframe1)
    X2, y2 = process_data(get_dataframe2)
    assert not np.array_equal(X1, X2)
