'''
Author: Abhijeet Ambekar
Date: 06/26/2023
'''
import os
import pytest
import process_data as prd
import train_model as trm
import evaluate as evl

@pytest.fixture
def load_data():
    """This is a ficture function to load the data used by unit test cases
    """
    file_path = './data/census.csv'
    columns = ['salary']
    prd.main(file_path, columns)

@pytest.fixture
def train_model(load_data):
    """This is a fixture function which trains model and makes it available 
    for the unit test case
    Args:
        load_data : fixture function to load the data
    """
    trm.main("LR")

def test_process_data(load_data):
    """Unit test case for process data script

    Args:
        load_data: fixture function to load the data
    """
    artifacts_list = [
    './artifacts/data_train.joblib',
    './artifacts/labels_train.joblib',
    './artifacts/data_test.joblib',
    './artifacts/labels_test.joblib',
    './artifacts/category_encoder.joblib']

    for artifact in artifacts_list:
        assert os.path.exists(artifact)


def test_train_model(train_model):
    """Unit test case for train model script

    Args:
        train_model (_type_): fixture function to train the model
    """
    artifacts_list = ["./artifacts/model.pkl", "./artifacts/metrics_train.json"]
    for artifact in artifacts_list:
        assert os.path.exists(artifact)


def test_evaluate(train_model):
    """Unit test case for evaluate script
    Args:
        train_model (_type_): fixture function to train the model
    """
    artifacts_list = ["./artifacts/metrics_evaluate.json"]
    evl.evaluate()
    for artifact in artifacts_list:
        assert os.path.exists(artifact)
