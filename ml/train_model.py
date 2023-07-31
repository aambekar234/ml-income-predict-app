'''
Author: Abhijeet Ambekar
Date: 06/26/2023
'''
import os
import time
import logging.config
import argparse
import joblib
import dvc.api
from sklearn.model_selection import train_test_split
from model import train_random_forest_model
from model import train_logistic_regression_model

logging.config.fileConfig("log_config.ini")
logger = logging.getLogger()

params = dvc.api.params_show()
artifacts_path = params['artifacts-path']


def train_model(classifier: str) -> None:
    """ train model with provided classifier option
    Args:
        classifier (str): classfier option, either LR (Logistic Regression)
        or RF (Random Forest)
    """
    try:
        # load the data and labels artifacts generated by processing data
        X_train = joblib.load(os.path.join(
            artifacts_path, "data_train.joblib"))
        y_train = joblib.load(os.path.join(
            artifacts_path, "labels_train.joblib"))
    except FileNotFoundError:
        logger.error(
            "Artifatcs not found. Please run the process data module first!!")
        return

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    if classifier == "RF":
        train_random_forest_model(X_train, X_test, y_train, y_test)
    else:
        train_logistic_regression_model(X_train, X_test, y_train, y_test)


def main(classifier: str):
    """This is a main function of train model script.
    Args:
        classifier (str): classifier to train the model (either LR/RF)
    """
    logger.info("Training a model with %s classifier option.", classifier)
    started = int(time.time() * 1000)
    train_model(classifier)
    time_took = int(time.time() * 1000) - started
    logger.info(
        "It took %s ms to train the classifer %s",
        time_took,
        classifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train model, either with Logistic Regression or \
            Random Forest!')

    parser.add_argument('-c', '--classifier', type=str,
                        default='RF', help='Classifier name, either LR or RF')

    args = parser.parse_args()
    classifier = args.classifier
    main(classifier)
