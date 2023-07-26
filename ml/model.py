'''
Author: Abhijeet Ambekar
Date: 06/26/2023
'''

import os
import json
import joblib
import dvc.api
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

params = dvc.api.params_show()
artifacts_path = params['artifacts-path']

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using
    precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def save_model(model):
    """ Save the model object with model_name at the provided path

    Args:
        model : model/classifier object
    Returns:
        None
    """
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)
    joblib.dump(model, os.path.join(artifacts_path, "model.pkl"))


def train_logistic_regression_model(X_train, X_test, y_train, y_test):
    """ train logistic regression model

    Args:
        X_train (_type_): np.array used for training
        y_train (_type_): np.array labels used for training
        X_test (_type_): np.array used for testing
        y_test (_type_): np.array labels used for testing

    Returns:
        LogisticRegression: sklearn logistic regression model
    """
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)
    y_test_preds = lrc.predict(X_test)
    generate_report(y_test, y_test_preds, "metrics_train.json")
    save_model(lrc)


def train_random_forest_model(X_train, X_test, y_train, y_test):
    """ train random forest classifier model

    Args:
        X_train (_type_): np.array used for training
        y_train (_type_): np.array labels used for training
        X_test (_type_): np.array used for testing
        y_test (_type_): np.array labels used for testing

    Returns:
        GridSearchCV: returns instance of GridSearchCV
    """

    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    y_test_preds = cv_rfc.best_estimator_.predict(X_test)
    generate_report(y_test, y_test_preds, "metrics_train.json")
    save_model(cv_rfc.best_estimator_)


def generate_report(y_test, y_test_preds, filename: str) -> None:
    """_summary_

    Args:
        y_test: y test numpy array
        y_test_preds: y test numpy array prediction
        filename (str): metrics file name
    """
    report = classification_report(
        y_test, y_test_preds, output_dict=True)
    print(report)
    new_report = {}
    new_report['accuracy'] = report['accuracy']
    new_report['0'] = {}
    new_report['1'] = {}
    new_report['0']['f1-score'] = report['0']['f1-score']
    new_report['1']['f1-score'] = report['1']['f1-score']
    with open(os.path.join(artifacts_path, filename), mode="w") as fp:
        json.dump(new_report, fp)
