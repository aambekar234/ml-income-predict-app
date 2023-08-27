'''
Author: Abhijeet Ambekar
Date: 06/26/2023
'''

import os
import logging.config
import json
import joblib
import dvc.api
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

params = dvc.api.params_show()
artifacts_path = params['artifacts-path']
logging.config.fileConfig("log_config.ini")
logger = logging.getLogger()


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
    fbeta = round(fbeta_score(y, preds, beta=1, zero_division=1), 3)
    precision = round(precision_score(y, preds, zero_division=1), 3)
    recall = round(recall_score(y, preds, zero_division=1), 3)
    accuracy = round(accuracy_score(y, preds), 3)
    return precision, recall, fbeta, accuracy


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


def train_logistic_regression_model(X, y):
    """ train logistic regression model with k-fold cross validation

    Args:
        X (_type_): np.array used for training
        y (_type_): np.array labels used for training

    Returns:
        LogisticRegression: sklearn logistic regression model
    """
    model = LogisticRegression(solver='lbfgs', max_iter=3000)
    kfoldTraining(X, y, model, 10)


def train_random_forest_model(X_train, X_test, y_train, y_test, folds=5):
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
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=folds)
    cv_rfc.fit(X_train, y_train)
    y_test_preds = cv_rfc.best_estimator_.predict(X_test)
    generate_report(
        y_test,
        y_test_preds,
        "Random Forest",
        "metrics_train.json")
    save_model(cv_rfc.best_estimator_)


def kfoldTraining(X, y, model, folds=5):
    """Trains the model with k-fold cross validation strategy

    Args:
        X (np.array): np.array used for training
        y (np.array): np.array labels used for training
        model: sklearn model classifier
        folds: No. of folds to perform in cross-validation
    """

    # Create a KFold instance
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)

    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Perform cross-validation and collect true and predicted labels
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        true_labels.extend(y_test)
        predicted_labels.extend(y_pred)

    # Calculate the classification report
    generate_report(
        true_labels,
        predicted_labels,
        "Logistic Regression",
        "metrics_train.json")
    save_model(model)


def generate_report(y_test, y_test_preds, classifier, filename: str) -> None:
    """generates model evaluation report

    Args:
        y_test: y test numpy array
        y_test_preds: y test numpy array prediction
        classifier: model classifier
        filename (str): metrics file name
    """
    report = classification_report(
        y_test, y_test_preds, output_dict=True)
    precision, recall, fbeta, accuracy = compute_model_metrics(y_test,
                                                               y_test_preds)
    new_report = {}
    new_report['accuracy'] = accuracy
    new_report['fbeta'] = fbeta
    new_report['precision'] = precision
    new_report['recall'] = recall
    new_report['0'] = {}
    new_report['1'] = {}
    new_report['0']['f1-score'] = report['0']['f1-score']
    new_report['1']['f1-score'] = report['1']['f1-score']

    with open(os.path.join(artifacts_path, filename), mode="w") as fp:
        json.dump(new_report, fp)
