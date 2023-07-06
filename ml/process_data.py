import logging.config
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd
import joblib
import os
import argparse
from sklearn.model_selection import train_test_split
import dvc.api
params = dvc.api.params_show()
artifacts_path = params['artifacts-path']

logging.config.fileConfig("log_config.ini")
logger = logging.getLogger()


def process_data(X, categorical_features, label, training):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    None
    """
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)

    y = X[label]
    X = X.drop([label], axis=1)

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)
    encoder, lb = None, None

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        # save encoder artifacts
        joblib.dump(encoder, os.path.join(
            artifacts_path, 'category_encoder.joblib'))
        joblib.dump(lb, os.path.join(
            artifacts_path, 'binary_encoder.joblib'))

    else:
        # load encoders
        encoder = joblib.load(os.path.join(
            artifacts_path, 'category_encoder.joblib'))
        lb = joblib.load(os.path.join(
            artifacts_path, 'binary_encoder.joblib'))

    X_categorical = encoder.fit_transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    y = lb.fit_transform(y.values).ravel()

    if training:
        joblib.dump(X, os.path.join(
            artifacts_path, 'data_train.joblib'))
        joblib.dump(y, os.path.join(
            artifacts_path, 'labels_train.joblib'))
    else:
        joblib.dump(X, os.path.join(
            artifacts_path, 'data_test.joblib'))
        joblib.dump(y, os.path.join(
            artifacts_path, 'labels_test.joblib'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Process the data. Get data and labels objects with encoders!')

    parser.add_argument('-f', '--file', type=str,
                        default='./data/census.csv', help='Path to data file.')

    args = parser.parse_args()
    file_path = args.file

    try:
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.lower().str.replace(' ', '')
        data = data.drop_duplicates()
        # Optional enhancement, use K-fold cross validation instead of a train-test split.
        train, test = train_test_split(
            data, test_size=0.20, stratify=data['salary'])

        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        # generates training artifacts
        process_data(train, cat_features, 'salary', True)
        # generates testing artifacts
        process_data(test, cat_features, 'salary', False)

    except FileNotFoundError:
        logger.error("Provided file path is not valid or file does not exist!")
