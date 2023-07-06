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


def process_data(df, categorical_features, label):
    """_summary_

    Args:
        df (pd.dataframe): dataframe
        categorical_features ([]): pytho list of categorical columns
        label (_type_): label column as string in dataframe df

    Returns:
        X (numpy.ndarray): numpy array of data
        y (numpy.ndarray): numpy array of labels
    """
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)

    y = df[label]
    X = df.drop(label, axis=1)

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(categorical_features, axis=1)
    X_continuous = X_continuous.values
    encoder, lb = None, None

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    lb = LabelBinarizer()

    X_categorical = encoder.fit_transform(X_categorical)
    logger.info(
        f"Categorical features shape - {X_categorical.shape}, Numerical features shape - {X_continuous.shape}")
    X = np.concatenate((X_categorical, X_continuous), axis=1)
    logger.info(
        f"Shape of training data after stitching categorical and numerical features - {X.shape}")
    y = lb.fit_transform(y.values).ravel()

    # save encoder artifacts so that it can be used during inference
    joblib.dump(encoder, os.path.join(
        artifacts_path, 'category_encoder.joblib'))
    joblib.dump(lb, os.path.join(
        artifacts_path, 'binary_encoder.joblib'))

    return X, y


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
        X = df.drop("education", axis=1)
        cat_features = [
            "workclass",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        X, y = process_data(data, cat_features, 'salary')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=data['salary'])

        # save the test, train artifacts
        joblib.dump(X_train, os.path.join(artifacts_path, 'data_train.joblib'))
        joblib.dump(X_test, os.path.join(artifacts_path, 'data_test.joblib'))
        joblib.dump(y_train, os.path.join(
            artifacts_path, 'labels_train.joblib'))
        joblib.dump(y_test, os.path.join(artifacts_path, 'labels_test.joblib'))
        logger.info(f"Shape of training artifact is {X_train.shape}")
        logger.info(f"Shape of testing artifact is {X_test.shape}")

    except FileNotFoundError:
        logger.error("Provided file path is not valid or file does not exist!")
