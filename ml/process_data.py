'''
Author: Abhijeet Ambekar
Date: 06/26/2023
'''
import os
import logging.config
import argparse
import joblib
import dvc.api
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split

params = dvc.api.params_show()
artifacts_path = params['artifacts-path']
logging.config.fileConfig("log_config.ini")
logger = logging.getLogger()


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


def save_category_encoder(df):
    """ creates One hot encoder of category columns and saves the encoder
    Args:
        df (pandas.DataFrame):
    Return:
        None
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df[categorical_features])
    joblib.dump(
        encoder,
        os.path.join(
            artifacts_path,
            'category_encoder.joblib'))


def process_data(df, label=None):
    """ This function encodes the category columns into one hot encoding and
        creates new dataframe with numerical only data for purpose of training.

    Args:
        df (pd.dataframe): dataframe
        label (String): label column as string in dataframe df

    Returns:
        X (numpy.ndarray): numpy array of data
        y (numpy.ndarray): numpy array of labels
    """
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)

    y = []
    if label is not None:
        y = df[label]

    X = None
    lb = LabelBinarizer()
    encoder = joblib.load(os.path.join(
        artifacts_path, 'category_encoder.joblib'))

    if label is not None:
        y = lb.fit_transform(y.values).ravel()
    X_numerical = df[numerical_features].to_numpy()
    X_categorical = encoder.transform(df[categorical_features])
    X = np.hstack((X_numerical, X_categorical))
    return X, y


def combine_columns_for_stratify(df, columns):
    """ This function combines the data frame columns which need to stratify
    into same single column. It return the dataframe with the combined column

    Args:
        df (pd.DataFrame): pandas dataframe
        label (str): label column in df
        columns ([]): columns to stratify

    Returns:
        pd.DataFrame: pandas dataframe
    """
    combined_column = df[columns[0]].astype(str)

    for i in range(1, len(columns)):
        combined_column += df[columns[i]].astype(str)

    unique_counts = combined_column.value_counts()

    # Convert to DataFrame
    unique_counts_df = unique_counts.to_frame().reset_index()
    unique_counts_df.columns = ['Element', 'Count']

    # Get the minimum count
    minimum_count = unique_counts_df['Count'].min()
    if minimum_count > 1:
        df['combined_column'] = combined_column
    else:
        df['combined_column'] = df['salary']
        logger.info(
            "The provided stratify configuration is incorrect, method will \
                stratify by the label now.")
    return df


def main(file_path: str, stratify_columns):
    """This is a main function of process data script.

    Args:
        file_path (str): path to data file
        stratify_columns ([str]): space seprate columns names to stratify data
    Return:
        None
    """
    try:
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.lower().str.replace(' ', '')
        # Trim leading and trailing whitespace from all string values
        data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        data = data.drop_duplicates()
        save_category_encoder(data)
        data = combine_columns_for_stratify(data, stratify_columns)
        train, test = train_test_split(
            data,
            test_size=0.15,
            stratify=data['combined_column'],
            random_state=100)
        train = train.drop('combined_column', axis=1)
        test = test.drop('combined_column', axis=1)
        X_train, y_train = process_data(train, label='salary')
        X_test, y_test = process_data(test, label='salary')
        # save the test, train artifacts
        joblib.dump(X_train, os.path.join(artifacts_path, 'data_train.joblib'))
        joblib.dump(y_train, os.path.join(artifacts_path,
                                          'labels_train.joblib'))
        joblib.dump(test, os.path.join(artifacts_path, "test.joblib"))

    except FileNotFoundError:
        logger.error("Provided file path is not valid or file does not exist!")


if __name__ == "__main__":
    """script entry point
    """

    parser = argparse.ArgumentParser(
        description='Process the data. Get data and labels \
        objects with encoders!')

    parser.add_argument('-f', '--file', type=str,
                        default='./data/census.csv', help='Path to data file.')
    parser.add_argument('-s', '--stratify', nargs='+',
                        default=['salary'],
                        help='List of columns as string to stratify the data.')

    args = parser.parse_args()
    file_path = args.file
    columns = args.stratify
    main(file_path, columns)
