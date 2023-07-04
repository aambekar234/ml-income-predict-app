from sklearn.model_selection import train_test_split
import pandas as pd
from data_lib import process_data
from model import train_logistic_regression_model
from model import train_random_forest_model
import argparse
import logging

# logging.basicConfig(
#     filename='../logs/application.log',
#     level=logging.INFO,
#     filemode='w',
#     format='%(name)s - %(levelname)s - %(message)s')


def train_model(classifier: str, file_path: str) -> None:
    """ train model with provided classifier option
    Args:
        classifier (str): classfier option, either LR (Logistic Regression) or RF (Random Forest)
        file_path (str): file path for the data csv file
    """

    try:
        data = pd.read_csv(file_path, sep=', ')
    except FileNotFoundError:
        print("todo: fix logging")
        # logging.log("Provided file path is not valid or file does not exist!")

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

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    if classifier == "RF":
        train_random_forest_model(X_train, X_test, y_train, y_test)
    else:
        train_logistic_regression_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train model, either with Logistic Regression or Random Forest!')

    parser.add_argument('-c', '--classifier', type=str,
                        default='LR', help='Classifier name, either LR or RF')
    parser.add_argument('-f', '--file', type=str,
                        default='LR', help='Path to data file.')

    args = parser.parse_args()
    classifier = args.classifier
    file_path = args.file
    train_model(classifier, file_path)
