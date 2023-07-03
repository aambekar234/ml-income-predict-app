from sklearn.model_selection import train_test_split
import pandas as pd
from data_lib import process_data
from model import train_logistic_regression_model
from model import train_random_forest_model

# Add code to load in the data.


data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# model1 = train_logistic_regression_model(X_train, X_test, y_train, y_test)

model2 = train_random_forest_model(X_train, X_test, y_train, y_test)

# Train and save a model.
