# Project

This project covers below concepts.

    1. Train model to predict whether income exceeds 50k based on census data.
    2. Evaluate the model on different data slices to minimize the bias towards certain datapoints. 
    3. Run unit test cases for Machine learning operations. 
    4. Use github actions for CI/CD and deploy app on render cloud. 
    5. Use FastAPI to serve model online for inference

## Project Description

In this project, you will find implementation of classification model on publicly available Census Bureau data. You will find the unit tests to monitor the model performance on various data slices. You can also deploy your model using the FastAPI package and create API tests. The slice validation and the API tests will be incorporated into a CI/CD framework using GitHub Actions.

## Setup environment

- Install conda/minicoda. [miniconda](https://docs.conda.io/en/latest/miniconda.html#installing)
- Create conda environement by below command and activate the environment
    ```
    conda env create -f environment.yml --force
    conda activate ml-income-predict-app
    ```
## Run experiments with DVC
Change the parameters in params.yaml file according to your needs and run dvc experiments by below commands. 
```
dvc exp run
```

## Serve model with FastAPI

Use below command to start the FastAPI app 
    ```
    uvicorn main:app --reload
    ```
