'''
Author: Abhijeet Ambekar
Date: 08/26/2023
'''

import requests

base_url = "https://demo-ey3d.onrender.com/"
payload = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 83311,
        "education": "Masters",
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 14084,
        "capital_loss": 0,
        "hours_per_week": 80,
        "native_country": "United-States"}

response = requests.post(base_url+"predict", json=payload)
print(response.text)
if response.status_code == 200:
    print(response)
else:
    print("Inference request failed!!!")
