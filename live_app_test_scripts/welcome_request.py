'''
Author: Abhijeet Ambekar
Date: 08/26/2023
'''

import requests

base_url = "https://demo-ey3d.onrender.com/"

response = requests.get(base_url)
print(response.text)
if response.status_code == 200:
    print(response)
else:
    print("Welcome request failed!!!")
