import requests

url = "http://localhost:5000/api/model-comparison"
print(requests.get(url).json())
