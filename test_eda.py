import requests
try:
    r = requests.get('http://localhost:5000/api/eda')
    print("Status Code:", r.status_code)
    print("Response JSON length:", len(r.text))
    if r.status_code != 200:
        print("Error content:", r.text)
except Exception as e:
    print("Error:", e)
