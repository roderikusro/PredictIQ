import requests

url = "http://localhost:5000/api/predict"
data_lr = {
    "Tahun": 2025, "Kuartal": 1, "Populasi_Juta": 280.0, 
    "Inflasi_Persen": 3.0, "Suku_Bunga_Persen": 5.75, 
    "Pengangguran_Persen": 5.0, "Investasi_Triliun": 275.0, 
    "Ekspor_Miliar_USD": 58.0, "Konsumsi_RT_Triliun": 1950.0, 
    "model_type": "linear_regression"
}
data_rf = data_lr.copy()
data_rf["model_type"] = "random_forest"

print("LR:", requests.post(url, json=data_lr).json())
print("RF:", requests.post(url, json=data_rf).json())
