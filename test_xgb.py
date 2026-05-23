"""
Test script untuk validasi integrasi XGBoost.
Jalankan: python test_xgb.py saat server aktif di localhost:5000.
"""

import requests


BASE = "http://localhost:5000"

SAMPLE_INPUT = {
    "Tahun": 2025,
    "Kuartal": 2,
    "Populasi_Juta": 280.5,
    "Inflasi_Persen": 3.2,
    "Suku_Bunga_Persen": 5.75,
    "Pengangguran_Persen": 5.1,
    "Investasi_Triliun": 278.0,
    "Ekspor_Miliar_USD": 59.5,
    "Konsumsi_RT_Triliun": 1975.0,
}


def test_predict_xgboost():
    """AC-BE-03: Prediksi single model XGBoost."""
    payload = {**SAMPLE_INPUT, "model_type": "xgboost"}
    response = requests.post(f"{BASE}/api/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()["data"]
    assert "prediction" in data
    assert data["model_type"] == "xgboost"
    assert isinstance(data["prediction"], float)
    print(f"OK XGBoost prediction: {data['prediction']:.4f}")


def test_predict_compare_three_models():
    """AC-BE-05: Compare tiga model sekaligus."""
    response = requests.post(f"{BASE}/api/predict-compare", json=SAMPLE_INPUT)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()["data"]
    assert "xgboost" in data, "Key 'xgboost' tidak ada di response"
    assert data["xgboost"] is not None
    assert data["xgboost"]["prediction"] is not None
    assert data["best_model"] in ["Linear Regression", "Random Forest", "XGBoost"]
    assert len(data["insight"]) >= 3
    print(f"OK Best model: {data['best_model']} ({data['best_prediction']:.4f})")


def test_model_info_xgboost():
    """AC-BE-06: Model info XGBoost."""
    response = requests.get(f"{BASE}/api/model-info?type=xgboost")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()["data"]
    assert "feature_importance" in data
    assert "hyperparameters" in data
    assert data["metrics"]["r2_score"] > 0
    print(f"OK XGBoost R2: {data['metrics']['r2_score']:.4f}")


def test_model_comparison_three_models():
    """AC-BE-07: Comparison endpoint tiga model."""
    response = requests.get(f"{BASE}/api/model-comparison")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()["data"]
    assert all(key in data for key in ["lr", "rf", "xgb"])
    assert data["xgb"]["available"] is True
    print(f"OK Best model from comparison: {data['best_model']}")


def test_invalid_model_type():
    """AC-BE-04: model_type tidak valid."""
    payload = {**SAMPLE_INPUT, "model_type": "neural_network"}
    response = requests.post(f"{BASE}/api/predict", json=payload)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
    print("OK Invalid model_type correctly rejected (400)")


if __name__ == "__main__":
    tests = [
        test_predict_xgboost,
        test_predict_compare_three_models,
        test_model_info_xgboost,
        test_model_comparison_three_models,
        test_invalid_model_type,
    ]
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"FAIL {test.__name__}: {exc}")
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
