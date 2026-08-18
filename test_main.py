from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_homepage_loads():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_rejects_empty_request():
    response = client.post("/predict")
    assert response.status_code == 422