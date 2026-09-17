from fastapi.testclient import TestClient
from main import app
from PIL import Image
import io

client = TestClient(app)

def test_homepage_loads():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_rejects_empty_request():
    response = client.post("/predict")
    assert response.status_code == 422

def test_predict_returns_predictions():
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post("/predict", files={"file": ("leaf.jpg", buf, "image/jpeg")})
    data = response.json()

    assert response.status_code == 200
    assert len(data["predictions"]) == 3

    for pred in data["predictions"]:
        assert "species" in pred
        assert "status" in pred
        assert "confidence" in pred
        assert "symptoms" in pred
        assert "causes" in pred
        assert "cures" in pred