import json
import pytest

from src.entities import HeartModel
from fastapi.testclient import TestClient
from app import app, load_model


@pytest.fixture()
def init_data():
    data = HeartModel(
        data=[
            [36, 0, 0, 90, 200, 0, 0, 150, 0, 0, 0, 0, 3],
            [36, 0, 0, 90, 200, 0, 0, 150, 0, 0, 0, 0, 3],
        ],
        features=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    )
    return data


def test_entry_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert 200 == response.status_code


def test_status_endpoint():
    with TestClient(app) as client:
        response = client.get("/status")
        assert 200 == response.status_code
        assert "Model is ready: True" == response.json()


def test_predict_endpoint(init_data):
    with TestClient(app) as client:
        response = client.get(
            "/predict/",
            json={"data": init_data.data, "features": init_data.features}
        )
        assert 200 == response.status_code
        assert len(init_data.data) == len(response.json())
        assert '0' == response.json()[0]["id"]
        assert response.json()[0]["target"] in [0, 1]
