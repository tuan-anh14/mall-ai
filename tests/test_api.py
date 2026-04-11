"""
Integration tests for FastAPI endpoints (no DB required).
Uses TestClient with mocked engine.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


@patch("api.routes.engine")
def test_recommend(mock_engine):
    mock_engine.recommend_for_user.return_value = ["p1", "p2", "p3"]
    response = client.post("/recommend", json={"userId": "user1", "limit": 3})
    assert response.status_code == 200
    assert response.json()["productIds"] == ["p1", "p2", "p3"]
    mock_engine.recommend_for_user.assert_called_once_with("user1", 3)


@patch("api.routes.engine")
def test_similar(mock_engine):
    mock_engine.similar_products.return_value = ["p2", "p3"]
    response = client.post("/similar", json={"productId": "p1", "limit": 2})
    assert response.status_code == 200
    assert response.json()["productIds"] == ["p2", "p3"]
