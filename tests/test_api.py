"""
Integration tests for the Sentiment Analysis API.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "text",
    [
        "The company reported record quarterly profits and raised dividends.",
        "The startup filed for bankruptcy following substantial financial losses.",
        "Markets closed flat on Monday amid cautious trading.",
    ],
)
def test_predict_returns_valid_response(client: TestClient, text: str):
    """Test that POST /predict returns the correct response schema and valid sentiment."""
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200

    data = response.json()
    assert "text" in data
    assert "sentiment" in data
    assert "confidence" in data
    assert "latency_ms" in data

    assert data["text"] == text
    assert data["sentiment"] in {"positive", "negative", "neutral"}
    assert isinstance(data["confidence"], (float, int))
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["latency_ms"], (float, int))
    assert data["latency_ms"] >= 0


def test_predict_batch(client: TestClient):
    """Test that POST /predict/batch returns a list of results matching the input length and schema."""
    texts = [
        "Company quarterly earnings exceeded analyst forecasts.",
        "Shares plummeted following unexpected executive departures.",
        "Central bank kept interest rates unchanged this period.",
    ]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == len(texts)

    for i, item in enumerate(data):
        assert item["text"] == texts[i]
        assert item["sentiment"] in {"positive", "negative", "neutral"}
        assert 0.0 <= item["confidence"] <= 1.0
        assert item["latency_ms"] >= 0


def test_predict_batch_empty_list_returns_422(client: TestClient):
    """Test that POST /predict/batch returns HTTP 422 for an empty texts list."""
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


def test_predict_empty_text_returns_422(client: TestClient):
    """Test that POST /predict returns HTTP 422 for empty text."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_health_check(client: TestClient):
    """Test that GET /health returns status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
