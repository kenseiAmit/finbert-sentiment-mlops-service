"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

try:
    from utils import load_classifier
except ImportError:
    from app.utils import load_classifier

# Task 6: Configure standard Python logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("finbert_api")


# Task 6: Structured logging helper
def log(level: str, message: str, **kwargs) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "message": message,
        **kwargs,
    }
    log_line = json.dumps(entry)
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.log(lvl, log_line)


# Task 6: Prometheus metrics definition
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["sentiment"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds",
    buckets=[10, 25, 50, 75, 100, 250, 500, 1000, 2500, 5000],
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

load_dotenv()

classifiers = {}


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("INFO", "Loading model...")
    classifiers["sentiment"] = load_classifier()
    log("INFO", "Model loaded successfully")
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    try:
        # Task 3: Warn if any text exceeds 2000 chars
        for text in texts:
            if len(text) > 2000:
                log(
                    "WARNING",
                    "Input text exceeds 2000 characters and may be truncated",
                    text_length=len(text),
                )

        classifier = classifiers.get("sentiment")
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")

        # Task 3: Measure latency and run batch inference
        start_time = time.perf_counter()
        results = classifier(texts, truncation=True, max_length=512)
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        per_item_latency_ms = (
            total_latency_ms / len(texts) if texts else total_latency_ms
        )

        # Task 6: Observe latency metric
        PREDICTION_LATENCY.observe(total_latency_ms)

        prediction_results: list[PredictionResult] = []
        for text, res in zip(texts, results):
            label = str(res["label"]).lower()
            confidence = float(res["score"])

            # Task 6: Increment request metric by sentiment and log structured entry
            PREDICTION_REQUESTS.labels(sentiment=label).inc()
            log(
                "INFO",
                "Prediction generated",
                sentiment=label,
                confidence=confidence,
                latency_ms=per_item_latency_ms,
            )

            prediction_results.append(
                PredictionResult(
                    text=text,
                    sentiment=label,
                    confidence=confidence,
                    latency_ms=per_item_latency_ms,
                )
            )

        return prediction_results
    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_ERRORS.inc()
        log("ERROR", f"Prediction error: {e!s}", error=str(e))
        raise


def run_prediction(text: str) -> PredictionResult:
    """Helper function that accepts a single text, runs inference, and returns PredictionResult."""
    return run_predictions([text])[0]


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    """Health check endpoint."""
    if "sentiment" not in classifiers or classifiers["sentiment"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    """Predict sentiment for a single headline."""
    if "sentiment" not in classifiers or classifiers["sentiment"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")
    return run_prediction(request.text)


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    """Predict sentiment for a batch of headlines."""
    if "sentiment" not in classifiers or classifiers["sentiment"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=422, detail="Texts list cannot be empty")
    return run_predictions(request.texts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
    )
