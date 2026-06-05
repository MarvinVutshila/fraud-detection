# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from fraud_detection.core.config import MODELS_DIR, DB_DSN, LOG_LEVEL, APPROVE_THRESHOLD, BLOCK_THRESHOLD
from fraud_detection.models.model_loader import load_artefacts
from fraud_detection.services.prediction_service import PredictionService
from fraud_detection.services.decision_service import DecisionService
from fraud_detection.services.storage_service import StorageService
from fraud_detection.database.postgres_db import Database
from fraud_detection.api.routes import router, set_services

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

class Services:
    """Simple container for all shared services."""
    pass

services = Services()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artefacts…")
    artefacts = load_artefacts(MODELS_DIR)

    # Use the thresholds from config, not the model’s optimal_threshold
    decision_service = DecisionService(
        approve_threshold=APPROVE_THRESHOLD,
        block_threshold=BLOCK_THRESHOLD
    )
    db = Database(DB_DSN)
    storage_service = StorageService(db)
    prediction_service = PredictionService(artefacts, decision_service, storage_service)

    services.prediction_service = prediction_service
    services.decision_service = decision_service
    services.storage_service = storage_service

    # Make services available to the routes module
    set_services(services)

    yield
    logger.info("Shutting down")

app = FastAPI(title="Fraud Detection API", version="3.0.0", lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)