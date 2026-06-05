# fraud_detection/api/routes.py
from fastapi import APIRouter, HTTPException
from typing import List, Optional
import logging

from fraud_detection.models.schemas import (
    TransactionRequest,
    PredictionResponse,
    BatchRequest,
    BatchResponse,
    HistoryResponse,
)
from fraud_detection.core.config import MAX_KNOWN_AMOUNT

logger = logging.getLogger(__name__)

router = APIRouter()
_services = None

def set_services(services):
    global _services
    _services = services

def get_services():
    if _services is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    return _services

@router.get("/")
async def root():
    return {
        "message": "Fraud Detection API is running",
        "docs": "/docs",
        "endpoints": ["/health", "/model/info", "/predict", "/predict/batch", "/history"]
    }

@router.get("/health")
async def health():
    svc = get_services()
    return {"status": "ok" if svc.prediction_service else "down"}

@router.get("/model/info")
async def model_info():
    svc = get_services()
    info = svc.prediction_service.model_info()
    info["max_allowed_amount"] = MAX_KNOWN_AMOUNT
    # Add 'threshold' key for Streamlit compatibility
    info["threshold"] = info.get("optimal_threshold", 0.5)
    return info

@router.post("/predict", response_model=PredictionResponse)
async def predict(tx: TransactionRequest):
    svc = get_services()
    if tx.Amount > MAX_KNOWN_AMOUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Amount ${tx.Amount:,.2f} exceeds maximum ${MAX_KNOWN_AMOUNT:,.2f}."
        )
    try:
        result = svc.prediction_service.predict(tx, explain=True)
        return PredictionResponse(
            transaction_id=result.transaction_id,
            fraud_probability=result.fraud_probability,
            decision=result.decision,
            risk_level=result.risk_level,
            threshold=result.threshold,
            explanation=result.explanation,
            is_fraud=(result.decision == "BLOCK")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest):
    svc = get_services()
    for idx, tx in enumerate(req.transactions):
        if tx.Amount > MAX_KNOWN_AMOUNT:
            raise HTTPException(
                status_code=400,
                detail=f"Transaction {idx} amount ${tx.Amount:,.2f} exceeds limit."
            )
    results = []
    for tx in req.transactions:
        pred = svc.prediction_service.predict(tx, explain=False)
        results.append(PredictionResponse(
            transaction_id=pred.transaction_id,
            fraud_probability=pred.fraud_probability,
            decision=pred.decision,
            risk_level=pred.risk_level,
            threshold=pred.threshold,
            explanation=pred.explanation,
            is_fraud=(pred.decision == "BLOCK")
        ))
    fraud_count = sum(1 for r in results if r.is_fraud)
    return BatchResponse(results=results, total=len(results), fraud_count=fraud_count)

@router.get("/history", response_model=HistoryResponse)
async def get_history(limit: int = 100, offset: int = 0):
    svc = get_services()
    records = svc.storage_service.get_recent(limit=limit, offset=offset)
    return HistoryResponse(records=records, total=len(records))