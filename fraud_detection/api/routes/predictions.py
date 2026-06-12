from fastapi import APIRouter, HTTPException
import uuid
from fraud_detection.schemas import TransactionRequest, PredictionResponse, BatchRequest, BatchResponse
from fraud_detection.core.config import MAX_KNOWN_AMOUNT
from fraud_detection.api.dependencies import get_services

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(tx: TransactionRequest):
    # Auto‑generate transaction_id if missing
    if not tx.transaction_id:
        tx.transaction_id = str(uuid.uuid4())
    
    svc = get_services()
    if tx.Amount > MAX_KNOWN_AMOUNT:
        raise HTTPException(status_code=400, detail=f"Amount ${tx.Amount:,.2f} exceeds maximum ${MAX_KNOWN_AMOUNT:,.2f}.")
    try:
        result = svc.prediction_service.predict(tx, explain=True)
        # Ensure the response includes the (possibly generated) transaction_id
        if not result.transaction_id:
            result.transaction_id = tx.transaction_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest):
    svc = get_services()
    for idx, tx in enumerate(req.transactions):
        if tx.Amount > MAX_KNOWN_AMOUNT:
            raise HTTPException(status_code=400, detail=f"Transaction {idx} amount ${tx.Amount:,.2f} exceeds limit.")
        # Optionally generate IDs for batch transactions too
        if not tx.transaction_id:
            tx.transaction_id = str(uuid.uuid4())
    results = []
    for tx in req.transactions:
        pred = svc.prediction_service.predict(tx, explain=False)
        if not pred.transaction_id:
            pred.transaction_id = tx.transaction_id
        results.append(pred)
    fraud_count = sum(1 for r in results if r.is_fraud)
    return BatchResponse(results=results, total=len(results), fraud_count=fraud_count)
