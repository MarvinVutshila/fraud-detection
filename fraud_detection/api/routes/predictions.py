from fastapi import APIRouter, HTTPException
from fraud_detection.schemas import TransactionRequest, PredictionResponse, BatchRequest, BatchResponse
from fraud_detection.core.config import MAX_KNOWN_AMOUNT
from fraud_detection.api.dependencies import get_services

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(tx: TransactionRequest):
    svc = get_services()
    if tx.Amount > MAX_KNOWN_AMOUNT:
        raise HTTPException(status_code=400, detail=f"Amount ${tx.Amount:,.2f} exceeds maximum ${MAX_KNOWN_AMOUNT:,.2f}.")
    try:
        result = svc.prediction_service.predict(tx, explain=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest):
    svc = get_services()
    for idx, tx in enumerate(req.transactions):
        if tx.Amount > MAX_KNOWN_AMOUNT:
            raise HTTPException(status_code=400, detail=f"Transaction {idx} amount ${tx.Amount:,.2f} exceeds limit.")
    results = []
    for tx in req.transactions:
        pred = svc.prediction_service.predict(tx, explain=False)
        results.append(pred)
    fraud_count = sum(1 for r in results if r.is_fraud)
    return BatchResponse(results=results, total=len(results), fraud_count=fraud_count)
