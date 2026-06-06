from fastapi import APIRouter, HTTPException
import requests
from fraud_detection.schemas import TransactionRequest
from fraud_detection.api.dependencies import get_services

router = APIRouter()

MOCKAROO_URL = "https://api.mockaroo.com/api/your_endpoint?key=your_key"

@router.post("/ingest")
async def ingest_from_mockaroo():
    try:
        resp = requests.get(MOCKAROO_URL, timeout=30)
        resp.raise_for_status()
        tx_list = resp.json()
        svc = get_services()
        count = 0
        for tx_dict in tx_list:
            tx = TransactionRequest(
                Time=tx_dict.get("Time", 0.0),
                Amount=tx_dict.get("Amount", 0.0),
                transaction_id=tx_dict.get("transaction_id"),
                **{f"V{i}": tx_dict.get(f"V{i}", 0.0) for i in range(1, 29)}
            )
            svc.prediction_service.predict(tx, explain=False)
            count += 1
        return {"status": "ok", "ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
