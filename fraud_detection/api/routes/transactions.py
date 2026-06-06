from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from fraud_detection.api.dependencies import get_services, verify_token
from pydantic import BaseModel

router = APIRouter()

class OverrideRequest(BaseModel):
    new_decision: str
    reason: str

@router.get("/transactions")
async def get_transactions(limit: int = 50, offset: int = 0, decision: Optional[str] = None, user=Depends(verify_token)):
    svc = get_services()
    records = svc.storage_service.get_transactions(limit, offset, decision)
    for rec in records:
        override = svc.storage_service.get_override(rec["transaction_id"])
        rec["overridden"] = override is not None
        rec["effective_decision"] = override["new_decision"] if override else rec["decision"]
        rec["overridden_by"] = override["overridden_by"] if override else None
    return {"transactions": records, "total": len(records)}

@router.post("/transactions/{tx_id}/override")
async def override_transaction(tx_id: str, req: OverrideRequest, user=Depends(verify_token)):
    svc = get_services()
    original = svc.storage_service.get_transaction(tx_id)
    if not original:
        raise HTTPException(status_code=404, detail="Transaction not found")
    svc.storage_service.set_override(tx_id, original["decision"], req.new_decision, user["sub"], req.reason)
    return {"status": "ok", "new_decision": req.new_decision}
