from fastapi import APIRouter
from fraud_detection.api.dependencies import get_services

router = APIRouter()

@router.get("/history")
async def get_history(limit: int = 100, offset: int = 0):
    svc = get_services()
    records = svc.storage_service.get_recent(limit=limit, offset=offset)
    return {"records": records, "total": len(records)}
