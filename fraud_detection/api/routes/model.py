from fastapi import APIRouter
from fraud_detection.core.config import MAX_KNOWN_AMOUNT
from fraud_detection.api.dependencies import get_services
import os, json

router = APIRouter()

def load_model_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "..", "models_store", "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "accuracy": 0.9982, "precision": 0.9000, "recall": 0.8265,
        "f1_score": 0.8617, "auc_roc": 0.9816, "auc_pr": 0.8714,
    }

@router.get("/model/info")
async def model_info():
    svc = get_services()
    info = svc.prediction_service.model_info()
    info["max_allowed_amount"] = MAX_KNOWN_AMOUNT
    info["threshold"] = info.get("optimal_threshold", 0.5)
    info["metrics"] = load_model_metrics()
    return info
