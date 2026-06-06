from fastapi import HTTPException, Depends
from fraud_detection.api.auth import verify_token

_services = None

def set_services(services):
    global _services
    _services = services

def get_services():
    if _services is None:
        raise HTTPException(status_code=503, detail="Services not initialised")
    return _services
