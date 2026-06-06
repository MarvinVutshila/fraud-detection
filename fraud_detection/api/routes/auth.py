from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fraud_detection.api.auth import create_access_token

router = APIRouter()

VALID_CREDENTIALS = {"analyst": "analyst123", "admin": "admin123"}

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/auth/login")
async def login(req: LoginRequest):
    if req.username in VALID_CREDENTIALS and VALID_CREDENTIALS[req.username] == req.password:
        token = create_access_token({"sub": req.username, "role": "analyst"})
        return {"access_token": token, "token_type": "bearer", "role": "analyst"}
    raise HTTPException(status_code=401, detail="Invalid credentials")
