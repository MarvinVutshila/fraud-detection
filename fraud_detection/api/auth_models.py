from pydantic import BaseModel
from typing import Optional

class LoginRequest(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    username: str
    password: str
    avatar_url: Optional[str] = None
