"""Autenticación y límite de peticiones compartidos por todos los routers."""

import os

from dotenv import load_dotenv
from fastapi import HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()

# Limitador de tasa de peticiones por IP
limiter = Limiter(key_func=get_remote_address)

SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")


# Autenticacion basada en Bearer token
def verify_api_key(authorization: str | None) -> None:
    if not SERVICE_API_KEY:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Falta el header Authorization: Bearer <key>")

    token = authorization.removeprefix("Bearer ").strip()
    if token != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida")
