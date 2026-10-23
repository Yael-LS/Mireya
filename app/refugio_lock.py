"""Clave del Refugio: distinta a la del sitio y cambiable por Mireya.

Se guarda solo el hash (PBKDF2) en una tabla de una fila; la clave nunca vive
en el código ni en variables de entorno.
"""

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.database import get_supabase
from app.security import limiter, verify_api_key

router = APIRouter(prefix="/api/refugio/lock")

TABLE = "refugio_lock"
ITERATIONS = 200_000
PIN_PATTERN = re.compile(r"^\d{6}$")


def hash_pin(pin: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}"


def check_pin(pin: str, stored: str) -> bool:
    try:
        _, iterations, salt, expected = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), int(iterations))
    except ValueError:
        return False
    return hmac.compare_digest(digest.hex(), expected)


def _current() -> dict | None:
    rows = get_supabase().table(TABLE).select("*").eq("id", 1).execute().data
    return rows[0] if rows else None


def save_pin(pin: str, hint: str) -> None:
    if not PIN_PATTERN.match(pin):
        raise HTTPException(status_code=422, detail="La clave debe tener 6 números.")
    get_supabase().table(TABLE).upsert({
        "id": 1,
        "pin_hash": hash_pin(pin),
        "hint": hint.strip()[:200] or None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


class PinIn(BaseModel):
    pin: str = Field(min_length=6, max_length=6)


class PinChange(BaseModel):
    current_pin: str = Field(min_length=6, max_length=6)
    new_pin: str = Field(min_length=6, max_length=6)
    hint: str = Field(default="", max_length=200)


@router.get("")
def lock_status(authorization: str | None = Header(default=None)):
    """Pista para la pantalla de entrada (nunca la clave)."""
    verify_api_key(authorization)
    row = _current()
    return {"configured": bool(row), "hint": row.get("hint") if row else None}


@router.post("/verify")
@limiter.limit("10/minute")
def verify_pin(request: Request, body: PinIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    row = _current()
    if not row or not check_pin(body.pin, row["pin_hash"]):
        raise HTTPException(status_code=401, detail="Esa no es la clave del Refugio.")
    return {"ok": True}


@router.post("/change")
@limiter.limit("5/minute")
def change_pin(request: Request, body: PinChange, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    row = _current()
    if not row or not check_pin(body.current_pin, row["pin_hash"]):
        raise HTTPException(status_code=401, detail="La clave actual no es correcta.")
    save_pin(body.new_pin, body.hint)
    return {"ok": True}
