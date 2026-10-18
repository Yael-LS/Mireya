"""Foto de perfil de Mireya: la que ella elija reemplaza a la original del sitio.

Vive en el bucket privado del Refugio en una ruta fija, así no hace falta una
tabla: si el archivo existe, es su foto; si no, el frontend usa /mireya.jpg.
"""

import logging

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel

from app.database import get_supabase
from app.refugio import BUCKET, SIGNED_URL_SECONDS, _decode_photo, _ensure_bucket
from app.security import limiter, verify_api_key

logger = logging.getLogger("profile_photo")

router = APIRouter(prefix="/api/profile/photo")

PHOTO_PATH = "perfil/avatar.jpg"


class AvatarIn(BaseModel):
    photo_base64: str
    photo_type: str


def _signed_url() -> str | None:
    try:
        signed = get_supabase().storage.from_(BUCKET).create_signed_url(PHOTO_PATH, SIGNED_URL_SECONDS)
        return signed.get("signedURL") or signed.get("signedUrl")
    except Exception:
        # No existe todavía: se usa la foto original.
        return None


@router.get("")
def get_photo(authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    return {"url": _signed_url()}


@router.post("")
@limiter.limit("10/minute")
def set_photo(request: Request, body: AvatarIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    data = _decode_photo(body.photo_base64, body.photo_type)
    _ensure_bucket()
    get_supabase().storage.from_(BUCKET).upload(
        PHOTO_PATH, data, {"content-type": body.photo_type, "upsert": "true"}
    )
    return {"url": _signed_url()}


@router.delete("")
def reset_photo(authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    try:
        get_supabase().storage.from_(BUCKET).remove([PHOTO_PATH])
    except Exception:
        logger.exception("No se pudo quitar la foto de perfil")
    return {"url": None}
