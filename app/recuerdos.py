"""Los recuerdos de Mireya: lo que ella le ha contado al biógrafo, para verlo y ponerle foto.

Los recuerdos viven en Qdrant porque son parte del RAG. La foto se guarda en el
bucket privado del Refugio y en el payload del punto solo queda su ruta y su pie
de foto; ninguno de los dos se vectoriza, así que cambiarlos no gasta embeddings.
"""

import logging
import uuid

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

import app.vectorstore
from app.config import COLLECTION_NAME
from app.database import get_supabase
from app.refugio import BUCKET, SIGNED_URL_SECONDS, remove_photo, upload_photo_to
from app.security import limiter, verify_api_key

logger = logging.getLogger("recuerdos")

router = APIRouter(prefix="/api/recuerdos")

MAX_CAPTION = 40


def upload_photo(photo_base64: str, photo_type: str) -> str:
    return upload_photo_to("recuerdos", photo_base64, photo_type)


def clean_caption(caption: str | None) -> str | None:
    return (caption or "").strip()[:MAX_CAPTION] or None


def _signed_urls(paths: list[str]) -> dict[str, str]:
    paths = list(dict.fromkeys(p for p in paths if p))
    if not paths:
        return {}
    try:
        signed = get_supabase().storage.from_(BUCKET).create_signed_urls(paths, SIGNED_URL_SECONDS)
    except Exception:
        logger.exception("No se pudieron firmar las fotos de los recuerdos")
        return {}
    return {entry.get("path"): entry.get("signedURL") or entry.get("signedUrl") for entry in signed}


def _to_memory(point_id, payload: dict, urls: dict[str, str]) -> dict:
    path = payload.get("photo_path")
    return {
        "id": str(point_id),
        "text": payload.get("text", ""),
        # Los recuerdos anteriores al 26/oct/2026 no tienen fecha.
        "created_at": payload.get("created_at"),
        "photo_url": urls.get(path) if path else None,
        "photo_caption": payload.get("photo_caption") if path else None,
    }


def memory_out(point_id, payload: dict) -> dict:
    return _to_memory(point_id, payload, _signed_urls([payload.get("photo_path")]))


def _memory_payload(client, memory_id: uuid.UUID) -> dict:
    points = client.retrieve(COLLECTION_NAME, ids=[str(memory_id)], with_payload=True, with_vectors=False)
    payload = (points[0].payload or {}) if points else {}
    # Solo los recuerdos: el perfil base no se edita desde aquí.
    if payload.get("section") != app.vectorstore.MEMORY_SECTION:
        raise HTTPException(status_code=404, detail="Ese recuerdo ya no existe.")
    return payload


def read_memory(memory_id) -> dict | None:
    client = app.vectorstore.get_client()
    points = client.retrieve(COLLECTION_NAME, ids=[str(memory_id)], with_payload=True, with_vectors=False)
    return memory_out(memory_id, points[0].payload or {}) if points else None


@router.get("")
def list_memories(authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    client = app.vectorstore.get_client()
    app.vectorstore.ensure_collection(client)
    found = []
    offset = None
    while True:
        points, offset = client.scroll(COLLECTION_NAME, limit=200, offset=offset, with_payload=True, with_vectors=False)
        found += [(p.id, p.payload or {}) for p in points if (p.payload or {}).get("section") == app.vectorstore.MEMORY_SECTION]
        if offset is None:
            break
    urls = _signed_urls([payload.get("photo_path") for _, payload in found])
    memories = [_to_memory(point_id, payload, urls) for point_id, payload in found]
    # Los más nuevos arriba; los que no tienen fecha, al final.
    dated = sorted((m for m in memories if m["created_at"]), key=lambda m: m["created_at"], reverse=True)
    return dated + [m for m in memories if not m["created_at"]]


class MemoryPatch(BaseModel):
    photo_base64: str | None = None
    photo_type: str | None = None
    photo_caption: str | None = Field(default=None, max_length=200)
    remove_photo: bool = False


@router.patch("/{memory_id}")
@limiter.limit("30/minute")
def update_memory(request: Request, memory_id: uuid.UUID, body: MemoryPatch, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    client = app.vectorstore.get_client()
    payload = _memory_payload(client, memory_id)
    old_path = payload.get("photo_path")

    if body.remove_photo:
        client.delete_payload(COLLECTION_NAME, keys=["photo_path", "photo_caption"], points=[str(memory_id)])
        remove_photo(old_path)
        payload.pop("photo_path", None)
        payload.pop("photo_caption", None)
        return memory_out(memory_id, payload)

    changes: dict = {}
    new_path = None
    if body.photo_base64:
        new_path = upload_photo(body.photo_base64, body.photo_type or "")
        changes["photo_path"] = new_path
    if body.photo_caption is not None:
        if not (new_path or old_path):
            raise HTTPException(status_code=422, detail="Primero ponle una foto al recuerdo.")
        changes["photo_caption"] = clean_caption(body.photo_caption)
    if not changes:
        return memory_out(memory_id, payload)

    try:
        client.set_payload(COLLECTION_NAME, payload=changes, points=[str(memory_id)])
    except Exception:
        logger.exception("No se pudo actualizar el recuerdo")
        remove_photo(new_path)
        raise HTTPException(status_code=500, detail="No se pudo guardar la foto del recuerdo.")
    if new_path:
        remove_photo(old_path)
    payload.update(changes)
    return memory_out(memory_id, payload)
