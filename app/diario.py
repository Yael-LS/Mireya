"""El diario de Mireya: hojas por día, carpetitas, cartas al futuro y calendario.

Vive aparte del biógrafo: nada de esto se indexa en Qdrant ni llega a la IA.
"""

import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

import requests
from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.database import get_supabase
from app.refugio import BUCKET, PHOTO_TYPES, SIGNED_URL_SECONDS, _decode_photo, _ensure_bucket
from app.security import limiter, verify_api_key
from app.spotify_client import SEARCH_URL, _get_access_token

router = APIRouter(prefix="/api/diario")

ENTRIES = "diary_entries"
FOLDERS = "diary_folders"
LETTERS = "diary_letters"
SETTINGS = "diary_settings"
MAX_TEXT = 100_000
# Ciudad de México ya no cambia de horario: el "hoy" de sus cartas no depende del servidor.
CDMX = timezone(timedelta(hours=-6))


def today() -> date:
    return datetime.now(CDMX).date()


def _signed_urls(paths: list[str]) -> dict[str, str]:
    paths = list(dict.fromkeys(p for p in paths if p))
    if not paths:
        return {}
    signed = get_supabase().storage.from_(BUCKET).create_signed_urls(paths, SIGNED_URL_SECONDS)
    return {entry.get("path"): entry.get("signedURL") or entry.get("signedUrl") for entry in signed}


def _visible_letter(row: dict) -> dict:
    """Una carta sellada no sale del servidor hasta su fecha."""
    locked = row["sealed"] and date.fromisoformat(row["open_on"]) > today()
    return {**row, "body": None if locked else row["body"], "locked": locked}


def _photo_token(value: str | None) -> str | None:
    return value[6:] if value and value.startswith("photo:") else None


# ---------- Todo el diario ----------

@router.get("")
def load_diary(authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    db = get_supabase()
    entries = db.table(ENTRIES).select("*").order("day").execute().data or []
    folders = db.table(FOLDERS).select("*").order("used_at", desc=True).execute().data or []
    letters = db.table(LETTERS).select("*").order("open_on").execute().data or []
    rows = db.table(SETTINGS).select("*").eq("id", 1).execute().data or []
    settings = rows[0] if rows else {}

    photos = [item.get("path") for entry in entries for item in entry.get("items") or [] if item.get("type") == "photo"]
    backgrounds = [settings.get("bg_all"), settings.get("my_bg"), *(settings.get("bg_month") or {}).values()]
    photos += [_photo_token(value) for value in backgrounds]
    urls = _signed_urls(photos)
    for entry in entries:
        for item in entry.get("items") or []:
            if item.get("type") == "photo":
                item["url"] = urls.get(item.get("path"))
    return {
        "today": today().isoformat(),
        "entries": entries,
        "folders": folders,
        "letters": [_visible_letter(row) for row in letters],
        "settings": settings,
        "bg_urls": urls,
    }


# ---------- Hojas ----------

class EntryIn(BaseModel):
    body: str | None = Field(default=None, max_length=MAX_TEXT)
    feel: str | None = Field(default=None, max_length=300)
    moods: list[str] | None = None
    items: list[dict[str, Any]] | None = None
    folder_id: uuid.UUID | None = None
    clear_folder: bool = False
    ink: str | None = Field(default=None, max_length=20)
    font: str | None = Field(default=None, max_length=20)


@router.patch("/entries/{day}")
def save_entry(day: date, body: EntryIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    if day > today():
        raise HTTPException(status_code=422, detail="Ese día todavía no llega.")
    row: dict[str, Any] = {"day": day.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat()}
    for field in ("body", "feel", "moods", "ink", "font"):
        value = getattr(body, field)
        if value is not None:
            row[field] = value
    if body.items is not None:
        # El enlace firmado caduca: solo se guarda la ruta de la foto.
        row["items"] = [{k: v for k, v in item.items() if k not in ("url", "fresh")} for item in body.items]
    if body.clear_folder:
        row["folder_id"] = None
    elif body.folder_id is not None:
        row["folder_id"] = str(body.folder_id)
    get_supabase().table(ENTRIES).upsert(row).execute()
    return {"ok": True}


class PhotoIn(BaseModel):
    photo_base64: str
    photo_type: str


@router.post("/photos", status_code=201)
def upload_photo(body: PhotoIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    data = _decode_photo(body.photo_base64, body.photo_type)
    _ensure_bucket()
    path = f"diario/{uuid.uuid4().hex}.{PHOTO_TYPES[body.photo_type]}"
    get_supabase().storage.from_(BUCKET).upload(path, data, {"content-type": body.photo_type})
    return {"path": path, "url": _signed_urls([path]).get(path)}


# ---------- Canciones ----------

@router.get("/spotify")
@limiter.limit("60/minute")
def search_songs(request: Request, q: str = Query(min_length=1, max_length=120), authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    try:
        response = requests.get(
            SEARCH_URL,
            params={"q": q, "type": "track", "limit": 8, "market": "MX"},
            headers={"Authorization": f"Bearer {_get_access_token()}"},
            timeout=10,
        )
        response.raise_for_status()
    except Exception:
        raise HTTPException(status_code=502, detail="Spotify no respondió. Intenta otra vez en un momento.")
    songs = []
    for track in response.json().get("tracks", {}).get("items", []):
        images = track.get("album", {}).get("images") or []
        songs.append({
            "id": track["id"],
            "t": track["name"],
            "a": ", ".join(artist["name"] for artist in track["artists"]),
            "art": images[-1]["url"] if images else None,
        })
    return songs


# ---------- Carpetitas ----------

class FolderIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    color: str = Field(max_length=20)


class FolderPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=40)
    color: str | None = Field(default=None, max_length=20)
    touch: bool = False


@router.post("/folders", status_code=201)
def add_folder(body: FolderIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    return get_supabase().table(FOLDERS).insert({"name": body.name.strip(), "color": body.color}).execute().data[0]


@router.patch("/folders/{folder_id}")
def update_folder(folder_id: uuid.UUID, body: FolderPatch, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    changes: dict[str, Any] = {}
    if body.name is not None:
        changes["name"] = body.name.strip()
    if body.color is not None:
        changes["color"] = body.color
    if body.touch:
        changes["used_at"] = datetime.now(timezone.utc).isoformat()
    if not changes:
        raise HTTPException(status_code=422, detail="No hay cambios.")
    result = get_supabase().table(FOLDERS).update(changes).eq("id", str(folder_id)).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="No encontré esa carpetita.")
    return result.data[0]


# ---------- Cartas a su yo del futuro ----------

class LetterIn(BaseModel):
    open_on: date
    sealed: bool
    body: str = Field(min_length=1, max_length=MAX_TEXT)
    ink: str | None = Field(default=None, max_length=20)
    font: str | None = Field(default=None, max_length=20)


@router.post("/letters", status_code=201)
def add_letter(body: LetterIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    if body.open_on <= today():
        raise HTTPException(status_code=422, detail="La carta tiene que ser para un día que todavía no llega.")
    row = {"open_on": body.open_on.isoformat(), "sealed": body.sealed, "body": body.body, "ink": body.ink, "font": body.font}
    return _visible_letter(get_supabase().table(LETTERS).insert(row).execute().data[0])


@router.patch("/letters/{letter_id}/open")
def open_letter(letter_id: uuid.UUID, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    db = get_supabase()
    rows = db.table(LETTERS).select("*").eq("id", str(letter_id)).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="No encontré esa carta.")
    letter = _visible_letter(rows[0])
    if letter["locked"]:
        raise HTTPException(status_code=403, detail="Esta carta sigue sellada.")
    db.table(LETTERS).update({"opened": True}).eq("id", str(letter_id)).execute()
    return {**letter, "opened": True}


# ---------- Calendario y preferencias ----------

class SettingsIn(BaseModel):
    era_by: list[int] | None = Field(default=None, min_length=12, max_length=12)
    cal_see: int | None = Field(default=None, ge=10, le=60)
    bg_all: str | None = Field(default=None, max_length=300)
    clear_bg_all: bool = False
    bg_month: dict[str, str] | None = None
    my_bg: str | None = Field(default=None, max_length=300)
    ink: str | None = Field(default=None, max_length=20)
    font: str | None = Field(default=None, max_length=20)


@router.patch("/settings")
def save_settings(body: SettingsIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    row = {"id": 1, **body.model_dump(exclude={"clear_bg_all"}, exclude_none=True)}
    if body.clear_bg_all:
        row["bg_all"] = None
    get_supabase().table(SETTINGS).upsert(row).execute()
    return {"ok": True}
