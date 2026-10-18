"""Caja de ideas: lo que a Mireya le gustaría que su scrapbook tuviera después."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.database import get_supabase
from app.security import verify_api_key

router = APIRouter(prefix="/api/ideas")

TABLE = "scrapbook_ideas"


class IdeaIn(BaseModel):
    text: str = Field(min_length=1, max_length=300)


class IdeaPatch(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=300)
    done: bool | None = None


@router.get("")
def list_ideas(authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    # Pendientes primero; dentro de cada grupo, las más recientes arriba.
    return (
        get_supabase().table(TABLE).select("*")
        .order("done")
        .order("created_at", desc=True)
        .execute().data or []
    )


@router.post("", status_code=201)
def add_idea(body: IdeaIn, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="La idea no puede estar vacía.")
    return get_supabase().table(TABLE).insert({"text": text}).execute().data[0]


@router.patch("/{idea_id}")
def update_idea(idea_id: uuid.UUID, body: IdeaPatch, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    changes: dict = {}
    if body.text is not None:
        changes["text"] = body.text.strip()
    if body.done is not None:
        changes["done"] = body.done
        changes["done_at"] = datetime.now(timezone.utc).isoformat() if body.done else None
    if not changes:
        raise HTTPException(status_code=422, detail="No hay cambios.")
    result = get_supabase().table(TABLE).update(changes).eq("id", str(idea_id)).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="No encontré esa idea.")
    return result.data[0]


@router.delete("/{idea_id}", status_code=204)
def delete_idea(idea_id: uuid.UUID, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    get_supabase().table(TABLE).delete().eq("id", str(idea_id)).execute()
