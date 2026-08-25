"""Acceso a Supabase para el historial persistente de Mireya."""

import os
from functools import lru_cache

from fastapi import HTTPException
from supabase import Client, create_client


@lru_cache
def get_supabase() -> Client:
    """Crea el cliente con la service role key, que nunca se expone al navegador."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise HTTPException(
            status_code=503,
            detail="La persistencia no está configurada. Define SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY.",
        )
    return create_client(url, key)
