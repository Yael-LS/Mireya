"""Acceso a Supabase para el historial persistente de Mireya."""

import logging
import os
from functools import lru_cache

import httpx
from fastapi import HTTPException
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

logger = logging.getLogger("database")


class _RetryOnDisconnect(httpx.HTTPTransport):
    """Supabase cierra las conexiones inactivas; si el cliente reutiliza una ya
    cerrada, la petición falla con "Server disconnected" sin haber llegado al
    servidor. Se reintenta una vez, ya con una conexión nueva."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return super().handle_request(request)
        except httpx.RemoteProtocolError:
            logger.warning("Conexión con Supabase cerrada; reintentando una vez")
            return super().handle_request(request)


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
    # HTTP/1.1 con keep-alive corto: menos conexiones viejas que reutilizar.
    http_client = httpx.Client(
        transport=_RetryOnDisconnect(limits=httpx.Limits(keepalive_expiry=20)),
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
    )
    return create_client(url, key, options=SyncClientOptions(httpx_client=http_client))
