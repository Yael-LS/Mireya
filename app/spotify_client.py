"""Cliente ligero para la Web API de Spotify (Client Credentials Flow).

Este flujo es 100% gratuito: no requiere que ningún usuario inicie sesión,
solo que la app se autentique con su propio Client ID/Secret para obtener
un token de acceso de solo-lectura a endpoints públicos como /search.
"""

import os
import time
import base64
import logging
import threading

import requests

logger = logging.getLogger("spotify")

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"

_token_lock = threading.Lock()
_cached_token: str | None = None
_token_expires_at: float = 0.0


def _get_credentials() -> tuple[str, str]:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Faltan SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET en el entorno."
        )
    return client_id, client_secret


def _get_access_token() -> str:
    """Obtiene (y cachea en memoria de proceso) un token de app-only."""
    global _cached_token, _token_expires_at

    with _token_lock:
        if _cached_token and time.time() < _token_expires_at:
            return _cached_token

        client_id, client_secret = _get_credentials()
        basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

        response = requests.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()

        _cached_token = payload["access_token"]
        # Margen de 60s para no usar el token justo al filo de su expiración.
        _token_expires_at = time.time() + payload.get("expires_in", 3600) - 60
        logger.info("Token de Spotify renovado.")
        return _cached_token


def buscar_cancion_spotify(nombre_cancion: str, artista: str = "") -> str:
    """Busca una canción en Spotify y devuelve su nombre, artista y URL reales.

    Úsala siempre que quieras mencionar y enlazar una canción o álbum en tu
    respuesta, en vez de escribir una URL de memoria. Nunca inventes un
    enlace de Spotify: si esta herramienta no encuentra la canción, dilo con
    naturalidad y no la enlaces.

    Args:
        nombre_cancion: título de la canción o álbum a buscar.
        artista: nombre del artista o intérprete, si se conoce (mejora la
            precisión de la búsqueda).
    """
    query = f"track:{nombre_cancion}"
    if artista:
        query += f" artist:{artista}"

    logger.info(f"[tool_call] buscar_cancion_spotify(nombre_cancion={nombre_cancion!r}, artista={artista!r})")

    try:
        token = _get_access_token()
        response = requests.get(
            SEARCH_URL,
            params={"q": query, "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        response.raise_for_status()
        items = response.json().get("tracks", {}).get("items", [])
    except Exception:
        logger.exception(f"Error buscando '{nombre_cancion}' en Spotify")
        return "No se pudo verificar esta canción en Spotify en este momento; menciónala sin enlace."

    if not items:
        return "No se encontró esta canción en Spotify; menciónala sin enlace."

    track = items[0]
    url = track["external_urls"]["spotify"]
    name = track["name"]
    artists = ", ".join(a["name"] for a in track["artists"])
    return f"Encontrada: \"{name}\" de {artists}. URL verificada: {url}"