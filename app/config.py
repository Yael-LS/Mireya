"""Configuración centralizada del servicio.

Los nombres nuevos se prefieren, pero se conserva ``GEMINI_API_KEY`` como
compatibilidad con despliegues anteriores.
"""

import os


def get_google_api_key() -> str | None:
    """Obtiene la clave de Google GenAI sin acoplarla a un proveedor concreto."""
    return os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mireya_collection")

