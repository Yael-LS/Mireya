"""Llena El Refugio con el perfil y los recuerdos que ya están en Qdrant.

Uso (desde cv-agent-main/): python -m scripts.refugio_backfill
Se puede volver a correr: salta los recuerdos que ya aportaron algo y no
duplica títulos.
"""

import logging

from dotenv import load_dotenv

load_dotenv()

from app.refugio import backfill  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(backfill())
