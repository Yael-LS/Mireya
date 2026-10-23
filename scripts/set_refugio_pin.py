"""Define (o restablece) la clave del Refugio sin pedir la anterior.

Uso (desde cv-agent-main/): python -m scripts.set_refugio_pin
Pide la clave sin mostrarla en pantalla y la pista. Solo se guarda el hash.
"""

from getpass import getpass

from dotenv import load_dotenv

load_dotenv()

from app.refugio_lock import save_pin  # noqa: E402

if __name__ == "__main__":
    pin = getpass("Clave del Refugio (6 números): ").strip()
    if pin != getpass("Repite la clave: ").strip():
        raise SystemExit("Las claves no coinciden.")
    hint = input("Pista: ").strip()
    save_pin(pin, hint)
    print("Clave del Refugio guardada.")
