from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_responses_endpoint_missing_payload():
    # Intento de petición inválida (cuerpo vacío)
    response = client.post("/v1/responses", json={})
    assert response.status_code in [400, 422], "Debe validar esquemas de entrada Pydantic."