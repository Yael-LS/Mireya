from types import SimpleNamespace

from google.genai import errors, types

import app.agent as agent_module
from app.agent import STREAM_RESET, CVAgent


def _chunk(text):
    return SimpleNamespace(text=text, candidates=[])


def _agent_with_streams(monkeypatch, streams):
    """Agente cuyo cliente de Gemini devuelve, en orden, cada stream de `streams`."""
    monkeypatch.setattr(agent_module, "RETRY_DELAY_SECONDS", 0)
    calls = []

    def generate_content_stream(model, contents, config):
        calls.append(model)
        yield from streams[len(calls) - 1]()

    agent = object.__new__(CVAgent)
    agent.history = []
    agent.client = SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate_content_stream))
    return agent, calls


def test_fallo_a_media_respuesta_reinicia_y_completa(monkeypatch):
    def cortado():
        yield _chunk("Hola, ")
        raise errors.ServerError(504, {"error": {"code": 504, "message": "Deadline expired", "status": "DEADLINE_EXCEEDED"}}, None)

    def completo():
        yield _chunk("respuesta completa")

    agent, calls = _agent_with_streams(monkeypatch, [cortado, completo])
    salida = list(agent._stream_with_resilience(types.GenerateContentConfig()))

    assert salida == ["Hola, ", STREAM_RESET, "respuesta completa"]
    assert len(calls) == 2


def test_chat_descarta_el_texto_previo_al_reinicio(monkeypatch):
    agent = object.__new__(CVAgent)
    agent.stream_chat = lambda message: iter(["a medias", STREAM_RESET, "final"])
    assert agent.chat("hola") == "final"
