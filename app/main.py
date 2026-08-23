import os
import re
import time
import uuid
import logging
import json
from contextlib import asynccontextmanager
from typing import Literal, Union, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.agent import CVAgent
from app.retrieval import index_cv
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Limitador de tasa de peticiones por IP
limiter = Limiter(key_func=get_remote_address)

MAX_HISTORY_MESSAGES = 6
MAX_MESSAGE_LENGTH = 2000

# Deteccion de patrones comunes de prompt injection para trazabilidad
SUSPICIOUS_PATTERNS = re.compile(
    r"(ignore (previous|all) instructions|olvida (tus |las )?instrucciones"
    r"|system prompt|actúa como|act as|you are now|eres ahora)",
    re.IGNORECASE,
)


# Inicializacion de la aplicacion e indexacion de documentos al arrancar
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Indexando CV en Qdrant...")
    n = index_cv("data/mireya_profile.md")
    logger.info(f"CV indexado correctamente: {n} chunks.")
    yield


app = FastAPI(title="CV Agent - Open Responses API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Por defecto acepta clientes web; en producción se puede restringir con CORS_ORIGINS.
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")


# Schemas para la especificacion Open Responses adaptada
class ContentPart(BaseModel):
    type: str = "input_text"
    text: str = ""


class InputMessage(BaseModel):
    role: Literal["user", "assistant"]
    type: str = "message"
    content: Union[str, List[ContentPart]]


class ResponsesRequest(BaseModel):
    model: str = "cv-agent"
    instructions: str | None = None
    input: Union[str, List[InputMessage]]
    stream: bool = False
    store: bool = False


class OutputTextContent(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class OutputItem(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["completed"] = "completed"
    content: list[OutputTextContent]


class ResponsesResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    model: str
    created_at: int
    status: Literal["completed"] = "completed"
    output: list[OutputItem]


# Autenticacion basada en Bearer token
def verify_api_key(authorization: str | None) -> None:
    if not SERVICE_API_KEY:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Falta el header Authorization: Bearer <key>")

    token = authorization.removeprefix("Bearer ").strip()
    if token != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida")


def _extract_text_from_content(content: Union[str, List[ContentPart]]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.text for part in content if hasattr(part, "text"))
    return ""


# Endpoint de respuestas de la API
@app.post("/v1/responses")
@limiter.limit("20/minute")
async def create_response(
    request: Request,
    authorization: str | None = Header(default=None),
):
    verify_api_key(authorization)

    raw_body = await request.body()
    logger.info(f"=== PAYLOAD RECIBIDO ===\n{raw_body.decode('utf-8', errors='ignore')}\n========================")

    try:
        body = ResponsesRequest.model_validate_json(raw_body)
    except Exception as e:
        logger.error(f"Error de validacion Pydantic: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    agent = CVAgent()

    if isinstance(body.input, str):
        user_message = body.input
    else:
        if not body.input:
            raise HTTPException(status_code=400, detail="input no puede ser una lista vacia")

        trimmed_input = body.input[-MAX_HISTORY_MESSAGES:]

        for msg in trimmed_input[:-1]:
            role = "model" if msg.role == "assistant" else "user"
            msg_text = _extract_text_from_content(msg.content)
            agent.history.append(types.Content(role=role, parts=[types.Part(text=msg_text)]))

        if trimmed_input[-1].role != "user":
            raise HTTPException(status_code=400, detail="El ultimo mensaje de 'input' debe ser del usuario")
        user_message = _extract_text_from_content(trimmed_input[-1].content)

    if len(user_message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Mensaje demasiado largo (máx {MAX_MESSAGE_LENGTH} caracteres)")

    if SUSPICIOUS_PATTERNS.search(user_message):
        logger.warning(f"[posible_injection] mensaje sospechoso recibido: {user_message[:200]!r}")

    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())

    if body.stream:
        def sse_event(event: str, payload: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def stream_events():
            yield sse_event("response.created", {
                "type": "response.created", "response": {"id": response_id, "object": "response", "model": body.model, "created_at": created_at, "status": "in_progress"},
            })
            try:
                for text in agent.stream_chat(user_message):
                    yield sse_event("response.output_text.delta", {
                        "type": "response.output_text.delta", "response_id": response_id, "delta": text,
                    })
                yield sse_event("response.completed", {
                    "type": "response.completed", "response": {"id": response_id, "object": "response", "model": body.model, "created_at": created_at, "status": "completed"},
                })
            except Exception:
                logger.exception("Error durante el streaming del agente")
                yield sse_event("error", {"type": "error", "message": "Error interno generando la respuesta"})

        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    try:
        answer_text = agent.chat(user_message)
    except Exception:
        logger.exception("Error generando respuesta del agente")
        raise HTTPException(status_code=500, detail="Error interno generando la respuesta")

    return ResponsesResponse(
        id=response_id,
        model=body.model,
        created_at=created_at,
        status="completed",
        output=[
            OutputItem(
                type="message",
                role="assistant",
                status="completed",
                content=[OutputTextContent(text=answer_text)],
            ),
        ],
    )


# Health check del servicio
@app.get("/health")
@app.head("/health")
def health():
    return {"status": "ok"}
