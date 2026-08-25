import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.chunking import Chunk
from app.config import COLLECTION_NAME

VECTOR_SIZE = 768


# Inicializa el cliente en nube si existen credenciales o en almacenamiento local
def get_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")

    if url:
        return QdrantClient(url=url, api_key=api_key)

    host = os.environ.get("QDRANT_HOST")
    if host:
        return QdrantClient(
            host=host,
            port=int(os.environ.get("QDRANT_PORT", "6333")),
            api_key=api_key,
            https=os.environ.get("QDRANT_HTTPS", "false").lower() == "true",
        )

    # Persistencia local para entorno de desarrollo
    return QdrantClient(path="./qdrant_local_data")


# Crea la colección en Qdrant si no existe previamente
def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


# Indexa los chunks y sus vectores almacenando el texto en el payload
def index_chunks(client: QdrantClient, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    points = [
        PointStruct(
            # UUID estable para el perfil base; los recuerdos reciben un id de
            # chunk único, de modo que no se sobrescriben ni se duplican en
            # cada reinicio del servicio.
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"mireya:{chunks[i].id}")),
            vector=vectors[i],
            payload={"chunk_id": chunks[i].id, "section": chunks[i].section, "text": chunks[i].text},
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


# Busca los k vectores mas cercanos a la consulta
def search(client: QdrantClient, query_vector: list[float], top_k: int = 4, score_threshold: float | None = None) -> list[dict]:
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    ).points

    return [
        {
            "section": r.payload["section"],
            "text": r.payload["text"],
            "score": r.score,
        }
        for r in results
    ]
