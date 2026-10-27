import logging
import uuid
from datetime import datetime, timezone

from app.chunking import Chunk, load_and_chunk
from app.embeddings import embed_documents, embed_query
import app.vectorstore


# Pipeline de indexacion desde archivo Markdown hacia la base vectorial
def index_cv(filepath: str) -> int:
    chunks = load_and_chunk(filepath)

    client = app.vectorstore.get_client()
    app.vectorstore.ensure_collection(client)

    # Solo se vectoriza lo que cambió: cada arranque gastaba 54 embeddings del
    # límite gratuito (100/min) aunque el perfil fuera idéntico.
    stored = app.vectorstore.stored_texts(client, [app.vectorstore.point_id(c.id) for c in chunks])
    changed = [c for c in chunks if stored.get(app.vectorstore.point_id(c.id)) != c.text]
    if changed:
        vectors = embed_documents([c.text for c in changed])
        app.vectorstore.index_chunks(client, changed, vectors)

    removed = app.vectorstore.delete_stale_profile_points(
        client, {app.vectorstore.point_id(c.id) for c in chunks}
    )
    logging.getLogger("retrieval").info(
        f"Perfil: {len(changed)} chunks nuevos o cambiados, {len(chunks) - len(changed)} sin cambios, {removed} borrados."
    )

    return len(chunks)


# Pipeline de recuperacion de contexto para consultas del usuario
def retrieve(query: str, top_k: int = 4) -> list[dict]:
    query_vector = embed_query(query)

    client = app.vectorstore.get_client()
    app.vectorstore.ensure_collection(client)
    
    results = app.vectorstore.search(client, query_vector, top_k=top_k)

    return results


def ingest_memory(text: str) -> list[str]:
    """Vectoriza un recuerdo libre, lo añade a la colección RAG existente y
    devuelve los ids de sus chunks."""
    clean_text = text.strip()
    if not clean_text:
        return []

    # Para una entrada de UI se conserva el texto completo como una unidad
    # semántica; se etiqueta para distinguirlo del perfil base en recuperación.
    chunks = [Chunk(id=f"memory-{uuid.uuid4()}", section="Nuevo recuerdo", text=clean_text)]
    vectors = embed_documents([clean_text])
    client = app.vectorstore.get_client()
    app.vectorstore.ensure_collection(client)
    # Fecha de alta: Qdrant no la guarda por sí solo y sin ella no se sabe cuándo llegó cada recuerdo.
    created_at = datetime.now(timezone.utc).isoformat()
    app.vectorstore.index_chunks(client, chunks, vectors, {"created_at": created_at})
    logging.getLogger("retrieval").info(f"[recuerdo] {chunks[0].id} guardado el {created_at} ({len(clean_text)} caracteres)")
    return [chunk.id for chunk in chunks]
