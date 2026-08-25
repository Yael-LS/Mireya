import uuid

from app.chunking import Chunk, load_and_chunk
from app.embeddings import embed_documents, embed_query
import app.vectorstore


# Pipeline de indexacion desde archivo Markdown hacia la base vectorial
def index_cv(filepath: str) -> int:
    chunks = load_and_chunk(filepath)

    texts = [c.text for c in chunks]
    vectors = embed_documents(texts)

    client = app.vectorstore.get_client()
    app.vectorstore.ensure_collection(client)
    app.vectorstore.index_chunks(client, chunks, vectors)

    return len(chunks)


# Pipeline de recuperacion de contexto para consultas del usuario
def retrieve(query: str, top_k: int = 4) -> list[dict]:
    query_vector = embed_query(query)

    client = app.vectorstore.get_client()
    app.vectorstore.ensure_collection(client)
    
    results = app.vectorstore.search(client, query_vector, top_k=top_k)

    return results


def ingest_memory(text: str) -> int:
    """Vectoriza un recuerdo libre y lo añade a la colección RAG existente."""
    clean_text = text.strip()
    if not clean_text:
        return 0

    # Para una entrada de UI se conserva el texto completo como una unidad
    # semántica; se etiqueta para distinguirlo del perfil base en recuperación.
    chunks = [Chunk(id=f"memory-{uuid.uuid4()}", section="Nuevo recuerdo", text=clean_text)]
    vectors = embed_documents([clean_text])
    client = app.vectorstore.get_client()
    app.vectorstore.ensure_collection(client)
    app.vectorstore.index_chunks(client, chunks, vectors)
    return len(chunks)
