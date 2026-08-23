from app.chunking import load_and_chunk

def test_load_and_chunk_structure():
    # Carga el archivo CV real y verifica la división
    chunks = load_and_chunk("data/cv.md")
    
    assert len(chunks) > 0, "El CV debería haber generado al menos 1 chunk."
    for chunk in chunks:
        assert chunk.section, "Cada chunk debe tener un título de sección."
        assert chunk.text.startswith(chunk.section), "El texto del chunk debe incluir el encabezado."