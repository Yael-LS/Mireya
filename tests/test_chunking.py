from app.chunking import load_and_chunk

def test_load_and_chunk_structure():
    # Carga el perfil real y verifica la división
    chunks = load_and_chunk("data/mireya_profile.md")

    assert len(chunks) > 0, "El perfil debería haber generado al menos 1 chunk."
    for chunk in chunks:
        assert chunk.section, "Cada chunk debe tener un título de sección."
        assert chunk.text.startswith(chunk.section), "El texto del chunk debe incluir el encabezado."


def test_load_and_chunk_splits_subsections(tmp_path):
    doc = tmp_path / "perfil.md"
    doc.write_text(
        "# Título\n\n"
        "## 1. Gustos\n\nIntro de gustos.\n\n"
        "### 1.1 Música\n\nTaylor Swift.\n\n"
        "### 1.2 Anime\n\nJujutsu Kaisen.\n\n---\n\n"
        "## 2. Frase\n\nUna frase.\n",
        encoding="utf-8",
    )

    chunks = load_and_chunk(str(doc))

    assert [c.section for c in chunks] == [
        "1. Gustos",
        "1. Gustos > 1.1 Música",
        "1. Gustos > 1.2 Anime",
        "2. Frase",
    ]
    assert chunks[1].text == "1. Gustos > 1.1 Música\nTaylor Swift."
    assert "---" not in chunks[2].text
    assert len({c.id for c in chunks}) == len(chunks)
