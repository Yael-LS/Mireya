import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    section: str
    text: str


def load_and_chunk(filepath: str) -> list[Chunk]:
    # Lee un archivo Markdown y divide el contenido en chunks según los encabezados "##"
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Parseo de secciones "## Título\nContenido"
    pattern = r"^##\s+(.+?)\n(.*?)(?=^##\s+|\Z)"
    matches = re.finditer(pattern, content, flags=re.MULTILINE | re.DOTALL)

    chunks = []
    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        section_text = match.group(2).strip()

        if not section_text:
            continue

        chunks.append(
            Chunk(
                id=f"chunk_{i}",
                section=section_title,
                text=f"{section_title}\n{section_text}",
            )
        )

    return chunks