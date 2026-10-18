import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    section: str
    text: str


SECTION_RE = re.compile(r"^##\s+(.+)$")
SUBSECTION_RE = re.compile(r"^###\s+(.+)$")


def load_and_chunk(filepath: str) -> list[Chunk]:
    # Lee un archivo Markdown y lo divide en un chunk por cada "###"; las
    # secciones "##" sin subtítulos (o el texto previo al primer "###") quedan
    # como un chunk propio. El título incluye la sección padre para que cada
    # chunk conserve su contexto, ej. "3. Gustos... > 3.1 Música".
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    chunks: list[Chunk] = []
    section: str | None = None
    subsection: str | None = None
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        lines.clear()
        if not section or not body:
            return
        title = f"{section} > {subsection}" if subsection else section
        chunks.append(Chunk(id=f"chunk_{len(chunks)}", section=title, text=f"{title}\n{body}"))

    for line in content.splitlines():
        if match := SUBSECTION_RE.match(line):
            flush()
            subsection = match.group(1).strip()
        elif match := SECTION_RE.match(line):
            flush()
            section = match.group(1).strip()
            subsection = None
        elif line.strip() != "---":
            lines.append(line)

    flush()
    return chunks
