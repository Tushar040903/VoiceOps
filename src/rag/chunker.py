"""Knowledge Base document parser and section-level chunker."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from src.config import KB_DIR


@dataclass
class KBChunk:
    """Represents a single indexed section of a knowledge base document."""
    section_id: str
    section_title: str
    doc_id: str
    doc_title: str
    doc_type: str
    status: str
    effective_from: str
    file_name: str
    content: str
    full_text: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "doc_type": self.doc_type,
            "status": self.status,
            "effective_from": self.effective_from,
            "file_name": self.file_name,
            "content": self.content,
        }


class KBChunker:
    """Chunks Markdown knowledge documents while strictly preserving section IDs and YAML metadata."""

    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = Path(kb_dir or KB_DIR)

    @staticmethod
    def _parse_frontmatter(text: str):
        """Parse YAML frontmatter enclosed in --- delimiters."""
        frontmatter = {}
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                except Exception:
                    frontmatter = {}
                body = parts[2].strip()
        return frontmatter, body

    def chunk_file(self, file_path: Path) -> List[KBChunk]:
        """Parse a single Markdown file into section chunks."""
        text = file_path.read_text(encoding="utf-8")
        meta, body = self._parse_frontmatter(text)

        doc_id = str(meta.get("doc_id", file_path.stem))
        doc_title = str(meta.get("title", file_path.stem))
        doc_type = str(meta.get("doc_type", "policy"))
        status = str(meta.get("status", "current"))
        effective_from = str(meta.get("effective_from", "2026-01-01"))

        # Regex to match ## [SECTION-ID] Section Title
        section_pattern = re.compile(r"^##\s*\[([A-Z0-9\.\-]+)\]\s*(.*)$", re.MULTILINE)
        matches = list(section_pattern.finditer(body))

        chunks: List[KBChunk] = []

        if not matches:
            # Fallback if document has no ## [SECTION-ID] headings
            chunks.append(
                KBChunk(
                    section_id=doc_id,
                    section_title=doc_title,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    doc_type=doc_type,
                    status=status,
                    effective_from=effective_from,
                    file_name=file_path.name,
                    content=body.strip(),
                    full_text=f"Document: {doc_title} ({doc_type}, status: {status})\n{body.strip()}",
                )
            )
            return chunks

        for i, match in enumerate(matches):
            sec_id = match.group(1).strip()
            sec_title = match.group(2).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            sec_content = body[start_pos:end_pos].strip()

            full_repr = (
                f"[{sec_id}] {sec_title}\n"
                f"Source: {doc_title} (Type: {doc_type}, Status: {status}, Effective: {effective_from})\n"
                f"{sec_content}"
            )

            chunks.append(
                KBChunk(
                    section_id=sec_id,
                    section_title=sec_title,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    doc_type=doc_type,
                    status=status,
                    effective_from=effective_from,
                    file_name=file_path.name,
                    content=sec_content,
                    full_text=full_repr,
                )
            )

        return chunks

    def chunk_all(self) -> List[KBChunk]:
        """Chunk all markdown files in the knowledge base directory."""
        all_chunks: List[KBChunk] = []
        for file_path in sorted(self.kb_dir.glob("*.md")):
            all_chunks.extend(self.chunk_file(file_path))
        return all_chunks

