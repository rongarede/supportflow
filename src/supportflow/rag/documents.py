from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import Field, model_validator

from supportflow.domain.models import StrictModel


class PolicyDocument(StrictModel):
    document_id: str
    version: str
    effective_from: datetime
    effective_to: datetime | None
    policy_type: str
    title: str
    body: str

    @model_validator(mode="after")
    def normalise_dates(self) -> PolicyDocument:
        if self.effective_from.tzinfo is None:
            self.effective_from = self.effective_from.replace(tzinfo=UTC)
        else:
            self.effective_from = self.effective_from.astimezone(UTC)
        if self.effective_to:
            if self.effective_to.tzinfo is None:
                self.effective_to = self.effective_to.replace(tzinfo=UTC)
            else:
                self.effective_to = self.effective_to.astimezone(UTC)
        return self

    def active_at(self, as_of: datetime) -> bool:
        current = as_of.astimezone(UTC)
        return self.effective_from <= current and (
            self.effective_to is None or current <= self.effective_to
        )


class PolicyChunk(StrictModel):
    evidence_id: str
    document: PolicyDocument
    heading: str
    text: str = Field(min_length=1)
    ordinal: int


def load_policy_documents(directory: Path) -> list[PolicyDocument]:
    documents: list[PolicyDocument] = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        if not raw.startswith("---\n"):
            raise ValueError(f"{path} has no YAML frontmatter")
        _, frontmatter, body = raw.split("---", 2)
        metadata = yaml.safe_load(frontmatter)
        documents.append(PolicyDocument(**metadata, body=body.strip()))
    return documents


def chunk_policy_document(document: PolicyDocument, chunk_size: int = 800, overlap: int = 100) -> list[PolicyChunk]:
    sections: list[tuple[str, str]] = []
    heading = "preamble"
    content: list[str] = []
    for line in document.body.splitlines():
        if line.startswith("#"):
            if content:
                sections.append((heading, "\n".join(content).strip()))
            heading = line.lstrip("#").strip()
            content = []
        else:
            content.append(line)
    if content:
        sections.append((heading, "\n".join(content).strip()))
    chunks: list[PolicyChunk] = []
    for section_heading, text in sections:
        start = 0
        while start < len(text):
            part = text[start : start + chunk_size]
            ordinal = len(chunks) + 1
            chunks.append(
                PolicyChunk(
                    evidence_id=f"{document.document_id}-{ordinal:03d}",
                    document=document,
                    heading=section_heading,
                    text=part,
                    ordinal=ordinal,
                )
            )
            if start + chunk_size >= len(text):
                break
            start += chunk_size - overlap
    return chunks
