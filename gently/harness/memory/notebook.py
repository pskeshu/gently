"""
The shared lab notebook — unified memory entry (Note) and file-backed store.

One Note kind taxonomy (observation / finding / question); everything else
(author, status, confidence, scope, links) is an orthogonal field. See
docs/superpowers/specs/2026-06-16-shared-lab-notebook-design.md.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .model import Confidence


class NoteKind(str, Enum):
    OBSERVATION = "observation"  # immutable record of what was seen/done/read/noted
    FINDING = "finding"          # revisable, supersedable believed claim
    QUESTION = "question"        # open inquiry; large ones are the thread spine


class Author(str, Enum):
    HUMAN = "human"
    AGENT = "agent"
    PERCEPTION = "perception"


class NoteStatus(str, Enum):
    OPEN = "open"              # questions not yet answered
    PROPOSED = "proposed"      # agent-drafted finding awaiting human confirm
    CONFIRMED = "confirmed"    # accepted observation/finding (default)
    ANSWERED = "answered"      # question resolved
    SUPERSEDED = "superseded"  # replaced by a newer note


@dataclass
class Note:
    id: str
    kind: NoteKind
    body: str
    author: Author = Author.AGENT
    title: str | None = None
    status: NoteStatus = NoteStatus.CONFIRMED
    confidence: Confidence | None = None
    strains: list[str] = field(default_factory=list)
    embryos: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    threads: list[str] = field(default_factory=list)
    basis: list[str] = field(default_factory=list)        # note ids this rests on
    links: list[dict] = field(default_factory=list)        # [{"rel": ..., "to": ...}]
    artifacts: list[dict] = field(default_factory=list)    # FileStore pointers
    superseded_by: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


def note_to_dict(n: Note) -> dict[str, Any]:
    return {
        "id": n.id,
        "kind": n.kind.value,
        "body": n.body,
        "author": n.author.value,
        "title": n.title,
        "status": n.status.value,
        "confidence": n.confidence.value if n.confidence else None,
        "strains": list(n.strains),
        "embryos": list(n.embryos),
        "sessions": list(n.sessions),
        "threads": list(n.threads),
        "basis": list(n.basis),
        "links": list(n.links),
        "artifacts": list(n.artifacts),
        "superseded_by": n.superseded_by,
        "created_at": n.created_at.isoformat(),
        "updated_at": n.updated_at.isoformat(),
    }


def note_from_dict(d: dict[str, Any]) -> Note:
    conf = d.get("confidence")
    return Note(
        id=d["id"],
        kind=NoteKind(d["kind"]),
        body=d.get("body", ""),
        author=Author(d.get("author", "agent")),
        title=d.get("title"),
        status=NoteStatus(d.get("status", "confirmed")),
        confidence=Confidence(conf) if conf else None,
        strains=list(d.get("strains") or []),
        embryos=list(d.get("embryos") or []),
        sessions=list(d.get("sessions") or []),
        threads=list(d.get("threads") or []),
        basis=list(d.get("basis") or []),
        links=list(d.get("links") or []),
        artifacts=list(d.get("artifacts") or []),
        superseded_by=d.get("superseded_by"),
        created_at=datetime.fromisoformat(d["created_at"]),
        updated_at=datetime.fromisoformat(d["updated_at"]),
    )


class NotebookStore:
    """File-backed store for notebook Notes. One YAML per note under notes/;
    flat pool, rebuildable reverse-indexes (added in Task 3)."""

    _FACETS = {"strain": "strains", "embryo": "embryos", "thread": "threads"}

    def __init__(self, notebook_dir: Path):
        self.root = Path(notebook_dir)
        self.notes_dir = self.root / "notes"
        self.index_dir = self.root / "index"
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # reverse indexes: facet -> {value: [note_id, ...]}
        self._index: dict[str, dict[str, list[str]]] = {
            "strain": {}, "embryo": {}, "thread": {}
        }
        self.rebuild_index()

    # ---- reverse indexes (notes are authoritative; indexes are caches) ----
    def _index_note(self, note: Note) -> None:
        for facet, attr in self._FACETS.items():
            for value in getattr(note, attr):
                bucket = self._index[facet].setdefault(value, [])
                if note.id not in bucket:
                    bucket.append(note.id)

    def rebuild_index(self) -> None:
        """Rebuild reverse-indexes by scanning notes/."""
        self._index = {"strain": {}, "embryo": {}, "thread": {}}
        for f in sorted(self.notes_dir.glob("*.yaml")):
            data = self._read_yaml(f)
            if data:
                self._index_note(note_from_dict(data))

    def ids_for_strain(self, strain: str) -> list[str]:
        return list(self._index["strain"].get(strain, []))

    def ids_for_embryo(self, embryo: str) -> list[str]:
        return list(self._index["embryo"].get(embryo, []))

    def ids_for_thread(self, thread: str) -> list[str]:
        return list(self._index["thread"].get(thread, []))

    # ---- helpers (mirror FileContextStore conventions) ----
    @staticmethod
    def _gen_id() -> str:
        return str(uuid.uuid4())[:8]

    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
        return slug[:30]

    def _write_yaml(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
        os.replace(str(tmp), str(path))

    def _read_yaml(self, path: Path) -> dict | None:
        try:
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        except OSError:
            return None

    def _note_path(self, note_id: str) -> Path | None:
        return next(self.notes_dir.glob(f"{note_id}_*.yaml"), None)

    # ---- read/write ----
    def write_note(self, note: Note) -> str:
        if not note.id:
            note.id = self._gen_id()
        note.updated_at = datetime.now()
        slug = self._slugify(note.title or note.body or note.kind.value)
        # remove any stale file for this id (slug may have changed)
        old = self._note_path(note.id)
        if old is not None:
            old.unlink()
        self._write_yaml(self.notes_dir / f"{note.id}_{slug}.yaml", note_to_dict(note))
        self._index_note(note)
        return note.id

    def get_note(self, note_id: str) -> Note | None:
        path = self._note_path(note_id)
        if path is None:
            return None
        data = self._read_yaml(path)
        return note_from_dict(data) if data else None
