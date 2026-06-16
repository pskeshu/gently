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

from .model import Confidence, Learning, Observation


class NoteKind(str, Enum):
    OBSERVATION = "observation"  # immutable record of what was seen/done/read/noted
    FINDING = "finding"  # revisable, supersedable believed claim
    QUESTION = "question"  # open inquiry; large ones are the thread spine


class Author(str, Enum):
    HUMAN = "human"
    AGENT = "agent"
    PERCEPTION = "perception"


class NoteStatus(str, Enum):
    OPEN = "open"  # questions not yet answered
    PROPOSED = "proposed"  # agent-drafted finding awaiting human confirm
    CONFIRMED = "confirmed"  # accepted observation/finding (default)
    ANSWERED = "answered"  # question resolved
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
    basis: list[str] = field(default_factory=list)  # note ids this rests on
    links: list[dict] = field(default_factory=list)  # [{"rel": ..., "to": ...}]
    artifacts: list[dict] = field(default_factory=list)  # FileStore pointers
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


def observation_to_note(obs: Observation) -> Note:
    """Bridge a legacy Observation into a notebook Note (kind=observation)."""
    return Note(
        id=obs.id,
        kind=NoteKind.OBSERVATION,
        body=obs.content,
        author=Author.AGENT,
        embryos=[obs.embryo_id] if obs.embryo_id else [],
        sessions=[obs.session_id] if obs.session_id else [],
        links=[{"rel": "relates_to", "to": r} for r in (obs.relates_to or [])],
        artifacts=[obs.gently_refs] if obs.gently_refs else [],
        created_at=obs.timestamp,
        updated_at=obs.timestamp,
    )


def learning_to_note(learning: Learning) -> Note:
    """Bridge a legacy Learning into a notebook Note (kind=finding, proposed)."""
    return Note(
        id=learning.id,
        kind=NoteKind.FINDING,
        body=learning.content,
        author=Author.AGENT,
        status=NoteStatus.PROPOSED,
        confidence=learning.confidence,
        created_at=learning.created_at,
        updated_at=learning.created_at,
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
        self._index: dict[str, dict[str, list[str]]] = {"strain": {}, "embryo": {}, "thread": {}}
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

    def query_notes(
        self,
        *,
        kind: NoteKind | None = None,
        author: Author | None = None,
        status: NoteStatus | None = None,
        strain: str | None = None,
        embryo: str | None = None,
        thread: str | None = None,
    ) -> list[Note]:
        """Structural query: narrow by scope via the indexes, then filter by
        kind/author/status. Returned newest-first. (No semantic ranking here —
        that's a later increment.)"""
        # 1. candidate ids — intersect any scope facets given, else all notes
        scope_sets: list[set[str]] = []
        if strain is not None:
            scope_sets.append(set(self.ids_for_strain(strain)))
        if embryo is not None:
            scope_sets.append(set(self.ids_for_embryo(embryo)))
        if thread is not None:
            scope_sets.append(set(self.ids_for_thread(thread)))
        candidate_ids: set[str] | None = set.intersection(*scope_sets) if scope_sets else None

        # 2. load + filter
        if candidate_ids is not None:
            notes = [n for n in (self.get_note(i) for i in candidate_ids) if n]
        else:
            notes = [
                note_from_dict(d)
                for d in (self._read_yaml(f) for f in self.notes_dir.glob("*.yaml"))
                if d
            ]
        results: list[Note] = []
        for n in notes:
            if kind is not None and n.kind != kind:
                continue
            if author is not None and n.author != author:
                continue
            if status is not None and n.status != status:
                continue
            results.append(n)
        results.sort(key=lambda n: n.created_at, reverse=True)
        return results

    # ---- links + supersession (append-only history) ----
    def link_notes(self, from_id: str, rel: str, to_id: str) -> None:
        """Add a typed edge from one note to another (append-only)."""
        note = self.get_note(from_id)
        if note is None:
            raise KeyError(from_id)
        edge = {"rel": rel, "to": to_id}
        if edge not in note.links:
            note.links.append(edge)
            self.write_note(note)

    def supersede_note(self, old_id: str, new_id: str) -> None:
        """Mark old as superseded (kept, never deleted) and link the new note
        back to it as a refinement — the chain is the intellectual history."""
        old = self.get_note(old_id)
        if old is None:
            raise KeyError(old_id)
        old.status = NoteStatus.SUPERSEDED
        old.superseded_by = new_id
        self.write_note(old)
        self.link_notes(new_id, "refines", old_id)
