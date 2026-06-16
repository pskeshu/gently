"""
The shared lab notebook — unified memory entry (Note) and file-backed store.

One Note kind taxonomy (observation / finding / question); everything else
(author, status, confidence, scope, links) is an orthogonal field. See
docs/superpowers/specs/2026-06-16-shared-lab-notebook-design.md.
"""

from __future__ import annotations

import copy
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
