"""Pages of the notebook.

The notes are one flat pool. A page is a saved query over it, with a
conversation pinned to the page: what a person and the agent said while
looking at exactly that set of notes. Open the page again and both are there,
the way a chat keeps its history, except the context is live — a new note
that matches the query joins the page on its own, and a note set aside
leaves it.

A page holds only its query and its talk. The same note can be on several
pages; deleting a page loses a conversation and nothing else.

Built-in pages come from the facets every note already carries, with no
setup: All notes, Agent's findings, Asks, Session summaries, one per project,
and one per thread. A built-in is not on disk until someone talks on it; from
then on its file keeps the conversation under the same id.
"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .notebook import Author, Note, NotebookStore, NoteKind, NoteStatus

# The facets a page query may name. Each is a NotebookStore.query_notes
# argument, except ``title`` (an exact title) and ``asks`` (notes that
# mention someone), which are filtered after.
QUERY_FACETS = (
    "kind",
    "author",
    "status",
    "strain",
    "embryo",
    "thread",
    "project",
    "session",
    "person",
    "title",
    "asks",
)


@dataclass
class Turn:
    role: str  # "person" | "agent"
    text: str
    by: str | None = None  # the person's id on a person's turn
    note_ids: list[str] = field(default_factory=list)  # what the agent's reply rested on
    points: list[dict] = field(default_factory=list)  # the reply's supporting points
    coverage: str | None = None  # covered | partial | not_in_notebook
    kept_as: str | None = None  # the note this reply was filed as, once kept
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Page:
    id: str
    name: str
    query: dict = field(default_factory=dict)
    builtin: bool = False
    by: str | None = None  # who saved it; None for a built-in
    turns: list[Turn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


def clean_query(raw: Any) -> dict:
    """Only the facets a page may name, with empty values dropped."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for k in QUERY_FACETS:
        v = raw.get(k)
        if k == "asks":
            if v:
                out[k] = True
        elif isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def turn_to_dict(t: Turn) -> dict:
    return {
        "role": t.role,
        "text": t.text,
        "by": t.by,
        "note_ids": list(t.note_ids),
        "points": list(t.points),
        "coverage": t.coverage,
        "kept_as": t.kept_as,
        "created_at": t.created_at.isoformat(),
    }


def turn_from_dict(d: dict) -> Turn:
    return Turn(
        role=str(d.get("role") or "person"),
        text=str(d.get("text") or ""),
        by=d.get("by"),
        note_ids=[str(x) for x in (d.get("note_ids") or [])],
        points=[p for p in (d.get("points") or []) if isinstance(p, dict)],
        coverage=d.get("coverage"),
        kept_as=d.get("kept_as"),
        created_at=datetime.fromisoformat(d["created_at"])
        if d.get("created_at")
        else datetime.now(),
    )


def page_to_dict(p: Page) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "query": dict(p.query),
        "builtin": bool(p.builtin),
        "by": p.by,
        "turns": [turn_to_dict(t) for t in p.turns],
        "created_at": p.created_at.isoformat(),
        "updated_at": p.updated_at.isoformat(),
    }


def page_from_dict(d: dict) -> Page:
    return Page(
        id=str(d["id"]),
        name=str(d.get("name") or d["id"]),
        query=clean_query(d.get("query")),
        builtin=bool(d.get("builtin", False)),
        by=d.get("by"),
        turns=[turn_from_dict(t) for t in (d.get("turns") or []) if isinstance(t, dict)],
        created_at=datetime.fromisoformat(d["created_at"])
        if d.get("created_at")
        else datetime.now(),
        updated_at=datetime.fromisoformat(d["updated_at"])
        if d.get("updated_at")
        else datetime.now(),
    )


# ---- built-ins -------------------------------------------------------------

_FIXED: list[tuple[str, str, dict]] = [
    ("all", "All notes", {}),
    ("agent-findings", "Agent's findings", {"author": "agent", "kind": "finding"}),
    ("asks", "Asks", {"asks": True}),
    ("session-summaries", "Session summaries", {"author": "agent", "title": "Session summary"}),
]


def builtin_pages(cs: Any) -> list[Page]:
    """The pages that need no setup: fixed ones, then one per project. A
    thread's page is built on demand by ``builtin_page`` (its id is
    ``thread:<name>``), so the rail can list threads from the notes."""
    pages = [Page(id=i, name=n, query=dict(q), builtin=True) for i, n, q in _FIXED]
    try:
        roots = cs.get_root_campaigns() if hasattr(cs, "get_root_campaigns") else []
    except Exception:
        roots = []
    for c in roots:
        name = c.shorthand or (c.description or c.id)[:48]
        pages.append(Page(id=f"project:{c.id}", name=name, query={"project": c.id}, builtin=True))
    return pages


def builtin_page(cs: Any, page_id: str) -> Page | None:
    if page_id.startswith("thread:") and len(page_id) > 7:
        t = page_id[7:]
        return Page(id=page_id, name=f"# {t}", query={"thread": t}, builtin=True)
    for p in builtin_pages(cs):
        if p.id == page_id:
            return p
    return None


# ---- the store --------------------------------------------------------------


class PageStore:
    """One YAML per page under ``<notebook>/pages/``: the query and the
    conversation. A built-in page gets a file the first time it is talked on."""

    def __init__(self, pages_dir: Path):
        self.dir = Path(pages_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _gen_id() -> str:
        return str(uuid.uuid4())[:8]

    @staticmethod
    def _slugify(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:30]

    @staticmethod
    def _fname_id(page_id: str) -> str:
        # ``project:abc`` / ``thread:t1`` are ids; a colon is not a file name.
        return page_id.replace(":", "__")

    def _path(self, page_id: str) -> Path | None:
        return next(self.dir.glob(f"{self._fname_id(page_id)}_*.yaml"), None)

    def list_saved(self) -> list[Page]:
        pages: list[Page] = []
        for f in sorted(self.dir.glob("*.yaml")):
            try:
                with open(f, encoding="utf-8") as fh:
                    d = yaml.safe_load(fh)
            except (OSError, ValueError, yaml.YAMLError):
                continue
            if isinstance(d, dict) and d.get("id"):
                pages.append(page_from_dict(d))
        pages.sort(key=lambda p: p.updated_at, reverse=True)
        return pages

    def get(self, page_id: str) -> Page | None:
        path = self._path(page_id)
        if path is None:
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                d = yaml.safe_load(fh)
        except (OSError, ValueError, yaml.YAMLError):
            return None
        return page_from_dict(d) if isinstance(d, dict) and d.get("id") else None

    def save(self, page: Page) -> str:
        if not page.id:
            page.id = self._gen_id()
        page.updated_at = datetime.now()
        old = self._path(page.id)
        if old is not None:
            old.unlink()
        path = self.dir / f"{self._fname_id(page.id)}_{self._slugify(page.name) or 'page'}.yaml"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(
                page_to_dict(page),
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        os.replace(str(tmp), str(path))
        return page.id

    def delete(self, page_id: str) -> bool:
        path = self._path(page_id)
        if path is None:
            return False
        path.unlink()
        return True


def pages_of(nb: NotebookStore) -> PageStore:
    """The notebook's page store, beside its notes."""
    store = getattr(nb, "_pages", None)
    if store is None:
        store = PageStore(nb.root / "pages")
        nb._pages = store  # type: ignore[attr-defined]
    return store


# ---- resolving and reading a page --------------------------------------------


def resolve_page(cs: Any, nb: NotebookStore, page_id: str) -> Page | None:
    """The saved page under this id, else the built-in it names."""
    saved = pages_of(nb).get(page_id)
    if saved is not None:
        return saved
    return builtin_page(cs, page_id)


def list_pages(cs: Any, nb: NotebookStore) -> list[Page]:
    """Built-ins first (with their saved conversations, if any), then the
    pages people saved, newest talked-on first."""
    store = pages_of(nb)
    saved = {p.id: p for p in store.list_saved()}
    out: list[Page] = []
    for b in builtin_pages(cs):
        out.append(saved.pop(b.id, b))
    out.extend(p for p in saved.values() if not p.id.startswith("thread:"))
    return out


def _enum(enum_cls, value):
    try:
        return enum_cls(value) if value else None
    except ValueError:
        return None


def page_notes(nb: NotebookStore, page: Page) -> list[Note]:
    """The page's notes now: the query over what is in the notebook."""
    q = page.query
    notes = nb.query_notes(
        kind=_enum(NoteKind, q.get("kind")),
        author=_enum(Author, q.get("author")),
        status=_enum(NoteStatus, q.get("status")),
        strain=q.get("strain"),
        embryo=q.get("embryo"),
        thread=q.get("thread"),
        project=q.get("project"),
        person=q.get("person"),
        session=q.get("session"),
        in_notebook=True,
    )
    if q.get("title"):
        notes = [n for n in notes if n.title == q["title"]]
    if q.get("asks"):
        notes = [n for n in notes if n.mentions]
    return notes


def note_from_reply(page: Page, turn: Turn) -> Note:
    """An agent reply kept as a finding: proposed until a person confirms,
    resting on the notes it cited, filed where the page is."""
    q = page.query
    return Note(
        id="",
        kind=NoteKind.FINDING,
        body=turn.text,
        author=Author.AGENT,
        status=NoteStatus.PROPOSED,
        basis=list(turn.note_ids),
        projects=[q["project"]] if q.get("project") else [],
        threads=[q["thread"]] if q.get("thread") else [],
        strains=[q["strain"]] if q.get("strain") else [],
        sessions=[q["session"]] if q.get("session") else [],
    )
