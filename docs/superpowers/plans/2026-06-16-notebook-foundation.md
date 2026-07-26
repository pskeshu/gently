# Notebook Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the unit-testable foundation of the shared lab notebook — the unified `Note` model and a file-backed `NotebookStore` (write / read / scope-query / rebuildable reverse-indexes / link & supersede) — with no UI, API, or agent wiring.

**Architecture:** A new self-contained module `gently/harness/memory/notebook.py`. One `Note` dataclass (three kinds: observation/finding/question) with author, status, confidence, scope facets (strains/embryos/sessions/threads), typed links, basis, and artifact pointers — orthogonal fields, not subtypes. `NotebookStore` persists one YAML per note under `notebook/notes/{id}_{slug}.yaml` (atomic write, mirroring `FileContextStore`), maintains rebuildable reverse-indexes by strain/embryo/thread, and answers scope+kind+status queries. This is Increment 1's keystone from the design doc (`docs/superpowers/specs/2026-06-16-shared-lab-notebook-design.md`).

**Tech Stack:** Python 3.11, dataclasses, `str`-Enums, PyYAML, pytest (fixtures in `tests/conftest.py`, flat `tests/` layout).

**Follow-on plans (NOT in scope here):** producer wiring (`apply_updates` → notebook), `/api/notebook` + Notebook tab (UI), retrieval/embeddings + brainstorm. Each ships on top of this foundation.

---

### Task 1: The `Note` model

**Files:**
- Create: `gently/harness/memory/notebook.py`
- Test: `tests/test_notebook_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_store.py
"""Tests for the shared lab notebook: Note model + NotebookStore."""

from datetime import datetime

from gently.harness.memory.model import Confidence
from gently.harness.memory.notebook import (
    Author,
    Note,
    NoteKind,
    NoteStatus,
    note_from_dict,
    note_to_dict,
)


class TestNoteModel:
    def test_round_trip_minimal(self):
        n = Note(id="abc123", kind=NoteKind.OBSERVATION, body="dim rings at 10 ms")
        d = note_to_dict(n)
        assert d["kind"] == "observation"
        assert d["author"] == "agent"  # default
        assert d["status"] == "confirmed"  # default
        back = note_from_dict(d)
        assert back == n

    def test_round_trip_full(self):
        n = Note(
            id="def456",
            kind=NoteKind.FINDING,
            body="temperature shifts timing ~12 min/degC",
            author=Author.AGENT,
            title="Temp shifts timing",
            status=NoteStatus.PROPOSED,
            confidence=Confidence.MEDIUM,
            strains=["N2", "OH904"],
            embryos=["emb_0007"],
            sessions=["20260615_1432_x"],
            threads=["q_division_temp"],
            basis=["obs_1", "obs_2"],
            links=[{"rel": "supports", "to": "q_division_temp"}],
            artifacts=[{"kind": "projection", "session": "s1", "embryo": "emb_0007", "t": 42}],
            created_at=datetime(2026, 6, 16, 11, 0, 0),
            updated_at=datetime(2026, 6, 16, 11, 0, 0),
        )
        back = note_from_dict(note_to_dict(n))
        assert back == n
        assert note_to_dict(n)["confidence"] == "medium"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_store.py::TestNoteModel -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gently.harness.memory.notebook'`

- [ ] **Step 3: Write minimal implementation**

```python
# gently/harness/memory/notebook.py
"""
The shared lab notebook — unified memory entry (Note) and file-backed store.

One Note kind taxonomy (observation / finding / question); everything else
(author, status, confidence, scope, links) is an orthogonal field. See
docs/superpowers/specs/2026-06-16-shared-lab-notebook-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .model import Confidence


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook_store.py::TestNoteModel -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/harness/memory/notebook.py tests/test_notebook_store.py
git commit -m "feat(notebook): unified Note model + dict serialization"
```

---

### Task 2: `NotebookStore` — write & read a note

**Files:**
- Modify: `gently/harness/memory/notebook.py` (append `NotebookStore`)
- Test: `tests/test_notebook_store.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_store.py
from gently.harness.memory.notebook import NotebookStore


class TestNotebookStoreReadWrite:
    def test_write_assigns_id_and_persists(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        n = Note(id="", kind=NoteKind.OBSERVATION, body="bean stage at t40")
        note_id = store.write_note(n)
        assert note_id  # non-empty id assigned
        files = list((tmp_path / "notebook" / "notes").glob("*.yaml"))
        assert len(files) == 1
        assert files[0].name.startswith(note_id + "_")

    def test_get_note_round_trip(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        n = Note(id="", kind=NoteKind.FINDING, body="x", strains=["N2"], threads=["t1"])
        note_id = store.write_note(n)
        got = store.get_note(note_id)
        assert got is not None
        assert got.id == note_id
        assert got.kind == NoteKind.FINDING
        assert got.strains == ["N2"]

    def test_get_missing_returns_none(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        assert store.get_note("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_store.py::TestNotebookStoreReadWrite -v`
Expected: FAIL — `ImportError: cannot import name 'NotebookStore'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to gently/harness/memory/notebook.py
import copy
import os
import re
import uuid
from pathlib import Path

import yaml


class NotebookStore:
    """File-backed store for notebook Notes. One YAML per note under notes/;
    flat pool, rebuildable reverse-indexes (added in Task 3)."""

    def __init__(self, notebook_dir: Path):
        self.root = Path(notebook_dir)
        self.notes_dir = self.root / "notes"
        self.index_dir = self.root / "index"
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

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
        return note.id

    def get_note(self, note_id: str) -> Note | None:
        path = self._note_path(note_id)
        if path is None:
            return None
        data = self._read_yaml(path)
        return note_from_dict(data) if data else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook_store.py::TestNotebookStoreReadWrite -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/harness/memory/notebook.py tests/test_notebook_store.py
git commit -m "feat(notebook): NotebookStore write_note/get_note with atomic YAML"
```

---

### Task 3: Reverse-indexes (by strain / embryo / thread) + rebuild

**Files:**
- Modify: `gently/harness/memory/notebook.py` (`NotebookStore`)
- Test: `tests/test_notebook_store.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_store.py
class TestNotebookIndex:
    def test_index_updated_on_write(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        a = store.write_note(Note(id="", kind=NoteKind.OBSERVATION, body="a", strains=["N2"]))
        b = store.write_note(
            Note(id="", kind=NoteKind.OBSERVATION, body="b", strains=["N2", "OH904"])
        )
        assert set(store.ids_for_strain("N2")) == {a, b}
        assert store.ids_for_strain("OH904") == [b]
        assert store.ids_for_strain("missing") == []

    def test_index_by_embryo_and_thread(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        a = store.write_note(
            Note(id="", kind=NoteKind.FINDING, body="a", embryos=["e1"], threads=["t1"])
        )
        assert store.ids_for_embryo("e1") == [a]
        assert store.ids_for_thread("t1") == [a]

    def test_rebuild_index_from_disk(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        a = store.write_note(Note(id="", kind=NoteKind.OBSERVATION, body="a", strains=["N2"]))
        # a fresh store over the same dir must rebuild the index by scanning notes/
        store2 = NotebookStore(tmp_path / "notebook")
        assert store2.ids_for_strain("N2") == [a]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_store.py::TestNotebookIndex -v`
Expected: FAIL — `AttributeError: 'NotebookStore' object has no attribute 'ids_for_strain'`

- [ ] **Step 3: Write minimal implementation**

Modify `NotebookStore.__init__` to add index state + rebuild, extend `write_note` to update the index, and add the index methods. Replace the existing `__init__` and `write_note` with these versions and add the new methods:

```python
# --- replace __init__ ---
def __init__(self, notebook_dir: Path):
    self.root = Path(notebook_dir)
    self.notes_dir = self.root / "notes"
    self.index_dir = self.root / "index"
    self.notes_dir.mkdir(parents=True, exist_ok=True)
    self.index_dir.mkdir(parents=True, exist_ok=True)
    # reverse indexes: facet -> {value: [note_id, ...]}
    self._index: dict[str, dict[str, list[str]]] = {"strain": {}, "embryo": {}, "thread": {}}
    self.rebuild_index()


# --- add: facet extraction + index maintenance ---
_FACETS = {"strain": "strains", "embryo": "embryos", "thread": "threads"}


def _index_note(self, note: Note) -> None:
    for facet, attr in self._FACETS.items():
        for value in getattr(note, attr):
            bucket = self._index[facet].setdefault(value, [])
            if note.id not in bucket:
                bucket.append(note.id)


def rebuild_index(self) -> None:
    """Rebuild reverse-indexes by scanning notes/ (the notes are authoritative;
    indexes are disposable caches)."""
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
```

Then add an index-update at the end of `write_note` (just before `return note.id`):

```python
        self._index_note(note)
        return note.id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook_store.py::TestNotebookIndex -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/harness/memory/notebook.py tests/test_notebook_store.py
git commit -m "feat(notebook): rebuildable reverse-indexes by strain/embryo/thread"
```

---

### Task 4: `query_notes` — filter by kind / author / status / scope

**Files:**
- Modify: `gently/harness/memory/notebook.py` (`NotebookStore`)
- Test: `tests/test_notebook_store.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_store.py
class TestNotebookQuery:
    def _seed(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        store.write_note(Note(id="o1", kind=NoteKind.OBSERVATION, body="o", strains=["N2"]))
        store.write_note(
            Note(
                id="f1",
                kind=NoteKind.FINDING,
                body="f",
                status=NoteStatus.PROPOSED,
                strains=["N2"],
                threads=["t1"],
            )
        )
        store.write_note(
            Note(id="q1", kind=NoteKind.QUESTION, body="q", status=NoteStatus.OPEN, threads=["t1"])
        )
        return store

    def test_query_by_kind(self, tmp_path):
        store = self._seed(tmp_path)
        ids = {n.id for n in store.query_notes(kind=NoteKind.FINDING)}
        assert ids == {"f1"}

    def test_query_by_thread_scope(self, tmp_path):
        store = self._seed(tmp_path)
        ids = {n.id for n in store.query_notes(thread="t1")}
        assert ids == {"f1", "q1"}

    def test_query_by_thread_and_kind(self, tmp_path):
        store = self._seed(tmp_path)
        ids = {n.id for n in store.query_notes(thread="t1", kind=NoteKind.QUESTION)}
        assert ids == {"q1"}

    def test_query_by_status(self, tmp_path):
        store = self._seed(tmp_path)
        ids = {n.id for n in store.query_notes(status=NoteStatus.OPEN)}
        assert ids == {"q1"}

    def test_query_all_sorted_newest_first(self, tmp_path):
        store = self._seed(tmp_path)
        notes = store.query_notes()
        assert len(notes) == 3
        ts = [n.created_at for n in notes]
        assert ts == sorted(ts, reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_store.py::TestNotebookQuery -v`
Expected: FAIL — `AttributeError: 'NotebookStore' object has no attribute 'query_notes'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to NotebookStore
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
        if scope_sets:
            candidate_ids: set[str] | None = set.intersection(*scope_sets)
        else:
            candidate_ids = None  # means "all"

        # 2. load + filter
        results: list[Note] = []
        if candidate_ids is not None:
            notes = [n for n in (self.get_note(i) for i in candidate_ids) if n]
        else:
            notes = [
                note_from_dict(d)
                for d in (self._read_yaml(f) for f in self.notes_dir.glob("*.yaml"))
                if d
            ]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook_store.py::TestNotebookQuery -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/harness/memory/notebook.py tests/test_notebook_store.py
git commit -m "feat(notebook): query_notes by kind/author/status/scope"
```

---

### Task 5: `link_notes` and `supersede_note`

**Files:**
- Modify: `gently/harness/memory/notebook.py` (`NotebookStore`)
- Test: `tests/test_notebook_store.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_store.py
class TestNotebookLinkSupersede:
    def test_link_notes_adds_typed_edge(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        a = store.write_note(Note(id="", kind=NoteKind.FINDING, body="a"))
        b = store.write_note(Note(id="", kind=NoteKind.QUESTION, body="b"))
        store.link_notes(a, "supports", b)
        got = store.get_note(a)
        assert {"rel": "supports", "to": b} in got.links

    def test_supersede_marks_old_and_points_new(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        old = store.write_note(Note(id="", kind=NoteKind.FINDING, body="old claim"))
        new = store.write_note(Note(id="", kind=NoteKind.FINDING, body="better claim"))
        store.supersede_note(old, new)
        old_n = store.get_note(old)
        new_n = store.get_note(new)
        assert old_n.status == NoteStatus.SUPERSEDED
        assert old_n.superseded_by == new
        assert {"rel": "refines", "to": old} in new_n.links
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_store.py::TestNotebookLinkSupersede -v`
Expected: FAIL — `AttributeError: 'NotebookStore' object has no attribute 'link_notes'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to NotebookStore
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook_store.py -v`
Expected: PASS (all tests across all classes pass)

- [ ] **Step 5: Commit**

```bash
git add gently/harness/memory/notebook.py tests/test_notebook_store.py
git commit -m "feat(notebook): link_notes + supersede_note (append-only history)"
```

---

## Self-Review

**Spec coverage (against the design doc §2 data model):**
- Three kinds (Observation/Finding/Question) → Task 1 `NoteKind`. ✓
- Orthogonal fields (author/status/confidence/scope/links/artifacts) → Task 1 `Note`. ✓
- Flat note pool + rebuildable reverse-indexes (strain/embryo/thread) → Tasks 2-3. ✓
- "By question + by strain + links coexist over flat YAML, no DB" → Tasks 3-4 (indexes + scope-intersect query). ✓
- Append-only / supersede-never-overwrite → Task 5 `supersede_note`. ✓
- Typed links graph → Tasks 1 (`links`) + 5 (`link_notes`). ✓
- *Deferred to follow-on plans (correctly out of scope):* inquiry-thread object, working-memory split, producer wiring, API/tab, embeddings/retrieval, consolidation. Noted in header.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type consistency:** `Note`, `NoteKind`, `Author`, `NoteStatus`, `note_to_dict`/`note_from_dict`, and `NotebookStore.{write_note,get_note,rebuild_index,ids_for_strain,ids_for_embryo,ids_for_thread,query_notes,link_notes,supersede_note}` are named identically across all tasks and tests. `Confidence` is imported from `.model` (confirmed to exist). ✓
