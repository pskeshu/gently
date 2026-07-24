# Notebook Read API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Expose the shared notebook over HTTP so the frontend can read it — list/filter notes, fetch one note, and list inquiry-threads with counts.

**Architecture:** A new route module `gently/ui/web/routes/notebook.py` exposing `create_router(server)` (the established pattern), reading `server.context_store.notebook` (the `NotebookStore` added in the producer-bridge plan) and serializing via `note_to_dict`. Registered in `routes/__init__.py`. Read-only; authoring/curation is a later increment.

**Tech Stack:** FastAPI `APIRouter`, pytest + `fastapi.testclient.TestClient`, the `file_context_store` fixture (`tests/conftest.py`).

**Out of scope:** the Notebook tab UI (next plan, browser-verified); the Agent's-View rewire; retrieval/embeddings.

---

### Task 1: Route module — list & get notes

**Files:**
- Create: `gently/ui/web/routes/notebook.py`
- Modify: `gently/ui/web/routes/__init__.py`
- Test: `tests/test_notebook_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_api.py
"""Tests for the notebook read API."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.harness.memory.notebook import Note, NoteKind, NoteStatus


def _make_app(context_store):
    from gently.ui.web.routes.notebook import create_router

    app = FastAPI()

    class _Server:
        pass

    server = _Server()
    server.context_store = context_store
    app.include_router(create_router(server))
    return app


def _seed(cs):
    nb = cs.notebook
    nb.write_note(Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed", strains=["N2"]))
    nb.write_note(
        Note(
            id="f1",
            kind=NoteKind.FINDING,
            body="rings by comma",
            status=NoteStatus.PROPOSED,
            strains=["N2"],
            threads=["t1"],
        )
    )
    return cs


class TestListNotes:
    def test_no_store_available_false(self):
        client = TestClient(_make_app(None))
        data = client.get("/api/notebook/notes").json()
        assert data == {"available": False, "notes": []}

    def test_list_all(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes").json()
        assert data["available"] is True
        assert {n["id"] for n in data["notes"]} == {"o1", "f1"}

    def test_filter_by_kind(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes?kind=finding").json()
        assert {n["id"] for n in data["notes"]} == {"f1"}

    def test_filter_by_strain(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes?strain=N2").json()
        assert {n["id"] for n in data["notes"]} == {"o1", "f1"}

    def test_invalid_kind_is_ignored(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes?kind=bogus").json()
        assert {n["id"] for n in data["notes"]} == {"o1", "f1"}


class TestGetNote:
    def test_get_existing(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes/o1").json()
        assert data["id"] == "o1"
        assert data["body"] == "ring formed"

    def test_get_missing_404(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        resp = client.get("/api/notebook/notes/nope")
        assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gently.ui.web.routes.notebook'`

- [ ] **Step 3: Write minimal implementation**

Create `gently/ui/web/routes/notebook.py`:

```python
"""Notebook (shared lab notebook) read routes.

Exposes the notebook's Notes for the Notebook tab + Agent's-View live edge.
Read-only here; authoring/curation come in a later increment.
"""

from fastapi import APIRouter, HTTPException

from gently.harness.memory.notebook import Author, NoteKind, NoteStatus, note_to_dict


def _coerce(enum_cls, value):
    """Parse a query-param string into an enum; invalid/None → None (no filter)."""
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        return None


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _nb():
        cs = getattr(server, "context_store", None)
        return cs.notebook if cs is not None else None

    @router.get("/api/notebook/notes")
    async def list_notes(
        kind: str | None = None,
        author: str | None = None,
        status: str | None = None,
        strain: str | None = None,
        embryo: str | None = None,
        thread: str | None = None,
    ):
        nb = _nb()
        if nb is None:
            return {"available": False, "notes": []}
        notes = nb.query_notes(
            kind=_coerce(NoteKind, kind),
            author=_coerce(Author, author),
            status=_coerce(NoteStatus, status),
            strain=strain,
            embryo=embryo,
            thread=thread,
        )
        return {"available": True, "notes": [note_to_dict(n) for n in notes]}

    @router.get("/api/notebook/notes/{note_id}")
    async def get_note(note_id: str):
        nb = _nb()
        if nb is None:
            raise HTTPException(status_code=404, detail="notebook unavailable")
        note = nb.get_note(note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="note not found")
        return note_to_dict(note)

    return router
```

Then register it in `gently/ui/web/routes/__init__.py`. Add the import after the `images` import line:

```python
from .notebook import create_router as create_notebook_router
```

And add `create_notebook_router,` to the factory tuple in `register_all_routes` (after `create_context_router,`):

```python
        create_context_router,
        create_notebook_router,
    ):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook_api.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/ui/web/routes/notebook.py gently/ui/web/routes/__init__.py tests/test_notebook_api.py
git commit -m "feat(notebook): read API — GET /api/notebook/notes + /notes/{id}"
```

---

### Task 2: `GET /api/notebook/threads`

**Files:**
- Modify: `gently/ui/web/routes/notebook.py`
- Test: `tests/test_notebook_api.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_api.py
class TestThreads:
    def test_no_store(self):
        client = TestClient(_make_app(None))
        assert client.get("/api/notebook/threads").json() == {"available": False, "threads": []}

    def test_thread_counts(self, file_context_store):
        cs = file_context_store
        nb = cs.notebook
        nb.write_note(Note(id="a", kind=NoteKind.QUESTION, body="q", threads=["t1"]))
        nb.write_note(Note(id="b", kind=NoteKind.FINDING, body="f", threads=["t1", "t2"]))
        client = TestClient(_make_app(cs))
        data = client.get("/api/notebook/threads").json()
        assert data["available"] is True
        assert data["threads"] == [{"id": "t1", "count": 2}, {"id": "t2", "count": 1}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_api.py::TestThreads -v`
Expected: FAIL — 404 (route not defined)

- [ ] **Step 3: Write minimal implementation**

Add this endpoint inside `create_router`, just before `return router`:

```python
    @router.get("/api/notebook/threads")
    async def list_threads():
        nb = _nb()
        if nb is None:
            return {"available": False, "threads": []}
        counts: dict[str, int] = {}
        for n in nb.query_notes():
            for t in n.threads:
                counts[t] = counts.get(t, 0) + 1
        threads = [{"id": t, "count": c} for t, c in sorted(counts.items())]
        return {"available": True, "threads": threads}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook_api.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add gently/ui/web/routes/notebook.py tests/test_notebook_api.py
git commit -m "feat(notebook): read API — GET /api/notebook/threads with counts"
```

---

## Self-Review

**Spec coverage:** read surface API for the notebook (design increment 1c, backend half) — query notes by kind/author/status/scope, fetch one, list threads. ✓ Reuses `NotebookStore.query_notes` + `note_to_dict` from the foundation. ✓ Follows `create_router(server)` + `server.context_store` convention (`context.py`). ✓
**Placeholder scan:** none — complete code + commands. ✓
**Type consistency:** `create_router`, `note_to_dict`, `NoteKind`/`Author`/`NoteStatus`, `server.context_store.notebook`, `nb.query_notes`/`get_note` all match the foundation + producer-bridge modules. Registration matches the existing tuple in `routes/__init__.py`. ✓
