# "Ask the Notebook" Implementation Plan (Increment 2, backend)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let the notebook be *reasoned with* — given a question (+ optional scope), retrieve relevant Notes, ask Claude to synthesize a **grounded, cited** answer, and return it as a validated structured object.

**Architecture:** A new module `gently/harness/memory/notebook_ask.py`: structural retrieval (`select_notes`) + a forced-`tool_choice` synthesis call (`answer_question`) that reuses gently's conventions — `anthropic.AsyncAnthropic` (per `chat.py:198`), `settings.models.main` (Opus 4.8), structured output via a pinned tool (per `verifier.py`), and **no self-rated confidence** (per the lab rule). A `POST /api/notebook/ask` route wires retrieval → synthesis. The Claude client is injected so everything is unit-testable with a fake.

**Tech Stack:** Python 3.11, `anthropic` SDK, FastAPI, pytest + TestClient (venv).

**Out of scope (later plans):** embeddings/semantic recall (structural-only here); the "Ask" UI box on the Notebook tab; proactive surfacing.

---

### Task 1: Structural retrieval — `select_notes`

**Files:**
- Create: `gently/harness/memory/notebook_ask.py`
- Test: `tests/test_notebook_ask.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notebook_ask.py
"""Tests for 'Ask the notebook' — retrieval + grounded synthesis."""

from gently.harness.memory.notebook import Note, NoteKind
from gently.harness.memory.notebook_ask import select_notes


def _seed(cs):
    nb = cs.notebook
    nb.write_note(
        Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed", strains=["N2"], threads=["t1"])
    )
    nb.write_note(
        Note(id="f1", kind=NoteKind.FINDING, body="12 min/degC", strains=["N2"], threads=["t1"])
    )
    nb.write_note(Note(id="x1", kind=NoteKind.OBSERVATION, body="unrelated", strains=["OH904"]))
    return nb


class TestSelectNotes:
    def test_scope_by_thread(self, file_context_store):
        nb = _seed(file_context_store)
        ids = {n.id for n in select_notes(nb, thread="t1")}
        assert ids == {"o1", "f1"}

    def test_scope_by_strain(self, file_context_store):
        nb = _seed(file_context_store)
        ids = {n.id for n in select_notes(nb, strain="OH904")}
        assert ids == {"x1"}

    def test_no_scope_returns_recent_capped(self, file_context_store):
        nb = _seed(file_context_store)
        notes = select_notes(nb, limit=2)
        assert len(notes) == 2  # newest-first, capped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_notebook_ask.py::TestSelectNotes -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gently.harness.memory.notebook_ask'`

- [ ] **Step 3: Write minimal implementation**

```python
# gently/harness/memory/notebook_ask.py
"""Ask the notebook — retrieve relevant Notes and synthesize a grounded,
cited answer with Claude. Structural retrieval only (semantic recall is a
later increment). See docs/superpowers/specs/2026-06-16-shared-lab-notebook-design.md §4.
"""

from __future__ import annotations

from .notebook import Note, NotebookStore


def select_notes(
    store: NotebookStore,
    *,
    thread: str | None = None,
    strain: str | None = None,
    limit: int = 12,
) -> list[Note]:
    """Structural narrowing: scope by thread/strain when given, else recent.
    Returns newest-first, capped at `limit`."""
    notes = store.query_notes(thread=thread, strain=strain)
    return notes[:limit]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_notebook_ask.py::TestSelectNotes -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/harness/memory/notebook_ask.py tests/test_notebook_ask.py
git commit -m "feat(notebook): select_notes — structural retrieval for ask"
```

---

### Task 2: Grounded synthesis — `answer_question` (forced tool)

**Files:**
- Modify: `gently/harness/memory/notebook_ask.py`
- Test: `tests/test_notebook_ask.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_ask.py
import asyncio

from gently.harness.memory.notebook_ask import ASK_TOOL, answer_question, build_ask_messages


class _FakeBlock:
    def __init__(self, inp):
        self.type = "tool_use"
        self.input = inp


class _FakeResp:
    def __init__(self, inp):
        self.content = [_FakeBlock(inp)]
        self.stop_reason = "tool_use"


class _FakeMessages:
    def __init__(self, captured, inp):
        self._captured, self._inp = captured, inp

    async def create(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeResp(self._inp)


class _FakeClient:
    def __init__(self, inp):
        self.captured = {}
        self.messages = _FakeMessages(self.captured, inp)


class TestAnswerQuestion:
    def test_build_messages_embeds_note_ids(self):
        notes = [Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed")]
        msgs = build_ask_messages("what formed?", notes)
        text = msgs[0]["content"]
        assert "o1" in text and "ring formed" in text and "what formed?" in text

    def test_answer_returns_structured_and_forces_tool(self):
        canned = {
            "answer": "A ring formed.",
            "points": [{"text": "ring formed", "note_ids": ["o1"]}],
            "suggested_next": [],
            "coverage": "covered",
        }
        client = _FakeClient(canned)
        notes = [Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed")]
        out = asyncio.run(answer_question(client, "m", "what formed?", notes))
        assert out == canned
        # tool_choice is pinned to the ask tool (forced structured output)
        assert client.captured["tool_choice"] == {"type": "tool", "name": ASK_TOOL["name"]}
        assert client.captured["model"] == "m"

    def test_answer_no_notes_short_circuits_without_api(self):
        client = _FakeClient({"should": "not be used"})
        out = asyncio.run(answer_question(client, "m", "anything?", []))
        assert out["coverage"] == "not_in_notebook"
        assert client.captured == {}  # no API call when nothing to ground on
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_notebook_ask.py::TestAnswerQuestion -v`
Expected: FAIL — `ImportError: cannot import name 'ASK_TOOL'`

- [ ] **Step 3: Write minimal implementation**

Append to `gently/harness/memory/notebook_ask.py`:

```python
# ASK_TOOL pins the structured output. No confidence field — we don't ask the
# model to self-rate (lab rule); "coverage" is a factual grounding classification.
ASK_TOOL = {
    "name": "answer_from_notebook",
    "description": "Return a grounded answer built ONLY from the provided notebook entries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Direct synthesis grounded in the notes."},
            "points": {
                "type": "array",
                "description": "Supporting points, each citing the note ids it rests on.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "note_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "note_ids"],
                },
            },
            "suggested_next": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete next experiments/moves if the question asks what to do; else empty.",
            },
            "coverage": {
                "type": "string",
                "enum": ["covered", "partial", "not_in_notebook"],
                "description": "How well the provided notes cover the question.",
            },
        },
        "required": ["answer", "points", "coverage"],
    },
}

_SYSTEM = (
    "You reason over a shared lab notebook. Answer ONLY from the notebook entries "
    "provided — every claim must cite the note id(s) it rests on. If the notes do "
    "not contain the answer, say so plainly and set coverage to 'not_in_notebook'. "
    "Never invent facts not in the notes. Call the answer_from_notebook tool."
)


def _render_notes(notes: list[Note]) -> str:
    lines = []
    for n in notes:
        scope = []
        if n.strains:
            scope.append("strains=" + ",".join(n.strains))
        if n.embryos:
            scope.append("embryos=" + ",".join(n.embryos))
        tag = f" [{'; '.join(scope)}]" if scope else ""
        lines.append(f"[{n.id}] ({n.kind.value}){tag} {n.body}")
    return "\n".join(lines)


def build_ask_messages(question: str, notes: list[Note]) -> list[dict]:
    body = (
        "Notebook entries:\n" + _render_notes(notes) + f"\n\nQuestion: {question}\n\n"
        "Answer using only these entries, citing note ids."
    )
    return [{"role": "user", "content": body}]


async def answer_question(client, model: str, question: str, notes: list[Note]) -> dict:
    """Force the ask tool and return its validated input dict. Short-circuits
    (no API call) when there are no notes to ground on."""
    if not notes:
        return {
            "answer": "The notebook doesn't cover this yet.",
            "points": [],
            "suggested_next": [],
            "coverage": "not_in_notebook",
        }
    resp = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM,
        tools=[ASK_TOOL],
        tool_choice={"type": "tool", "name": ASK_TOOL["name"]},
        messages=build_ask_messages(question, notes),
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input
    return {"answer": "", "points": [], "suggested_next": [], "coverage": "not_in_notebook"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_notebook_ask.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add gently/harness/memory/notebook_ask.py tests/test_notebook_ask.py
git commit -m "feat(notebook): answer_question — forced-tool grounded synthesis"
```

---

### Task 3: `POST /api/notebook/ask`

**Files:**
- Modify: `gently/ui/web/routes/notebook.py`
- Test: `tests/test_notebook_api.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_api.py
class _AskBlock:
    def __init__(self, inp):
        self.type = "tool_use"
        self.input = inp


class _AskResp:
    def __init__(self, inp):
        self.content = [_AskBlock(inp)]
        self.stop_reason = "tool_use"


class _AskMessages:
    def __init__(self, inp):
        self._inp = inp

    async def create(self, **kwargs):
        return _AskResp(self._inp)


class _AskClient:
    def __init__(self, inp):
        self.messages = _AskMessages(inp)


def _make_app_with_client(context_store, client):
    from gently.ui.web.routes.notebook import create_router

    app = FastAPI()

    class _Server:
        pass

    server = _Server()
    server.context_store = context_store
    server.claude_async = client
    app.include_router(create_router(server))
    return app


class TestAsk:
    def test_ask_returns_grounded_answer(self, file_context_store):
        cs = _seed(file_context_store)
        canned = {
            "answer": "A ring formed.",
            "points": [{"text": "ring", "note_ids": ["o1"]}],
            "suggested_next": [],
            "coverage": "covered",
        }
        client = TestClient(_make_app_with_client(cs, _AskClient(canned)))
        resp = client.post("/api/notebook/ask", json={"question": "what happened?"})
        assert resp.status_code == 200
        assert resp.json()["coverage"] == "covered"

    def test_ask_no_store(self):
        client = TestClient(_make_app(None))
        resp = client.post("/api/notebook/ask", json={"question": "x"})
        assert resp.json() == {"available": False}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_notebook_api.py::TestAsk -v`
Expected: FAIL — 404/405 (route not defined)

- [ ] **Step 3: Write minimal implementation**

In `gently/ui/web/routes/notebook.py`, add `Body` to the fastapi import and add the route inside `create_router`, before `return router`:

```python
    @router.post("/api/notebook/ask")
    async def ask(
        question: str = Body(..., embed=True),
        thread: str | None = Body(None, embed=True),
        strain: str | None = Body(None, embed=True),
    ):
        nb = _nb()
        if nb is None:
            return {"available": False}
        from gently.harness.memory.notebook_ask import answer_question, select_notes
        from gently.settings import settings

        notes = select_notes(nb, thread=thread, strain=strain)
        client = getattr(server, "claude_async", None)
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic()
        result = await answer_question(client, settings.models.main, question, notes)
        result["available"] = True
        result["note_ids"] = [n.id for n in notes]
        return result
```

Update the import line at the top of the file:

```python
from fastapi import APIRouter, Body, HTTPException
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_notebook_api.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add gently/ui/web/routes/notebook.py tests/test_notebook_api.py
git commit -m "feat(notebook): POST /api/notebook/ask — grounded notebook Q&A"
```

---

## Self-Review
**Spec coverage (design §4):** structural retrieval (`select_notes`) → grounded synthesis (`answer_question`, forced tool, cited, "not_in_notebook" valid) → endpoint. ✓ No self-rated confidence (`coverage` is grounding, not correctness-confidence). ✓ Reuses `settings.models.main`, `anthropic.AsyncAnthropic`, the `verifier.py` forced-tool pattern. ✓ Client injected → unit-testable without real API. ✓
**Deferred (correct):** embeddings/semantic recall; the "Ask" UI; proactive surfacing.
**Placeholder scan:** none — complete code + commands. ✓
**Type consistency:** `select_notes`, `answer_question`, `ASK_TOOL`, `build_ask_messages` named consistently across tasks/tests; route uses `server.claude_async` (tests inject) with a real `AsyncAnthropic` fallback. ✓
