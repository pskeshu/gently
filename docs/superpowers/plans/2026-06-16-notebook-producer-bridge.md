# Notebook Producer Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the existing agent-memory write path actually populate the shared notebook — when `FileContextStore.apply_updates()` records observations and learnings, mirror them into the `NotebookStore` as Notes.

**Architecture:** Pure converters (`observation_to_note`, `learning_to_note`) in `notebook.py`; a lazy `FileContextStore.notebook` property rooted at `agent_dir/notebook`; a guarded mirror step at the end of `apply_updates`. Builds on the foundation plan (`2026-06-16-notebook-foundation.md`). Backend-only, no UI/agent-loop changes. Transitional dual-write (legacy YAML + notebook) — legacy silos retire in a later increment.

**Tech Stack:** Python 3.11, dataclasses, PyYAML, pytest (`file_context_store` fixture in `tests/conftest.py`).

**Out of scope:** wiring the live loop to *call* `apply_updates` (separate increment); read API + Notebook tab; mapping expectations/watchpoints (they're working memory, not notebook entries — see design doc §2).

---

### Task 1: Converters — Observation/Learning → Note

**Files:**
- Modify: `gently/harness/memory/notebook.py`
- Test: `tests/test_notebook_store.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_store.py
from datetime import datetime as _dt

from gently.harness.memory.model import Learning, Observation
from gently.harness.memory.notebook import learning_to_note, observation_to_note


class TestConverters:
    def test_observation_to_note(self):
        obs = Observation(
            id="o1",
            timestamp=_dt(2026, 6, 16, 9, 0, 0),
            type="milestone",
            content="nerve ring formed",
            embryo_id="e1",
            session_id="s1",
            relates_to=["o0"],
            gently_refs={"kind": "projection", "t": 42},
        )
        n = observation_to_note(obs)
        assert n.id == "o1"
        assert n.kind == NoteKind.OBSERVATION
        assert n.body == "nerve ring formed"
        assert n.author == Author.AGENT
        assert n.embryos == ["e1"]
        assert n.sessions == ["s1"]
        assert {"rel": "relates_to", "to": "o0"} in n.links
        assert n.artifacts == [{"kind": "projection", "t": 42}]
        assert n.created_at == _dt(2026, 6, 16, 9, 0, 0)

    def test_learning_to_note(self):
        lrn = Learning(id="l1", content="rings form by comma", confidence=Confidence.HIGH)
        n = learning_to_note(lrn)
        assert n.id == "l1"
        assert n.kind == NoteKind.FINDING
        assert n.body == "rings form by comma"
        assert n.status == NoteStatus.PROPOSED  # agent-drafted, awaits confirm
        assert n.confidence == Confidence.HIGH
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_store.py::TestConverters -v`
Expected: FAIL — `ImportError: cannot import name 'observation_to_note'`

- [ ] **Step 3: Write minimal implementation**

Add the model imports and converters to `notebook.py`. Extend the existing model import line:

```python
from .model import Confidence, Learning, Observation
```

Append at end of `notebook.py` (module-level functions, after `note_from_dict`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook_store.py::TestConverters -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/harness/memory/notebook.py tests/test_notebook_store.py
git commit -m "feat(notebook): Observation/Learning -> Note converters"
```

---

### Task 2: `FileContextStore.notebook` property

**Files:**
- Modify: `gently/harness/memory/file_store.py` (add property near `apply_updates`)
- Test: `tests/test_notebook_store.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_store.py
class TestContextStoreNotebook:
    def test_notebook_property_rooted_under_agent_dir(self, file_context_store):
        nb = file_context_store.notebook
        assert nb.root == file_context_store.agent_dir / "notebook"

    def test_notebook_property_is_cached(self, file_context_store):
        assert file_context_store.notebook is file_context_store.notebook
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_store.py::TestContextStoreNotebook -v`
Expected: FAIL — `AttributeError: 'FileContextStore' object has no attribute 'notebook'`

- [ ] **Step 3: Write minimal implementation**

In `gently/harness/memory/file_store.py`, add this property immediately **before** `def apply_updates(self, updates: ContextUpdates):` (line ~2178):

```python
@property
def notebook(self):
    """The shared lab notebook, rooted at agent_dir/notebook (lazy)."""
    nb = getattr(self, "_notebook", None)
    if nb is None:
        from .notebook import NotebookStore

        nb = NotebookStore(self.agent_dir / "notebook")
        self._notebook = nb
    return nb
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook_store.py::TestContextStoreNotebook -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/harness/memory/file_store.py tests/test_notebook_store.py
git commit -m "feat(notebook): FileContextStore.notebook lazy property"
```

---

### Task 3: Mirror observations & learnings in `apply_updates`

**Files:**
- Modify: `gently/harness/memory/file_store.py` (`apply_updates`)
- Test: `tests/test_notebook_store.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_store.py
class TestApplyUpdatesMirror:
    def test_apply_updates_mirrors_observations_and_learnings(self, file_context_store):
        from gently.harness.memory.model import ContextUpdates

        cs = file_context_store
        obs = Observation(
            id="o1",
            timestamp=_dt(2026, 6, 16, 9, 0, 0),
            type="milestone",
            content="ring formed",
            embryo_id="e1",
        )
        lrn = Learning(id="l1", content="rings form by comma", confidence=Confidence.HIGH)
        cs.apply_updates(ContextUpdates(new_observations=[obs], new_learnings=[lrn]))

        bodies = {n.body for n in cs.notebook.query_notes()}
        assert "ring formed" in bodies
        assert "rings form by comma" in bodies
        assert cs.notebook.ids_for_embryo("e1") == ["o1"]

    def test_apply_updates_empty_is_noop_for_notebook(self, file_context_store):
        from gently.harness.memory.model import ContextUpdates

        cs = file_context_store
        cs.apply_updates(ContextUpdates())
        assert cs.notebook.query_notes() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notebook_store.py::TestApplyUpdatesMirror -v`
Expected: FAIL — `assert "ring formed" in set()` (notebook not populated yet)

- [ ] **Step 3: Write minimal implementation**

In `gently/harness/memory/file_store.py`, at the END of `apply_updates` (after the `if updates.new_focus is not None:` block), append:

```python
        # Mirror new observations & learnings into the shared notebook
        # (best-effort — a notebook failure never breaks the legacy write).
        from .notebook import learning_to_note, observation_to_note

        try:
            for obs in updates.new_observations:
                self.notebook.write_note(observation_to_note(obs))
            for learning in updates.new_learnings:
                self.notebook.write_note(learning_to_note(learning))
        except Exception:
            logger.warning("notebook mirror failed", exc_info=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notebook_store.py::TestApplyUpdatesMirror -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run full notebook suite + commit**

Run: `python -m pytest tests/test_notebook_store.py -q`
Expected: all pass

```bash
git add gently/harness/memory/file_store.py tests/test_notebook_store.py
git commit -m "feat(notebook): apply_updates mirrors observations & learnings into notebook"
```

---

## Self-Review

**Spec coverage:** Producer wiring (design doc increment 1b) — `apply_updates` now populates the notebook. ✓ Converters honor the model (Observation→observation note, Learning→finding/proposed). ✓ Working-memory types (expectation/watchpoint) intentionally not mirrored (design §2). ✓
**Placeholder scan:** none; complete code + commands throughout. ✓
**Type consistency:** `observation_to_note`/`learning_to_note`, `FileContextStore.notebook`, `NoteKind`/`Author`/`NoteStatus`/`Confidence` match the foundation module and `model.py` (`Observation`, `Learning`, `ContextUpdates` confirmed at `file_store.py:2178-2203`). ✓
