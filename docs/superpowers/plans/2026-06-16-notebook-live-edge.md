# Notebook Live Edge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the Home "Agent's view" panel surface recent notebook activity — the ambient "live edge" of the notebook (design §3, two-faced presentation), complementing the Notebook tab (the reading room).

**Architecture:** Add an optional `limit` to the notes read API (TDD). Extend `context-surface.js` to also fetch recent notes and render a "From the notebook" section whose rows click through to the Notebook tab. Reuse the existing `cx-dot` colors (amber/blue/green) for kinds — no new CSS.

**Tech Stack:** FastAPI, pytest + TestClient (venv), vanilla JS.

**Out of scope:** retrieval/"Ask the notebook" (next increment); proactive surfacing.

---

### Task 1: `limit` param on `GET /api/notebook/notes`

**Files:**
- Modify: `gently/ui/web/routes/notebook.py`
- Test: `tests/test_notebook_api.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_notebook_api.py
class TestLimit:
    def test_limit_returns_newest(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes?limit=1").json()
        assert len(data["notes"]) == 1  # newest-first, capped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_notebook_api.py::TestLimit -v`
Expected: FAIL — returns 2 notes, not 1.

- [ ] **Step 3: Write minimal implementation**

In `gently/ui/web/routes/notebook.py`, add a `limit` param to `list_notes` and slice. Replace the `list_notes` signature and the final return:

```python
    @router.get("/api/notebook/notes")
    async def list_notes(
        kind: str | None = None,
        author: str | None = None,
        status: str | None = None,
        strain: str | None = None,
        embryo: str | None = None,
        thread: str | None = None,
        limit: int | None = None,
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
        if limit is not None and limit >= 0:
            notes = notes[:limit]
        return {"available": True, "notes": [note_to_dict(n) for n in notes]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_notebook_api.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add gently/ui/web/routes/notebook.py tests/test_notebook_api.py
git commit -m "feat(notebook): limit param on GET /api/notebook/notes"
```

---

### Task 2: "From the notebook" section in the Agent's-view panel

**Files:**
- Modify: `gently/ui/web/static/js/context-surface.js`

Verification is browser-based (chrome-devtools), not pytest.

- [ ] **Step 1: Extend `fetchAndRender` to also pull recent notes**

Replace `fetchAndRender` with a version that fetches both endpoints and passes notes to `render`:

```javascript
    async function fetchAndRender() {
        if (!el || loading) return;
        loading = true;
        try {
            const [ctx, nb] = await Promise.all([
                fetch('/api/context').then(r => r.json()).catch(() => ({})),
                fetch('/api/notebook/notes?limit=5').then(r => r.json()).catch(() => ({})),
            ]);
            render(ctx || {}, (nb && nb.notes) || []);
        } catch (e) { /* keep last render */ }
        finally { loading = false; }
    }
```

- [ ] **Step 2: Render the notebook section**

Replace `render(data)` with `render(data, notes)`. Add the notebook section and include notes in the empty-state check. Replace the whole `render` function body:

```javascript
    function render(data, notes) {
        if (!el) return;
        notes = notes || [];
        const hc = hasControl();
        const questions = data.questions || [], watchpoints = data.watchpoints || [], expectations = data.expectations || [];
        el.classList.remove('hidden');
        if (!questions.length && !watchpoints.length && !expectations.length && !notes.length) {
            el.innerHTML = '<div class="cx-title">Agent’s view</div>' +
                '<div class="cx-empty">Nothing yet — the agent’s notes, expectations, and open questions appear here as it works.</div>';
            return;
        }

        const qHtml = questions.map(it => `
            <div class="cx-item" data-kind="questions" data-id="${esc(it.id)}">
                <span class="cx-dot cx-q"></span>
                <span class="cx-text">${esc(it.content)}</span>
                ${hc ? '<button class="cx-act" data-act="answer">Answer</button>' : ''}
                ${hc ? '<div class="cx-answer hidden"><input class="cx-answer-input" placeholder="Answer / resolve…"><button class="cx-answer-go">→</button></div>' : ''}
            </div>`).join('');
        const wHtml = watchpoints.map(it => `
            <div class="cx-item" data-kind="watchpoints" data-id="${esc(it.id)}">
                <span class="cx-dot cx-w"></span>
                <span class="cx-text">${esc(it.target)}${it.condition ? ' — ' + esc(it.condition) : ''}</span>
                ${hc ? '<button class="cx-act" data-act="resolve">Resolve</button>' : ''}
            </div>`).join('');
        const eHtml = expectations.map(it => `
            <div class="cx-item" data-kind="expectations" data-id="${esc(it.id)}">
                <span class="cx-dot cx-e"></span>
                <span class="cx-text">${esc(it.target)}${it.prediction ? ': ' + esc(it.prediction) : ''}</span>
                ${hc ? '<button class="cx-act" data-act="confirm" title="Mark confirmed">✓</button>' : ''}
            </div>`).join('');
        // kind → existing cx-dot color: observation=blue, finding=green, question=amber
        const dotFor = (k) => k === 'finding' ? 'cx-e' : (k === 'question' ? 'cx-q' : 'cx-w');
        const nHtml = notes.map(n => `
            <div class="cx-item cx-note" data-note="1">
                <span class="cx-dot ${dotFor(n.kind)}"></span>
                <span class="cx-text">${esc(n.title || n.body)}</span>
            </div>`).join('');

        el.innerHTML = '<div class="cx-title">Agent’s view</div>' +
            section('Open questions', qHtml) + section('Watching', wHtml) +
            section('Expectations', eHtml) + section('From the notebook', nHtml);
        wire();
    }
```

- [ ] **Step 3: Make notebook rows click through to the Notebook tab**

In `wire()`, after the existing `.cx-item` loop, add a handler for note rows (append inside `wire`, before its closing brace):

```javascript
        el.querySelectorAll('.cx-note').forEach(row => {
            row.style.cursor = 'pointer';
            row.addEventListener('click', () => {
                if (typeof switchTab === 'function') switchTab('notebook');
            });
        });
```

- [ ] **Step 4: Verify in the browser**

Restart the server, open `http://localhost:8080`, confirm the Home "Agent's view" panel shows a "From the notebook" section with recent notes, and clicking a row switches to the Notebook tab. Use chrome-devtools (navigate, evaluate_script to click, take_screenshot, list_console_messages → zero errors).

- [ ] **Step 5: Commit**

```bash
git add gently/ui/web/static/js/context-surface.js
git commit -m "feat(notebook): Agent's-view live edge — recent notes section -> Notebook tab"
```

---

## Self-Review
**Spec coverage:** design §3 two-faced presentation — the ambient live edge now surfaces recent notebook notes on Home, clicking through to the tab. ✓ Reuses `/api/notebook/notes` (+ new `limit`) and existing `cx-dot` colors. ✓
**Placeholder scan:** none. ✓
**Type consistency:** `render(data, notes)`, `dotFor`, `limit` param, `switchTab('notebook')` all consistent; kinds map to existing cx classes. ✓
