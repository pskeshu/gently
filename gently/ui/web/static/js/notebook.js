/**
 * NotebookApp — the LIBRARY "Notebook" tab.
 *
 * The reading room for the shared lab notebook: a thread rail (the inquiry
 * spine) + kind filter, rendering Notes from the read API (/api/notebook).
 * Read-only for now; authoring/curation arrive in a later increment.
 */
const NotebookApp = (() => {
    let inited = false;
    let kindFilter = '';      // '' | observation | finding | question
    let threadFilter = '';    // '' = all notes

    const $ = (id) => document.getElementById(id);

    async function fetchJSON(url) {
        try {
            const r = await fetch(url);
            if (!r.ok) return null;
            return await r.json();
        } catch (e) {
            return null;
        }
    }

    function kindMeta(kind) {
        return ({
            observation: { label: 'Observation', cls: 'nb-k-obs' },
            finding: { label: 'Finding', cls: 'nb-k-find' },
            question: { label: 'Question', cls: 'nb-k-q' },
        })[kind] || { label: kind || 'note', cls: '' };
    }

    async function loadThreads() {
        const rail = $('nb-threads');
        if (!rail) return;
        const data = await fetchJSON('/api/notebook/threads');
        const threads = (data && data.threads) || [];
        rail.innerHTML = '';
        const mk = (id, label, count, active) => {
            const b = document.createElement('button');
            b.className = 'nb-thread' + (active ? ' active' : '');
            b.textContent = label + (count != null ? `  ${count}` : '');
            b.addEventListener('click', () => { threadFilter = id; loadThreads(); loadNotes(); });
            return b;
        };
        rail.appendChild(mk('', 'All notes', null, threadFilter === ''));
        threads.forEach(t => rail.appendChild(mk(t.id, t.id, t.count, threadFilter === t.id)));
    }

    function card(n) {
        const km = kindMeta(n.kind);
        const el = document.createElement('div');
        el.className = 'nb-card';

        const head = document.createElement('div');
        head.className = 'nb-card-head';
        const badge = document.createElement('span');
        badge.className = 'nb-badge ' + km.cls;
        badge.textContent = km.label;
        const author = document.createElement('span');
        author.className = 'nb-author';
        author.textContent = n.author || '';
        const status = document.createElement('span');
        status.className = 'nb-status';
        status.textContent = n.status || '';
        head.append(badge, author, status);

        const body = document.createElement('div');
        body.className = 'nb-body-text';
        body.textContent = n.title || n.body || '';

        el.append(head, body);

        const chips = []
            .concat((n.strains || []).map(s => '🧬 ' + s))
            .concat((n.embryos || []).map(e => '◌ ' + e))
            .concat((n.threads || []).map(t => '# ' + t));
        if (chips.length) {
            const row = document.createElement('div');
            row.className = 'nb-chips';
            chips.forEach(text => {
                const c = document.createElement('span');
                c.className = 'nb-chip';
                c.textContent = text;
                row.appendChild(c);
            });
            el.appendChild(row);
        }
        return el;
    }

    async function loadNotes() {
        const list = $('nb-notes');
        if (!list) return;
        const params = new URLSearchParams();
        if (kindFilter) params.set('kind', kindFilter);
        if (threadFilter) params.set('thread', threadFilter);
        const qs = params.toString();
        const data = await fetchJSON('/api/notebook/notes' + (qs ? `?${qs}` : ''));
        if (!data || data.available === false) {
            list.innerHTML = '<div class="nb-empty">Notebook unavailable.</div>';
            return;
        }
        const notes = data.notes || [];
        if (!notes.length) {
            list.innerHTML =
                '<div class="nb-empty">No notes yet — the notebook fills as the agent ' +
                'records observations, findings, and open questions.</div>';
            return;
        }
        list.innerHTML = '';
        notes.forEach(n => list.appendChild(card(n)));
    }

    function setupFilters() {
        document.querySelectorAll('#notebook-content [data-nb-kind]').forEach(btn => {
            btn.addEventListener('click', () => {
                kindFilter = btn.dataset.nbKind;
                document.querySelectorAll('#notebook-content [data-nb-kind]')
                    .forEach(b => b.classList.toggle('active', b === btn));
                loadNotes();
            });
        });
    }

    // ── Ask the notebook ───────────────────────────────────────────────
    function renderAskResult(data) {
        const box = $('nb-ask-result');
        if (!box) return;
        box.hidden = false;
        if (!data || data.available === false) {
            box.innerHTML = '<div class="nb-ask-empty">The notebook is unavailable right now.</div>';
            return;
        }
        const cov = data.coverage || 'covered';
        const covLabel = { covered: 'Grounded in the notebook', partial: 'Partially covered', not_in_notebook: 'Not in the notebook yet' }[cov] || cov;
        const points = (data.points || []).map(p => `
            <li class="nb-ask-point">
                <span>${esc(p.text)}</span>
                ${(p.note_ids || []).map(id => `<span class="nb-cite">${esc(id)}</span>`).join('')}
            </li>`).join('');
        const nexts = (data.suggested_next || []).map(s => `<li>${esc(s)}</li>`).join('');
        box.innerHTML =
            `<div class="nb-ask-cov nb-cov-${esc(cov)}">${esc(covLabel)}</div>` +
            `<div class="nb-ask-answer">${esc(data.answer || '')}</div>` +
            (points ? `<div class="nb-ask-h">Why</div><ul class="nb-ask-points">${points}</ul>` : '') +
            (nexts ? `<div class="nb-ask-h">Try next</div><ul class="nb-ask-next">${nexts}</ul>` : '');
    }

    async function ask() {
        const input = $('nb-ask-input');
        const box = $('nb-ask-result');
        const q = (input && input.value || '').trim();
        if (!q) return;
        if (box) { box.hidden = false; box.innerHTML = '<div class="nb-ask-loading">Thinking over the notebook…</div>'; }
        const body = { question: q };
        if (threadFilter) body.thread = threadFilter;   // ask within the selected thread
        try {
            const r = await fetch('/api/notebook/ask', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
            });
            renderAskResult(await r.json());
        } catch (e) {
            if (box) box.innerHTML = '<div class="nb-ask-empty">Something went wrong asking the notebook.</div>';
        }
    }

    function setupAsk() {
        const go = $('nb-ask-go'), input = $('nb-ask-input');
        if (go) go.addEventListener('click', ask);
        if (input) input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); ask(); } });
    }

    function esc(s) {
        return (typeof escapeHtml === 'function') ? escapeHtml(String(s == null ? '' : s))
            : String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    }

    function refresh() { loadThreads(); loadNotes(); }

    function init() {
        if (inited) { refresh(); return; }
        inited = true;
        setupFilters();
        setupAsk();
        refresh();
        // Notebook writes ride the CONTEXT_UPDATED event — live-refresh if visible.
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('CONTEXT_UPDATED', () => {
                if (typeof state !== 'undefined' && state.tab === 'notebook') refresh();
            });
        }
    }

    return { init };
})();
