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

    function refresh() { loadThreads(); loadNotes(); }

    function init() {
        if (inited) { refresh(); return; }
        inited = true;
        setupFilters();
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
