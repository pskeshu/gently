/**
 * NotebookApp — the LIBRARY "Notebook" tab.
 *
 * The notes are one flat pool, read through pages. A page is a saved query
 * over the notes with a conversation pinned to it: what you and the agent
 * said while looking at exactly that set. Open the page again and both are
 * there, except the context is live — a new note that matches joins the
 * page on its own. Built-in pages come from the facets (All notes, Agent's
 * findings, Asks, Session summaries, one per project, one per thread); your
 * own are any query you save.
 *
 * The notebook is also the working set: what the agent reads. The one verb
 * on a card is to set a note aside — it leaves the notebook and the agent's
 * reading, and stays in the record (the stream, the inbox) — with Undo in
 * the toast, and a "Set aside" view to put any of them back.
 */
const NotebookApp = (() => {
    let inited = false;
    let pageId = 'all';       // the page open now
    let page = null;          // it, as last loaded: {id, name, query, builtin, turns}
    let pages = [];           // the rail: built-ins (with counts) and saved pages
    let kindFilter = '';      // '' | observation | finding | question — a view filter, not the query
    let shelf = 'notebook';   // 'notebook' = what the agent reads | 'aside' = set aside
    let sessionNames = null;  // session id -> name, so a chip reads "monday run"

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

    // A write: the server's reason on failure, not the number.
    async function send(url, body, method = 'POST') {
        const r = await fetch(url, {
            method,
            headers: body !== undefined ? { 'Content-Type': 'application/json' } : {},
            body: body !== undefined ? JSON.stringify(body) : undefined,
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || `the server answered ${r.status}`);
        return d;
    }

    function esc(s) {
        return (typeof escapeHtml === 'function') ? escapeHtml(String(s == null ? '' : s))
            : String(s == null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    }

    function kindMeta(kind) {
        return ({
            observation: { label: 'Observation', cls: 'nb-k-obs' },
            finding: { label: 'Finding', cls: 'nb-k-find' },
            question: { label: 'Question', cls: 'nb-k-q' },
        })[kind] || { label: kind || 'note', cls: '' };
    }

    // Only a control holder may set a note aside, keep a reply, save a page
    // or write a line; a viewer sees none of those controls.
    const hasControl = () =>
        (typeof AgentChat !== 'undefined' && AgentChat.hasControl) ? AgentChat.hasControl() : true;

    function toast(msg, actionLabel, actionFn, ms) {
        if (typeof showGentlyToast === 'function') showGentlyToast(msg, actionLabel, actionFn, ms);
    }
    function toastFail(msg) {
        if (typeof showGentlyToast === 'function') showGentlyToast(msg, null, null, 6000, 'error');
    }

    function firstWords(n, max = 48) {
        const text = String(n.title || n.body || '').replace(/\s+/g, ' ').trim();
        return text.length > max ? text.slice(0, max - 1) + '…' : text;
    }

    // Session names, so a chip reads "monday run" and not "session 6ef95558".
    // One fetch per load of the page; the id stays in the chip's title.
    async function loadSessionNames() {
        const data = await fetchJSON('/api/sessions');
        sessionNames = {};
        ((data && data.sessions) || []).forEach(s => {
            sessionNames[s.session_id] = s.name || s.suggested_name || '';
        });
    }
    function sessionLabel(sid) {
        const name = sessionNames && sessionNames[sid];
        // An unnamed session's "name" is its id; say what it is.
        return name && name !== sid ? name : 'session ' + String(sid).slice(0, 8);
    }

    // ── The rail: pages, then threads ──────────────────────────────────
    async function loadPages() {
        const [pg, th] = await Promise.all([
            fetchJSON('/api/notebook/pages'),
            fetchJSON(`/api/notebook/threads?shelf=notebook`),
        ]);
        pages = (pg && pg.pages) || [];
        renderRail((th && th.threads) || []);
    }

    function railButton(id, label, count, talked) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'nb-thread' + (id === pageId ? ' active' : '');
        b.innerHTML = `<span class="nb-rail-name">${esc(label)}</span>`
            + (talked ? `<span class="nb-rail-talk" title="This page has a conversation">…</span>` : '')
            + (count != null ? `<span class="nb-rail-count">${esc(count)}</span>` : '');
        b.addEventListener('click', () => openPage(id));
        return b;
    }

    function renderRail(threads) {
        const host = $('nb-pages'), rail = $('nb-threads');
        if (!host || !rail) return;
        host.innerHTML = '';
        const builtins = pages.filter(p => p.builtin), mine = pages.filter(p => !p.builtin);
        builtins.forEach(p => host.appendChild(railButton(p.id, p.name, p.count, p.turn_count > 0)));
        if (mine.length) {
            const h = document.createElement('div');
            h.className = 'nb-rail-h';
            h.textContent = 'Your pages';
            host.appendChild(h);
            mine.forEach(p => host.appendChild(railButton(p.id, p.name, p.count, p.turn_count > 0)));
        }
        rail.innerHTML = '';
        if (!threads.length) {
            rail.innerHTML = '<div class="nb-rail-none">No threads yet.</div>';
        }
        threads.forEach(t => rail.appendChild(railButton('thread:' + t.id, '# ' + t.id, t.count, false)));
    }

    function openPage(id) {
        pageId = id;
        loadPages();
        loadPage();
    }

    // ── The page: head, notes, conversation ────────────────────────────
    async function loadPage() {
        const data = await fetchJSON(`/api/notebook/pages/${encodeURIComponent(pageId)}`);
        if (!data || data.available === false) {
            if (pageId !== 'all') { pageId = 'all'; return loadPage(); }
            page = null;
            $('nb-notes').innerHTML = '<div class="nb-empty">Notebook unavailable.</div>';
            $('nb-conv').innerHTML = '';
            renderHead();
            return;
        }
        page = data.page;
        renderHead();
        if (shelf === 'aside') {
            await loadAsideNotes();
        } else {
            renderNotes(filterKind(data.notes || []));
        }
        renderConversation();
    }

    function filterKind(notes) {
        return kindFilter ? notes.filter(n => n.kind === kindFilter) : notes;
    }

    // The set-aside view of a page: the same query over what was set aside.
    async function loadAsideNotes() {
        const params = new URLSearchParams({ shelf: 'aside' });
        const q = (page && page.query) || {};
        ['kind', 'author', 'status', 'strain', 'embryo', 'thread', 'project', 'session'].forEach(k => {
            if (q[k]) params.set(k, q[k]);
        });
        const data = await fetchJSON(`/api/notebook/notes?${params}`);
        let notes = (data && data.notes) || [];
        if (q.title) notes = notes.filter(n => n.title === q.title);
        if (q.asks) notes = notes.filter(n => (n.mentions || []).length);
        renderNotes(filterKind(notes));
    }

    function queryWords(q) {
        const parts = [];
        if (q.author) parts.push(`by the ${q.author === 'human' ? 'people' : q.author}`);
        if (q.kind) parts.push(q.kind + 's');
        if (q.status) parts.push(q.status);
        if (q.title) parts.push(`titled “${q.title}”`);
        if (q.asks) parts.push('that ask someone');
        if (q.thread) parts.push('# ' + q.thread);
        if (q.project) parts.push('on this project');
        if (q.strain) parts.push('🧬 ' + q.strain);
        if (q.session) parts.push('on one session');
        if (q.embryo) parts.push('◌ ' + q.embryo);
        if (q.person) parts.push('asking ' + q.person);
        return parts.length ? parts.join(' · ') : 'every note in the notebook';
    }

    function renderHead() {
        const host = $('nb-page-head');
        if (!host) return;
        if (!page) { host.innerHTML = ''; return; }
        const q = page.query || {};
        const talked = (page.turns || []).length;
        const actions = [];
        if (hasControl()) {
            if (page.builtin || kindFilter) {
                actions.push(`<button type="button" class="nb-head-btn" data-act="save" title="Save this view as a page of your own, with its own conversation">Save as page</button>`);
            }
            if (talked) {
                actions.push(`<button type="button" class="nb-head-btn" data-act="clear" title="Start this page’s conversation over; the notes stay">Clear conversation</button>`);
            }
            if (!page.builtin) {
                actions.push(`<button type="button" class="nb-head-btn is-danger" data-act="delete" title="Delete this page: its conversation goes, the notes stay">Delete page</button>`);
            }
        }
        host.innerHTML = `
            <div class="nb-page-title">
                <h3>${esc(page.name)}</h3>
                <span class="nb-page-q">${esc(queryWords(q))}${kindFilter ? ` · ${esc(kindFilter)}s only` : ''}</span>
            </div>
            <div class="nb-page-actions">${actions.join('')}</div>
            <div class="nb-page-save" id="nb-page-save" hidden>
                <input id="nb-page-name" placeholder="Name this page…" maxlength="60">
                <button type="button" class="nb-head-btn" data-act="save-go">Save</button>
                <button type="button" class="nb-head-btn" data-act="save-no">Cancel</button>
            </div>`;
        host.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', () => headAction(b.dataset.act)));
    }

    async function headAction(act) {
        if (act === 'save') {
            const box = $('nb-page-save');
            box.hidden = false;
            const inp = $('nb-page-name');
            inp.value = page.builtin ? '' : page.name;
            inp.focus();
            inp.onkeydown = e => { if (e.key === 'Enter') headAction('save-go'); if (e.key === 'Escape') headAction('save-no'); };
            return;
        }
        if (act === 'save-no') { $('nb-page-save').hidden = true; return; }
        if (act === 'save-go') {
            const name = ($('nb-page-name').value || '').trim();
            if (!name) return;
            const query = Object.assign({}, page.query || {});
            if (kindFilter) query.kind = kindFilter;
            try {
                const d = await send('/api/notebook/pages', { name, query });
                kindFilter = '';
                document.querySelectorAll('#notebook-content [data-nb-kind]')
                    .forEach(b => b.classList.toggle('active', b.dataset.nbKind === ''));
                openPage(d.page.id);
                toast(`Saved “${name}” as a page.`, null, null, 4000);
            } catch (e) {
                toastFail(`Not saved (${e.message || 'unknown error'})`);
            }
            return;
        }
        if (act === 'clear') {
            try {
                await send(`/api/notebook/pages/${encodeURIComponent(pageId)}/turns/clear`);
                openPage(pageId);
            } catch (e) {
                toastFail(`Not cleared (${e.message || 'unknown error'})`);
            }
            return;
        }
        if (act === 'delete') {
            try {
                await send(`/api/notebook/pages/${encodeURIComponent(pageId)}`, undefined, 'DELETE');
                toast(`Deleted the page “${page.name}”. Its notes are still in the notebook.`, null, null, 5000);
                openPage('all');
            } catch (e) {
                toastFail(`Not deleted (${e.message || 'unknown error'})`);
            }
        }
    }

    // ── Notes ──────────────────────────────────────────────────────────
    function renderNotes(notes) {
        const list = $('nb-notes');
        if (!list) return;
        if (!notes.length) {
            list.innerHTML = shelf === 'aside'
                ? '<div class="nb-empty">Nothing set aside here. A note’s × takes it out of the ' +
                  'notebook and the agent’s reading; it comes here, and can be put back.</div>'
                : '<div class="nb-empty">Nothing on this page yet — it fills as notes that match are ' +
                  'written, by you or by the agent.</div>';
            return;
        }
        list.innerHTML = '';
        notes.forEach(n => list.appendChild(card(n)));
    }

    function card(n) {
        const km = kindMeta(n.kind);
        const el = document.createElement('div');
        el.className = 'nb-card';
        el.dataset.noteId = n.id;

        const head = document.createElement('div');
        head.className = 'nb-card-head';
        const badge = document.createElement('span');
        badge.className = 'nb-badge ' + km.cls;
        badge.textContent = km.label;
        const author = document.createElement('span');
        author.className = 'nb-author';
        author.innerHTML = n.author === 'agent' || !n.by
            ? esc(n.author || '')
            : (typeof Who !== 'undefined' ? Who.chip(n.by) : esc(n.by));
        const status = document.createElement('span');
        status.className = 'nb-status';
        status.textContent = n.status || '';
        head.append(badge, author, status);
        if (hasControl()) {
            const b = document.createElement('button');
            b.type = 'button';
            if (n.in_notebook === false) {
                b.className = 'nb-putback';
                b.textContent = 'Put back';
                b.title = 'Put this note back in the notebook, so the agent reads it again';
                b.addEventListener('click', () => setAside(n, el, false));
            } else {
                b.className = 'nb-del';
                b.textContent = '×';
                b.title = 'Set aside: out of the notebook and the agent’s reading; still in the stream';
                b.setAttribute('aria-label', 'Set this note aside');
                b.addEventListener('click', () => setAside(n, el, true));
            }
            head.appendChild(b);
        }

        const body = document.createElement('div');
        body.className = 'nb-body-text';
        body.textContent = n.title && n.title !== 'Session summary' ? `${n.title}: ${n.body || ''}` : (n.body || '');

        el.append(head, body);

        // Each chip goes somewhere. A session chip (or an embryo's, which
        // names its session) opens the session in the Sessions tab, landing
        // on the very line this note is — its source. A thread chip opens
        // the thread's page.
        const embryos = Array.isArray(n.embryo_refs)
            ? n.embryo_refs.map(r => ({ label: r.label, sid: r.session || r.session_id || null }))
            : (Array.isArray(n.embryos) ? n.embryos.map(e => ({ label: e, sid: null })) : []);
        const sessions = Array.isArray(n.sessions) ? n.sessions : [];
        const home = sessions[0] || null;
        const chips = []
            .concat((n.strains || []).map(s => ({ text: '🧬 ' + s })))
            .concat(embryos.length ? [] : sessions.map(s => ({ text: sessionLabel(s), sid: s })))
            .concat(embryos.map(e => ({ text: '◌ ' + e.label, sid: e.sid || home })))
            .concat((n.threads || []).map(t => ({ text: '# ' + t, thread: t })))
            .concat((n.basis || []).map(b => ({ text: 'rests on ' + b, note: b })));
        if (chips.length) {
            const row = document.createElement('div');
            row.className = 'nb-chips';
            chips.forEach(ch => {
                let c;
                if (ch.sid) {
                    c = document.createElement('a');
                    // The href is the same place by URL (a new tab, a middle
                    // click); a plain click goes there within this page.
                    c.href = `/#sessions:${encodeURIComponent(ch.sid)}:${encodeURIComponent(n.id)}`;
                    c.className = 'nb-chip nb-chip-link';
                    c.title = `Open session ${ch.sid} where this was written`;
                    c.addEventListener('click', e => {
                        if (e.ctrlKey || e.metaKey || e.shiftKey || e.button !== 0) return;
                        if (typeof switchTab !== 'function' || typeof ReviewApp === 'undefined') return;
                        e.preventDefault();
                        switchTab('sessions');
                        ReviewApp.openNote(ch.sid, n.id);
                    });
                } else if (ch.thread) {
                    c = document.createElement('button');
                    c.type = 'button';
                    c.className = 'nb-chip nb-chip-link';
                    c.title = 'Open this thread’s page';
                    c.addEventListener('click', () => openPage('thread:' + ch.thread));
                } else if (ch.note) {
                    c = document.createElement('button');
                    c.type = 'button';
                    c.className = 'nb-chip nb-chip-link';
                    c.title = 'The note this rests on';
                    c.addEventListener('click', () => flashNote(ch.note));
                } else {
                    c = document.createElement('span');
                    c.className = 'nb-chip';
                }
                c.textContent = ch.text;
                row.appendChild(c);
            });
            el.appendChild(row);
        }
        return el;
    }

    // A citation clicked: the card lit and brought into view, if it is on the page.
    function flashNote(id) {
        const el = document.querySelector(`#nb-notes .nb-card[data-note-id="${CSS.escape(id)}"]`);
        if (!el) { toast(`Note ${id} is not on this page.`, null, null, 3000); return; }
        el.classList.add('is-target');
        el.scrollIntoView({ block: 'center', behavior: 'smooth' });
        setTimeout(() => el.classList.remove('is-target'), 3000);
    }

    // ── Set a note aside, and put it back ──────────────────────────────
    // One flag on the note. Aside: out of every page, out of what the agent
    // is told; still in the stream and the inbox, untouched. Undo and "Put
    // back" are the same call the other way.
    async function setAside(n, cardEl, aside) {
        try {
            await send(`/api/notebook/notes/${encodeURIComponent(n.id)}/in_notebook`, { value: !aside });
            if (cardEl) cardEl.remove();                        // it leaves whichever view it was in
            if (!$('nb-notes').children.length) loadPage();    // the empty state, not a blank
            loadPages();                                        // the counts on the rail
            if (aside) {
                toast(`Set aside “${firstWords(n)}”. Still in the stream; the agent no longer reads it.`,
                    'Undo', () => setAside(n, null, false), 12000);
            } else {
                toast(`“${firstWords(n)}” is back in the notebook.`, null, null, 4000);
                if (shelf === 'notebook') loadPage();
            }
        } catch (e) {
            toastFail(`${aside ? 'Not set aside' : 'Not put back'} (${e && e.message ? e.message : 'unknown error'})`);
        }
    }

    // ── The conversation on the page ───────────────────────────────────
    function renderConversation() {
        const host = $('nb-conv');
        if (!host) return;
        const turns = (page && page.turns) || [];
        if (!turns.length) {
            host.innerHTML = `<div class="nb-conv-empty">Ask the agent about what is on this page. The answer rests only on these notes, and the conversation stays with the page.</div>`;
            return;
        }
        const covLabel = { covered: 'Grounded in these notes', partial: 'Partially covered', not_in_notebook: 'Not in these notes' };
        host.innerHTML = turns.map((t, i) => {
            if (t.role === 'person') {
                const who = typeof Who !== 'undefined' ? Who.head(t.by) : `<b>${esc(t.by || 'someone')}</b>`;
                return `<div class="nb-turn is-person"><div class="nb-turn-head">${who}<span class="nb-turn-when">${esc(when(t.created_at))}</span></div><div class="nb-turn-text">${esc(t.text)}</div></div>`;
            }
            const cov = t.coverage || '';
            const points = (t.points || []).map(p => `
                <li class="nb-ask-point"><span>${esc(p.text)}</span>${(p.note_ids || []).map(id => `<button type="button" class="nb-cite" data-note="${esc(id)}" title="The note this rests on">${esc(id)}</button>`).join('')}</li>`).join('');
            let keep = '';
            if (t.kept_as) {
                keep = `<button type="button" class="nb-kept" data-note="${esc(t.kept_as)}" title="This reply is a proposed finding now; click to see it">Kept as a finding · ${esc(t.kept_as)}</button>`;
            } else if (hasControl() && t.text) {
                keep = `<button type="button" class="nb-keep" data-keep="${i}" title="File this reply as a finding, proposed until you confirm it, resting on the notes it cites">Keep as finding</button>`;
            }
            return `<div class="nb-turn is-agent">
                <div class="nb-turn-head">${typeof Who !== 'undefined' ? Who.head('agent') : '<b>agent</b>'}<span class="nb-turn-when">${esc(when(t.created_at))}</span>${cov ? `<span class="nb-ask-cov nb-cov-${esc(cov)}">${esc(covLabel[cov] || cov)}</span>` : ''}</div>
                <div class="nb-turn-text">${esc(t.text)}</div>
                ${points ? `<ul class="nb-ask-points">${points}</ul>` : ''}
                <div class="nb-turn-foot">${keep}</div>
            </div>`;
        }).join('');
        host.querySelectorAll('[data-note]').forEach(b => b.addEventListener('click', () => flashNote(b.dataset.note)));
        host.querySelectorAll('[data-keep]').forEach(b => b.addEventListener('click', () => keepReply(Number(b.dataset.keep))));
        host.scrollTop = host.scrollHeight;
    }

    function when(iso) {
        return (typeof formatDate === 'function') ? formatDate(iso) : String(iso || '').slice(0, 16);
    }

    async function keepReply(index) {
        try {
            const d = await send(`/api/notebook/pages/${encodeURIComponent(pageId)}/turns/${index}/keep`);
            page = d.page;
            renderNotes(filterKind(d.notes || []));
            renderConversation();
            loadPages();
            toast(`Kept as a proposed finding. Confirm it from the project’s review when you are sure.`, null, null, 5000);
        } catch (e) {
            toastFail(`Not kept (${e.message || 'unknown error'})`);
        }
    }

    async function ask() {
        const input = $('nb-ask-input'), go = $('nb-ask-go'), note = $('nb-note-go');
        const q = (input && input.value || '').trim();
        if (!q || !page) return;
        const host = $('nb-conv');
        if (go) go.disabled = true;
        if (note) note.disabled = true;
        // The question is on the page as soon as it is asked.
        const pending = document.createElement('div');
        pending.className = 'nb-turn is-person is-pending';
        pending.innerHTML = `<div class="nb-turn-text">${esc(q)}</div><div class="nb-ask-loading">Reading ${esc(page.name)}…</div>`;
        if (host) { if (host.querySelector('.nb-conv-empty')) host.innerHTML = ''; host.appendChild(pending); host.scrollTop = host.scrollHeight; }
        try {
            const d = await send(`/api/notebook/pages/${encodeURIComponent(pageId)}/ask`, { question: q });
            if (d.ok === false) throw new Error(d.error || 'the notebook could not answer');
            input.value = '';
            page = d.page;
            renderConversation();
            loadPages();
        } catch (e) {
            pending.remove();
            toastFail(`Asking failed: ${e && e.message ? e.message : 'unknown error'}`);
        } finally {
            if (go) go.disabled = false;
            if (note) note.disabled = false;
        }
    }

    async function noteIt() {
        const input = $('nb-ask-input');
        const text = (input && input.value || '').trim();
        if (!text || !page) return;
        try {
            const d = await send(`/api/notebook/pages/${encodeURIComponent(pageId)}/notes`, { body: text });
            input.value = '';
            page = d.page;
            if (shelf === 'notebook') renderNotes(filterKind(d.notes || []));
            loadPages();
            const on = d.note && d.note.id && (d.notes || []).some(n => n.id === d.note.id);
            toast(on ? 'Filed on this page.' : 'Filed to the notebook (it does not match this page’s query).', null, null, 4000);
        } catch (e) {
            toastFail(`Not filed (${e.message || 'unknown error'})`);
        }
    }

    // ── Wiring ─────────────────────────────────────────────────────────
    function setupFilters() {
        document.querySelectorAll('#notebook-content [data-nb-kind]').forEach(btn => {
            btn.addEventListener('click', () => {
                kindFilter = btn.dataset.nbKind;
                document.querySelectorAll('#notebook-content [data-nb-kind]')
                    .forEach(b => b.classList.toggle('active', b === btn));
                loadPage();
            });
        });
        // The shelf is a second axis: what the agent reads, or what was set aside.
        document.querySelectorAll('#notebook-content [data-nb-shelf]').forEach(btn => {
            btn.addEventListener('click', () => {
                shelf = btn.dataset.nbShelf;
                document.querySelectorAll('#notebook-content [data-nb-shelf]')
                    .forEach(b => b.classList.toggle('active', b === btn));
                loadPage();
            });
        });
    }

    function setupAsk() {
        const go = $('nb-ask-go'), note = $('nb-note-go'), input = $('nb-ask-input');
        if (go) go.addEventListener('click', ask);
        if (note) note.addEventListener('click', noteIt);
        if (input) input.addEventListener('keydown', e => {
            if (e.key !== 'Enter') return;
            e.preventDefault();
            if (e.shiftKey) noteIt(); else ask();
        });
        if (note && !hasControl()) note.hidden = true;
    }

    async function refresh() {
        if (sessionNames === null) await loadSessionNames();
        if (typeof Who !== 'undefined') await Who.load();
        loadPages();
        loadPage();
    }

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

    return { init, openPage };
})();
