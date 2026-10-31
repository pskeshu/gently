/**
 * Sessions — what a saved session holds, before it is restored.
 *
 * The list says how much each session has (embryos, timepoints, brightfield frames,
 * whether its run was cut short). Selecting one shows the plan it ran with,
 * the latest picture of every embryo, the brightfield overview frames, the stage
 * calls, and the conversation — all read from the session's folder, nothing
 * restored. Resuming is the one button that changes the live agent.
 */

function ageDays(iso) {
    const t = new Date(iso).getTime();
    return Number.isFinite(t) ? (Date.now() - t) / 86400000 : Infinity;
}

// Errors are said in a toast, not a modal that stops the page.
function reviewToast(msg) {
    if (typeof showGentlyToast === 'function') showGentlyToast(msg, null, null, 6000, 'error');
    else console.error(msg);
}

const ReviewApp = {
    sessions: [],
    currentSession: null,
    currentTab: 'embryos',

    _initialized: false,
    _loadRun: 0,            // the click a session detail answers; an older answer is dropped
    _stale: false,          // rows changed while the tab was hidden

    async init() {
        if (this._initialized) return;
        this._initialized = true;
        fetch('/api/tools').then(r => r.json()).then(d => { window.GentlyToolCatalog = d.tools || []; }).catch(() => {});
        await this.loadSessions();
        this.updateStatusbar();

        // Check for session query parameter to auto-load
        const params = new URLSearchParams(window.location.search);
        const sessionId = params.get('session');
        if (sessionId) {
            this.loadSession(sessionId);
        }
    },

    async loadSessions() {
        const list = document.getElementById('session-list');
        // Rows that are already drawn stay until the new ones replace them.
        if (!this.sessions.length) list.innerHTML = '<div class="loading">Loading sessions...</div>';

        try {
            const [resp] = await Promise.all([fetch('/api/sessions'), this.loadProjects()]);
            const data = await resp.json();
            this.sessions = data.sessions || [];
            this.sessions.forEach(s => {
                s._hay = [s.name, s.suggested_name, s.session_id, s.description, this.formatDate(s.created_at)]
                    .filter(Boolean).join(' ').toLowerCase();
            });
            this.renderSessionList();
            this.fillSizes();
        } catch (e) {
            if (!this.sessions.length) list.innerHTML = '<div class="error">Failed to load sessions</div>';
            console.error('Failed to load sessions:', e);
        }
        this.watchForFiling();
    },

    /** Is the Sessions tab on screen? Work for a hidden tab waits for it. */
    showing() {
        const panel = document.getElementById('sessions-content');
        return !!(panel && panel.classList.contains('active'));
    },

    // ---- sizes come after the list. A walk over a session's files is the one
    // slow read, and a rig has hundreds of sessions: "i went to sessions tab -
    // it took forever to load". The list carries the sizes the server has
    // cached; the rest are asked for in batches, newest first, and each batch
    // repaints only the rows it filled, so nothing scrolls or flickers. ----
    _sizesRun: 0,

    async fillSizes() {
        const run = ++this._sizesRun;
        const todo = this.sessions.filter(s => s.bytes == null).map(s => s.session_id);
        const BATCH = 24;
        for (let i = 0; i < todo.length; i += BATCH) {
            if (run !== this._sizesRun) return;   // a newer load took over
            const ids = todo.slice(i, i + BATCH);
            let sizes = {};
            try {
                const r = await fetch(`/api/sessions/sizes?ids=${encodeURIComponent(ids.join(','))}`);
                sizes = (await r.json()).sizes || {};
            } catch (e) {
                console.warn('Session sizes:', e);
                return;
            }
            if (run !== this._sizesRun) return;
            const filled = this.sessions.filter(s => s.bytes == null && sizes[s.session_id] != null);
            filled.forEach(s => { s.bytes = sizes[s.session_id]; });
            this.repaintHoldings(filled);
        }
    },

    /** What a row says it holds, repainted in place for these sessions only. */
    repaintHoldings(sessions) {
        const list = document.getElementById('session-list');
        if (!list) return;
        sessions.forEach(s => {
            const meta = list.querySelector(`.session-item[data-session-id="${CSS.escape(s.session_id)}"] .session-meta`);
            if (!meta) return;
            meta.innerHTML = `<span>${this.formatDate(s.created_at)}</span>`
                + this.holdings(s).map(h => `<span class="dot"></span><span>${h}</span>`).join('')
                + this.runBadge(this.runOf(s));
        });
    },

    // ---- projects: a session has one home, and may serve items elsewhere ----

    projects: [],          // root campaign trees from /api/campaigns
    _projectRoot: {},      // any campaign id (root or phase) -> its root id

    async loadProjects() {
        try {
            const data = await (await fetch('/api/campaigns')).json();
            this.projects = data.campaigns || [];
        } catch (_) { this.projects = []; }
        const roots = {};
        const walk = (node, rootId) => {
            const c = node.campaign || {};
            if (c.id) roots[c.id] = rootId;
            (node.children || []).forEach(ch => walk(ch, rootId));
        };
        this.projects.forEach(t => walk(t, (t.campaign || {}).id));
        this._projectRoot = roots;
    },

    projectName(id) {
        const t = this.projects.find(p => (p.campaign || {}).id === id);
        const c = t ? t.campaign : null;
        return c ? (c.shorthand || c.description || id) : (id ? id.slice(0, 8) : '');
    },

    rootOf(campaignId) { return this._projectRoot[campaignId] || campaignId; },

    // Every plan item in every project, for the "also serves" select.
    allPlanItems() {
        const out = [];
        const walk = (node, rootId) => {
            (node.items || []).forEach(it => out.push({ id: it.id, title: it.title, campaign_id: it.campaign_id || (node.campaign || {}).id, root: rootId }));
            (node.children || []).forEach(ch => walk(ch, rootId));
        };
        this.projects.forEach(t => walk(t, (t.campaign || {}).id));
        return out;
    },

    // The links a session carries, the home first. The list's row has them;
    // a session.yaml from before the list reads as one link to its project_id.
    linksOf(s) {
        if (!s) return [];
        const src = Array.isArray(s.projects) ? s : (this.sessions.find(x => x.session_id === s.session_id) || s);
        const links = (src.projects || []).filter(x => x && x.id);
        if (links.length) return links;
        return src.project_id ? [{ id: src.project_id, phase: null, role: null, by: null, at: null }] : [];
    },

    projectChip(s) {
        const links = this.linksOf(s);
        if (!links.length) {
            return `<span class="chip chip-unfiled">unfiled</span> <a href="#" class="session-file-link" onclick="event.preventDefault(); event.stopPropagation(); ReviewApp.openFiling('${this.escapeHtml(s.session_id)}')">file under…</a>`;
        }
        const chips = links.map((l, i) => `<span class="chip chip-proj" title="${i === 0 ? 'Its home project' : 'Also in this project'}${l.role ? ' · ' + this.escapeHtml(l.role) : ''}">${this.escapeHtml(this.projectName(l.id))}${l.role ? `<small>${this.escapeHtml(l.role)}</small>` : ''}</span>`).join(' ');
        const items = (s.plan_items || []).filter(it => this.rootOf(it.campaign_id) === links[0].id);
        const served = items.length
            ? `<span class="session-plan-served">plan: ${items.map(it => this.escapeHtml(it.title) + (it.origin === 'manual' ? ' · manual' : '')).join(', ')}</span>`
            : '';
        return `${chips}${served}`;
    },

    // Filing done elsewhere (the agent, the Projects tab) shows up here.
    _filingWatched: false,
    watchForFiling() {
        if (this._filingWatched || typeof ClientEventBus === 'undefined') return;
        this._filingWatched = true;
        let timer = null;
        const bump = () => {
            if (timer) clearTimeout(timer);
            timer = setTimeout(() => {
                timer = null;
                if (!this.showing()) { this._stale = true; return; }
                this.loadSessions().catch(() => {});
            }, 500);
        };
        // What changes a row: a filing or a plan link (PLAN_UPDATED), a session
        // starting, restored or ending. A note or a tactic changes no row, so
        // CONTEXT_UPDATED no longer reloads the list.
        ['PLAN_UPDATED', 'SESSION_STARTED', 'SESSION_RESTORED', 'ACQUISITION_COMPLETED', 'ACQUISITION_STOPPED']
            .forEach(ev => ClientEventBus.on(ev, bump));
        ClientEventBus.on('TAB_CHANGED', tab => {
            if (tab === 'sessions' && this._stale) { this._stale = false; this.loadSessions().catch(() => {}); }
        });
        // A line written elsewhere on the open session shows up in its stream.
        ClientEventBus.on('CONTEXT_UPDATED', d => {
            if ((!d || !d.kind || d.kind === 'note') && this.currentSession && this.showing()) this.mountStream(this.currentSession.session_id, true);
        });
    },

    // ---- filing: a session informs several projects, in a role per link; the
    // first link is its home. Adding one is here, so is taking one out, and so
    // is making another the home. ----

    _filing: null,   // {sid, where: 'list'|'detail', choice, role, also, newName, error}

    openFiling(sid, where) {
        this._filing = { sid, where: where || 'list', choice: '', role: '', also: '', newName: '', error: '' };
        this._paintFiling();
    },

    closeFiling() {
        const sid = this._filing && this._filing.sid;
        this._filing = null;
        if (sid) this.repaintRow(sid);
        if (sid && this.currentSession && this.currentSession.session_id === sid) this.repaintFiled();
    },

    // Redraw where the popover lives, that row and the filed line, and keep
    // the cursor in its text box. The rest of the page does not move.
    _paintFiling() {
        const f = this._filing;
        if (f) this.repaintRow(f.sid);
        if (f && this.currentSession && this.currentSession.session_id === f.sid) this.repaintFiled();
        const box = document.querySelector('.filing-pop input[type="text"]');
        if (box) { box.focus(); box.setSelectionRange(box.value.length, box.value.length); }
    },

    repaintRow(sid) {
        const el = document.querySelector(`.session-item[data-session-id="${CSS.escape(sid)}"]`);
        const s = this.sessions.find(x => x.session_id === sid);
        if (!el || !s) { this.renderSessionList(); return; }
        el.outerHTML = this.rowHtml(s);
    },

    repaintFiled() {
        const host = document.getElementById('session-filed');
        if (host && this.currentSession) host.innerHTML = this.renderFiled(this.currentSession);
    },

    filingChoice(value) { if (this._filing) { this._filing.choice = value; this._paintFiling(); } },
    filingRole(value) { if (this._filing) this._filing.role = value; },
    filingAlso(value) { if (this._filing) this._filing.also = value; },
    filingNewName(value) { if (this._filing) this._filing.newName = value; },

    renderFilingPopover(s) {
        const f = this._filing;
        if (!f) return '';
        const esc = v => this.escapeHtml(v);
        const links = this.linksOf(s);
        const home = links.length ? links[0].id : '';
        const already = new Set(links.map(l => l.id));
        const opts = this.projects.filter(t => (t.campaign || {}).id && !already.has(t.campaign.id)).map(t => {
            const c = t.campaign, p = t.project || {};
            return `<option value="${esc(c.id)}" ${f.choice === c.id ? 'selected' : ''}>${esc(c.shorthand || c.description || c.id)} · ${p.sessions || 0} session${p.sessions === 1 ? '' : 's'}</option>`;
        }).join('');
        const newOn = f.choice === '__new__';
        const servedAlready = new Set((s.plan_items || []).map(it => it.id));
        const itemOpts = this.allPlanItems().filter(it => !servedAlready.has(it.id)).map(it =>
            `<option value="${esc(it.root)}|${esc(it.campaign_id)}|${esc(it.id)}" ${f.also === `${it.root}|${it.campaign_id}|${it.id}` ? 'selected' : ''}>${esc(this.projectName(it.root))} › ${esc(it.title)}</option>`).join('');
        return `<div class="filing-pop" onclick="event.stopPropagation()">
            <div class="filing-title">${home ? 'Add to a project' : 'File under a project'}</div>
            <select onchange="ReviewApp.filingChoice(this.value)">
                <option value="">a project…</option>${opts}
                <option value="__new__" ${newOn ? 'selected' : ''}>New project…</option>
            </select>
            ${newOn ? `<input type="text" placeholder="What the project asks, in one line" value="${esc(f.newName)}" oninput="ReviewApp.filingNewName(this.value)">` : ''}
            <div class="filing-also"><span>As</span>
                <select onchange="ReviewApp.filingRole(this.value)">${['', 'control', 'test', 'reference', 'pilot'].map(r => `<option value="${r}" ${f.role === r ? 'selected' : ''}>${r || '— no role —'}</option>`).join('')}</select></div>
            ${itemOpts ? `<div class="filing-also"><span>Serves a plan item?</span>
                <select onchange="ReviewApp.filingAlso(this.value)"><option value="">— no —</option>${itemOpts}</select></div>` : ''}
            ${home ? `<div class="filing-note">Its home stays <b>${esc(this.projectName(home))}</b>.</div>` : ''}
            ${f.error ? `<div class="filing-err">${esc(f.error)}</div>` : ''}
            <div class="filing-btns">
                <button type="button" class="go" ${f.busy ? 'disabled' : ''} onclick="ReviewApp.submitFiling()">${f.busy ? 'Filing…' : (home ? 'Add' : 'File')}</button>
                <button type="button" onclick="ReviewApp.closeFiling()">Cancel</button>
            </div>
        </div>`;
    },

    async _post(url, body) {
        const resp = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const d = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(d.detail || `HTTP ${resp.status}`);
        return d;
    },

    async submitFiling() {
        const f = this._filing;
        if (!f || f.busy) return;          // one press files once
        f.busy = true;
        this._paintFiling();
        try {
            let campaignId = f.choice;
            if (campaignId === '__new__') {
                const name = (f.newName || '').trim();
                if (!name) { f.error = 'Give the new project a line of description.'; this._paintFiling(); return; }
                const made = await this._post('/api/campaigns', { description: name });
                campaignId = made.campaign.id;
            }
            if (!campaignId && !f.also) { f.error = 'Pick a project.'; this._paintFiling(); return; }
            if (campaignId) {
                // No move: the first link is the home, a later one is added beside it.
                await this._post(`/api/sessions/${encodeURIComponent(f.sid)}/project`, { campaign_id: campaignId, role: f.role || null });
            }
            if (f.also) {
                const [, itemCampaign, itemId] = f.also.split('|');
                await this._post(`/api/campaigns/${encodeURIComponent(itemCampaign)}/items/${encodeURIComponent(itemId)}/sessions`, { session_id: f.sid });
            }
            this._filing = null;
            await this._refiled(f.sid);
        } catch (e) {
            f.busy = false;
            f.error = `Could not file: ${e.message || e}`;
            this._paintFiling();
        }
    },

    // After a link changes: the rows again from a fresh list, drawn in place,
    // and the open session's filed line. No blanking, no tab reset.
    async _refiled(sid) {
        await this.loadSessions();
        if (this.currentSession && this.currentSession.session_id === sid) {
            const row = this.sessions.find(x => x.session_id === sid);
            if (row) Object.assign(this.currentSession, { projects: row.projects, project_id: row.project_id, plan_items: row.plan_items });
            this.repaintFiled();
        }
    },

    // Take the session out of one project. Its other links stay, and if that
    // one was the home the next link is.
    async unlinkProject(sid, pid) {
        try {
            await this._post(`/api/sessions/${encodeURIComponent(sid)}/project`, { campaign_id: null, from: pid });
            await this._refiled(sid);
        } catch (e) {
            reviewToast(`Could not take it out of the project: ${e.message || e}`);
        }
    },

    // Make one of its projects the home: what a reader that takes one id sees.
    async makeHome(sid, pid) {
        try {
            await this._post(`/api/sessions/${encodeURIComponent(sid)}/project`, { campaign_id: pid, move: true });
            await this._refiled(sid);
        } catch (e) {
            reviewToast(`Could not make it the home: ${e.message || e}`);
        }
    },

    // The detail header: its home and each project it also informs, each with
    // its role and who linked it; then any plan item served somewhere else.
    renderFiled(s) {
        const esc = v => this.escapeHtml(v);
        const links = this.linksOf(s);
        const home = links.length ? links[0].id : null;
        const items = s.plan_items || [];
        const own = items.filter(it => this.rootOf(it.campaign_id) === home);
        const elsewhere = items.filter(it => !links.some(l => l.id === this.rootOf(it.campaign_id)));
        const sid = esc(s.session_id);
        const openBtn = (label) => `<button class="session-edit-btn" onclick="ReviewApp.openFiling('${sid}', 'detail')">${label}</button>`;
        const name = (pid) => `<b><a href="#" onclick="event.preventDefault(); ReviewApp.goToProject('${esc(pid)}')">${esc(this.projectName(pid))}</a></b>`;
        const tail = (l, isHome) => [
            l.role ? `<span class="chip" title="Its role in this project">${esc(l.role)}</span>` : '',
            l.by ? `<span class="session-filed-by">by ${Who.chip(l.by, { when: l.at })}</span>` : '',
            isHome ? '' : `<button class="session-edit-btn" onclick="ReviewApp.makeHome('${sid}', '${esc(l.id)}')" title="Make this the session's home project">make home</button>`,
            `<button class="session-edit-btn" onclick="ReviewApp.unlinkProject('${sid}', '${esc(l.id)}')" title="Take the session out of this project${links.length > 1 ? '; its other links stay' : ''}">${links.length > 1 ? 'remove' : 'unfile'}</button>`,
        ].filter(Boolean).join(' ');
        const rows = [];
        if (home) {
            const ownText = own.length
                ? ` · plan item ${own.map(it => `<a href="#" onclick="event.preventDefault(); ReviewApp.goToProject('${esc(home)}')">${esc(it.title)}</a>${it.origin === 'manual' ? ' <span class="chip">manual</span>' : ''}`).join(', ')}`
                : '';
            rows.push(`<div class="session-filed">in ${name(home)}${ownText} ${tail(links[0], true)} ${openBtn('Add to a project…')}</div>`);
            links.slice(1).forEach(l => rows.push(`<div class="session-filed">also in ${name(l.id)} ${tail(l, false)}</div>`));
        } else {
            rows.push(`<div class="session-filed"><span class="chip chip-unfiled">unfiled</span> in no project yet ${openBtn('File under…')}</div>`);
        }
        if (elsewhere.length) {
            rows.push(`<div class="session-filed">also serves ${elsewhere.map(it => `<a href="#" onclick="event.preventDefault(); ReviewApp.goToProject('${esc(this.rootOf(it.campaign_id))}')">${esc(this.projectName(this.rootOf(it.campaign_id)))} › ${esc(it.title)}</a>`).join(', ')}</div>`);
        }
        const pop = this._filing && this._filing.sid === s.session_id && this._filing.where === 'detail' ? this.renderFilingPopover(s) : '';
        return rows.join('') + pop;
    },

    goToProject(id) {
        try { sessionStorage.setItem('gently-open-project', id); } catch (_) { /* fine */ }
        if (typeof switchTab === 'function') switchTab('plans');
    },

    async loadSession(sessionId) {
        const content = document.getElementById('session-content');
        const same = !!(this.currentSession && this.currentSession.session_id === sessionId);
        this.highlightActiveSession(sessionId);                 // the row answers the click now
        if (!same) content.innerHTML = '<div class="loading-content">Loading session...</div>';
        const run = ++this._loadRun;

        try {
            const resp = await fetch(`/api/sessions/${sessionId}`);
            if (!resp.ok) throw new Error('Session not found');
            const data = await resp.json();
            if (run !== this._loadRun) return;                  // a later click already answered
            this.currentSession = data;
            if (!same) this.currentTab = 'embryos';
            this.renderSessionContent();
            this.updateStatusbar();
        } catch (e) {
            if (run !== this._loadRun) return;
            content.innerHTML = '<div class="error">Failed to load session</div>';
            console.error('Failed to load session:', e);
        }
    },

    highlightActiveSession(sessionId) {
        document.querySelectorAll('.session-item').forEach(el => {
            el.classList.toggle('active', el.dataset.sessionId === sessionId);
        });
    },

    // ---- what a session holds, in words ----------------------------------

    // The run a session holds: an interrupted one (``run``, what the gate
    // offers to carry on) or else whatever its checkpoint last said.
    runOf(s) {
        if (s.run && s.run.status) return s.run;
        const last = s.last_run;
        if (!last || !last.status) return null;
        const ended = ['completed', 'complete', 'stopped', 'idle'].includes(String(last.status));
        return Object.assign({}, last, { status: ended || !last.embryos_going ? 'complete' : last.status });
    },

    runBadge(run) {
        if (!run || !run.status) return '';
        const map = {
            interrupted: ['is-interrupted', 'cut short'],
            complete: ['is-complete', 'complete'],
            running: ['is-interrupted', 'was running'],
            paused: ['is-interrupted', 'paused'],
        };
        const [cls, label] = map[run.status] || ['is-stopped', run.status];
        return `<span class="session-run-badge ${cls}">${this.escapeHtml(label)}</span>`;
    },

    holdings(s) {
        const parts = [];
        if (s.embryo_count) parts.push(`${s.embryo_count} embryo${s.embryo_count !== 1 ? 's' : ''}`);
        if (s.timepoints) parts.push(`${s.timepoints} timepoint${s.timepoints !== 1 ? 's' : ''}`);
        if (s.bf_frames) parts.push(`${s.bf_frames} brightfield`);
        if (s.brightfield_references) parts.push('dark/flat ✓');
        if (s.bytes) parts.push(this.fmtBytes(s.bytes));
        if ((s.operators || []).length) parts.push(s.operators.map(o => Who.name(o)).join(', '));
        return parts;
    },

    fmtInterval(sec) {
        const s = Number(sec);
        if (!Number.isFinite(s) || s <= 0) return '';
        if (s < 60) return `${s} s`;
        if (s < 3600) return `${Math.round(s / 60)} min`;
        const h = s / 3600;
        return `${Number.isInteger(h) ? h : h.toFixed(1)} h`;
    },

    stopText(stop) {
        if (!stop) return '';
        const kind = stop.kind || stop.condition_type || stop.type;
        const v = stop.value;
        switch (kind) {
            case 'manual': return 'until stopped';
            case 'timepoints': return `after ${v} timepoints`;
            case 'duration': return `after ${v} h`;
            case 'hatching': return 'at hatching';
            case 'comma': return 'at comma stage';
            case 'all_test_hatched': return 'when every test embryo has hatched';
            default: return stop.spec || (kind ? String(kind) : '');
        }
    },

    stageText(stage) {
        if (!stage) return '';
        return String(stage).replace(/_/g, ' ').replace(/^1 5 fold$/, '1.5-fold').replace(/(\d) fold$/, '$1-fold');
    },

    // ---- the list -----------------------------------------------------------

    // ---- filters ------------------------------------------------------------
    // Chips, in groups. Within a group a session passes if it matches ANY chip
    // that is on; across groups every group must pass; the search box narrows
    // further. Each chip shows how many of the searched sessions it would keep,
    // so a filter never hides sessions silently.

    FILTERS: {
        // What a session holds: anything at all (the usual view hides the
        // empties), and by channel, SPIM volumes or brightfield frames.
        content: {
            label: 'Content',
            chips: {
                nonempty: { label: 'non-empty', test: s => s.timepoints > 0 || s.bf_frames > 0 },
                empty: { label: 'empty', test: s => !(s.timepoints > 0) && !(s.bf_frames > 0) },
            },
            defaults: ['nonempty'],
        },
        channel: {
            label: 'Channel',
            chips: {
                spim: { label: 'SPIM', test: s => s.timepoints > 0 },
                brightfield: { label: 'brightfield', test: s => s.bf_frames > 0 },
            },
            defaults: [],
        },
        run: {
            label: 'Run',
            chips: {
                interrupted: { label: 'cut short', test: (s, app) => (app.runOf(s) || {}).status === 'interrupted' },
                complete: { label: 'complete', test: (s, app) => (app.runOf(s) || {}).status === 'complete' },
                never: { label: 'never ran', test: (s, app) => !app.runOf(s) },
            },
            defaults: ['interrupted', 'complete', 'never'],
        },
        when: {
            label: 'When',
            chips: {
                today: { label: 'today', test: s => ageDays(s.created_at) < 1 },
                week: { label: 'this week', test: s => ageDays(s.created_at) >= 1 && ageDays(s.created_at) < 7 },
                older: { label: 'older', test: s => ageDays(s.created_at) >= 7 },
            },
            defaults: ['today', 'week', 'older'],
        },
        has: {
            label: 'Has',
            any: true,   // these are extra requirements, not a partition: on = required
            chips: {
                diagnostics: { label: 'diagnostics recording', test: s => !!s.advanced_diagnostics },
                references: { label: 'dark/flat refs', test: s => s.brightfield_references > 0 },
            },
            defaults: [],
        },
        // The chips come from the sessions loaded: one per project present, plus unfiled.
        project: { label: 'Project', dynamic: true, chips: {}, defaults: [] },
    },

    SORTS: {
        newest: { label: 'newest', key: s => s.created_at || '', desc: true },
        timepoints: { label: 'most timepoints', key: s => Number(s.timepoints) || 0, desc: true },
        size: { label: 'largest', key: s => Number(s.bytes) || 0, desc: true },
    },

    // A group's chips: the static table, or the projects the loaded sessions belong to.
    chipsOf(group) {
        const def = this.FILTERS[group];
        return def.dynamic ? this.projectChips() : def.chips;
    },

    projectChips() {
        const chips = {};
        const seen = new Set();
        this.sessions.forEach(s => this.linksOf(s).forEach(l => seen.add(l.id)));
        [...seen].sort((a, b) => this.projectName(a).localeCompare(this.projectName(b))).forEach(pid => {
            chips[pid] = { label: this.projectName(pid), test: s => this.linksOf(s).some(l => l.id === pid) };
        });
        chips.unfiled = { label: 'unfiled', test: s => this.linksOf(s).length === 0 };
        return chips;
    },

    _filterState: null,

    filterState() {
        if (this._filterState) return this._filterState;
        let saved = null;
        try { saved = JSON.parse(localStorage.getItem('gently-session-filters') || 'null'); } catch (_) { saved = null; }
        const state = { on: {}, sort: 'newest', open: false };
        Object.entries(this.FILTERS).forEach(([g, def]) => {
            const want = saved && saved.on && Array.isArray(saved.on[g]) ? saved.on[g] : def.defaults;
            state.on[g] = def.dynamic ? want : want.filter(k => def.chips[k]);
        });
        if (saved && this.SORTS[saved.sort]) state.sort = saved.sort;
        if (saved && typeof saved.open === 'boolean') state.open = saved.open;
        this._filterState = state;
        return state;
    },

    saveFilterState() {
        try { localStorage.setItem('gently-session-filters', JSON.stringify(this._filterState)); } catch (_) { /* fine */ }
    },

    toggleChip(group, key) {
        const st = this.filterState();
        const on = new Set(st.on[group] || []);
        if (on.has(key)) on.delete(key); else on.add(key);
        st.on[group] = [...on];
        this.saveFilterState();
        this.renderSessionList();
    },

    setSort(key) {
        if (!this.SORTS[key]) return;
        this.filterState().sort = key;
        this.saveFilterState();
        this.renderSessionList();
    },

    resetFilters() {
        this._filterState = null;
        try { localStorage.removeItem('gently-session-filters'); } catch (_) { /* fine */ }
        this.renderSessionList();
    },

    // Every group's chips, built once per render; the project chips come from the rows.
    chipTable() {
        const t = {};
        Object.keys(this.FILTERS).forEach(g => { t[g] = this.chipsOf(g); });
        return t;
    },

    passesGroup(s, group, on, table) {
        const def = this.FILTERS[group];
        const chips = table ? table[group] : this.chipsOf(group);
        on = on.filter(k => chips[k]);                                         // a project that is gone filters nothing
        if (def.any) return on.every(k => chips[k].test(s, this));             // requirements
        if (!on.length) return true;                                           // nothing chosen = no filter
        return on.some(k => chips[k].test(s, this));                           // partition
    },

    searched() {
        const searchBox = document.getElementById('session-search');
        const needle = (searchBox ? searchBox.value : '').trim().toLowerCase();
        const out = needle
            ? this.sessions.filter(s => (s._hay || '').includes(needle))
            : this.sessions.slice();
        return { needle, sessions: out };
    },

    // The search box redraws a little after the last keystroke, not on each one.
    searchInput() {
        clearTimeout(this._searchTimer);
        this._searchTimer = setTimeout(() => this.renderSessionList(), 120);
    },

    // What the filters say, in a line: only what differs from the default.
    filterSummary(st) {
        const bits = [];
        Object.entries(this.FILTERS).forEach(([g, def]) => {
            const chips = this.chipsOf(g);
            const on = (st.on[g] || []).filter(k => chips[k]);
            const same = [...on].sort().join() === [...def.defaults].sort().join();
            if (same) return;
            if (def.any) { on.forEach(k => bits.push(chips[k].label)); return; }
            if (!on.length) bits.push(`any ${def.label.toLowerCase()}`);
            else bits.push(on.map(k => chips[k].label).join(' or '));
        });
        if (st.sort !== 'newest') bits.push(this.SORTS[st.sort].label + ' first');
        return bits;
    },

    toggleFilters() {
        const st = this.filterState();
        st.open = !st.open;
        this.saveFilterState();
        this.renderSessionList();
    },

    renderFilters(searchedSessions, table) {
        const host = document.getElementById('session-filters');
        if (!host) return;
        const st = this.filterState();
        const summary = this.filterSummary(st);
        const isDefault = summary.length === 0;
        table = table || this.chipTable();
        // The counts are only drawn when the panel is open, so they are only counted then.
        const groups = !st.open ? '' : Object.entries(this.FILTERS).map(([g, def]) => {
            // The count a chip shows: the searched sessions that pass every OTHER
            // group and this chip — what turning it on alone would keep.
            const others = s => Object.keys(this.FILTERS).filter(o => o !== g).every(o => this.passesGroup(s, o, st.on[o] || [], table));
            const chips = Object.entries(table[g]).map(([k, chip]) => {
                const n = searchedSessions.filter(s => others(s) && chip.test(s, this)).length;
                const on = (st.on[g] || []).includes(k);
                return `<button type="button" class="session-chip${on ? ' is-on' : ''}${n ? '' : ' is-empty'}" onclick="ReviewApp.toggleChip('${g}','${k}')" title="${def.any ? 'Require' : 'Show'} sessions: ${this.escapeHtml(chip.label)}">${this.escapeHtml(chip.label)}${n ? `<span class="n">${n}</span>` : ''}</button>`;
            }).join('');
            return `<div class="session-filter-label">${def.label}</div><div class="session-filter-chips">${chips}</div>`;
        }).join('');
        const sorts = Object.entries(this.SORTS).map(([k, so]) =>
            `<button type="button" class="session-chip${st.sort === k ? ' is-on' : ''}" onclick="ReviewApp.setSort('${k}')">${so.label}</button>`).join('');
        host.innerHTML = `
            <button type="button" class="session-filters-toggle${isDefault ? '' : ' is-set'}" onclick="ReviewApp.toggleFilters()" aria-expanded="${st.open ? 'true' : 'false'}">
                <span class="session-filters-caret">${st.open ? '▾' : '▸'}</span>
                <span class="session-filters-word">Filters</span>
                <span class="session-filters-summary">${isDefault ? 'non-empty · newest' : this.escapeHtml(summary.join(' · '))}</span>
            </button>
            ${st.open ? `<div class="session-filters-grid">${groups}<div class="session-filter-label">Sort</div><div class="session-filter-chips">${sorts}</div></div>
            ${isDefault ? '' : '<button type="button" class="session-filters-reset" onclick="ReviewApp.resetFilters()">Back to the usual view</button>'}` : ''}`;
    },

    applyFilters(sessions, table) {
        const st = this.filterState();
        table = table || this.chipTable();
        const kept = sessions.filter(s => Object.keys(this.FILTERS).every(g => this.passesGroup(s, g, st.on[g] || [], table)));
        const so = this.SORTS[st.sort] || this.SORTS.newest;
        kept.sort((a, b) => {
            const ka = so.key(a), kb = so.key(b);
            const c = ka < kb ? -1 : ka > kb ? 1 : 0;
            return so.desc ? -c : c;
        });
        return kept;
    },

    renderSessionList() {
        const list = document.getElementById('session-list');
        const { needle, sessions: searchedSessions } = this.searched();
        const table = this.chipTable();
        this.renderFilters(searchedSessions, table);
        const filtered = this.applyFilters(searchedSessions, table);
        // Whatever scrolls the list keeps its place across a redraw.
        const scroller = [list, list.parentElement, list.parentElement && list.parentElement.parentElement]
            .find(el => el && el.scrollHeight > el.clientHeight + 1);
        const top = scroller ? scroller.scrollTop : 0;

        if (this.sessions.length === 0) {
            list.innerHTML = '<div class="no-sessions">No sessions found</div>';
            return;
        }

        if (filtered.length === 0) {
            const hidden = searchedSessions.length;
            list.innerHTML = needle && !hidden
                ? `<div class="no-sessions">Nothing matches “${this.escapeHtml(needle)}”</div>`
                : `<div class="no-sessions">Nothing passes these filters<br><small>${hidden} session${hidden !== 1 ? 's' : ''} hidden · <a href="#" onclick="event.preventDefault(); ReviewApp.resetFilters()">reset</a></small></div>`;
            return;
        }

        list.innerHTML = filtered.map(s => this.rowHtml(s)).join('');
        if (scroller) scroller.scrollTop = top;
    },

    rowHtml(s) {
            const held = this.holdings(s);
            const open = this.currentSession && this.currentSession.session_id === s.session_id;
            return `
            <div class="session-item ${s.active ? 'active-session' : ''}${open ? ' active' : ''}" data-session-id="${s.session_id}" onclick="ReviewApp.loadSession('${s.session_id}')">
                <div class="session-name${s.name ? '' : ' is-suggested'}" title="${s.name ? '' : 'Not named yet — this is read off the session itself'}">${this.escapeHtml(s.name || s.suggested_name || s.session_id)}${s.active ? ' <span class="session-active-badge">active</span>' : ''}${s.advanced_diagnostics ? ' <span class="session-active-badge session-diag-badge" title="Recorded with Advanced diagnostics on">advanced diagnostics</span>' : ''}</div>
                <div class="session-meta">
                    <span>${this.formatDate(s.created_at)}</span>
                    ${held.map(h => `<span class="dot"></span><span>${h}</span>`).join('')}
                    ${this.runBadge(this.runOf(s))}
                </div>
                <div class="session-project">${this.projectChip(s)}</div>
                ${this._filing && this._filing.sid === s.session_id && this._filing.where === 'list' ? this.renderFilingPopover(s) : ''}
                ${s.description ? `<div class="session-desc">${this.escapeHtml(s.description)}</div>` : ''}
                <div class="session-row-actions">
                    ${s.active ? '' : `<button class="session-resume-btn" onclick="event.stopPropagation(); ReviewApp.resumeSession('${s.session_id}')" title="New images, chat and stage targets will belong to this session">Make live</button>`}
                    ${typeof Reveal !== 'undefined' ? Reveal.button(
                        { what: 'session', session_id: s.session_id }, 'show',
                        { label: 'Folder', title: 'Open this session’s folder', cls: 'session-folder-btn' }) : ''}
                </div>
            </div>`;
    },

    // Two intents, said apart. Looking at a session is this tab: nothing
    // changes. Making one live is the button, and it says what that means.
    async resumeSession(sessionId) {
        const s = this.sessions.find(x => x.session_id === sessionId) || {};
        const what = [
            `Make “${s.name || s.suggested_name || sessionId}” the live session?`,
            '',
            'From then on, new images, the chat and the stage targets belong to it.',
            s.embryo_count ? `Its ${s.embryo_count} embryo position${s.embryo_count !== 1 ? 's' : ''} come from the sample that was on the scope then — re-centre before trusting them.` : '',
            'The current session is saved first. To just look, you are already in the right place.',
        ].filter(Boolean).join('\n');
        if (!confirm(what)) return;
        await this._switchTo(`/api/sessions/${encodeURIComponent(sessionId)}/resume`, 'Could not make it live');
    },

    async newSession() {
        if (!confirm('Open a new, empty session?\n\nThe current session is saved. New images and the chat go to the new one.')) return;
        await this._switchTo('/api/sessions/new', 'Could not open a new session');
    },

    async _switchTo(url, failure) {
        try {
            const resp = await fetch(url, { method: 'POST' });
            if (resp.ok) {
                // Server broadcasts session_changed to reload all clients; we
                // navigate home as well so the operator lands on the session.
                window.location.href = '/';
            } else {
                const d = await resp.json().catch(() => ({}));
                reviewToast(failure + ': ' + (d.detail || ('HTTP ' + resp.status)));
            }
        } catch (e) {
            reviewToast(failure + ': ' + e);
        }
    },

    fmtBytes(n) {
        if (n == null || !Number.isFinite(Number(n))) return '';
        const units = ['B', 'kB', 'MB', 'GB', 'TB'];
        let v = Number(n), i = 0;
        while (v >= 1000 && i < units.length - 1) { v /= 1000; i++; }
        return `${i === 0 ? v : v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
    },

    // ---- one session, before restoring ------------------------------------

    renderSessionContent() {
        const content = document.getElementById('session-content');
        const s = this.currentSession;
        const embryos = s.embryos || [];
        const frames = s.bf_frames || [];

        content.innerHTML = `
            <div class="session-header">
                <div class="session-header-row">
                    <div class="session-title">
                        <h2 class="${s.name ? '' : 'is-suggested'}" title="${s.name ? '' : 'Not named yet — read off the session itself'}">${this.escapeHtml(s.name || s.suggested_name || s.session_id)}</h2>
                        ${s.name ? '' : '<span class="session-title-hint">not named yet</span>'}
                        <button class="session-edit-btn" onclick="ReviewApp.editName()" title="Name this session">${s.name ? 'Rename' : 'Name it'}</button>
                    </div>
                    ${s.active ? '<span class="session-active-badge">live now</span>'
                        : `<button class="session-resume-btn session-resume-main" onclick="ReviewApp.resumeSession('${s.session_id}')" title="New images, chat and stage targets will belong to this session">Make live</button>`}
                </div>
                <div id="session-name-form"></div>
                <div id="session-filed">${this.renderFiled(s)}</div>
                ${s.description ? `<p class="session-description">${this.escapeHtml(s.description)}</p>` : ''}
                <div class="session-stats">
                    <span>Created: ${this.formatDateTime(s.created_at)}</span>
                    ${s.last_active ? `<span>Last active: ${this.formatDateTime(s.last_active)}</span>` : ''}
                    ${(s.operators || []).length ? `<span>Worked by ${s.operators.map(o => Who.chip(o)).join(' ')}</span>` : ''}
                    ${s.edited && s.edited.by ? `<span>Last edit ${Who.chip(s.edited.by, { when: s.edited.at })} (${this.escapeHtml(s.edited.what || '')})</span>` : ''}
                    ${s.bytes != null ? `<span title="Everything in the session's folder">On disk: ${this.fmtBytes(s.bytes)}</span>` : ''}
                </div>
                ${this.renderRun(this.runOf(s))}
                <div class="session-export" id="session-export"></div>
                <div class="session-movies" id="session-movies"></div>
            </div>

            <div class="session-stream" id="session-stream"></div>

            ${this.renderPlan(s.acquisition)}

            <div class="session-tabs">
                <button class="tab ${this.currentTab === 'embryos' ? 'active' : ''}" data-tab="embryos">
                    Embryos
                    <span class="tab-count">${embryos.length}</span>
                </button>
                <button class="tab ${this.currentTab === 'conversation' ? 'active' : ''}" data-tab="conversation">
                    Conversation
                    <span class="tab-count">${(s.conversation || []).length}</span>
                </button>
                <button class="tab ${this.currentTab === 'tools' ? 'active' : ''}" data-tab="tools">
                    Tools
                    <span class="tab-count">${(s.tools || []).length}</span>
                </button>
            </div>

            <div class="session-tab-content" id="session-tab-content">
                ${this.renderCurrentTab()}
            </div>
        `;

        this.setupTabHandlers();
        this.wireScrub();
        this.pollExport(true);
        this.pollMovie(true);
        this.mountStream(s.session_id);
        void frames;
    },

    // ---- the stream: what people wrote on this session, and the box -------
    // The same lines the project's Overview shows for this session, written
    // here. @ a person in a line to ask them; it reaches their inbox. A tool
    // note (saved marks) is a line too. `quiet` is a redraw from an event:
    // it leaves a line someone is still typing alone.
    async mountStream(sid, quiet) {
        const host = document.getElementById('session-stream');
        if (!host || !sid) return;
        const typing = document.getElementById('session-stream-line');
        if (quiet && typing && document.activeElement === typing && typing.value) return;
        let notes = [];
        try {
            const [n, t] = await Promise.all([
                fetch(`/api/notebook/notes?session=${encodeURIComponent(sid)}`).then(r => r.json()),
                this._toolTitles ? Promise.resolve(null) : fetch('/api/tools').then(r => r.json()).catch(() => ({})),
            ]);
            notes = n.notes || [];
            if (t) this._toolTitles = new Set((t.tools || []).map(x => x.title));
        } catch (e) { /* the box still works */ }
        if (!this.currentSession || this.currentSession.session_id !== sid) return;
        notes.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
        host.innerHTML = `<div class="pj-card session-stream-card">
            <div class="pj-card-head"><h3>Stream</h3><span class="pj-muted">what you noticed here, in your words; @ a person to ask them</span></div>
            <div class="rd-add"><input id="session-stream-line" placeholder="One line, in your words. Enter files it to this session." autocomplete="off"></div>
            <div class="pj-stack rd-stream">${notes.length ? notes.map(n => streamLineHtml(n, this._toolTitles, { cite: false })).join('') : '<div class="pj-none">Nothing written on this session yet.</div>'}</div>
        </div>`;
        this.landOnNote();
        const box = document.getElementById('session-stream-line');
        Mentions.attach(box);
        box.onkeydown = async e => {
            if (e.key !== 'Enter' || !box.value.trim() || box.disabled) return;
            const text = box.value.trim();
            box.disabled = true;                                   // one press, one line
            try { await this._post('/api/notebook/notes', { body: text, sessions: [sid] }); }
            catch (err) { box.disabled = false; reviewToast(`Could not file the line: ${err.message || err}`); return; }
            this.mountStream(sid);
        };
    },

    // A link in from the Notebook lands on the line itself, not just the
    // session. openNote(sid, noteId) opens the session (the Sessions tab is
    // inited if it is not yet) and remembers the line; the stream, once
    // drawn, scrolls to it and lights it for a moment. Once — the memory is
    // cleared so a redraw does not jump again. The same thing reached by URL
    // is /#sessions:<sid>:<noteId>, which app.js routes here.
    _landOn: null,
    async openNote(sessionId, noteId) {
        this._landOn = noteId || null;
        await this.init();
        await this.loadSession(sessionId);
    },

    landOnNote() {
        const id = this._landOn;
        if (!id) return;
        const line = document.querySelector(`.rd-line[data-note-id="${CSS.escape(id)}"]`);
        if (!line) return;
        this._landOn = null;
        line.classList.add('is-target');
        line.scrollIntoView({ block: 'center', behavior: 'smooth' });
        setTimeout(() => line.classList.remove('is-target'), 4000);
    },

    // ---- export, for Fiji and for finding the experiment again -----------

    async pollExport(once) {
        const s = this.currentSession;
        const host = document.getElementById('session-export');
        if (!s || !host) return;
        let job = null;
        try {
            const r = await fetch(`/api/sessions/${encodeURIComponent(s.session_id)}/export`);
            job = r.ok ? await r.json() : null;
        } catch (_) { /* leave the button */ }
        if (!job || job.session_id !== s.session_id) return;
        this._exportDefault = job.default_dest || '';
        this._exportPath = job.state === 'done' && job.path ? job.path : (this._exportPath || '');
        this.renderExport(job);
        if (job.state === 'running' && !once) {
            clearTimeout(this._exportTimer);
            this._exportTimer = setTimeout(() => this.pollExport(false), 1000);
        } else if (job.state === 'running' && once) {
            clearTimeout(this._exportTimer);
            this._exportTimer = setTimeout(() => this.pollExport(false), 1000);
        }
    },

    renderExport(job) {
        const host = document.getElementById('session-export');
        if (!host) return;
        const s = this.currentSession;
        const open = typeof Reveal !== 'undefined'
            ? Reveal.button({ what: 'export', session_id: s.session_id }, 'show', { label: 'Open export folder', title: 'Show the export in the file manager', cls: 'session-edit-btn' })
            : '';
        if (job.state === 'running') {
            const pct = job.total ? Math.round((job.done / job.total) * 100) : 0;
            host.innerHTML = `<span class="session-export-status">Exporting… ${job.done}${job.total ? ` of ${job.total}` : ''} files${job.current ? ` · ${this.escapeHtml(job.current)}` : ''}</span>
                <div class="dose-bar session-export-bar"><span style="width:${pct}%"></span></div>`;
            return;
        }
        const last = job.state === 'done'
            ? `<span class="session-export-status">Exported ${job.done} files to <code>${this.escapeHtml(job.path)}</code></span> ${open}`
            : job.state === 'error'
                ? `<span class="session-export-status is-error">Export failed: ${this.escapeHtml(job.error || '')}</span>`
                : '';
        host.innerHTML = `
            <div class="session-export-row">
                <span class="session-export-label">Export</span>
                <button class="session-edit-btn" onclick="ReviewApp.askExport()" title="One folder per embryo, files named in time order, the record beside them; opens in Fiji with Import › Image Sequence">Export for Fiji…</button>
                ${last}
            </div>
            <div id="session-export-form"></div>`;
    },

    askExport() {
        const host = document.getElementById('session-export-form');
        if (!host) return;
        host.innerHTML = `
            <form class="session-name-form" onsubmit="event.preventDefault(); ReviewApp.startExport()">
                <div class="session-export-dest-row">
                    <input id="session-export-dest" placeholder="Destination folder (leave empty for ${this.escapeHtml(this._exportDefault || 'the data folder\'s exports/')})" autocomplete="off">
                    <button type="button" class="session-edit-btn reveal-local" id="session-export-browse" onclick="ReviewApp.browseExportDest()" title="Choose the folder in a dialog">Browse…</button>
                </div>
                <label class="session-movie-option"><input type="checkbox" id="session-export-crops" checked> Also cut each embryo out of the brightfield frames into a Hugging Face folder of its own (needs the Operate tab's embryo marking)</label>
                <div class="session-name-form-row">
                    <button type="submit" class="session-resume-btn">Export</button>
                    <button type="button" class="session-edit-btn" onclick="document.getElementById('session-export-form').innerHTML = ''">Cancel</button>
                    <span class="hint">Copies, not links. A folder per embryo, named by nickname; volumes and brightfield frames named so a sort is time order; plan, stage calls, events and temperature as CSV; a README pointing back at the originals.</span>
                </div>
            </form>`;
        const input = document.getElementById('session-export-dest');
        if (input) input.focus();
    },

    async browseExportDest() {
        const input = document.getElementById('session-export-dest');
        const btn = document.getElementById('session-export-browse');
        if (!input || typeof Reveal === 'undefined') return;
        if (btn) { btn.disabled = true; btn.textContent = 'Choosing…'; }
        try {
            const path = await Reveal.pickFolder({
                title: 'Export the session to…',
                initial: input.value.trim() || this._exportDefault || null,
            });
            if (path) { input.value = path; input.focus(); }
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = 'Browse…'; }
        }
    },

    async startExport() {
        const s = this.currentSession;
        const input = document.getElementById('session-export-dest');
        const dest = input && input.value.trim() ? input.value.trim() : null;
        const cropsBox = document.getElementById('session-export-crops');
        const body = { crops: !cropsBox || cropsBox.checked };
        if (dest) body.dest = dest;
        try {
            const r = await fetch(`/api/sessions/${encodeURIComponent(s.session_id)}/export`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
            });
            if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || ('HTTP ' + r.status)); }
            this.renderExport(await r.json());
            this.pollExport(false);
        } catch (e) {
            const host = document.getElementById('session-export');
            if (host) host.insertAdjacentHTML('beforeend', `<div class="session-export-status is-error">Export failed to start: ${this.escapeHtml(e.message)}</div>`);
        }
    },

    // ---- movies, from any folder: brightfield frames, or SPIM volumes ---------------
    // Two buttons with two behaviours. A brightfield movie is frames over time, with
    // the flat divided out where the dark and flat sit beside them. A SPIM
    // movie is volumes: max projections over time, or every slice, stack by
    // stack. Any folder: an export on a share, a session's own volumes.

    async pollMovie(once) {
        const host = document.getElementById('session-movies');
        if (!host) return;
        let job = null;
        try {
            const r = await fetch('/api/movies');
            job = r.ok ? await r.json() : null;
        } catch (_) { /* leave it */ }
        if (!job) return;
        this.renderMovies(job);
        clearTimeout(this._movieTimer);
        if (job.state === 'running') this._movieTimer = setTimeout(() => this.pollMovie(false), 1000);
    },

    renderMovies(job) {
        const host = document.getElementById('session-movies');
        if (!host) return;
        if (!document.getElementById('session-movie-status')) {
            host.innerHTML = `
                <div class="session-export-row">
                    <button class="session-edit-btn" id="session-movie-bf" onclick="ReviewApp.askMovie('bf')" title="A folder of brightfield frames (an export's bf/) as a movie: every frame in time order, and flat-fielded where the dark and flat are beside them">brightfield movie…</button>
                    <button class="session-edit-btn" id="session-movie-spim" onclick="ReviewApp.askMovie('spim')" title="A folder of volumes (an export's <embryo>/volumes) as a movie: max projections over time, or every slice stack by stack">SPIM movie…</button>
                    <button class="session-edit-btn" id="session-movie-crops" onclick="ReviewApp.askMovie('crops')" title="Cut each embryo out of a folder of brightfield frames (an export's bf/) into a Hugging Face folder of its own: raw and corrected crops, the references, one metadata table. Where the embryos are comes from this session's Operate marking">Embryo crops…</button>
                    <span id="session-movie-status"></span>
                </div>
                <div id="session-movie-form"></div>`;
        }
        const status = document.getElementById('session-movie-status');
        const running = job.state === 'running';
        for (const id of ['session-movie-bf', 'session-movie-spim', 'session-movie-crops']) {
            const b = document.getElementById(id);
            if (b) b.disabled = running;
        }
        if (running) {
            const pct = job.total ? Math.round((job.done / job.total) * 100) : 0;
            const doing = job.kind === 'crops' ? 'Cutting the embryos out' : `Making ${this.escapeHtml(job.current || 'the movie')}`;
            status.innerHTML = `<span class="session-export-status">${doing}… ${job.done}${job.total ? ` of ${job.total}` : ''}</span>
                <div class="dose-bar session-export-bar"><span style="width:${pct}%"></span></div>`;
            return;
        }
        const open = typeof Reveal !== 'undefined'
            ? Reveal.button({ what: 'movie' }, 'show', { label: 'Open folder', title: 'Show the movie in the file manager', cls: 'session-edit-btn' })
            : '';
        if (job.state === 'done') {
            status.innerHTML = `<span class="session-export-status">Wrote ${this.escapeHtml((job.outputs || []).join(', '))}${job.note ? ` (${this.escapeHtml(job.note)})` : ''} in <code>${this.escapeHtml(job.folder)}</code></span> ${open}`;
        } else if (job.state === 'error') {
            status.innerHTML = `<span class="session-export-status is-error">Movie failed: ${this.escapeHtml(job.error || '')}</span>`;
        } else {
            status.innerHTML = '';
        }
    },

    askMovie(kind) {
        const host = document.getElementById('session-movie-form');
        if (!host) return;
        const bf = kind === 'bf';
        const crops = kind === 'crops';
        const prefill = (bf || crops) && this._exportPath ? `${this._exportPath}\\bf` : '';
        const choice = bf
            ? `<label class="session-movie-option"><input type="checkbox" id="session-movie-corrected" checked> Also flat-fielded, with the dark and flat beside the frames (bf_corrected.avi)</label>`
            : crops
                ? ''
                : `<label class="session-movie-option"><input type="radio" name="session-movie-view" value="projection" checked> Max projection, one frame per timepoint (spim_projection.avi)</label>
               <label class="session-movie-option"><input type="radio" name="session-movie-view" value="slices"> Every slice, stack by stack (spim_slices.avi)</label>`;
        const makeLabel = bf ? 'Make brightfield movie' : crops ? 'Cut the embryos out' : 'Make SPIM movie';
        const hint = bf
            ? 'Every frame in time order, one brightness stretch for the run, frame and time in the corner. Written beside the frames; Fiji opens it with File › Import › AVI.'
            : crops
                ? 'One fixed box per embryo, from this session\'s Operate marking. Writes bf/embryos/<embryo>/ with raw/ and corrected/ crops, the dark and flat cropped the same, metadata.csv, two movies, and a README that says how to load or upload it to Hugging Face.'
                : 'One brightness stretch for the run, timepoint (and slice) in the corner. Written beside the volumes; Fiji opens it with File › Import › AVI.';
        host.innerHTML = `
            <form class="session-name-form" onsubmit="event.preventDefault(); ReviewApp.startMovie('${kind}')">
                <div class="session-export-dest-row">
                    <input id="session-movie-folder" value="${this.escapeHtml(prefill)}" placeholder="${bf || crops ? "Folder of brightfield frames: an export's bf/ folder" : "Folder of volumes: an export's <embryo>/volumes folder"}" autocomplete="off">
                    <button type="button" class="session-edit-btn reveal-local" id="session-movie-browse" onclick="ReviewApp.browseMovieFolder()" title="Choose the folder in a dialog">Browse…</button>
                </div>
                ${choice}
                <div class="session-name-form-row">
                    <button type="submit" class="session-resume-btn">${makeLabel}</button>
                    <button type="button" class="session-edit-btn" onclick="document.getElementById('session-movie-form').innerHTML = ''">Cancel</button>
                    <span class="hint">${hint}</span>
                </div>
            </form>`;
        const input = document.getElementById('session-movie-folder');
        if (input) input.focus();
    },

    async browseMovieFolder() {
        const input = document.getElementById('session-movie-folder');
        const btn = document.getElementById('session-movie-browse');
        if (!input || typeof Reveal === 'undefined') return;
        if (btn) { btn.disabled = true; btn.textContent = 'Choosing…'; }
        try {
            const path = await Reveal.pickFolder({ title: 'The folder to make a movie of', initial: input.value.trim() || this._exportPath || null });
            if (path) { input.value = path; input.focus(); }
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = 'Browse…'; }
        }
    },

    async startMovie(kind) {
        const input = document.getElementById('session-movie-folder');
        const folder = input ? input.value.trim() : '';
        const body = { kind, folder };
        if (kind === 'bf') {
            const c = document.getElementById('session-movie-corrected');
            body.corrected = !c || c.checked;
        } else if (kind === 'crops') {
            body.session_id = this.currentSession ? this.currentSession.session_id : null;
        } else {
            const v = document.querySelector('input[name="session-movie-view"]:checked');
            body.view = v ? v.value : 'projection';
        }
        const status = document.getElementById('session-movie-status');
        try {
            const r = await fetch('/api/movies', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
            });
            if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || ('HTTP ' + r.status)); }
            const form = document.getElementById('session-movie-form');
            if (form) form.innerHTML = '';
            this.renderMovies(await r.json());
            this.pollMovie(false);
        } catch (e) {
            if (status) status.innerHTML = `<span class="session-export-status is-error">Not started: ${this.escapeHtml(e.message)}</span>`;
        }
    },

    // ---- naming -------------------------------------------------------------

    editName(prefill) {
        const s = this.currentSession;
        const host = document.getElementById('session-name-form');
        if (!host) return;
        const name = prefill && prefill.name != null ? prefill.name : (s.name || s.suggested_name || '');
        const desc = prefill && prefill.description != null ? prefill.description : (s.description || '');
        host.innerHTML = `
            <form class="session-name-form" onsubmit="event.preventDefault(); ReviewApp.saveName()">
                <input id="session-name-input" maxlength="120" value="${this.escapeHtml(name)}" placeholder="Name" autocomplete="off">
                <textarea id="session-desc-input" maxlength="600" placeholder="One line about this session: what, why, how it went">${this.escapeHtml(desc)}</textarea>
                <div class="session-name-form-row">
                    <button type="submit" class="session-resume-btn">Save</button>
                    <button type="button" class="session-edit-btn" id="session-suggest-btn" onclick="ReviewApp.suggestName()" title="Written by the model from what the session holds, or read off its facts when the agent has no API key">Suggest</button>
                    <button type="button" class="session-edit-btn" onclick="document.getElementById('session-name-form').innerHTML = ''">Cancel</button>
                    <span class="hint" id="session-name-hint"></span>
                </div>
            </form>`;
        const input = document.getElementById('session-name-input');
        if (input) { input.focus(); input.select(); }
    },

    async suggestName() {
        const btn = document.getElementById('session-suggest-btn');
        const hint = document.getElementById('session-name-hint');
        if (btn) { btn.disabled = true; btn.textContent = 'Thinking…'; }
        try {
            const resp = await fetch(`/api/sessions/${this.currentSession.session_id}/suggest-name`, { method: 'POST' });
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            const d = await resp.json();
            const input = document.getElementById('session-name-input');
            const desc = document.getElementById('session-desc-input');
            if (input && d.name) input.value = d.name;
            if (desc && d.description) desc.value = d.description;
            if (hint) hint.textContent = d.source === 'model' ? 'Suggested by the model — edit freely, then save.' : 'Read off the session (the agent has no API key for a written one).';
        } catch (e) {
            if (hint) hint.textContent = 'Could not suggest a name: ' + e;
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = 'Suggest'; }
        }
    },

    async saveName() {
        const s = this.currentSession;
        const input = document.getElementById('session-name-input');
        const desc = document.getElementById('session-desc-input');
        const hint = document.getElementById('session-name-hint');
        const body = { name: input ? input.value : '', description: desc ? desc.value : '' };
        try {
            const resp = await fetch(`/api/sessions/${s.session_id}`, {
                method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
            });
            if (!resp.ok) { const d = await resp.json().catch(() => ({})); throw new Error(d.detail || ('HTTP ' + resp.status)); }
            const d = await resp.json();
            s.name = d.name; s.description = d.description;
            const row = this.sessions.find(x => x.session_id === s.session_id);
            if (row) { row.name = d.name; row.description = d.description; }
            this.renderSessionContent();
            this.renderSessionList();
            this.highlightActiveSession(s.session_id);
        } catch (e) {
            if (hint) hint.textContent = 'Not saved: ' + e.message;
        }
    },

    // ---- scrubbing through an embryo's timepoints ------------------------

    wireScrub() {
        document.querySelectorAll('.embryo-thumb[data-embryo]').forEach(el => {
            const eid = el.dataset.embryo;
            // The card's picture is the counts, drawn by the viewer at the size a
            // card has (gently.core.u16); the JPEG if they cannot be had.
            const host = el.querySelector('.embryo-pix');
            let view = null;
            if (host) {
                if (typeof U16View !== 'undefined') view = U16View.tile(host, host.dataset.u16, host.dataset.jpg, host.dataset.alt);
                else host.replaceWith(Object.assign(document.createElement('img'), { src: host.dataset.jpg, alt: host.dataset.alt, loading: 'lazy' }));
            }
            const e = (this.currentSession.embryos || []).find(x => x.embryo_id === eid);
            const tps = e && e.projection_timepoints;
            if (!tps || tps.length < 2) return;
            el.classList.add('is-scrubbing');
            const img = el.querySelector('img');
            const cap = el.querySelector('.embryo-thumb-cap');
            // One frame in flight at a time: a sweep across 300 timepoints asks
            // for the frames it can show, not every one it crossed.
            let want = null;
            const apply = t => {
                const url = `/api/sessions/${this.currentSession.session_id}/projection?embryo=${encodeURIComponent(eid)}&t=${t}`;
                img.dataset.loading = '1'; img.dataset.t = String(t); img.src = url; el.href = url;
            };
            const landed = () => {
                img.dataset.loading = '';
                if (want != null && String(want) !== img.dataset.t) { const t = want; want = null; apply(t); }
            };
            if (img) { img.addEventListener('load', landed); img.addEventListener('error', landed); }
            let shownT = null;
            const show = t => {
                if (cap) cap.textContent = `t${t} · ${tps.indexOf(t) + 1}/${tps.length}`;
                if (view && view.host.isConnected) {
                    // the viewer: a later load wins, and the range holds across timepoints
                    if (shownT === t) return;
                    shownT = t;
                    const keep = view.data ? [view.lo, view.hi, view.gamma] : null;
                    el.href = `/api/sessions/${this.currentSession.session_id}/projection?embryo=${encodeURIComponent(eid)}&t=${t}`;
                    view.load(host.dataset.u16.replace(/\/volumes\/\d+\//, `/volumes/${t}/`)).then(m => { if (m && keep) view.setRange(keep[0], keep[1], keep[2]); }).catch(() => {});
                    return;
                }
                if (!img || img.dataset.t === String(t)) return;
                if (img.dataset.loading) { want = t; return; }
                apply(t);
            };
            el.addEventListener('mousemove', ev => {
                const r = el.getBoundingClientRect();
                const i = Math.min(tps.length - 1, Math.max(0, Math.floor(((ev.clientX - r.left) / r.width) * tps.length)));
                show(tps[i]);
            });
            el.addEventListener('mouseleave', () => show(tps[tps.length - 1]));
        });
    },

    // A stage's WormBase life-stage id, from the session payload's stage_ontology.
    stageTerm(stage) {
        const map = (this.currentSession && this.currentSession.stage_ontology) || {};
        return map[stage] || map[String(stage).replace(/-/g, '')] || null;
    },

    // Consecutive calls of one stage, as one segment: a strip of 300 calls is a few spans.
    stageRuns(preds) {
        const runs = [];
        preds.forEach(p => {
            const r = runs[runs.length - 1];
            if (r && r.stage === p.stage) { r.n++; r.last = p; } else runs.push({ stage: p.stage, n: 1, first: p, last: p });
        });
        return runs;
    },

    stageBar(e) {
        const preds = e.predictions || [];
        if (!preds.length || typeof stageColor !== 'function') return '';
        const last = preds[preds.length - 1];
        const term = this.stageTerm(last.stage);
        const tip = p => `t${p.timepoint} · ${this.stageText(p.stage)}${this.stageTerm(p.stage) ? ` · ${this.stageTerm(p.stage)}` : ''}`;
        return `
            <div class="stage-bar" title="Stage call per timepoint">
                ${this.stageRuns(preds).map(r => `<span style="background:${stageColor(r.stage)};flex-grow:${r.n}" title="${this.escapeHtml(r.n === 1 ? tip(r.first) : `t${r.first.timepoint}–t${r.last.timepoint} · ${this.stageText(r.stage)}`)}"></span>`).join('')}
            </div>
            <div class="stage-bar-key"><span>t${preds[0].timepoint} ${this.escapeHtml(this.stageText(preds[0].stage))}</span><span>${this.escapeHtml(this.stageText(last.stage))} t${last.timepoint}</span></div>
            ${term ? `<div class="stage-ontology">${this.escapeHtml(this.stageText(last.stage))} · <a href="https://wormbase.org/species/all/life_stage/${encodeURIComponent(term)}" target="_blank" rel="noopener" title="WormBase life stage">${this.escapeHtml(term)}</a></div>` : ''}`;
    },

    dose(e) {
        if (e.dose_ms == null) return '';
        const used = Number(e.dose_ms); const budget = Number(e.dose_budget_ms);
        const fmt = ms => ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
        if (!Number.isFinite(budget) || budget <= 0) return `<div class="dose">Light dose ${fmt(used)}</div>`;
        const frac = used / budget;
        const cls = frac >= 1 ? ' is-over' : frac >= 0.75 ? ' is-high' : '';
        return `
            <div class="dose${cls}" title="Total exposure so far against the role's photodose budget">
                Light dose ${fmt(used)} of ${fmt(budget)} (${Math.round(frac * 100)}%)
                <div class="dose-bar"><span style="width:${Math.min(100, frac * 100).toFixed(0)}%"></span></div>
            </div>`;
    },

    renderRemoved() {
        const removed = this.currentSession.removed_embryos || [];
        if (!removed.length) return '';
        return `
            <div class="session-removed">Set aside: ${removed.map(r =>
                `<b>${this.escapeHtml(r.nickname || r.embryo_id)}</b>${r.reason ? ` (${this.escapeHtml(r.reason)})` : ''}${r.removed_at ? ` at ${this.formatDateTime(r.removed_at)}` : ''}`
            ).join(' · ')}</div>`;
    },



    renderRun(run) {
        if (!run || !run.status) return '';
        const bits = [];
        if (run.total_timepoints) bits.push(`${run.total_timepoints} timepoints`);
        if (run.rounds) bits.push(`${run.rounds} rounds`);
        if (run.interval_seconds) bits.push(`every ${this.fmtInterval(run.interval_seconds)}`);
        if (run.started_at) bits.push(`started ${this.formatDateTime(run.started_at)}`);
        if (run.status === 'interrupted' && run.embryos_going) bits.push(`${run.embryos_going} embryo${run.embryos_going !== 1 ? 's' : ''} still going`);
        return `<div class="session-run">${this.runBadge(run)} ${bits.map(b => this.escapeHtml(b)).join(' · ')}</div>`;
    },

    renderPlan(plan) {
        if (!plan) return '<div class="session-plan session-plan-empty">No acquisition plan recorded for this session.</div>';
        const rows = [];
        if (plan.interval_seconds) rows.push(['Interval', `every ${this.fmtInterval(plan.interval_seconds)}`]);
        if (plan.volumes === false) {
            rows.push(['Volumes', 'off (brightfield only)']);
        } else {
            const vol = [];
            if (plan.num_slices != null) vol.push(`${plan.num_slices} slices`);
            if (plan.exposure_ms != null) vol.push(`${plan.exposure_ms} ms`);
            if (vol.length) rows.push(['Volumes', vol.join(' · ')]);
            const lasers = [];
            if (plan.laser_config) lasers.push(String(plan.laser_config).replace(/_/g, ' '));
            Object.entries(plan.laser_powers || {}).forEach(([wl, pct]) => lasers.push(`${wl} nm ${pct}%`));
            if (lasers.length) rows.push(['Laser', lasers.join(' · ')]);
            if (plan.monitoring_mode && plan.monitoring_mode !== 'idle') rows.push(['Monitoring', plan.monitoring_mode]);
        }
        const stop = this.stopText(plan.stop_condition);
        if (stop) rows.push(['Stop', stop]);
        const bf = plan.bf;
        if (bf && bf.enabled) {
            const d = [];
            if (bf.every_seconds) d.push(`every ${this.fmtInterval(bf.every_seconds)}`);
            if (bf.light) d.push(bf.light === 'led' ? `LED${bf.led_intensity_pct != null ? ` ${bf.led_intensity_pct}%` : ''}` : bf.light === 'room' ? 'room light' : 'light as it is');
            if (bf.exposure_ms != null) d.push(`${bf.exposure_ms} ms`);
            rows.push(['Brightfield overview', d.join(' · ')]);
        } else {
            rows.push(['Brightfield overview', 'off']);
        }
        if (Array.isArray(plan.embryo_ids) && plan.embryo_ids.length) rows.push(['Embryos', plan.embryo_ids.join(', ')]);
        return `
            <div class="session-plan">
                <div class="session-plan-title">Acquisition plan</div>
                <dl class="session-plan-grid">
                    ${rows.map(([k, v]) => `<dt>${this.escapeHtml(k)}</dt><dd>${this.escapeHtml(v)}</dd>`).join('')}
                </dl>
            </div>`;
    },

    setupTabHandlers() {
        document.querySelectorAll('.session-tabs .tab').forEach(tab => {
            tab.addEventListener('click', () => {
                this.currentTab = tab.dataset.tab;
                document.querySelectorAll('.session-tabs .tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById('session-tab-content').innerHTML = this.renderCurrentTab();
                this.wireScrub();
                this.mountTools();
            });
        });
    },

    renderCurrentTab() {
        switch (this.currentTab) {
            case 'embryos': return this.renderEmbryosTab();
            case 'conversation': return this.renderConversationTab();
            case 'tools': return this.renderToolsTab();
            default: return '';
        }
    },

    // ---- tools: optional panes a session imports at any time ---------------
    // A tool is off until the session names it; importing is one PUT. Each
    // imported tool mounts itself into its own box (static/js/tools/*.js).
    renderToolsTab() {
        const s = this.currentSession;
        const on = s.tools || [];
        const avail = (window.GentlyToolCatalog || []).filter(t => (t.scope || 'session') === 'session' && !on.includes(t.name));
        return `
            <div class="session-tools-bar">
                ${avail.length ? `<select id="session-tool-pick">${avail.map(t => `<option value="${t.name}" title="${this.escapeHtml(t.summary)}">${this.escapeHtml(t.title)}</option>`).join('')}</select>
                <button class="session-edit-btn" onclick="ReviewApp.importTool()">Import</button>` : ''}
                ${!on.length ? '<span class="hint">No tools on this session yet.</span>' : ''}
            </div>
            ${on.map(name => {
                const t = (window.GentlyToolCatalog || []).find(x => x.name === name) || { title: name };
                return `<div class="session-tool">
                    <div class="session-bf-head">${this.escapeHtml(t.title)} <button class="session-edit-btn session-tool-drop" onclick="ReviewApp.dropTool('${name}')" title="Take this tool off the session; its file stays">remove</button></div>
                    <div class="session-tool-body" data-tool="${name}"></div>
                </div>`;
            }).join('')}`;
    },

    mountTools() {
        const s = this.currentSession;
        if (!s || this.currentTab !== 'tools') return;
        document.querySelectorAll('.session-tool-body[data-tool]').forEach(el => {
            const tool = (window.GentlyTools || {})[el.dataset.tool];
            if (tool) tool.mount(el, s.session_id);
            else el.innerHTML = '<div class="empty-tab">This tool has no pane on this page</div>';
        });
    },

    async setTools(tools) {
        const s = this.currentSession;
        const r = await fetch(`/api/sessions/${encodeURIComponent(s.session_id)}/tools`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tools }),
        });
        if (!r.ok) { const d = await r.json().catch(() => ({})); reviewToast(d.detail || `HTTP ${r.status}`); return; }
        s.tools = (await r.json()).tools;
        document.getElementById('session-tab-content').innerHTML = this.renderCurrentTab();
        this.mountTools();
    },

    importTool() {
        const pick = document.getElementById('session-tool-pick');
        if (pick) this.setTools([...(this.currentSession.tools || []), pick.value]);
    },

    dropTool(name) { this.setTools((this.currentSession.tools || []).filter(t => t !== name)); },

    renderBfStrip() {
        const frames = this.currentSession.bf_frames || [];
        if (!frames.length) return '';
        return `
            <div class="session-bf">
                <div class="session-bf-head">Brightfield overview <span class="session-bf-count">${frames.length} frame${frames.length !== 1 ? 's' : ''}</span></div>
                <div class="session-bf-frames">
                    ${frames.map(f => `
                        <a class="session-bf-frame" href="${f.url}" title="Open frame ${f.frame ?? ''} with its values" onclick="event.preventDefault(); U16View.overlay('${f.url.replace(/\.png$/, '.u16')}', 'frame ${f.frame ?? ''}${f.captured_at ? ' · ' + this.formatTime(f.captured_at) : ''}')">
                            <img src="${f.url}?max=256" alt="brightfield overview frame ${f.frame ?? ''}" loading="lazy">
                            <span class="session-bf-cap">${f.frame ?? ''}${f.captured_at ? ` · ${this.formatTime(f.captured_at)}` : ''}</span>
                        </a>`).join('')}
                </div>
            </div>`;
    },

    renderEmbryosTab() {
        const embryos = this.currentSession.embryos || [];
        const strip = this.renderBfStrip();

        if (embryos.length === 0) {
            return strip + '<div class="empty-tab">No embryos in this session.</div>';
        }

        return strip + `
            <div class="embryo-grid">
                ${embryos.map(e => {
                    const name = e.nickname || e.embryo_id;
                    const sub = e.nickname ? e.embryo_id : '';
                    const stage = this.stageText(e.stage);
                    const conf = e.stage_confidence != null ? ` ${Math.round(e.stage_confidence * 100)}%` : '';
                    return `
                    <div class="embryo-card">
                        ${e.thumbnail
                            ? `<a class="embryo-thumb" data-embryo="${this.escapeHtml(e.embryo_id)}" href="${e.thumbnail}" target="_blank" rel="noopener" title="Move across to scrub through the timepoints; click to open"><span class="embryo-pix" data-u16="/api/sessions/${encodeURIComponent(this.currentSession.session_id)}/embryos/${encodeURIComponent(e.embryo_id)}/volumes/${e.latest_timepoint}/projection.u16?max=256" data-jpg="${e.thumbnail}" data-alt="${this.escapeHtml(name)}, timepoint ${e.latest_timepoint}"></span><span class="embryo-thumb-cap">t${e.latest_timepoint}</span></a>`
                            : '<div class="embryo-thumb embryo-thumb-empty">no image</div>'}
                        <div class="embryo-header">
                            <h3>${this.escapeHtml(name)}${sub ? ` <small>${this.escapeHtml(sub)}</small>` : ''}</h3>
                            ${e.role ? `<span class="chip embryo-role" title="Its role in the run">${this.escapeHtml(e.role)}</span>` : ''}
                            ${e.is_complete ? '<span class="status-badge complete">Complete</span>' : ''}
                        </div>
                        <div class="embryo-details">
                            ${stage ? `<div class="embryo-stage">${this.escapeHtml(stage)}${conf ? `<span class="embryo-conf">${conf.trim()}</span>` : ''}</div>` : '<div class="embryo-stage is-none">no stage call yet</div>'}
                            <div class="embryo-meta">${e.timepoints || 0} timepoint${(e.timepoints || 0) === 1 ? '' : 's'}${e.strain ? ` · ${this.escapeHtml(e.strain)}` : ''}${e.position && e.position.x != null ? ` · <span class="embryo-pos" title="Stage position">${Number(e.position.x).toFixed(0)}, ${Number(e.position.y).toFixed(0)}</span>` : ''}</div>
                        </div>
                        ${this.stageBar(e)}
                        ${this.dose(e)}
                    </div>`;
                }).join('')}
            </div>
            ${this.renderRemoved()}
        `;
    },


    renderConversationTab() {
        const conversation = this.currentSession.conversation || [];

        if (conversation.length === 0) {
            return '<div class="empty-tab">No conversation in this session.</div>';
        }

        return `
            <div class="conversation-list">
                ${conversation.map(msg => {
                    const blocks = Array.isArray(msg.content) ? msg.content : null;
                    const onlyTools = blocks && blocks.length && blocks.every(b => b.type === 'tool_result' || b.type === 'tool_use');
                    const role = onlyTools ? 'tool' : msg.role;
                    const label = role === 'user' ? 'User' : role === 'assistant' ? 'Assistant' : role === 'tool' ? 'Tool' : 'System';
                    return `
                    <div class="message ${role}">
                        <div class="message-role">${label}</div>
                        <div class="message-content">${this.formatMessageContent(msg.content)}</div>
                        ${msg.timestamp ? `<div class="message-time">${this.formatDateTime(msg.timestamp)}</div>` : ''}
                    </div>`;
                }).join('')}
            </div>
        `;
    },

    formatMessageContent(content) {
        if (typeof content === 'string') {
            return this.escapeHtml(content).replace(/\n/g, '<br>');
        }
        if (Array.isArray(content)) {
            return content.map(block => {
                if (block.type === 'text') {
                    return `<p>${this.escapeHtml(block.text || '').replace(/\n/g, '<br>')}</p>`;
                }
                if (block.type === 'tool_use') {
                    const input = block.input ? JSON.stringify(block.input) : '';
                    return `<div class="msg-tool">⚙ ${this.escapeHtml(block.name || 'tool')}${input ? ` <code>${this.escapeHtml(input.length > 300 ? input.slice(0, 300) + '…' : input)}</code>` : ''}</div>`;
                }
                if (block.type === 'tool_result') {
                    const c = block.content;
                    const text = typeof c === 'string' ? c
                        : Array.isArray(c) ? c.map(x => x.text || (x.type === 'image' ? '[image]' : '')).join('\n')
                        : c == null ? '' : JSON.stringify(c);
                    return `<div class="msg-tool-result">↳ ${this.escapeHtml(text.length > 600 ? text.slice(0, 600) + '…' : text).replace(/\n/g, '<br>')}</div>`;
                }
                if (block.type === 'image') return '<span class="content-block">[image]</span>';
                return `<span class="content-block">[${this.escapeHtml(block.type || 'block')}]</span>`;
            }).join('');
        }
        return this.escapeHtml(JSON.stringify(content));
    },

    formatDate(isoString) { return formatDate(isoString); },

    formatDateTime(isoString) { return formatDate(isoString); },

    formatTime(isoString) {
        const d = new Date(isoString);
        return isNaN(d) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },

    escapeHtml(str) { return escapeHtml(str == null ? '' : String(str)); },

    updateStatusbar() {
        const left = document.getElementById('status-left');
        const right = document.getElementById('status-right');
        if (!left) return;
        const total = this.sessions.length;
        const withContent = this.sessions.filter(s => s.embryo_count > 0 || s.bf_frames > 0).length;
        left.textContent = `${total} session${total !== 1 ? 's' : ''} · ${withContent} with content`;
        if (right && this.currentSession) {
            const embryos = (this.currentSession.embryos || []).length;
            right.textContent = `${embryos} embryo${embryos !== 1 ? 's' : ''}`;
        } else if (right) {
            right.textContent = '';
        }
    }
};

// Auto-init on standalone review page (detected via data-page attribute)
if (document.body?.dataset.page === 'review') {
    ReviewApp.init();
} else {
    document.addEventListener('DOMContentLoaded', () => {
        if (document.body.dataset.page === 'review') ReviewApp.init();
    });
}
