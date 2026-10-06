/**
 * Sessions — what a saved session holds, before it is restored.
 *
 * The list says how much each session has (embryos, timepoints, DIC frames,
 * whether its run was cut short). Selecting one shows the plan it ran with,
 * the latest picture of every embryo, the DIC overview frames, the stage
 * calls, and the conversation — all read from the session's folder, nothing
 * restored. Resuming is the one button that changes the live agent.
 */

const ReviewApp = {
    sessions: [],
    currentSession: null,
    currentTab: 'embryos',

    _initialized: false,

    async init() {
        if (this._initialized) return;
        this._initialized = true;
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
        list.innerHTML = '<div class="loading">Loading sessions...</div>';

        try {
            const resp = await fetch('/api/sessions');
            const data = await resp.json();
            this.sessions = data.sessions || [];
            this.renderSessionList();
        } catch (e) {
            list.innerHTML = '<div class="error">Failed to load sessions</div>';
            console.error('Failed to load sessions:', e);
        }
    },

    async loadSession(sessionId) {
        const content = document.getElementById('session-content');
        content.innerHTML = '<div class="loading-content">Loading session...</div>';

        try {
            const resp = await fetch(`/api/sessions/${sessionId}`);
            if (!resp.ok) throw new Error('Session not found');
            this.currentSession = await resp.json();
            this.currentTab = 'embryos';
            this.renderSessionContent();
            this.highlightActiveSession(sessionId);
            this.updateStatusbar();
        } catch (e) {
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
        if (s.dic_frames) parts.push(`${s.dic_frames} DIC`);
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

    renderSessionList() {
        const list = document.getElementById('session-list');
        const filterCheckbox = document.getElementById('filter-with-content');
        const filterWithContent = filterCheckbox ? filterCheckbox.checked : true;
        const searchBox = document.getElementById('session-search');
        const needle = (searchBox ? searchBox.value : '').trim().toLowerCase();

        // Filter sessions based on checkbox, then on the search box
        let filtered = this.sessions;
        if (filterWithContent) {
            filtered = this.sessions.filter(s => s.embryo_count > 0);
        }
        if (needle) {
            filtered = filtered.filter(s => [s.name, s.suggested_name, s.session_id, s.description, this.formatDate(s.created_at)]
                .some(v => v && String(v).toLowerCase().includes(needle)));
        }

        if (this.sessions.length === 0) {
            list.innerHTML = '<div class="no-sessions">No sessions found</div>';
            return;
        }

        if (filtered.length === 0) {
            list.innerHTML = needle
                ? `<div class="no-sessions">Nothing matches “${this.escapeHtml(needle)}”</div>`
                : `<div class="no-sessions">No sessions with content<br><small>${this.sessions.length} empty session${this.sessions.length !== 1 ? 's' : ''} hidden</small></div>`;
            return;
        }

        list.innerHTML = filtered.map(s => {
            const held = this.holdings(s);
            return `
            <div class="session-item ${s.active ? 'active-session' : ''}" data-session-id="${s.session_id}" onclick="ReviewApp.loadSession('${s.session_id}')">
                <div class="session-name${s.name ? '' : ' is-suggested'}" title="${s.name ? '' : 'Not named yet — this is read off the session itself'}">${this.escapeHtml(s.name || s.suggested_name || s.session_id)}${s.active ? ' <span class="session-active-badge">active</span>' : ''}${s.advanced_diagnostics ? ' <span class="session-active-badge session-diag-badge" title="Recorded with Advanced diagnostics on">advanced diagnostics</span>' : ''}</div>
                <div class="session-meta">
                    <span>${this.formatDate(s.created_at)}</span>
                    ${held.map(h => `<span class="dot"></span><span>${h}</span>`).join('')}
                    ${this.runBadge(this.runOf(s))}
                </div>
                ${s.description ? `<div class="session-desc">${this.escapeHtml(s.description)}</div>` : ''}
                ${s.active ? '' : `<button class="session-resume-btn" onclick="event.stopPropagation(); ReviewApp.resumeSession('${s.session_id}')">Resume in agent</button>`}
                ${typeof Reveal !== 'undefined' ? Reveal.button(
                    { what: 'session', session_id: s.session_id }, 'show',
                    { label: 'Folder', title: 'Open this session’s folder', cls: 'session-folder-btn' }) : ''}
            </div>`;
        }).join('');
    },

    async resumeSession(sessionId) {
        if (!confirm('Switch the live agent to this session?\nThe current session is saved first.')) return;
        try {
            const resp = await fetch(`/api/sessions/${sessionId}/resume`, { method: 'POST' });
            if (resp.ok) {
                // Server broadcasts session_changed to reload all clients; we
                // navigate home as well so the operator lands on the new session.
                window.location.href = '/';
            } else {
                const d = await resp.json().catch(() => ({}));
                alert('Resume failed: ' + (d.detail || ('HTTP ' + resp.status)));
            }
        } catch (e) {
            alert('Resume failed: ' + e);
        }
    },

    // ---- one session, before restoring ------------------------------------

    renderSessionContent() {
        const content = document.getElementById('session-content');
        const s = this.currentSession;
        const embryos = s.embryos || [];
        const frames = s.dic_frames || [];
        const predictions = embryos.reduce((n, e) => n + ((e.predictions || []).length), 0);

        content.innerHTML = `
            <div class="session-header">
                <div class="session-header-row">
                    <div class="session-title">
                        <h2 class="${s.name ? '' : 'is-suggested'}" title="${s.name ? '' : 'Not named yet — read off the session itself'}">${this.escapeHtml(s.name || s.suggested_name || s.session_id)}</h2>
                        ${s.name ? '' : '<span class="session-title-hint">not named yet</span>'}
                        <button class="session-edit-btn" onclick="ReviewApp.editName()" title="Name this session">${s.name ? 'Rename' : 'Name it'}</button>
                    </div>
                    ${s.active ? '<span class="session-active-badge">live now</span>'
                        : `<button class="session-resume-btn session-resume-main" onclick="ReviewApp.resumeSession('${s.session_id}')">Resume in agent</button>`}
                </div>
                <div id="session-name-form"></div>
                ${s.description ? `<p class="session-description">${this.escapeHtml(s.description)}</p>` : ''}
                <div class="session-stats">
                    <span>Created: ${this.formatDateTime(s.created_at)}</span>
                    ${s.last_active ? `<span>Last active: ${this.formatDateTime(s.last_active)}</span>` : ''}
                </div>
                ${this.renderRun(this.runOf(s))}
            </div>

            ${this.renderPlan(s.acquisition)}

            <div class="session-tabs">
                <button class="tab ${this.currentTab === 'embryos' ? 'active' : ''}" data-tab="embryos">
                    Embryos
                    <span class="tab-count">${embryos.length}</span>
                </button>
                <button class="tab ${this.currentTab === 'detections' ? 'active' : ''}" data-tab="detections">
                    Stage calls
                    <span class="tab-count">${predictions}</span>
                </button>
                <button class="tab ${this.currentTab === 'conversation' ? 'active' : ''}" data-tab="conversation">
                    Conversation
                    <span class="tab-count">${(s.conversation || []).length}</span>
                </button>
                <button class="tab ${this.currentTab === 'events' ? 'active' : ''}" data-tab="events">
                    What happened
                    <span class="tab-count">${(s.events || []).length}</span>
                </button>
            </div>

            <div class="session-tab-content" id="session-tab-content">
                ${this.renderCurrentTab()}
            </div>
        `;

        this.setupTabHandlers();
        this.wireScrub();
        void frames;
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
            const e = (this.currentSession.embryos || []).find(x => x.embryo_id === eid);
            const tps = e && e.projection_timepoints;
            if (!tps || tps.length < 2) return;
            el.classList.add('is-scrubbing');
            const img = el.querySelector('img');
            const cap = el.querySelector('.embryo-thumb-cap');
            const show = t => {
                const url = `/api/sessions/${this.currentSession.session_id}/projection?embryo=${encodeURIComponent(eid)}&t=${t}`;
                if (img && img.dataset.t !== String(t)) { img.src = url; img.dataset.t = String(t); el.href = url; }
                if (cap) cap.textContent = `t${t} · ${tps.indexOf(t) + 1}/${tps.length}`;
            };
            el.addEventListener('mousemove', ev => {
                const r = el.getBoundingClientRect();
                const i = Math.min(tps.length - 1, Math.max(0, Math.floor(((ev.clientX - r.left) / r.width) * tps.length)));
                show(tps[i]);
            });
            el.addEventListener('mouseleave', () => show(tps[tps.length - 1]));
        });
    },

    stageBar(e) {
        const preds = e.predictions || [];
        if (!preds.length || typeof stageColor !== 'function') return '';
        return `
            <div class="stage-bar" title="Stage call per timepoint">
                ${preds.map(p => `<span style="background:${stageColor(p.stage)}" title="t${p.timepoint} · ${this.escapeHtml(this.stageText(p.stage))}"></span>`).join('')}
            </div>
            <div class="stage-bar-key"><span>t${preds[0].timepoint} ${this.escapeHtml(this.stageText(preds[0].stage))}</span><span>${this.escapeHtml(this.stageText(preds[preds.length - 1].stage))} t${preds[preds.length - 1].timepoint}</span></div>`;
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

    renderEventsTab() {
        const s = this.currentSession;
        const events = s.events || [];
        const temp = s.temperature || [];
        const spark = this.temperatureSpark(temp);
        if (!events.length && !spark) return '<div class="empty-tab">Nothing recorded beyond the images</div>';
        return `
            ${spark}
            ${events.length ? `<div class="event-list">
                ${events.map(ev => `<time>${this.formatDateTime(ev.at)}</time><div class="${ev.level === 'warn' ? 'is-warn' : ev.level === 'error' ? 'is-error' : ''}">${this.escapeHtml(ev.text)}</div>`).join('')}
            </div>` : '<div class="empty-tab">No notable events recorded</div>'}`;
    },

    temperatureSpark(samples) {
        const pts = samples.filter(p => Number.isFinite(Number(p.water_c)));
        if (pts.length < 2) return '';
        const W = 600, H = 72, pad = 4;
        const ys = pts.map(p => Number(p.water_c));
        const sp = pts.map(p => Number(p.setpoint_c)).filter(Number.isFinite);
        const lo = Math.min(...ys, ...sp) - 0.2, hi = Math.max(...ys, ...sp) + 0.2;
        const x = i => pad + (i / (pts.length - 1)) * (W - 2 * pad);
        const y = v => H - pad - ((v - lo) / (hi - lo)) * (H - 2 * pad);
        const line = ys.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
        const set = sp.length ? `<line x1="${pad}" x2="${W - pad}" y1="${y(sp[sp.length - 1]).toFixed(1)}" y2="${y(sp[sp.length - 1]).toFixed(1)}" stroke="var(--temp-setpoint-color)" stroke-dasharray="4 4" stroke-width="1"/>` : '';
        const minV = Math.min(...ys), maxV = Math.max(...ys);
        return `
            <div class="session-temp">
                <div class="session-temp-head">Water temperature <span>${minV.toFixed(1)}–${maxV.toFixed(1)} °C</span>${sp.length ? `<span>setpoint ${sp[sp.length - 1].toFixed(1)} °C</span>` : ''}<span>${this.formatDateTime(pts[0].t)} → ${this.formatDateTime(pts[pts.length - 1].t)}</span></div>
                <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Water temperature over the session">
                    ${set}
                    <polyline fill="none" stroke="var(--temp-water-color)" stroke-width="1.5" points="${line}"/>
                </svg>
            </div>`;
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
        const dic = plan.dic;
        if (dic && dic.enabled) {
            const d = [];
            if (dic.every_seconds) d.push(`every ${this.fmtInterval(dic.every_seconds)}`);
            if (dic.light) d.push(dic.light === 'led' ? `LED${dic.led_intensity_pct != null ? ` ${dic.led_intensity_pct}%` : ''}` : dic.light === 'room' ? 'room light' : 'light as it is');
            if (dic.exposure_ms != null) d.push(`${dic.exposure_ms} ms`);
            rows.push(['DIC overview', d.join(' · ')]);
        } else {
            rows.push(['DIC overview', 'off']);
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
            });
        });
    },

    renderCurrentTab() {
        switch (this.currentTab) {
            case 'embryos': return this.renderEmbryosTab();
            case 'detections': return this.renderDetectionsTab();
            case 'conversation': return this.renderConversationTab();
            case 'events': return this.renderEventsTab();
            default: return '';
        }
    },

    renderDicStrip() {
        const frames = this.currentSession.dic_frames || [];
        if (!frames.length) return '';
        return `
            <div class="session-dic">
                <div class="session-dic-head">DIC overview <span class="session-dic-count">${frames.length} frame${frames.length !== 1 ? 's' : ''}</span></div>
                <div class="session-dic-frames">
                    ${frames.map(f => `
                        <a class="session-dic-frame" href="${f.url}" target="_blank" rel="noopener" title="Open frame ${f.frame ?? ''}">
                            <img src="${f.url}?max=256" alt="DIC overview frame ${f.frame ?? ''}" loading="lazy">
                            <span class="session-dic-cap">${f.frame ?? ''}${f.captured_at ? ` · ${this.formatTime(f.captured_at)}` : ''}</span>
                        </a>`).join('')}
                </div>
            </div>`;
    },

    renderEmbryosTab() {
        const embryos = this.currentSession.embryos || [];
        const strip = this.renderDicStrip();

        if (embryos.length === 0) {
            return strip + '<div class="empty-tab">No embryo data recorded</div>';
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
                            ? `<a class="embryo-thumb" data-embryo="${this.escapeHtml(e.embryo_id)}" href="${e.thumbnail}" target="_blank" rel="noopener" title="Move across to scrub through the timepoints; click to open"><img src="${e.thumbnail}" alt="${this.escapeHtml(name)}, timepoint ${e.latest_timepoint}" loading="lazy"><span class="embryo-thumb-cap">t${e.latest_timepoint}</span></a>`
                            : '<div class="embryo-thumb embryo-thumb-empty">no image</div>'}
                        <div class="embryo-header">
                            <h3>${this.escapeHtml(name)}${sub ? ` <small>${this.escapeHtml(sub)}</small>` : ''}</h3>
                            ${e.is_complete ? '<span class="status-badge complete">Complete</span>' : ''}
                        </div>
                        <div class="embryo-details">
                            ${e.role ? `<div class="detail"><span>Role:</span> ${this.escapeHtml(e.role)}</div>` : ''}
                            ${stage ? `<div class="detail"><span>Stage:</span> ${this.escapeHtml(stage)}${conf}</div>` : ''}
                            <div class="detail"><span>Timepoints:</span> ${e.timepoints || 0}</div>
                            ${e.strain ? `<div class="detail"><span>Strain:</span> ${this.escapeHtml(e.strain)}</div>` : ''}
                            ${e.position && e.position.x != null ? `<div class="detail"><span>Position:</span> (${Number(e.position.x).toFixed(1)}, ${Number(e.position.y).toFixed(1)})</div>` : ''}
                        </div>
                        ${this.stageBar(e)}
                        ${this.dose(e)}
                    </div>`;
                }).join('')}
            </div>
            ${this.renderRemoved()}
        `;
    },

    renderDetectionsTab() {
        const embryos = (this.currentSession.embryos || []).filter(e => (e.predictions || []).length);

        if (embryos.length === 0) {
            return '<div class="empty-tab">No stage calls recorded</div>';
        }

        return `
            <div class="detections-list">
                ${embryos.map(e => `
                    <div class="detection-group">
                        <h4>${this.escapeHtml(e.nickname || e.embryo_id)}</h4>
                        <div class="stage-calls">
                            ${e.predictions.map(p => `
                                <a class="stage-call" href="/api/sessions/${this.currentSession.session_id}/projection?embryo=${encodeURIComponent(e.embryo_id)}&t=${p.timepoint}" target="_blank" rel="noopener" title="Open timepoint ${p.timepoint}">
                                    <span class="stage-call-t">t${p.timepoint}</span>
                                    <span class="stage-call-stage">${this.escapeHtml(this.stageText(p.stage))}</span>
                                    ${p.confidence != null ? `<span class="stage-call-conf">${Math.round(p.confidence * 100)}%</span>` : ''}
                                </a>`).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    },

    renderConversationTab() {
        const conversation = this.currentSession.conversation || [];

        if (conversation.length === 0) {
            return '<div class="empty-tab">No conversation history</div>';
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
        const withContent = this.sessions.filter(s => s.embryo_count > 0).length;
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
