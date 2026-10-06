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

        // Filter sessions based on checkbox
        let filtered = this.sessions;
        if (filterWithContent) {
            filtered = this.sessions.filter(s => s.embryo_count > 0);
        }

        if (this.sessions.length === 0) {
            list.innerHTML = '<div class="no-sessions">No sessions found</div>';
            return;
        }

        if (filtered.length === 0) {
            list.innerHTML = `<div class="no-sessions">No sessions with content<br><small>${this.sessions.length} empty session${this.sessions.length !== 1 ? 's' : ''} hidden</small></div>`;
            return;
        }

        list.innerHTML = filtered.map(s => {
            const held = this.holdings(s);
            return `
            <div class="session-item ${s.active ? 'active-session' : ''}" data-session-id="${s.session_id}" onclick="ReviewApp.loadSession('${s.session_id}')">
                <div class="session-name">${this.escapeHtml(s.name || s.session_id)}${s.active ? ' <span class="session-active-badge">active</span>' : ''}${s.advanced_diagnostics ? ' <span class="session-active-badge session-diag-badge" title="Recorded with Advanced diagnostics on">advanced diagnostics</span>' : ''}</div>
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
                    <h2>${this.escapeHtml(s.name || s.session_id)}</h2>
                    ${s.active ? '<span class="session-active-badge">live now</span>'
                        : `<button class="session-resume-btn session-resume-main" onclick="ReviewApp.resumeSession('${s.session_id}')">Resume in agent</button>`}
                </div>
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
            </div>

            <div class="session-tab-content" id="session-tab-content">
                ${this.renderCurrentTab()}
            </div>
        `;

        this.setupTabHandlers();
        void frames;
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
            });
        });
    },

    renderCurrentTab() {
        switch (this.currentTab) {
            case 'embryos': return this.renderEmbryosTab();
            case 'detections': return this.renderDetectionsTab();
            case 'conversation': return this.renderConversationTab();
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
                            ? `<a class="embryo-thumb" href="${e.thumbnail}" target="_blank" rel="noopener" title="Open timepoint ${e.latest_timepoint}"><img src="${e.thumbnail}" alt="${this.escapeHtml(name)}, timepoint ${e.latest_timepoint}" loading="lazy"><span class="embryo-thumb-cap">t${e.latest_timepoint}</span></a>`
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
                    </div>`;
                }).join('')}
            </div>
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
