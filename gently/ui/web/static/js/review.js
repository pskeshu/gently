/**
 * Session Review Application
 * Browse and review past experiment sessions
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
        this.setupTabHandlers();
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

        list.innerHTML = filtered.map(s => `
            <div class="session-item ${s.active ? 'active-session' : ''}" data-session-id="${s.session_id}" onclick="ReviewApp.loadSession('${s.session_id}')">
                <div class="session-name">${this.escapeHtml(s.name || s.session_id)}${s.active ? ' <span class="session-active-badge">active</span>' : ''}</div>
                <div class="session-meta">
                    ${this.formatDate(s.created_at)}
                    ${s.embryo_count ? `<span class="dot"></span>${s.embryo_count} embryo${s.embryo_count !== 1 ? 's' : ''}` : ''}
                </div>
                ${s.description ? `<div class="session-desc">${this.escapeHtml(s.description)}</div>` : ''}
                ${s.active ? '' : `<button class="session-resume-btn" onclick="event.stopPropagation(); ReviewApp.resumeSession('${s.session_id}')">Resume in agent</button>`}
            </div>
        `).join('');
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

    renderSessionContent() {
        const content = document.getElementById('session-content');
        const s = this.currentSession;

        content.innerHTML = `
            <div class="session-header">
                <h2>${this.escapeHtml(s.name || s.session_id)}</h2>
                ${s.description ? `<p class="session-description">${this.escapeHtml(s.description)}</p>` : ''}
                <div class="session-stats">
                    <span>Created: ${this.formatDateTime(s.created_at)}</span>
                    ${s.last_active ? `<span>Last active: ${this.formatDateTime(s.last_active)}</span>` : ''}
                </div>
            </div>

            <div class="session-tabs">
                <button class="tab ${this.currentTab === 'embryos' ? 'active' : ''}" data-tab="embryos">
                    Embryos
                    <span class="tab-count">${Object.keys(s.embryo_states || {}).length}</span>
                </button>
                <button class="tab ${this.currentTab === 'detections' ? 'active' : ''}" data-tab="detections">
                    Detections
                </button>
                <button class="tab ${this.currentTab === 'conversation' ? 'active' : ''}" data-tab="conversation">
                    Conversation
                    <span class="tab-count">${(s.conversation || []).length}</span>
                </button>
            </div>

            <div class="tab-content" id="tab-content">
                ${this.renderCurrentTab()}
            </div>
        `;

        this.setupTabHandlers();
    },

    setupTabHandlers() {
        document.querySelectorAll('.session-tabs .tab').forEach(tab => {
            tab.addEventListener('click', () => {
                this.currentTab = tab.dataset.tab;
                document.querySelectorAll('.session-tabs .tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById('tab-content').innerHTML = this.renderCurrentTab();
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

    renderEmbryosTab() {
        const embryos = this.currentSession.embryo_states || {};
        const entries = Object.entries(embryos);

        if (entries.length === 0) {
            return '<div class="empty-tab">No embryo data recorded</div>';
        }

        return `
            <div class="embryo-grid">
                ${entries.map(([id, state]) => `
                    <div class="embryo-card">
                        <div class="embryo-header">
                            <h3>${this.escapeHtml(state.nickname || id)}</h3>
                            ${state.is_complete ? '<span class="status-badge complete">Complete</span>' : ''}
                        </div>
                        <div class="embryo-details">
                            ${state.current_stage ? `<div class="detail"><span>Stage:</span> ${state.current_stage}</div>` : ''}
                            ${state.timepoints_acquired ? `<div class="detail"><span>Timepoints:</span> ${state.timepoints_acquired}</div>` : ''}
                            ${state.hatching_status ? `<div class="detail"><span>Hatching:</span> ${state.hatching_status.detected ? 'Yes' : 'No'}</div>` : ''}
                            ${state.stage_position ? `<div class="detail"><span>Position:</span> (${state.stage_position.x?.toFixed(1)}, ${state.stage_position.y?.toFixed(1)})</div>` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    },

    renderDetectionsTab() {
        const history = this.currentSession.detection_history || {};
        const entries = Object.entries(history);

        if (entries.length === 0) {
            return '<div class="empty-tab">No detection history recorded</div>';
        }

        return `
            <div class="detections-list">
                ${entries.map(([embryoId, detections]) => `
                    <div class="detection-group">
                        <h4>${this.escapeHtml(embryoId)}</h4>
                        ${(detections || []).map(d => `
                            <div class="detection-item ${d.detected ? 'positive' : ''}">
                                <div class="detection-header">
                                    <span class="detector-name">${this.escapeHtml(d.detector_name || 'Unknown')}</span>
                                    <span class="detection-result">${d.detected ? 'Detected' : 'Not detected'}</span>
                                    ${d.confidence ? `<span class="confidence">${(d.confidence * 100).toFixed(0)}%</span>` : ''}
                                </div>
                                ${d.reasoning ? `<div class="detection-reasoning">${this.escapeHtml(d.reasoning)}</div>` : ''}
                                ${d.timepoint !== undefined ? `<div class="detection-meta">Timepoint: ${d.timepoint}</div>` : ''}
                            </div>
                        `).join('')}
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
                ${conversation.map(msg => `
                    <div class="message ${msg.role}">
                        <div class="message-role">${msg.role === 'user' ? 'User' : msg.role === 'assistant' ? 'Assistant' : 'System'}</div>
                        <div class="message-content">${this.formatMessageContent(msg.content)}</div>
                        ${msg.timestamp ? `<div class="message-time">${this.formatDateTime(msg.timestamp)}</div>` : ''}
                    </div>
                `).join('')}
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
                    return this.escapeHtml(block.text || '').replace(/\n/g, '<br>');
                }
                return `<span class="content-block">[${block.type}]</span>`;
            }).join('');
        }
        return JSON.stringify(content);
    },

    formatDate(isoString) { return formatDate(isoString); },

    formatDateTime(isoString) { return formatDate(isoString); },

    escapeHtml(str) { return escapeHtml(str); },

    updateStatusbar() {
        const left = document.getElementById('status-left');
        const right = document.getElementById('status-right');
        if (!left) return;
        const total = this.sessions.length;
        const withContent = this.sessions.filter(s => s.embryo_count > 0).length;
        left.textContent = `${total} session${total !== 1 ? 's' : ''} \u00B7 ${withContent} with content`;
        if (right && this.currentSession) {
            const embryos = Object.keys(this.currentSession.embryo_states || {}).length;
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
