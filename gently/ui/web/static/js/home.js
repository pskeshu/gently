/**
 * HomeApp — the landing tab.
 *
 * A light at-a-glance landing surface: recent sessions, recent plans, recent
 * images, a thin status line, and a "Start / continue an experiment" button
 * that launches the setup flow (the wizard, which no longer auto-pops in chat).
 *
 * Read-only fetches against existing endpoints (/api/sessions, /api/campaigns,
 * /api/snapshots); mirrors the ReviewApp/CampaignsApp module pattern.
 */
const HomeApp = (() => {
    let _inited = false;
    const SESSIONS_N = 5;
    const CAMPAIGNS_N = 5;
    const IMAGES_N = 8;

    function relTime(iso) {
        if (!iso) return '';
        const t = Date.parse(iso);
        if (isNaN(t)) return '';
        const s = Math.max(0, (Date.now() - t) / 1000);
        if (s < 60) return 'just now';
        if (s < 3600) return `${Math.floor(s / 60)}m ago`;
        if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
        const d = Math.floor(s / 86400);
        return d < 30 ? `${d}d ago` : new Date(t).toLocaleDateString();
    }

    function empty(el, msg) {
        el.innerHTML = `<div class="empty-state">${escapeHtml(msg)}</div>`;
    }

    function wireGoTab(scope) {
        (scope || document).querySelectorAll('[data-go-tab]').forEach(el => {
            if (el._goWired) return;
            el._goWired = true;
            el.addEventListener('click', (e) => {
                e.preventDefault();
                if (typeof switchTab === 'function') switchTab(el.dataset.goTab);
            });
        });
    }

    async function loadSessions() {
        const el = document.getElementById('home-recent-sessions');
        if (!el) return;
        try {
            const data = await (await fetch('/api/sessions')).json();
            const sessions = (data.sessions || []).slice(0, SESSIONS_N);
            if (!sessions.length) { empty(el, 'No sessions yet.'); return; }
            el.innerHTML = sessions.map(s => {
                const live = s.active ? '<span class="home-tag home-tag-live">live</span>' : '';
                const resume = s.active ? '' :
                    `<button class="home-resume" data-resume="${escapeHtml(s.session_id)}">Resume</button>`;
                return `<div class="home-item">
                    <div class="home-item-main">
                        <div class="home-item-row"><span class="home-item-name">${escapeHtml(s.name || s.session_id)}</span>${live}</div>
                        <span class="home-item-meta">${escapeHtml(relTime(s.last_active))} · ${s.embryo_count || 0} embryos</span>
                    </div>${resume}
                </div>`;
            }).join('');
            el.querySelectorAll('[data-resume]').forEach(b => b.addEventListener('click', async () => {
                b.disabled = true;
                b.textContent = 'Resuming…';
                try {
                    await fetch(`/api/sessions/${encodeURIComponent(b.dataset.resume)}/resume`, { method: 'POST' });
                } catch (_) { b.disabled = false; b.textContent = 'Resume'; }
            }));
        } catch (e) { empty(el, 'Could not load sessions.'); }
    }

    async function loadCampaigns() {
        const el = document.getElementById('home-recent-campaigns');
        if (!el) return;
        try {
            const data = await (await fetch('/api/campaigns')).json();
            const items = (data.campaigns || []).slice(0, CAMPAIGNS_N);
            if (!items.length) { empty(el, 'No plans yet.'); return; }
            el.innerHTML = items.map(t => {
                const c = t.campaign || {};
                const st = t.status || {};
                const name = c.shorthand || c.description || 'Untitled plan';
                const total = st.total || 0;
                const chip = total ? `<span class="home-chip">${st.completed || 0}/${total} done</span>` : '';
                return `<div class="home-item home-item-clickable" data-go-tab="plans">
                    <span class="home-item-name">${escapeHtml(name)}</span>${chip}
                </div>`;
            }).join('');
            wireGoTab(el);
        } catch (e) { empty(el, 'Could not load plans.'); }
    }

    async function loadImages() {
        const el = document.getElementById('home-recent-images');
        if (!el) return;
        try {
            const data = await (await fetch('/api/snapshots')).json();
            // /api/snapshots is timestamp-ASCENDING; take the tail for "recent".
            const recent = (data.snapshots || []).slice(-IMAGES_N).reverse();
            if (!recent.length) {
                empty(el, 'No images yet — they appear once a session is active.');
                return;
            }
            el.innerHTML = '<div class="home-image-strip">' + recent.map(s => {
                const m = s.metadata || {};
                const label = m.embryo_id
                    ? `${m.embryo_id}${m.timepoint != null ? ' · t' + m.timepoint : ''}` : '';
                return `<div class="home-image" title="${escapeHtml(label)}">
                    <img loading="lazy" src="/api/images/${encodeURIComponent(s.uid)}/png?size=96" alt="${escapeHtml(label)}">
                </div>`;
            }).join('') + '</div>';
        } catch (e) { empty(el, 'Could not load images.'); }
    }

    function updateStatus() {
        const el = document.getElementById('home-status');
        if (!el) return;
        const connected = (typeof state !== 'undefined' && state.connected);
        const n = (typeof state !== 'undefined' && Array.isArray(state.embryos)) ? state.embryos.length : 0;
        el.textContent = connected
            ? `Connected · ${n} embryo${n === 1 ? '' : 's'} in view`
            : 'Offline — start the agent to connect.';
    }

    function refresh() {
        updateStatus();
        loadSessions();
        loadCampaigns();
        loadImages();
    }

    function init() {
        if (!_inited) {
            _inited = true;
            wireGoTab(document.getElementById('home-content'));
            const start = document.getElementById('home-start-btn');
            if (start) start.addEventListener('click', () => {
                if (typeof AgentChat !== 'undefined' && AgentChat.togglePanel) {
                    AgentChat.togglePanel(true);
                    // Let the panel's WS connect before sending the command.
                    if (AgentChat.runCommand) setTimeout(() => AgentChat.runCommand('/wizard'), 250);
                }
            });
        }
        refresh();  // re-fetch on every entry to the tab
    }

    // Self-initialise on load when Home is the default-active tab (switchTab's
    // lazy-init hook only fires on a tab click / hash route, not initial paint).
    document.addEventListener('DOMContentLoaded', () => {
        const home = document.getElementById('home-content');
        if (home && home.classList.contains('active')) init();
    });

    return { init, refresh };
})();
