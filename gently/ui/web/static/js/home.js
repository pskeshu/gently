/**
 * HomeApp — the landing tab.
 *
 * A light at-a-glance landing surface: recent sessions, recent plans, recent
 * images, a thin status line, and a "Start / continue an experiment" button
 * that launches the setup flow (the wizard, which no longer auto-pops in chat).
 *
 * Read-only fetches against existing endpoints (/api/sessions, /api/campaigns,
 * /api/home/recent-images); mirrors the ReviewApp/CampaignsApp module pattern.
 */
const HomeApp = (() => {
    let _inited = false;
    const SESSIONS_N = 5;
    const CAMPAIGNS_N = 5;
    const IMAGES_N = 8;
    // Recent images are stable (latest projection per embryo). refresh() runs on
    // every Home-tab entry, so guard against redundant disk-walking fetches:
    // skip if one is in flight or the strip was loaded within IMAGES_TTL_MS.
    const IMAGES_TTL_MS = 15000;
    let _imgState = { at: 0, inflight: false };

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

    async function loadImages(force) {
        const el = document.getElementById('home-recent-images');
        if (!el) return;
        if (_imgState.inflight) return;
        // _imgState.at is set only after a completed fetch (images or empty),
        // never after an error — so failures still retry on the next entry.
        if (!force && _imgState.at && (Date.now() - _imgState.at) < IMAGES_TTL_MS) return;
        _imgState.inflight = true;
        try {
            const data = await (await fetch(`/api/home/recent-images?limit=${IMAGES_N}`)).json();
            // Latest projection per embryo across recent sessions (server orders
            // most-recent session first).
            const recent = (data.images || []).slice(0, IMAGES_N);
            if (!recent.length) {
                empty(el, 'No images yet — they appear once a session has captured volumes.');
                _imgState.at = Date.now();
                return;
            }
            el.innerHTML = '<div class="home-image-strip">' + recent.map(s => {
                const tp = (s.timepoint != null) ? ` · t${s.timepoint}` : '';
                const label = `${s.embryo_id || ''}${tp}`;
                const sub = s.session_name && s.session_name !== s.session_id
                    ? ` (${s.session_name})` : '';
                const src = `/api/sessions/${encodeURIComponent(s.session_id)}`
                    + `/projection?embryo=${encodeURIComponent(s.embryo_id)}`
                    + `&t=${encodeURIComponent(s.timepoint)}`;
                return `<div class="home-image" title="${escapeHtml(label + sub)}">
                    <img loading="lazy" src="${src}" alt="${escapeHtml(label)}">
                </div>`;
            }).join('') + '</div>';
            _imgState.at = Date.now();
        } catch (e) {
            empty(el, 'Could not load images.');
        } finally {
            _imgState.inflight = false;
        }
    }

    function updateStatus() {
        const el = document.getElementById('home-status');
        if (!el) return;
        // Read the shared ConnectionStatus store, not a one-shot snapshot of
        // state.connected — the latter was read once at tab init (before the
        // /ws handshake) and never corrected, showing "Offline" while the
        // header pill said "Online".
        const connected = (typeof ConnectionStatus !== 'undefined')
            ? ConnectionStatus.get().gentlyConnected
            : (typeof state !== 'undefined' && state.connected);
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
            // Re-render the status line on every connection change. subscribe()
            // replays the current snapshot immediately, so a late init still
            // renders correct state. Registered once (inside the _inited guard).
            if (typeof ConnectionStatus !== 'undefined') {
                ConnectionStatus.subscribe(() => updateStatus());
            }
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
