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
    // The last few sessions whole, embryo by embryo: up to this many
    // sessions, and this many images among them.
    const IMAGES_N = 24;
    const IMAGE_SESSIONS_N = 3;
    // Recent images are stable (latest projection per embryo). refresh() runs on
    // every Home-tab entry, so guard against redundant disk-walking fetches:
    // skip if one is in flight or the strip was loaded within IMAGES_TTL_MS.
    const IMAGES_TTL_MS = 15000;
    let _imgState = { at: 0, inflight: false };
    let _recent = [];          // the strip as last rendered, for the Lightbox
    let _liveTimer = null;

    /** Is Home the tab on screen? The strip only redraws live when it is seen. */
    function homeIsShowing() {
        const panel = document.getElementById('home-content');
        return !!(panel && panel.classList.contains('active'));
    }

    /**
     * A volume or an overview frame landed. The strip is "latest projection
     * per embryo", so it is out of date the moment one does. Redrawn now if
     * Home is on screen, otherwise marked stale so the next entry skips the
     * throttle — the strip used to refresh only on entry, at most every 15 s,
     * and never on the event that changes it.
     */
    function onImagery() {
        _imgState.at = 0;
        if (!homeIsShowing()) return;
        clearTimeout(_liveTimer);
        _liveTimer = setTimeout(() => loadImages(true), 1500);
    }
    if (typeof ClientEventBus !== 'undefined') {
        ['VOLUME_ACQUIRED', 'IMAGE_ACQUIRED', 'ACQUISITION_COMPLETED'].forEach(ev => ClientEventBus.on(ev, onImagery));
    }

    /**
     * The images under their sessions, in the order they came: the latest
     * session first, its embryos in order. `index` is the image's place in
     * the whole list, which is what the Lightbox walks.
     */
    function groupBySession(images) {
        const groups = [];
        images.forEach((image, index) => {
            let g = groups.find(x => x.session_id === image.session_id);
            if (!g) {
                g = {
                    session_id: image.session_id,
                    session_name: image.session_name,
                    created_at: image.session_created_at,
                    items: [],
                };
                groups.push(g);
            }
            g.items.push({ image, index });
        });
        groups.forEach(g => g.items.sort((a, b) =>
            String(a.image.embryo_id).localeCompare(String(b.image.embryo_id), undefined, { numeric: true })));
        return groups;
    }

    /** A session by its name if it was given one, else by when it was. */
    function sessionTitle(g) {
        const named = g.session_name && g.session_name !== g.session_id && g.session_name !== 'unnamed';
        if (named) return g.session_name;
        const t = g.created_at ? new Date(g.created_at) : null;
        if (t && !isNaN(t)) {
            return t.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        }
        return 'Session';
    }

    function embryoName(id) {
        const m = String(id || '').match(/^embryo_(\d+)$/);
        return m ? `embryo ${m[1]}` : String(id || '');
    }

    function liveSessionId() {
        const link = document.getElementById('session-id-link');
        return link ? link.textContent.trim() : '';
    }

    /** Open the strip's images in the Lightbox, starting at the one clicked. */
    function openRecent(index) {
        if (typeof Lightbox === 'undefined' || !_recent.length) return;
        Lightbox.open(_recent.map(s => ({
            url: s.url,
            data_type: 'projection',
            metadata: {
                embryo_id: s.embryo_id,
                timepoint: s.timepoint,
                session_id: s.session_id,
                session_name: s.session_name,
            },
        })), Math.max(0, Math.min(index, _recent.length - 1)), 'home');
    }

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
            const data = await (await fetch(
                `/api/home/recent-images?limit=${IMAGES_N}&sessions=${IMAGE_SESSIONS_N}`)).json();
            // Latest projection per embryo across recent sessions (server orders
            // most-recent session first).
            const recent = (data.images || []).slice(0, IMAGES_N);
            if (!recent.length) {
                empty(el, 'No images yet — they appear once a session has captured volumes.');
                _imgState.at = Date.now();
                return;
            }
            _recent = recent.map(s => Object.assign({}, s, {
                url: `/api/sessions/${encodeURIComponent(s.session_id)}`
                    + `/projection?embryo=${encodeURIComponent(s.embryo_id)}`
                    + `&t=${encodeURIComponent(s.timepoint)}`,
            }));
            // A tile is a button to the image, and looks like one. It used to
            // be a div: "thumbnail only … not able to click them really".
            // Under its session: embryo_1 is a different embryo in every
            // session, and a strip of them said nothing about which.
            const tile = (s, i) => {
                const tp = (s.timepoint != null) ? ` · t${s.timepoint}` : '';
                const label = `${embryoName(s.embryo_id)}${tp}`;
                // Two lines, on purpose: a tile is 84 px wide, and the name
                // and the timepoint on one line wrapped wherever they fell.
                const cap = escapeHtml(embryoName(s.embryo_id))
                    + (s.timepoint != null ? `<br>t${escapeHtml(String(s.timepoint))}` : '');
                const n = Number(s.timepoints) || 0;
                const title = `${s.session_id}/${s.embryo_id}${tp}`
                    + (n ? ` · ${n} timepoint${n === 1 ? '' : 's'}` : '');
                return `<button type="button" class="home-image" data-home-image="${i}" title="${escapeHtml(title)}">
                    <img loading="lazy" src="${s.url}" alt="${escapeHtml(label)}">
                    <span class="home-image-cap">${cap}</span>
                </button>`;
            };
            el.innerHTML = groupBySession(_recent).map(g => `
                <div class="home-image-group" data-session="${escapeHtml(g.session_id)}">
                    <div class="home-image-group-head">
                        <span class="home-image-session">${escapeHtml(sessionTitle(g))}</span>
                        <span class="home-image-session-id">${escapeHtml(g.session_id)}</span>
                        ${g.session_id === liveSessionId() ? '<span class="home-image-live">open now</span>' : ''}
                        <span class="home-image-group-count">${g.items.length} embryo${g.items.length === 1 ? '' : 's'}</span>
                        ${typeof Reveal !== 'undefined' ? Reveal.button(
                            { what: 'session', session_id: g.session_id }, 'show',
                            { title: 'Open this session’s folder' }) : ''}
                    </div>
                    <div class="home-image-strip">${g.items.map(x => tile(x.image, x.index)).join('')}</div>
                </div>`).join('');
            if (!el._openWired) {
                el._openWired = true;
                el.addEventListener('click', e => {
                    const b = e.target.closest('[data-home-image]');
                    if (b) openRecent(Number(b.dataset.homeImage));
                });
            }
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
