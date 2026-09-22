/**
 * Shell (ux_v2): the grouped left-rail nav (Now / Library / System) + the
 * session-context strip that replace the flat 8-tab bar.
 *
 * CRITICAL: the rail ROUTES THROUGH switchTab(tabId) for every reveal — it
 * never reimplements tab activation, so each tab's lazy-init side-effect
 * (HomeApp.init, EmbryosManager.clearDetectionBadge, CampaignsApp.init, …)
 * still fires. switchTab emits TAB_CHANGED, which keeps the rail's active
 * state in sync no matter who switched (rail, keyboard shortcut, home card,
 * hash route). No-ops unless body.ux-v2 is present (flag off → v1 untouched).
 */
const Shell = (() => {
    let railItems = [];

    const COLLAPSE_KEY = 'gently.rail.collapsed';

    function setActive(tabName) {
        railItems.forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));
    }

    // The rail collapses to an icon strip. The choice is per browser and
    // survives reloads: an operator who wants the width for the camera should
    // not have to reclaim it every session. localStorage can throw (private
    // window, blocked site data), so every access is guarded and the rail
    // simply stays expanded if it cannot be read.
    function applyCollapsed(on) {
        document.body.classList.toggle('rail-collapsed', on);
        const btn = document.getElementById('v2-rail-collapse');
        if (btn) {
            btn.setAttribute('aria-expanded', String(!on));
            btn.title = on ? 'Expand the sidebar' : 'Collapse the sidebar';
        }
    }

    function initCollapse() {
        const btn = document.getElementById('v2-rail-collapse');
        if (!btn) return;
        let saved = false;
        try { saved = localStorage.getItem(COLLAPSE_KEY) === '1'; } catch (e) { /* stays expanded */ }
        applyCollapsed(saved);
        btn.addEventListener('click', () => {
            const on = !document.body.classList.contains('rail-collapsed');
            applyCollapsed(on);
            try { localStorage.setItem(COLLAPSE_KEY, on ? '1' : '0'); } catch (e) { /* not fatal */ }
        });
    }

    function currentTab() {
        const active = document.querySelector('.tab.active');
        return (active && active.dataset.tab) ||
            (typeof state !== 'undefined' && state.tab) || 'home';
    }

    function renderStrip(status) {
        const el = document.getElementById('v2-strip-status');
        if (!el) return;
        const s = status || (typeof ConnectionStatus !== 'undefined' ? ConnectionStatus.get() : {});
        const n = (typeof state !== 'undefined' && Array.isArray(state.embryos)) ? state.embryos.length : 0;
        const conn = s.gentlyConnected ? (s.microscopeConnected ? 'Connected' : 'Online') : 'Offline';
        el.textContent = `${n} embryo${n === 1 ? '' : 's'} · ${conn}`;
    }

    function init() {
        if (!document.body.classList.contains('ux-v2')) return;  // flag off → no-op

        railItems = Array.from(document.querySelectorAll('.v2-nav-item'));
        railItems.forEach(btn => btn.addEventListener('click', () => {
            if (typeof switchTab === 'function') switchTab(btn.dataset.tab);
        }));
        setActive(currentTab());

        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('TAB_CHANGED', (tabName) => setActive(tabName));
            ClientEventBus.on('CONNECTION_STATUS', (s) => renderStrip(s));
            // Embryo count lives in state.embryos; re-render the strip whenever it
            // changes (including the initial bootstrap) so the header doesn't sit
            // at the pre-load 0.
            ClientEventBus.on('EMBRYOS_UPDATE', () => renderStrip());
        }

        initCollapse();

        renderStrip();
    }

    document.addEventListener('DOMContentLoaded', init);
    return {};
})();
