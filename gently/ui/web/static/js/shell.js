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

    function setActive(tabName) {
        railItems.forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));
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

        const chatBtn = document.getElementById('v2-rail-chat');
        if (chatBtn) chatBtn.addEventListener('click', () => {
            if (typeof AgentChat !== 'undefined' && AgentChat.togglePanel) AgentChat.togglePanel(true);
        });

        renderStrip();
    }

    document.addEventListener('DOMContentLoaded', init);
    return {};
})();
