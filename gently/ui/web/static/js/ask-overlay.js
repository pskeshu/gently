/**
 * AskOverlay — the agent's question when the chat is collapsed.
 *
 * The question card lives in one place: the sticky slot above the chat's
 * composer (agent-chat.js renderChoice), because that is where the eye is
 * after typing. This module covers the other case. While a question waits
 * and the panel is collapsed:
 *
 *   - the header beacon (#ask-beacon) shows, carrying the question, and stays
 *     until the question is answered — a waiting agent is never silent;
 *   - clicking it opens the overlay (#ask-overlay): the same card, built by the
 *     same builder, centred over a dimmed workspace. It never opens by itself;
 *     a question popping up over the camera while a hand is on a control is the
 *     wrong kind of attention. Esc, ×, or a click outside put it back to the
 *     beacon, not away. "open the chat" opens the panel, where the card already
 *     is, and the overlay closes;
 *   - after a moment, the tab title says "Gently is asking", for a window that
 *     is not in front.
 *
 * No timeout and no default: the agent waits for an answer. This module only
 * makes the waiting visible.
 */
const AskOverlay = (() => {
    const ASKING_TITLE = 'Gently is asking';
    const TITLE_DELAY_MS = 3000;

    let current = null;      // { reqId, data, isWake } while a question waits
    let wanted = false;      // the operator asked to see the overlay
    let overlay = null, cardHost = null, beacon = null, beaconQ = null;
    let titleTimer = null, savedTitle = null;

    function panelOpen() {
        return typeof AgentChat !== 'undefined' && AgentChat.isPanelOpen ? AgentChat.isPanelOpen() : true;
    }

    function render() {
        const pending = !!current;
        const collapsed = pending && !panelOpen();

        if (beacon) {
            beacon.classList.toggle('hidden', !collapsed);
            if (collapsed && beaconQ) beaconQ.textContent = (current.data && current.data.question) || '';
        }

        const show = collapsed && wanted;
        if (overlay) {
            cardHost.innerHTML = '';
            if (show && typeof AgentChat !== 'undefined' && AgentChat.buildAskCard) {
                cardHost.appendChild(AgentChat.buildAskCard(current.data, {
                    reqId: current.reqId,
                    isWake: current.isWake,
                    hasControl: AgentChat.hasControl ? AgentChat.hasControl() : true,
                    onPick: (sel) => AgentChat.answerChoice(current.reqId, sel),
                }));
            }
            overlay.classList.toggle('hidden', !show);
        }

        if (pending) {
            if (titleTimer === null) {
                titleTimer = setTimeout(() => {
                    titleTimer = null;
                    if (!current) return;
                    if (savedTitle === null) savedTitle = document.title;
                    document.title = ASKING_TITLE;
                }, TITLE_DELAY_MS);
            }
        } else {
            if (titleTimer !== null) { clearTimeout(titleTimer); titleTimer = null; }
            if (savedTitle !== null) { document.title = savedTitle; savedTitle = null; }
        }
    }

    function dismiss() { wanted = false; render(); }

    function init() {
        overlay = document.getElementById('ask-overlay');
        beacon = document.getElementById('ask-beacon');
        if (!overlay || typeof ClientEventBus === 'undefined') return;
        cardHost = document.getElementById('ask-overlay-card');
        beaconQ = beacon ? beacon.querySelector('.ask-beacon-q') : null;

        ClientEventBus.on('AGENT_ASK', ({ request_id, choice_data, origin }) => {
            current = { reqId: request_id, data: choice_data || {}, isWake: origin === 'wake' };
            wanted = false;   // never opens by itself
            render();
        });
        ClientEventBus.on('ASK_CLEARED', ({ request_id }) => {
            if (!current) return;
            if (request_id === '*' || request_id === current.reqId) { current = null; wanted = false; render(); }
        });
        ClientEventBus.on('AGENT_PANEL', render);
        // Control changing hands mid-question: the card re-renders read-only or live.
        ClientEventBus.on('AGENT_CONTROL', () => { if (current) render(); });

        if (beacon) beacon.addEventListener('click', () => { wanted = true; render(); });
        overlay.querySelectorAll('[data-dismiss]').forEach(el => el.addEventListener('click', dismiss));
        overlay.querySelector('.ask-overlay-open-chat').addEventListener('click', () => {
            wanted = false;
            if (typeof AgentChat !== 'undefined') AgentChat.togglePanel(true);
            render();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !overlay.classList.contains('hidden')) { e.preventDefault(); dismiss(); }
        });
    }

    document.addEventListener('DOMContentLoaded', init);
    return { dismiss };
})();
