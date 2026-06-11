/**
 * AskStage (ux_v2) — renders the agent's CURRENT pending ask prominently on the
 * main stage, in addition to the chat transcript. One payload, two renderers:
 * it reuses AgentChat.buildAskCard so the stage and the transcript can't drift,
 * and answering from either surface clears both (via the ASK_CLEARED event that
 * AgentChat fires off the CHOICE lifecycle — not stream_end, which arrives only
 * after an in-turn answer and never for a cancelled turn).
 *
 * No-ops unless #ask-stage is present (gated behind GENTLY_UX_V2 in the
 * template), so it never affects the v1 dashboard.
 */
const AskStage = (() => {
    let stageEl = null;
    let current = null;  // { reqId, data, isWake }

    function clear() {
        current = null;
        if (stageEl) { stageEl.innerHTML = ''; stageEl.classList.add('hidden'); }
    }

    function render() {
        if (!stageEl || !current || typeof AgentChat === 'undefined' || !AgentChat.buildAskCard) return;
        const hasControl = AgentChat.hasControl ? AgentChat.hasControl() : true;
        const card = AgentChat.buildAskCard(current.data, {
            reqId: current.reqId,
            isWake: current.isWake,
            hasControl,
            onPick: (sel) => AgentChat.answerChoice(current.reqId, sel),
        });
        stageEl.innerHTML = '';
        stageEl.appendChild(card);
        stageEl.classList.remove('hidden');
    }

    function init() {
        stageEl = document.getElementById('ask-stage');
        if (!stageEl || typeof ClientEventBus === 'undefined') return;  // ux_v2 off → no-op

        ClientEventBus.on('AGENT_ASK', ({ request_id, choice_data, origin }) => {
            current = { reqId: request_id, data: choice_data || {}, isWake: origin === 'wake' };
            render();
        });
        ClientEventBus.on('ASK_CLEARED', ({ request_id }) => {
            if (!current) return;
            if (request_id === '*' || request_id === current.reqId) clear();
        });
        // Re-render read-only / actionable when control changes hands mid-ask.
        ClientEventBus.on('AGENT_CONTROL', () => { if (current) render(); });
    }

    document.addEventListener('DOMContentLoaded', init);
    return { clear };
})();
