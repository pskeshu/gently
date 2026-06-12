/**
 * V2Landing (ux_v2): the agent-first welcome AND the in-place plan wizard.
 *
 * The key paradigm fix: the plan dialogue renders IN THE LANDING, not the chat
 * REPL. Clicking "Plan an experiment" doesn't recede to the dashboard — it
 * switches the landing to a plan screen, enters plan mode, and renders the
 * agent's ask_user_choice questions as button cards right there (#v2-plan-ask),
 * reusing AgentChat.buildAskCard so it's the same card/answer path as elsewhere.
 * The plan assembles on the right from each pick. Chat is never the surface
 * (it's one click away via "Open conversation" as the last resort).
 *
 *   Plan an experiment → switch to plan screen; /plan + kickoff; asks render here
 *   Take a quick look  → scope (Devices view)
 *   "just tell me…"    → free-text into the agent loop (then chat as transcript)
 *
 * No-ops unless #v2-landing is present (flag off → v1 untouched, overlay absent).
 */
const V2Landing = (() => {
    let el = null;
    let current = null;  // { reqId, data, isWake } — the ask showing in the plan stage

    const $ = (id) => document.getElementById(id);

    function greet() {
        const g = $('v2-landing-greeting');
        if (!g) return;
        const h = new Date().getHours();
        const t = h < 5 ? 'Still here.' : h < 12 ? 'Good morning.'
            : h < 18 ? 'Good afternoon.' : 'Good evening.';
        g.innerHTML = t + '<br><span class="dim">What are we doing today?</span>';
    }

    function setScreen(name) { if (el) el.dataset.screen = name; }
    function planActive() {
        return !!el && el.dataset.screen === 'plan' && !el.classList.contains('dismissed')
            && el.style.display !== 'none';
    }

    function dismiss() {
        if (!el || el.classList.contains('dismissed')) return;
        el.classList.add('dismissed');
        let done = false;
        const finish = () => {
            if (done) return;
            done = true;
            el.style.display = 'none';
            el.setAttribute('aria-hidden', 'true');
        };
        el.addEventListener('transitionend', finish, { once: true });
        setTimeout(finish, 650);
    }

    // ── plan stage helpers ────────────────────────────────────────────
    function showThinking(on) {
        const t = $('v2-plan-thinking');
        if (t) t.classList.toggle('hidden', !on);
    }
    function clearAsk() { const m = $('v2-plan-ask'); if (m) m.innerHTML = ''; }
    function resetSummary() {
        const list = $('v2-plan-summary');
        if (list) list.innerHTML = '<div class="v2-plan-side-empty">Assembling from your choices…</div>';
    }
    function labelFor(data, sel) {
        const opts = (data && data.options) || [];
        const one = (s) => {
            const o = opts.find(o => o && (o.id === s || o.value === s || o.label === s));
            return o ? o.label : String(s);
        };
        if (Array.isArray(sel)) return sel.map(one).join(', ');
        return one(sel);
    }
    function recordPick(data, sel) {
        const list = $('v2-plan-summary');
        if (!list) return;
        const empty = list.querySelector('.v2-plan-side-empty');
        if (empty) empty.remove();
        const row = document.createElement('div');
        row.className = 'v2-plan-row';
        row.innerHTML = '<span class="k"></span><span class="v"></span>';
        row.querySelector('.k').textContent = (data && data.question) || 'Choice';
        row.querySelector('.v').textContent = labelFor(data, sel);
        list.appendChild(row);
    }
    function renderAsk() {
        const mount = $('v2-plan-ask');
        if (!mount || !current || typeof AgentChat === 'undefined' || !AgentChat.buildAskCard) return;
        showThinking(false);
        const data = current.data, reqId = current.reqId;
        const hasControl = AgentChat.hasControl ? AgentChat.hasControl() : true;
        const card = AgentChat.buildAskCard(data, {
            reqId, isWake: current.isWake, hasControl,
            onPick: (sel) => {
                recordPick(data, sel);
                AgentChat.answerChoice(reqId, sel);
                current = null;
                clearAsk();
                showThinking(true);  // agent computing the next step
            },
        });
        mount.innerHTML = '';
        mount.appendChild(card);
    }

    function startPlan() {
        setScreen('plan');     // stay on the overlay; swap welcome → plan wizard
        resetSummary();
        clearAsk();
        current = null;
        showThinking(true);
        // /plan deterministically enters plan mode; the kickoff draws out the
        // first question, which the agent asks via ask_user_choice (its prompt
        // mandates it) → renders here, not in chat. runCommand connects on its
        // own and flushes both in order without opening the chat panel.
        if (typeof AgentChat !== 'undefined' && AgentChat.runCommand) {
            AgentChat.runCommand('/plan');
            AgentChat.runCommand("Let's design this run — what should it capture?");
        }
    }

    function openScope() {
        dismiss();
        if (typeof switchTab === 'function') switchTab('devices');
    }

    function sendFreeform(text) {
        const v = (text || '').trim();
        dismiss();  // free-text is the chat last-resort → recede, then open chat
        if (typeof AgentChat !== 'undefined' && AgentChat.togglePanel) {
            AgentChat.togglePanel(true);
            if (v && AgentChat.runCommand) setTimeout(() => AgentChat.runCommand(v), 300);
        }
    }

    function init() {
        el = $('v2-landing');
        if (!el || typeof ClientEventBus === 'undefined') return;  // flag off → no-op
        greet();

        // welcome choices
        el.querySelectorAll('[data-landing]').forEach(btn => btn.addEventListener('click', () => {
            const kind = btn.dataset.landing;
            if (kind === 'plan') startPlan();
            else if (kind === 'standalone') openScope();
        }));

        // welcome escape field
        const esc = $('v2-escape'), escToggle = $('v2-escape-toggle'),
            escInput = $('v2-escape-input'), escSend = $('v2-escape-send');
        if (escToggle && esc && escInput) {
            escToggle.addEventListener('click', () => {
                const open = esc.classList.toggle('open');
                if (open) setTimeout(() => escInput.focus(), 120);
            });
            const submit = () => sendFreeform(escInput.value);
            if (escSend) escSend.addEventListener('click', submit);
            escInput.addEventListener('keydown', e => {
                if (e.key === 'Enter') { e.preventDefault(); submit(); }
                else if (e.key === 'Escape') { e.stopPropagation(); esc.classList.remove('open'); }
            });
        }

        const skip = $('v2-landing-skip');
        if (skip) skip.addEventListener('click', dismiss);

        // plan-screen controls
        const back = $('v2-plan-back');
        if (back) back.addEventListener('click', () => setScreen('welcome'));
        const planChat = $('v2-plan-chat');
        if (planChat) planChat.addEventListener('click', () => {
            dismiss();  // chat panel lives under the overlay → recede first
            if (typeof AgentChat !== 'undefined' && AgentChat.togglePanel) {
                setTimeout(() => AgentChat.togglePanel(true), 300);
            }
        });
        const cont = $('v2-plan-continue');
        if (cont) cont.addEventListener('click', dismiss);

        // The agent's questions render in the plan stage while it's active; once
        // we've receded into the workspace, AskStage (#ask-stage) takes over.
        ClientEventBus.on('AGENT_ASK', ({ request_id, choice_data, origin }) => {
            if (!planActive()) return;
            current = { reqId: request_id, data: choice_data || {}, isWake: origin === 'wake' };
            renderAsk();
        });
        ClientEventBus.on('ASK_CLEARED', ({ request_id }) => {
            if (request_id === '*' || (current && request_id === current.reqId)) {
                current = null;
                clearAsk();
                if (planActive()) showThinking(true);
            }
        });
        ClientEventBus.on('AGENT_CONTROL', () => { if (current && planActive()) renderAsk(); });

        document.addEventListener('keydown', e => {
            if (e.key !== 'Escape' || !el || el.classList.contains('dismissed')) return;
            if (el.dataset.screen === 'plan') setScreen('welcome');  // step back, don't bail
            else dismiss();
        });
    }

    document.addEventListener('DOMContentLoaded', init);

    return {
        dismiss,
        show: () => {
            if (!el) return;
            el.style.display = '';
            el.removeAttribute('aria-hidden');
            el.classList.remove('dismissed');
            setScreen('welcome');
            greet();
        },
    };
})();
