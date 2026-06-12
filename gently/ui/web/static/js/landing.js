/**
 * V2Landing (ux_v2): the agent-first welcome AND the in-place plan wizard.
 *
 * Clicking "Plan an experiment" switches the landing to a plan screen, enters
 * plan mode, and renders the agent's work IN THE WIZARD — not the chat REPL:
 *   - the agent's reasoning + tool calls render as a tidy, claude.ai-style
 *     collapsible activity feed (#v2-plan-activity), fed by the AGENT_ACTIVITY
 *     event that agent-chat.js mirrors off the /ws/agent stream;
 *   - the agent's ask_user_choice questions render as button cards
 *     (#v2-plan-ask) via AgentChat.buildAskCard;
 *   - "THE PLAN" panel (#v2-plan-summary) mirrors the REAL plan (phases→tasks)
 *     fetched from /api/campaigns once a turn settles.
 * Chat is the last resort (the escape pill / "Open conversation").
 *
 * No-ops unless #v2-landing is present (flag off → v1 untouched, overlay absent).
 */
const V2Landing = (() => {
    let el = null;
    let current = null;            // the ask currently in #v2-plan-ask
    let feedTextEl = null;         // current accumulating prose paragraph in the feed
    let runningTools = {};         // tool name -> stack of running card elements
    let feedHadContent = false;    // did this turn surface anything in the feed?
    let capturedCampaignId = null; // best-effort id scraped from tool results

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

    // ── status / error helpers ────────────────────────────────────────
    function showThinking(on) { const t = $('v2-plan-thinking'); if (t) t.classList.toggle('hidden', !on); }
    function errorVisible() { const e = $('v2-plan-error'); return !!e && !e.classList.contains('hidden'); }
    function showPlanError(msg) {
        const e = $('v2-plan-error'); if (!e) return;
        e.textContent = msg; e.classList.remove('hidden');
        showThinking(false);
    }
    function hidePlanError() { const e = $('v2-plan-error'); if (e) e.classList.add('hidden'); }

    function clearAsk() { const m = $('v2-plan-ask'); if (m) m.innerHTML = ''; }
    function resetSummary() {
        const list = $('v2-plan-summary');
        if (list) list.innerHTML = '<div class="v2-plan-side-empty">The plan will take shape here as Gently designs it.</div>';
    }

    // ── activity feed (claude.ai-style collapsible tool cards) ─────────
    function feedEl() { return $('v2-plan-activity'); }
    function clearActivity() {
        const f = feedEl(); if (f) f.innerHTML = '';
        feedTextEl = null; runningTools = {}; feedHadContent = false;
        capturedCampaignId = null;
        hidePlanError();
    }
    function scrollFeedIfNearBottom() {
        const m = document.querySelector('.v2-screen-plan .v2-plan-main');
        if (m && (m.scrollHeight - m.scrollTop - m.clientHeight) < 140) m.scrollTop = m.scrollHeight;
    }
    function clearFallback() { feedEl()?.querySelectorAll('.v2-plan-fallback').forEach(n => n.remove()); }

    // Render the agent's prose like the chat does (reuses AgentChat.mdToHtml —
    // escapes then renders bold/italic/code/line-breaks), so the feed isn't raw
    // markdown. Falls back to escaped text if the helper isn't available.
    function renderMd(s) {
        if (typeof AgentChat !== 'undefined' && AgentChat.mdToHtml) return AgentChat.mdToHtml(s);
        const esc = (typeof escapeHtml === 'function') ? escapeHtml(String(s)) : String(s);
        return esc.replace(/\n/g, '<br>');
    }

    // Plan-writing tools → refresh THE PLAN panel during the turn (debounced),
    // not only at turn_end (ask_user_choice pauses the turn before it ends).
    const PLAN_TOOLS = new Set([
        'create_campaign', 'create_plan_item', 'link_plan_items', 'update_plan_item',
        'delete_plan_item', 'propose_plan', 'get_plan_status', 'validate_plan',
    ]);
    let planRefreshTimer = null;
    function schedulePlanRefresh() {
        if (planRefreshTimer) clearTimeout(planRefreshTimer);
        planRefreshTimer = setTimeout(() => { planRefreshTimer = null; refreshPlanPanel(); }, 600);
    }

    function safeStringify(v) {
        try {
            const s = (typeof v === 'string') ? v : JSON.stringify(v, null, 2);
            return s.length > 4000 ? s.slice(0, 4000) + '\n…' : s;
        } catch (e) { return String(v); }
    }
    function fillToolBody(body, act) {
        body.innerHTML = '';
        // grid-template-rows reveal (landing.css) needs ONE collapsible child —
        // append blocks into a single inner wrapper, not directly onto body.
        const inner = document.createElement('div');
        body.appendChild(inner);
        const inputStr = (act.input != null) ? safeStringify(act.input) : '';
        const full = act.full || act.summary || '';
        const block = (label, text) => {
            const l = document.createElement('div'); l.className = 'v2-act-block-label'; l.textContent = label;
            const b = document.createElement('pre'); b.className = 'v2-act-block'; b.textContent = text;
            inner.append(l, b);
        };
        if (inputStr) block('input', inputStr);
        if (full) block('result', full);
        if (!inputStr && !full) {
            const e = document.createElement('div'); e.className = 'v2-act-block-label'; e.textContent = 'no details';
            inner.append(e);
        }
    }
    function buildToolCard(act, done) {
        const card = document.createElement('div');
        card.className = 'v2-act-tool' + (done ? (act.is_error ? ' done err' : ' done') : '');
        const head = document.createElement('button');
        head.className = 'v2-act-tool-head'; head.type = 'button';
        head.setAttribute('aria-expanded', 'false');
        const ic = document.createElement('span'); ic.className = 'v2-act-ic';
        ic.innerHTML = done ? (act.is_error ? '⚠' : '✓') : '<span class="v2-act-spin"></span>';
        const label = document.createElement('span'); label.className = 'v2-act-label';
        label.textContent = act.label || act.name || 'tool';
        const sum = document.createElement('span'); sum.className = 'v2-act-summary';
        sum.textContent = done ? (act.summary || '') : '';
        const chev = document.createElement('span'); chev.className = 'v2-act-chev'; chev.textContent = '›';
        head.append(ic, label, sum, chev);
        const body = document.createElement('div'); body.className = 'v2-act-tool-body';
        fillToolBody(body, act);
        head.addEventListener('click', () => {
            const open = card.classList.toggle('open');
            head.setAttribute('aria-expanded', open ? 'true' : 'false');
        });
        card.append(head, body);
        return card;
    }
    function updateToolCard(card, act) {
        card.classList.add('done');
        if (act.is_error) card.classList.add('err');
        const ic = card.querySelector('.v2-act-ic'); if (ic) ic.textContent = act.is_error ? '⚠' : '✓';
        const sum = card.querySelector('.v2-act-summary'); if (sum) sum.textContent = act.summary || '';
        const body = card.querySelector('.v2-act-tool-body'); if (body) fillToolBody(body, act);
    }
    function captureCampaignId(text) {
        if (!text) return;
        const s = String(text);
        const m = s.match(/campaign_id[=:\s]+([0-9a-f]{6,})/i) || s.match(/\(id:\s*([0-9a-f]{6,})/i);
        if (m) capturedCampaignId = m[1];
    }

    function applyActivity(act) {
        if (!planActive() || !act) return;
        const f = feedEl(); if (!f) return;
        switch (act.kind) {
            case 'turn_start':
                feedTextEl = null; hidePlanError(); clearFallback(); showThinking(true);
                break;
            case 'thinking':
                showThinking(true);
                break;
            case 'text': {
                const chunk = act.text || '';
                if (!chunk) break;
                if (!feedTextEl) {
                    feedTextEl = document.createElement('div');
                    feedTextEl.className = 'v2-act-text';
                    feedTextEl._raw = '';
                    f.appendChild(feedTextEl);
                }
                feedTextEl._raw += chunk;
                feedTextEl.innerHTML = renderMd(feedTextEl._raw);
                feedHadContent = true; showThinking(true); scrollFeedIfNearBottom();
                break;
            }
            case 'tool_start': {
                // ask_user_choice IS the active question (rendered separately in
                // #v2-plan-ask) — don't also show it as a feed card.
                if (act.name === 'ask_user_choice') break;
                feedTextEl = null;
                const card = buildToolCard(act, false);
                f.appendChild(card);
                (runningTools[act.name] = runningTools[act.name] || []).push(card);
                feedHadContent = true; showThinking(true); scrollFeedIfNearBottom();
                break;
            }
            case 'tool_result': {
                captureCampaignId(act.summary);
                captureCampaignId(act.full);
                if (PLAN_TOOLS.has(act.name)) schedulePlanRefresh();
                if (act.name === 'ask_user_choice') break;
                feedTextEl = null;
                const arr = runningTools[act.name] || [];
                const card = arr.pop();
                if (card) updateToolCard(card, act);
                else f.appendChild(buildToolCard(act, true));
                feedHadContent = true; scrollFeedIfNearBottom();
                break;
            }
            case 'turn_end':
                showThinking(false); feedTextEl = null;
                refreshPlanPanel();
                if (!current && !feedHadContent) showFallback();
                break;
            case 'turn_error':
                showPlanError(act.error || 'Something went wrong — open the conversation for detail.');
                break;
        }
    }

    function showFallback() {
        const f = feedEl(); if (!f || f.querySelector('.v2-plan-fallback')) return;
        const d = document.createElement('div');
        d.className = 'v2-plan-fallback';
        d.innerHTML = 'Gently replied in prose — <a>open the conversation</a> to read it.';
        d.querySelector('a').addEventListener('click', openChat);
        f.appendChild(d);
    }

    // ── THE PLAN panel: mirror the real campaign tree ──────────────────
    async function refreshPlanPanel() {
        try {
            let tree = null;
            if (capturedCampaignId) {
                const r = await fetch(`/api/campaigns/${encodeURIComponent(capturedCampaignId)}/tree`);
                if (r.ok) tree = await r.json();
            }
            if (!tree) {
                const r = await fetch('/api/campaigns');
                if (r.ok) { const d = await r.json(); tree = (d.campaigns || [])[0] || null; }
            }
            if (tree) renderPlanTree(tree);
        } catch (e) { /* keep whatever is shown */ }
    }
    function planName(c) {
        c = c || {};
        return c.shorthand || c.display_name || c.description || 'Plan';
    }
    function renderPlanTree(tree) {
        const list = $('v2-plan-summary');
        if (!list || !tree) return;
        const phases = tree.children || [];
        const rootItems = tree.items || [];
        if (!phases.length && !rootItems.length) return;  // nothing to show yet — keep placeholder
        list.innerHTML = '';
        const title = document.createElement('div');
        title.className = 'v2-plan-title-row';
        title.textContent = planName(tree.campaign);
        list.appendChild(title);
        const addTask = (parent, it) => {
            const t = document.createElement('div');
            t.className = 'v2-plan-task';
            t.textContent = it.title || it.shorthand || '(task)';
            parent.appendChild(t);
        };
        rootItems.forEach(it => addTask(list, it));
        phases.forEach(phase => {
            if (!phase) return;
            const wrap = document.createElement('div');
            wrap.className = 'v2-plan-phase';
            const h = document.createElement('div');
            h.className = 'v2-plan-phase-h';
            h.textContent = planName(phase.campaign);
            wrap.appendChild(h);
            (phase.items || []).forEach(it => addTask(wrap, it));
            list.appendChild(wrap);
        });
    }

    // ── ask rendering (the active question) ────────────────────────────
    function labelFor(data, sel) {
        const opts = (data && data.options) || [];
        const one = (s) => {
            const o = opts.find(o => o && (o.id === s || o.value === s || o.label === s));
            return o ? o.label : String(s);
        };
        return Array.isArray(sel) ? sel.map(one).join(', ') : one(sel);
    }
    function recordPick(data, sel) {
        const list = $('v2-plan-summary');
        if (!list) return;
        const empty = list.querySelector('.v2-plan-side-empty');
        if (empty) empty.remove();
        const matched = (data && data.options || []).some(o => o && (o.id === sel || o.value === sel || o.label === sel));
        const row = document.createElement('div');
        row.className = 'v2-plan-row' + (matched ? '' : ' v2-plan-row-freetext');
        row.innerHTML = '<span class="k"></span><span class="v"></span>';
        row.querySelector('.k').textContent = (data && data.question) || 'Choice';
        row.querySelector('.v').textContent = labelFor(data, sel);
        list.appendChild(row);
    }
    function renderAsk() {
        const mount = $('v2-plan-ask');
        if (!mount || !current || typeof AgentChat === 'undefined' || !AgentChat.buildAskCard) return;
        showThinking(false); hidePlanError(); clearFallback();
        const data = current.data, reqId = current.reqId;
        const hasControl = AgentChat.hasControl ? AgentChat.hasControl() : true;
        const card = AgentChat.buildAskCard(data, {
            reqId, isWake: current.isWake, hasControl,
            onPick: (sel) => {
                recordPick(data, sel);
                AgentChat.answerChoice(reqId, sel);
                current = null; clearAsk(); showThinking(true);
            },
        });
        mount.innerHTML = '';
        mount.appendChild(card);
        const first = mount.querySelector('button:not([disabled])');
        if (first) setTimeout(() => first.focus(), 30);
    }

    let planKickedOff = false;   // guard: design-kickoff fires once per session
    async function startPlan() {
        setScreen('plan');
        // Re-entering the wizard (Back → Plan again) must NOT re-fire the
        // kickoff — that stacked duplicate "/plan" + design turns. Just show
        // the wizard with its existing state.
        if (planKickedOff) return;
        planKickedOff = true;
        resetSummary(); clearAsk(); clearActivity();
        current = null;
        showThinking(true);
        // Campaigns are persistent agent memory (not session state), so the
        // agent always builds on an existing one — which leaves a user wanting a
        // fresh plan stuck. So if an active campaign exists, ask up front:
        // continue it (the default) or start a brand-new one. With NO campaign
        // there's nothing to continue, so skip the gate and design straight away
        // (that path is fresh anyway).
        let campaign = null;
        try {
            const r = await fetch('/api/campaigns');
            if (r.ok) { const d = await r.json(); campaign = (d.campaigns || [])[0] || null; }
        } catch (e) { /* offline / no API — just design */ }
        if (campaign) renderCampaignChoice(campaign);
        else kickoffDesign('continue');
    }

    // Enter plan mode, then prompt design. The prompt differs by intent: build
    // on the active campaign, or set it aside and create a new one. A free-typed
    // answer from the choice card becomes the design brief directly.
    function kickoffDesign(mode) {
        showThinking(true);
        if (typeof AgentChat === 'undefined' || !AgentChat.runCommand) return;
        AgentChat.runCommand('/plan');
        if (mode === 'fresh') {
            AgentChat.runCommand(
                "I want to start a brand-new experiment, not continue any existing " +
                "campaign. Create a new campaign and let's design it from scratch — " +
                "what should we capture?"
            );
        } else if (mode === 'continue') {
            AgentChat.runCommand("Let's design this run — what should it capture?");
        } else {
            // free text from the choice card's "Something else…" escape
            AgentChat.runCommand(String(mode));
        }
    }

    // Continue-vs-fresh gate, shown only when an active campaign exists. Reuses
    // the agent ask-card styling so it's visually identical to the agent's own
    // questions; picking routes into kickoffDesign rather than the agent bridge.
    function renderCampaignChoice(tree) {
        const mount = $('v2-plan-ask');
        if (!mount || typeof AgentChat === 'undefined' || !AgentChat.buildAskCard) {
            kickoffDesign('continue');
            return;
        }
        showThinking(false); hidePlanError(); clearFallback();
        const name = planName((tree && tree.campaign) || {});
        const data = {
            question: `You have an active campaign — **${name}**. Design the next run inside it, or start something new?`,
            options: [
                { id: 'continue', label: `Continue ${name}`, description: 'Design the next run inside your existing campaign' },
                { id: 'fresh', label: 'Start a brand-new campaign', description: 'Set the existing plan aside and plan from scratch' },
            ],
        };
        const hasControl = AgentChat.hasControl ? AgentChat.hasControl() : true;
        const card = AgentChat.buildAskCard(data, {
            reqId: 'landing-campaign-choice', isWake: false, hasControl,
            onPick: (sel) => { clearAsk(); kickoffDesign(sel); },
        });
        mount.innerHTML = '';
        mount.appendChild(card);
        const first = mount.querySelector('button:not([disabled])');
        if (first) setTimeout(() => first.focus(), 30);
    }

    function openScope() {
        dismiss();
        if (typeof switchTab === 'function') switchTab('devices');
    }
    function openChat() {
        dismiss();
        if (typeof AgentChat !== 'undefined' && AgentChat.togglePanel) {
            setTimeout(() => AgentChat.togglePanel(true), 300);
        }
    }
    function sendFreeform(text) {
        const v = (text || '').trim();
        dismiss();
        if (typeof AgentChat !== 'undefined' && AgentChat.togglePanel) {
            AgentChat.togglePanel(true);
            if (v && AgentChat.runCommand) setTimeout(() => AgentChat.runCommand(v), 300);
        }
    }

    function init() {
        el = $('v2-landing');
        if (!el || typeof ClientEventBus === 'undefined') return;  // flag off → no-op
        greet();

        el.querySelectorAll('[data-landing]').forEach(btn => btn.addEventListener('click', () => {
            const kind = btn.dataset.landing;
            if (kind === 'plan') startPlan();
            else if (kind === 'standalone') openScope();
        }));

        const esc = $('v2-escape'), escToggle = $('v2-escape-toggle'),
            escInput = $('v2-escape-input'), escSend = $('v2-escape-send');
        if (escToggle && esc && escInput) {
            escToggle.addEventListener('click', () => {
                const open = esc.classList.toggle('open');
                escToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
                if (open) setTimeout(() => escInput.focus(), 120);
            });
            const submit = () => sendFreeform(escInput.value);
            if (escSend) escSend.addEventListener('click', submit);
            escInput.addEventListener('keydown', e => {
                if (e.key === 'Enter') { e.preventDefault(); submit(); }
                else if (e.key === 'Escape') { e.stopPropagation(); esc.classList.remove('open'); escToggle.setAttribute('aria-expanded', 'false'); }
            });
        }

        const skip = $('v2-landing-skip');
        if (skip) skip.addEventListener('click', dismiss);

        const back = $('v2-plan-back');
        if (back) back.addEventListener('click', () => setScreen('welcome'));
        const planChat = $('v2-plan-chat');
        if (planChat) planChat.addEventListener('click', openChat);
        const cont = $('v2-plan-continue');
        if (cont) cont.addEventListener('click', dismiss);

        // The agent's questions + work render in the plan stage while it's active;
        // once we've receded into the workspace, AskStage (#ask-stage) takes over.
        ClientEventBus.on('AGENT_ASK', ({ request_id, choice_data, origin }) => {
            if (!planActive()) return;
            current = { reqId: request_id, data: choice_data || {}, isWake: origin === 'wake' };
            renderAsk();
        });
        ClientEventBus.on('ASK_CLEARED', ({ request_id }) => {
            if (request_id === '*' || (current && request_id === current.reqId)) {
                current = null; clearAsk();
                if (planActive() && !errorVisible()) showThinking(true);
            }
        });
        ClientEventBus.on('AGENT_CONTROL', () => { if (current && planActive()) renderAsk(); });
        ClientEventBus.on('AGENT_ACTIVITY', (act) => applyActivity(act));

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
