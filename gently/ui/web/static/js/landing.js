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
    let feedThinkingEl = null;     // current accumulating reasoning (thinking) block
    let runningTools = {};         // tool name -> stack of running card elements
    let feedHadContent = false;    // did this turn surface anything in the feed?
    let capturedCampaignId = null; // best-effort id scraped from tool results
    let planProposed = false;      // propose_plan ran → plan is ready to commit

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
    function setThinkingLabel(text) {
        const l = document.querySelector('#v2-plan-thinking .v2-plan-thinking-label');
        if (l && text) l.textContent = text;
    }
    // Elapsed-time counter so a long think reads as progress, not a hang. Starts
    // when the thinking indicator first shows and runs until it's hidden (turn end).
    let _thinkTimer = null;
    let _thinkStart = 0;
    function _thinkTick() {
        const t = $('v2-plan-thinking');
        if (!t) return;
        let el = t.querySelector('.v2-plan-elapsed');
        if (!el) {
            el = document.createElement('span');
            el.className = 'v2-plan-elapsed';
            el.style.cssText = 'margin-left:6px;opacity:.55;font-variant-numeric:tabular-nums;';
            t.appendChild(el);
        }
        const s = Math.round((Date.now() - _thinkStart) / 1000);
        el.textContent = s > 0 ? s + 's' : '';
    }
    function showThinking(on, label) {
        const t = $('v2-plan-thinking');
        if (t) t.classList.toggle('hidden', !on);
        if (on && label) setThinkingLabel(label);
        if (on) {
            if (!_thinkTimer) {
                _thinkStart = Date.now();
                _thinkTick();
                _thinkTimer = setInterval(_thinkTick, 1000);
            }
        } else if (_thinkTimer) {
            clearInterval(_thinkTimer);
            _thinkTimer = null;
            const el = t && t.querySelector('.v2-plan-elapsed');
            if (el) el.textContent = '';
        }
    }
    // Human-readable "what's happening right now" from a tool activity event,
    // so the status line names the live operation instead of a static string.
    function prettyTool(act) {
        const raw = (act && (act.label || act.name)) || 'the next step';
        const s = String(raw).replace(/_/g, ' ').trim();
        return s.charAt(0).toUpperCase() + s.slice(1) + '…';
    }
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
        planPage = 0; planPages = []; planTitleText = '';
    }

    // ── activity feed: paginated, ONE agent step (turn) per page ───────
    // Instead of one ever-growing scroll, each agent turn — its reasoning +
    // the tool calls it made — is a page you flip through with ‹ Prev / Next ›.
    // The current question stays pinned below the feed (#v2-plan-ask). A new
    // turn auto-advances to its page; you can flip back to review earlier steps.
    let feedPages = [];        // .v2-act-page elements, one per turn
    let feedPage = 0;          // index currently shown
    let curPageEl = null;      // page receiving this turn's content
    let pendingNewPage = false; // a turn started; open a fresh page on first content

    function feedEl() { return $('v2-plan-activity'); }
    function feedPagesWrap() { return feedEl()?.querySelector('.v2-feed-pages'); }
    function clearActivity() {
        const f = feedEl();
        if (f) {
            f.innerHTML =
                '<div class="v2-plan-pager v2-feed-pager-bar" hidden></div>' +
                '<div class="v2-feed-pages"></div>' +
                '<div class="v2-plan-dots v2-feed-dots" hidden></div>';
        }
        feedPages = []; feedPage = 0; curPageEl = null; pendingNewPage = false;
        feedTextEl = null; feedThinkingEl = null; runningTools = {}; feedHadContent = false;
        capturedCampaignId = null; planProposed = false;
        clearPlanReady();
        hidePlanError();
    }
    function newFeedPage() {
        const wrap = feedPagesWrap(); if (!wrap) return null;
        const page = document.createElement('div');
        page.className = 'v2-act-page';
        wrap.appendChild(page);
        feedPages.push(page);
        curPageEl = page;
        feedPage = feedPages.length - 1;   // auto-advance to the live step
        feedTextEl = null;
        drawFeedPager();
        return page;
    }
    // Where this turn's prose/tool cards land. Opens a fresh page the first time
    // content arrives after a turn_start (so content-less command turns don't
    // leave empty pages), and lazily on the very first content.
    function feedTarget() {
        if (pendingNewPage || !curPageEl) { newFeedPage(); pendingNewPage = false; }
        return curPageEl;
    }
    function viewingLatest() { return feedPage >= feedPages.length - 1; }
    function drawFeedPager() {
        const f = feedEl(); if (!f) return;
        const n = feedPages.length;
        const i = Math.min(Math.max(feedPage, 0), Math.max(n - 1, 0));
        feedPages.forEach((p, idx) => p.classList.toggle('active', idx === i));
        const bar = f.querySelector('.v2-feed-pager-bar');
        const dots = f.querySelector('.v2-feed-dots');
        if (!bar || !dots) return;
        if (n <= 1) { bar.hidden = true; dots.hidden = true; return; }
        bar.hidden = false; dots.hidden = false;
        bar.innerHTML = '';
        const mkBtn = (txt, disabled, fn) => {
            const b = document.createElement('button');
            b.className = 'v2-plan-pager-btn'; b.type = 'button'; b.textContent = txt;
            b.disabled = disabled; b.addEventListener('click', fn);
            return b;
        };
        const pos = document.createElement('span');
        pos.className = 'v2-plan-pager-pos'; pos.textContent = `Step ${i + 1} of ${n}`;
        bar.append(
            mkBtn('‹ Prev', i === 0, () => { if (feedPage > 0) { feedPage--; drawFeedPager(); } }),
            pos,
            mkBtn('Next ›', i === n - 1, () => { if (feedPage < n - 1) { feedPage++; drawFeedPager(); } }),
        );
        dots.innerHTML = '';
        for (let d = 0; d < n; d++) {
            const dot = document.createElement('button');
            dot.className = 'v2-plan-dot' + (d === i ? ' active' : '');
            dot.type = 'button';
            dot.setAttribute('aria-label', `Step ${d + 1} of ${n}`);
            dot.addEventListener('click', () => { feedPage = d; drawFeedPager(); });
            dots.appendChild(dot);
        }
    }
    function scrollFeedIfNearBottom() {
        if (!viewingLatest()) return;   // don't yank the user off an earlier step
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
                feedTextEl = null; feedThinkingEl = null; pendingNewPage = true; hidePlanError(); clearFallback();
                clearPlanReady();   // new work in flight — drop any "ready" state
                showThinking(true, 'reviewing your campaign and plan…');
                break;
            case 'thinking': {
                // Stream the model's reasoning summary live into the feed as a dim
                // block, so the wait shows what the agent is actually considering.
                showThinking(true);
                const chunk = act.text || '';
                if (!chunk) { setThinkingLabel('thinking through the next step…'); break; }
                if (!feedThinkingEl) {
                    feedThinkingEl = document.createElement('div');
                    feedThinkingEl.className = 'v2-act-think';
                    feedThinkingEl.style.cssText =
                        'font-style:italic;opacity:.7;white-space:pre-wrap;margin:2px 0 8px;font-size:12.5px;line-height:1.5;';
                    feedThinkingEl._raw = '';
                    feedTarget().appendChild(feedThinkingEl);
                }
                feedThinkingEl._raw += chunk;
                feedThinkingEl.textContent = feedThinkingEl._raw;
                feedHadContent = true;
                setThinkingLabel('reasoning…');
                scrollFeedIfNearBottom();
                break;
            }
            case 'text': {
                const chunk = act.text || '';
                if (!chunk) break;
                // The reasoning that immediately precedes the spoken answer is
                // wrap-up meta ("let me wrap this up concisely and offer to
                // export…") — drop the block entirely so the feed keeps the
                // answer, not the narration of getting there. Reasoning that
                // precedes a TOOL is left in place (tool_start only nulls the
                // pointer) as the rationale for that action.
                if (feedThinkingEl) { feedThinkingEl.remove(); feedThinkingEl = null; }
                if (!feedTextEl) {
                    feedTextEl = document.createElement('div');
                    feedTextEl.className = 'v2-act-text';
                    feedTextEl._raw = '';
                    feedTarget().appendChild(feedTextEl);
                }
                feedTextEl._raw += chunk;
                feedTextEl.innerHTML = renderMd(feedTextEl._raw);
                feedHadContent = true; showThinking(true, 'composing the response…'); scrollFeedIfNearBottom();
                break;
            }
            case 'tool_start': {
                // ask_user_choice IS the active question (rendered separately in
                // #v2-plan-ask) — don't also show it as a feed card.
                if (act.name === 'ask_user_choice') break;
                feedTextEl = null; feedThinkingEl = null;
                const card = buildToolCard(act, false);
                feedTarget().appendChild(card);
                (runningTools[act.name] = runningTools[act.name] || []).push(card);
                feedHadContent = true; showThinking(true, prettyTool(act)); scrollFeedIfNearBottom();
                break;
            }
            case 'tool_result': {
                captureCampaignId(act.summary);
                captureCampaignId(act.full);
                if (PLAN_TOOLS.has(act.name)) schedulePlanRefresh();
                if (act.name === 'propose_plan' && !act.is_error) planProposed = true;
                if (act.name === 'ask_user_choice') break;
                feedTextEl = null; feedThinkingEl = null;
                const arr = runningTools[act.name] || [];
                const card = arr.pop();
                if (card) updateToolCard(card, act);
                else feedTarget().appendChild(buildToolCard(act, true));
                feedHadContent = true; setThinkingLabel('working through the next step…'); scrollFeedIfNearBottom();
                break;
            }
            case 'turn_end':
                showThinking(false); feedTextEl = null; feedThinkingEl = null;
                refreshPlanPanel();
                if (!current && !feedHadContent) showFallback();
                // Plan proposed and the agent has settled (no pending question) →
                // the design is done. Surface a clear "ready" state instead of
                // leaving the user parked on the last wizard step.
                if (planProposed && !current) showPlanReady();
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
        feedTarget().appendChild(d);
    }

    // ── plan-ready state ───────────────────────────────────────────────
    // Once the agent has proposed the plan and gone quiet, the wizard is done.
    // Mark the screen "ready": rename the header, count phases/items from the
    // panel, and promote "open workspace" to the obvious primary action — so the
    // finish line is signposted instead of looking like one more wizard step.
    function planCounts() {
        let phases = 0, items = 0;
        planPages.forEach(p => {
            if (p.name !== 'Tasks') phases++;
            items += (p.items || []).length;
        });
        return { phases, items };
    }
    function showPlanReady() {
        const sec = document.querySelector('.v2-screen-plan');
        if (!sec) return;
        sec.classList.add('ready');
        showThinking(false);
        const who = sec.querySelector('.v2-plan-who');
        const title = sec.querySelector('.v2-plan-title');
        if (who) who.textContent = 'Gently · plan ready';
        if (title) {
            const { phases, items } = planCounts();
            title.textContent = items
                ? `Your plan is ready — ${items} item${items === 1 ? '' : 's'} across ${phases} phase${phases === 1 ? '' : 's'}`
                : 'Your plan is ready';
        }
        const cont = $('v2-plan-continue');
        if (cont) cont.textContent = 'Open the workspace ›';
        const exp = $('v2-plan-export');
        if (exp) exp.hidden = false;   // the plan is final → offer the download
    }
    function clearPlanReady() {
        const sec = document.querySelector('.v2-screen-plan');
        if (!sec || !sec.classList.contains('ready')) return;
        sec.classList.remove('ready');
        const who = sec.querySelector('.v2-plan-who');
        const title = sec.querySelector('.v2-plan-title');
        if (who) who.textContent = 'Gently · planning';
        if (title) title.textContent = "Let's design your run";
        const cont = $('v2-plan-continue');
        if (cont) cont.textContent = 'Continue in workspace ›';
        const exp = $('v2-plan-export');
        if (exp) exp.hidden = true;
    }

    // ── export the finished plan as a shareable markdown doc ────────────
    // Replaces the agent's end-of-plan "want me to export this?" prose with a
    // real action: pull the enriched plan tree (/export) and render it to
    // markdown client-side so the biologist can drop it in a doc or share it.
    function specToLines(spec) {
        let s = spec;
        if (typeof s === 'string') { try { s = JSON.parse(s); } catch { return s ? ['- ' + s] : []; } }
        if (!s || typeof s !== 'object') return [];
        const out = [];
        const fmt = (v) => Array.isArray(v) ? v.join(', ') : (typeof v === 'object' ? JSON.stringify(v) : String(v));
        const pick = (k, label) => { if (s[k] != null && s[k] !== '') out.push(`- **${label}:** ${fmt(s[k])}`); };
        pick('strain', 'Strain'); pick('goal', 'Goal');
        if (Array.isArray(s.channels) && s.channels.length) {
            out.push('- **Channels:** ' + s.channels.map(c => `${c.name || '?'} (${c.excitation_nm || '?'} nm${c.exposure_ms ? `, ${c.exposure_ms} ms` : ''})`).join(', '));
        }
        pick('num_slices', 'Slices'); pick('interval_s', 'Interval (s)'); pick('temperature_c', 'Temperature (°C)');
        pick('num_embryos', 'Embryos'); pick('start_stage', 'Start stage'); pick('stop_condition', 'Stop condition');
        pick('criteria', 'Criteria'); pick('success_criteria', 'Success criteria');
        return out;
    }
    function buildPlanMarkdown(tree) {
        const L = [];
        L.push(`# ${tree.description || tree.shorthand || 'Experimental plan'}`, '');
        if (tree.target) L.push(`**Goal:** ${tree.target}`, '');
        if (tree.shorthand) L.push(`**Plan ID:** \`${tree.shorthand}\``, '');
        L.push(`_Exported from Gently — ${new Date().toLocaleString()}_`, '');
        const renderItems = (items, prefix) => {
            (items || []).slice().sort((a, b) => (a.phase_order || 0) - (b.phase_order || 0)).forEach((it, idx) => {
                L.push(`### ${prefix}${idx + 1} ${it.title || '(task)'}  \`${it.type || 'task'}\``, '');
                if (it.description) L.push(it.description, '');
                const sl = specToLines(it.spec);
                if (sl.length) L.push(...sl, '');
                const refs = it.references || [];
                if (refs.length) {
                    L.push('**References:**');
                    refs.forEach((r, i) => L.push(`${i + 1}. ${r.citation || r.id || ''}${r.source ? ` _(${r.source})_` : ''}`));
                    L.push('');
                }
            });
        };
        if ((tree.items || []).length) { L.push('## Tasks', ''); renderItems(tree.items, ''); }
        (tree.children || []).forEach((ph, pi) => {
            if (!ph) return;
            L.push(`## ${ph.display_name || ph.description || ph.shorthand || `Phase ${pi + 1}`}`, '');
            if (ph.target) L.push(ph.target, '');
            renderItems(ph.items, `${pi + 1}.`);
        });
        return L.join('\n').replace(/\n{3,}/g, '\n\n').trimEnd() + '\n';
    }
    async function resolveCampaignId() {
        if (capturedCampaignId) return capturedCampaignId;
        try {
            const r = await fetch('/api/campaigns');
            if (r.ok) { const d = await r.json(); const t = (d.campaigns || [])[0]; return (t && t.campaign && t.campaign.id) || null; }
        } catch (e) { /* offline */ }
        return null;
    }
    async function exportPlan() {
        const btn = $('v2-plan-export');
        const id = await resolveCampaignId();
        if (!id) { showPlanError('No plan to export yet.'); return; }
        if (btn) { btn.disabled = true; btn.textContent = '↓ Exporting…'; }
        try {
            const r = await fetch(`/api/campaigns/${encodeURIComponent(id)}/export`);
            if (!r.ok) throw new Error(`export ${r.status}`);
            const tree = await r.json();
            const md = buildPlanMarkdown(tree);
            const blob = new Blob([md], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${(tree.shorthand || 'plan').replace(/[^\w.-]+/g, '_')}.md`;
            document.body.appendChild(a); a.click(); a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch (e) {
            showPlanError('Could not export the plan — open the conversation to export it manually.');
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = '↓ Export plan'; }
        }
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
    // Phases read better by their human name ("Phase 1 — Reporter validation")
    // than by their code shorthand ("nrp-p1"), which looks like a machine id.
    function phaseName(c) {
        c = c || {};
        return c.display_name || c.description || c.shorthand || 'Phase';
    }
    // THE PLAN renders as a pager — one phase per page with ‹ Prev / Next ›,
    // a position label, and dots — instead of one long scroll. planPage is held
    // across re-renders (the panel refetches on every plan-writing tool) so the
    // page you're reading doesn't snap back to the start mid-design.
    let planPage = 0;
    let planPages = [];     // [{ name, items }]
    let planTitleText = '';

    function renderPlanTree(tree) {
        if (!tree) return;
        const phases = tree.children || [];
        const rootItems = tree.items || [];
        if (!phases.length && !rootItems.length) return;  // nothing to show yet — keep placeholder
        const pages = [];
        if (rootItems.length) pages.push({ name: 'Tasks', items: rootItems });
        phases.forEach(phase => {
            if (!phase) return;
            pages.push({ name: phaseName(phase.campaign), items: phase.items || [] });
        });
        planPages = pages;
        planTitleText = planName(tree.campaign);
        if (planPage >= pages.length) planPage = pages.length - 1;
        if (planPage < 0) planPage = 0;
        drawPlanPage();
    }

    function drawPlanPage() {
        const list = $('v2-plan-summary');
        if (!list) return;
        const pages = planPages;
        const n = pages.length;
        if (!n) return;
        const i = Math.min(Math.max(planPage, 0), n - 1);
        const page = pages[i];

        list.innerHTML = '';
        const title = document.createElement('div');
        title.className = 'v2-plan-title-row';
        title.textContent = planTitleText;
        list.appendChild(title);

        if (n > 1) {
            const bar = document.createElement('div');
            bar.className = 'v2-plan-pager';
            const prev = document.createElement('button');
            prev.className = 'v2-plan-pager-btn'; prev.type = 'button'; prev.textContent = '‹ Prev';
            prev.disabled = i === 0;
            prev.addEventListener('click', () => { if (planPage > 0) { planPage--; drawPlanPage(); } });
            const pos = document.createElement('span');
            pos.className = 'v2-plan-pager-pos';
            pos.textContent = page.name;   // position shown by the dots below
            pos.title = `${page.name} · ${i + 1} of ${n}`;
            const next = document.createElement('button');
            next.className = 'v2-plan-pager-btn'; next.type = 'button'; next.textContent = 'Next ›';
            next.disabled = i === n - 1;
            next.addEventListener('click', () => { if (planPage < n - 1) { planPage++; drawPlanPage(); } });
            bar.append(prev, pos, next);
            list.appendChild(bar);
        } else {
            const h = document.createElement('div');
            h.className = 'v2-plan-phase-h';
            h.textContent = page.name;
            list.appendChild(h);
        }

        const tasks = document.createElement('div');
        tasks.className = 'v2-plan-phase';
        const items = page.items || [];
        // phase ordinal (1-based) for "P.I" numbering; the rootItems "Tasks" page
        // isn't a phase, so it numbers items bare (1, 2, …).
        const phaseOrd = pages.slice(0, i + 1).filter(p => p.name !== 'Tasks').length;
        if (items.length) {
            items.forEach((it, idx) => {
                const type = String(it.type || '').toLowerCase();
                const t = document.createElement('div');
                t.className = 'v2-plan-task type-' + (type || 'other');
                const num = document.createElement('span');
                num.className = 'v2-task-num';
                num.textContent = phaseOrd ? `${phaseOrd}.${idx + 1}` : `${idx + 1}`;
                const dot = document.createElement('span');
                dot.className = 'v2-task-dot';
                dot.title = type || 'task';
                const ttl = document.createElement('span');
                ttl.className = 'v2-task-ttl';
                ttl.textContent = it.title || it.shorthand || '(task)';
                t.append(num, dot, ttl);
                if (it.estimated_days) {
                    const d = document.createElement('span');
                    d.className = 'v2-task-days';
                    d.textContent = `${it.estimated_days}d`;
                    t.append(d);
                }
                tasks.appendChild(t);
            });
        } else {
            const e = document.createElement('div');
            e.className = 'v2-plan-task v2-plan-task-empty';
            e.textContent = 'no items in this phase yet';
            tasks.appendChild(e);
        }
        list.appendChild(tasks);

        if (n > 1) {
            const dots = document.createElement('div');
            dots.className = 'v2-plan-dots';
            for (let d = 0; d < n; d++) {
                const dot = document.createElement('button');
                dot.className = 'v2-plan-dot' + (d === i ? ' active' : '');
                dot.type = 'button';
                dot.setAttribute('aria-label', `Go to ${pages[d].name} (${d + 1} of ${n})`);
                dot.addEventListener('click', () => { planPage = d; drawPlanPage(); });
                dots.appendChild(dot);
            }
            list.appendChild(dots);
        }
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

        // Theme toggle (header's is occluded by the overlay). Mirrors
        // _header.html: flip data-theme on both roots + persist.
        const themeBtn = $('v2-landing-theme');
        if (themeBtn) themeBtn.addEventListener('click', () => {
            const cur = document.documentElement.getAttribute('data-theme')
                || document.body.getAttribute('data-theme') || 'light';
            const next = cur === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            document.body.setAttribute('data-theme', next);
            localStorage.setItem('gently-theme', next);
        });

        const back = $('v2-plan-back');
        if (back) back.addEventListener('click', () => setScreen('welcome'));
        const planChat = $('v2-plan-chat');
        if (planChat) planChat.addEventListener('click', openChat);
        const cont = $('v2-plan-continue');
        if (cont) cont.addEventListener('click', dismiss);
        const exp = $('v2-plan-export');
        if (exp) exp.addEventListener('click', exportPlan);

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
                // A question was just answered — the agent's continuation is the
                // next step, so open a fresh feed page for it. (A turn stays one
                // stream across an ask_user_choice pause, so turn_start alone
                // would lump every step of the design into a single page.)
                pendingNewPage = true;
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
