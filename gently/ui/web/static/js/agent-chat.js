/**
 * Floating agent-chat window — the web-side control surface.
 *
 * Connects to the same /ws/agent bridge the TUI uses, streams the agent's
 * responses, and renders interactive choice pickers. A single-driver control
 * lock on the server arbitrates who may drive the microscope; this client
 * shows a banner and offers "Take control" when another client holds it.
 *
 * Self-contained IIFE (no build step). All untrusted text is escaped before
 * insertion — never assign agent/user/tool strings to innerHTML directly.
 */
const AgentChat = (() => {
    let ws = null;
    let reconnectDelay = 1000;
    const MAX_DELAY = 30000;

    let panelOpen = false;
    let hasControl = true;        // optimistic until the server says otherwise
    let holderLabel = null;
    let streaming = false;
    let currentAgentEl = null;    // the agent content element being streamed into
    let activityEl = null;        // the persistent "working…" indicator (reused)
    let me = null;                // { authenticated, username, role, can_control }

    // Autocomplete: slash-command + @tool registries (pushed by the server on
    // connect) and the live dropdown state.
    let commands = [];            // [{name, description, aliases, ...}]
    let tools = [];               // [{name, description, params, ...}]
    let acItems = [];             // current completion items shown in the dropdown
    let acIdx = -1;               // highlighted item index
    let autonomousTurn = false;   // true while rendering an autonomous (wake) turn
    let agentBusy = false;        // a turn (user or autonomous) is currently running
    let busySource = null;        // 'user' | 'wake' while busy
    let msgQueue = [];            // messages typed while busy, sent on idle
    let queuePanel = null;        // the "⏳ Queued (N)" panel element
    let stopBtn = null;           // explicit Stop button (separate from Send)

    // DOM refs (resolved in init)
    let panel, log, input, sendBtn, conn, banner, closeBtn, userEl, signoutBtn;
    let toggleBtn, pinBtn, resizeEl, toggleDot, toggleBadge;  // docked-panel chrome
    let pendingSlot = null;       // sticky slot for ASK approval proposals
    let acComplete = null;        // the autocomplete dropdown element

    // ── Safe rendering ────────────────────────────────────────
    function escapeHtml(s) {
        const d = document.createElement('div');
        d.textContent = String(s == null ? '' : s);
        return d.innerHTML;
    }

    /** Minimal, safe markdown: escape first, then a few inline transforms. */
    function mdToHtml(text) {
        let html = escapeHtml(text);
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        html = html.replace(/\n/g, '<br>');
        return html;
    }

    // Pin-to-bottom autoscroll: only follow new content if the user is already
    // near the bottom; otherwise count unseen items and show a "↓ N new" pill so
    // a streaming agent never yanks the operator away from something they're reading.
    let stickBottom = true;
    let newCount = 0;
    let jumpPill = null;
    function nearBottom() { return (log.scrollHeight - log.scrollTop - log.clientHeight) < 60; }
    function renderJumpPill() {
        if (!jumpPill) return;
        if (!stickBottom && newCount > 0) {
            jumpPill.textContent = `↓ ${newCount} new`;
            jumpPill.classList.remove('hidden');
        } else {
            jumpPill.classList.add('hidden');
        }
    }
    function scrollToBottom(isNewItem = true) {
        if (stickBottom) { log.scrollTop = log.scrollHeight; }
        // Only count genuinely new items (bubbles/rows), not in-place streaming
        // text edits — otherwise the "N new" pill inflates per chunk.
        else { if (isNewItem) newCount += 1; renderJumpPill(); }
    }
    function jumpToBottom() {
        stickBottom = true; newCount = 0;
        log.scrollTop = log.scrollHeight;
        renderJumpPill();
    }

    // ── Activity indicator ────────────────────────────────────
    // A single reusable "the agent is working" row, always pinned to the
    // bottom of the log. This is the trust signal — something is happening.
    function setActivity(label) {
        if (!activityEl) {
            activityEl = document.createElement('div');
            activityEl.className = 'ac-activity';
            activityEl.innerHTML =
                '<span class="ac-dots"><i></i><i></i><i></i></span>' +
                '<span class="ac-activity-label"></span>';
        }
        activityEl.querySelector('.ac-activity-label').textContent = label;
        log.appendChild(activityEl);  // (re)pin to bottom
        scrollToBottom();
    }
    function hideActivity() {
        if (activityEl && activityEl.parentNode) activityEl.parentNode.removeChild(activityEl);
    }

    // ── Message elements ──────────────────────────────────────
    function addTurn(role) {
        const wrap = document.createElement('div');
        wrap.className = `ac-turn ac-turn-${role}`;
        if (role === 'agent' && autonomousTurn) wrap.classList.add('ac-turn-autonomous');
        if (role === 'agent') {
            const label = document.createElement('div');
            label.className = 'ac-role';
            label.textContent = autonomousTurn ? 'Gently · autonomous' : 'Gently';
            wrap.appendChild(label);
        }
        const content = document.createElement('div');
        content.className = 'ac-content';
        wrap.appendChild(content);
        log.appendChild(wrap);
        scrollToBottom();
        return content;
    }

    function addUserMessage(text, author) {
        const wrap = document.createElement('div');
        wrap.className = 'ac-turn ac-turn-user';
        if (author) {
            const label = document.createElement('div');
            label.className = 'ac-role ac-role-user';
            label.textContent = author;
            wrap.appendChild(label);
        }
        const content = document.createElement('div');
        content.className = 'ac-content';
        content.textContent = text;
        wrap.appendChild(content);
        log.appendChild(wrap);
        scrollToBottom();
    }

    /** Rebuild the transcript from a persisted/replayed history list. */
    function renderHistory(items) {
        log.innerHTML = '';
        currentAgentEl = null;
        activityEl = null;
        stickBottom = true; newCount = 0;  // a full rebuild jumps to latest
        (items || []).forEach(it => {
            if (it.role === 'user') {
                addUserMessage(it.text, it.author);
            } else if (it.role === 'agent') {
                const c = addTurn('agent');
                c._raw = it.text || '';
                c.innerHTML = mdToHtml(c._raw);
            } else if (it.role === 'autonomous_start') {
                addAutonomousBanner(it.trigger || '');
            } else if (it.role === 'autonomous') {
                autonomousTurn = true;
                const c = addTurn('agent');
                c._raw = it.text || '';
                c.innerHTML = mdToHtml(c._raw);
                autonomousTurn = false;
            } else if (it.role === 'tool') {
                const el = document.createElement('div');
                el.className = 'ac-tool ac-tool-done';
                const dur = it.duration ? ` · ${(it.duration.toFixed ? it.duration.toFixed(1) : it.duration)}s` : '';
                const summary = it.summary ? ` — ${escapeHtml(it.summary)}` : '';
                el.innerHTML = `<span class="ac-tool-check">✓</span><span class="ac-tool-name">${escapeHtml(it.name || 'tool')}</span><span class="ac-tool-meta">${dur}${summary}</span>`;
                log.appendChild(el);
            } else if (it.role === 'system') {
                addSystemLine(it.text, it.level || 'info');
            }
        });
        scrollToBottom();
    }

    /** A divider announcing the agent woke itself, with the trigger reason. */
    function addAutonomousBanner(trigger) {
        const el = document.createElement('div');
        el.className = 'ac-autonomous-banner';
        const t = trigger ? `Gently woke up — ${trigger}` : 'Gently woke up';
        el.innerHTML = `<span class="ac-autonomous-dot"></span><span>${escapeHtml(t)}</span>`;
        log.appendChild(el);
        scrollToBottom();
    }

    function addSystemLine(text, level = 'info') {
        const el = document.createElement('div');
        el.className = `ac-system ac-level-${level}`;
        el.textContent = text;
        log.appendChild(el);
        scrollToBottom();
    }

    // ── Protocol handlers ─────────────────────────────────────
    function handle(msg) {
        switch (msg.type) {
            case 'connected':
                reconnectDelay = 1000;
                setConn(true, msg.version ? `Connected · v${msg.version}` : 'Connected');
                // The bridge ships the command + tool registries on connect.
                // Capture them so the composer can offer autocomplete — the
                // data was always on the wire; we just never used it.
                commands = Array.isArray(msg.commands) ? msg.commands : [];
                tools = Array.isArray(msg.tools) ? msg.tools : [];
                break;

            case 'control_status':
                hasControl = !!msg.you_have_control;
                holderLabel = msg.holder_label || null;
                renderControl();
                break;

            case 'history':
                renderHistory(msg.items || []);
                break;

            case 'user_message':
                hideActivity();
                addUserMessage(msg.text, msg.author);
                break;

            case 'stream_start':
                streaming = true;
                currentAgentEl = null;  // created lazily on first text
                setBusy(true, 'user');
                setActivity('Working…');
                break;

            case 'autonomous_start':
                // The agent woke itself — render a distinct banner + label the
                // following text as autonomous (no stream_start precedes this).
                hideActivity();
                autonomousTurn = true;
                currentAgentEl = null;
                setBusy(true, 'wake');
                addAutonomousBanner(msg.trigger || '');
                bumpBadge();
                break;

            case 'thinking':
                if (streaming) setActivity('Thinking…');
                break;

            case 'text': {
                if (!currentAgentEl) {
                    hideActivity();
                    currentAgentEl = addTurn('agent');
                    currentAgentEl._raw = '';
                }
                currentAgentEl._raw += (msg.text || '');
                currentAgentEl.innerHTML = mdToHtml(currentAgentEl._raw);
                scrollToBottom(false);  // in-place edit, not a new item
                break;
            }

            case 'tool_start': {
                hideActivity();           // the running tool row is the signal now
                currentAgentEl = null;    // text after a tool starts a fresh bubble
                const label = msg.tool_label || msg.tool_name || 'tool';
                const args = fmtArgs(msg.tool_input);
                const el = document.createElement('div');
                el.className = 'ac-tool ac-tool-running';
                el.dataset.tool = msg.tool_name || '';
                el.innerHTML =
                    `<div class="ac-tool-head"><span class="ac-tool-spin"></span>` +
                    `<span class="ac-tool-name">${escapeHtml(label)}</span></div>` +
                    (args ? `<div class="ac-tool-args">${escapeHtml(args)}</div>` : '');
                log.appendChild(el);
                scrollToBottom();
                break;
            }

            case 'tool_call': {
                const running = [...log.querySelectorAll('.ac-tool-running')]
                    .filter(e => e.dataset.tool === (msg.tool_name || ''));
                const el = running[running.length - 1];
                const label = msg.tool_name || 'tool';
                const dur = msg.duration
                    ? ` · ${(msg.duration.toFixed ? msg.duration.toFixed(1) : msg.duration)}s` : '';
                const args = fmtArgs(msg.tool_input);
                const summary = msg.result_summary || '';
                // Show ⚠ instead of ✓ when the tool errored or its result reads
                // like a failure — so the operator can tell when a tool did nothing.
                const isErr = !!msg.is_error || looksLikeError(summary);
                const icon = isErr
                    ? `<span class="ac-tool-warn">⚠</span>`
                    : `<span class="ac-tool-check">✓</span>`;
                const html =
                    `<div class="ac-tool-head">${icon}` +
                    `<span class="ac-tool-name">${escapeHtml(label)}</span>` +
                    `<span class="ac-tool-meta">${dur}</span></div>` +
                    (args ? `<div class="ac-tool-args">${escapeHtml(args)}</div>` : '') +
                    (summary ? `<div class="ac-tool-summary${isErr ? ' ac-tool-summary-err' : ''}">${escapeHtml(summary)}</div>` : '');
                if (el) {
                    el.className = 'ac-tool ac-tool-done' + (isErr ? ' ac-tool-err' : '');
                    el.innerHTML = html;
                } else {
                    // No matching running row (e.g. after a reconnect) — append fresh.
                    const fresh = document.createElement('div');
                    fresh.className = 'ac-tool ac-tool-done' + (isErr ? ' ac-tool-err' : '');
                    fresh.innerHTML = html;
                    log.appendChild(fresh);
                }
                if (streaming) setActivity('Working…');  // agent continues after the tool
                scrollToBottom();
                break;
            }

            case 'choice_request':
                hideActivity();
                renderChoice(msg);
                bumpBadge();
                break;

            case 'applied_spec':
                renderSpec(msg.spec || {});
                break;

            case 'stream_end':
                streaming = false;
                currentAgentEl = null;
                autonomousTurn = false;
                hideActivity();
                setBusy(false);
                break;

            case 'command_result':
                if (msg.error) addSystemLine(`${msg.command}: ${msg.error}`, 'error');
                else if (msg.content) addSystemLine(`${msg.command} ✓`, 'info');
                break;

            case 'notification':
                addSystemLine(msg.body ? `${msg.title} — ${msg.body}` : msg.title, msg.level || 'info');
                bumpBadge();
                break;

            case 'error':
                streaming = false;
                hideActivity();
                setBusy(false);
                addSystemLine(msg.error || 'Unknown error', 'error');
                break;

            case 'ping':
                send({ type: 'pong' });
                break;

            default:
                break;  // pong / state_update / browse_result / unknown — ignored
        }
    }

    function renderChoice(msg) {
        const data = msg.choice_data || {};
        const reqId = msg.request_id || data.request_id || '';
        const isWake = msg.origin === 'wake';
        const wrap = document.createElement('div');
        wrap.className = 'ac-choice' + (isWake ? ' ac-choice-wake' : '');
        if (isWake) {
            const tag = document.createElement('div');
            tag.className = 'ac-choice-origin';
            tag.textContent = 'Autonomy proposal — your approval needed';
            wrap.appendChild(tag);
        }
        const q = document.createElement('div');
        q.className = 'ac-choice-q';
        q.innerHTML = mdToHtml(data.question || 'Choose:');
        wrap.appendChild(q);

        (data.options || []).forEach(opt => {
            const btn = document.createElement('button');
            btn.className = 'ac-choice-opt';
            btn.disabled = !!opt.disabled || !hasControl;  // observers see it read-only
            const desc = opt.description ? `<span class="ac-choice-desc">${escapeHtml(opt.description)}</span>` : '';
            btn.innerHTML = `<span class="ac-choice-label">${escapeHtml(opt.label)}</span>${desc}`;
            btn.addEventListener('click', () => {
                send({ type: 'choice_response', request_id: reqId, selected: opt.id });
                [...wrap.querySelectorAll('button')].forEach(b => b.disabled = true);
                wrap.classList.add('ac-choice-answered');
                btn.classList.add('ac-choice-picked');
                if (streaming) setActivity('Working…');
                if (isWake && pendingSlot) {
                    setTimeout(() => { pendingSlot.classList.add('hidden'); pendingSlot.innerHTML = ''; }, 700);
                }
            });
            wrap.appendChild(btn);
        });
        // ASK approvals pin to the sticky slot above the composer so they can't
        // scroll out of reach; ordinary choices stay inline in the transcript.
        if (isWake && pendingSlot) {
            pendingSlot.innerHTML = '';
            pendingSlot.appendChild(wrap);
            pendingSlot.classList.remove('hidden');
            return;
        }
        log.appendChild(wrap);
        scrollToBottom();
    }

    function renderSpec(spec) {
        const rows = [];
        const add = (k, v) => { if (v !== undefined && v !== null && v !== '') rows.push([k, v]); };
        add('Strain', spec.strain);
        add('Temperature', spec.temperature_c != null ? `${spec.temperature_c} °C` : null);
        add('Slices', spec.num_slices);
        add('Exposure', spec.exposure_ms != null ? `${spec.exposure_ms} ms` : null);
        add('Interval', spec.interval_s != null ? `${spec.interval_s} s` : null);
        add('Stop at', spec.stop_condition);
        if (!rows.length) return;
        const el = document.createElement('div');
        el.className = 'ac-spec';
        el.innerHTML = '<div class="ac-spec-title">Imaging spec applied</div>' +
            rows.map(([k, v]) => `<div class="ac-spec-row"><span>${escapeHtml(k)}</span><span>${escapeHtml(v)}</span></div>`).join('');
        log.appendChild(el);
        scrollToBottom();
    }

    // ── Tool argument formatting ──────────────────────────────
    /** Compact, escaped "key=value" rendering of a tool's input for the chat. */
    function fmtArgs(input) {
        if (!input || typeof input !== 'object') return '';
        const parts = [];
        for (const [k, v] of Object.entries(input)) {
            if (k === 'context' || v === null || v === undefined || v === '') continue;
            let val = (typeof v === 'object') ? JSON.stringify(v) : String(v);
            if (val.length > 48) val = val.slice(0, 47) + '…';
            parts.push(`${k}=${val}`);
        }
        return parts.join('  ');
    }

    /** Heuristic: does a tool's result summary read like a failure?
     *  Used to show ⚠ for tools that return an error STRING (the agent only
     *  flags raised exceptions). Avoids false alarms like "No errors found". */
    function looksLikeError(s) {
        if (!s) return false;
        const t = s.trim();
        if (/^no\s+(errors?|issues?|problems?|anomal|changes?|warnings?)\b/i.test(t)) return false;
        if (/^(error|failed|failure|unable|cannot|can'?t|could\s?n'?t|could not|denied|invalid|no |not )/i.test(t)) return true;
        // mid-string failure markers, e.g. "Timepoint 7 not found for embryo_2".
        return /\bnot (found|available|connected|recognized|valid|supported)\b/i.test(t);
    }

    // ── Autocomplete ──────────────────────────────────────────
    /** The whitespace-delimited token immediately left of the caret. */
    function currentToken() {
        const v = input.value;
        const pos = (input.selectionStart != null) ? input.selectionStart : v.length;
        const before = v.slice(0, pos);
        const m = before.match(/(\S+)$/);
        return { token: m ? m[1] : '', start: m ? pos - m[1].length : pos, pos };
    }

    /** Compute completion items for the current input/caret, or []. */
    function computeCompletions() {
        const trimmed = input.value.trimStart().toLowerCase();
        // Slash commands: whole-input prefix (mirrors the TUI). A trailing space
        // (i.e. typing args) naturally yields no matches and hides the menu.
        if (trimmed.startsWith('/')) {
            return commands.filter(c =>
                (c.name && c.name.toLowerCase().startsWith(trimmed)) ||
                (c.aliases || []).some(a => String(a).toLowerCase().startsWith(trimmed))
            ).slice(0, 8).map(c => ({ kind: 'command', name: c.name, desc: c.description || '' }));
        }
        // @tool mention: complete the token under the caret against tool names.
        const tok = currentToken();
        if (tok.token.startsWith('@') && tools.length) {
            const q = tok.token.slice(1).toLowerCase();
            return tools.filter(t => t.name.toLowerCase().includes(q))
                .slice(0, 8)
                .map(t => ({ kind: 'tool', name: t.name, desc: t.description || '', token: tok }));
        }
        return [];
    }

    function renderCompletions(items) {
        acItems = items || [];
        acIdx = acItems.length ? 0 : -1;
        if (!acComplete) return;
        if (!acItems.length) { hideCompletions(); return; }
        acComplete.innerHTML = '';
        acItems.forEach((it, i) => {
            const row = document.createElement('div');
            row.className = 'ac-complete-item' + (i === acIdx ? ' active' : '');
            row.innerHTML =
                `<span class="ac-complete-name">${escapeHtml(it.name)}</span>` +
                (it.desc ? `<span class="ac-complete-desc">${escapeHtml(it.desc)}</span>` : '');
            // mousedown (not click) so it fires before the textarea blurs.
            row.addEventListener('mousedown', (e) => { e.preventDefault(); acceptCompletion(it); });
            acComplete.appendChild(row);
        });
        acComplete.classList.remove('hidden');
    }

    function hideCompletions() {
        acItems = [];
        acIdx = -1;
        if (acComplete) { acComplete.classList.add('hidden'); acComplete.innerHTML = ''; }
    }

    function updateCompletions() {
        renderCompletions(computeCompletions());
    }

    function moveCompletion(delta) {
        if (!acItems.length || !acComplete) return;
        acIdx = (acIdx + delta + acItems.length) % acItems.length;
        [...acComplete.children].forEach((c, i) => c.classList.toggle('active', i === acIdx));
    }

    function acceptCompletion(item) {
        if (!item) return;
        if (item.kind === 'command') {
            input.value = item.name + ' ';
            const p = input.value.length;
            try { input.setSelectionRange(p, p); } catch (_) {}
        } else if (item.kind === 'tool') {
            const tok = item.token || currentToken();
            const v = input.value;
            const insert = '@' + item.name + ' ';
            input.value = v.slice(0, tok.start) + insert + v.slice(tok.pos);
            const p = tok.start + insert.length;
            try { input.setSelectionRange(p, p); } catch (_) {}
        }
        hideCompletions();
        input.focus();
        autosize();
    }

    // ── Control / UI state ────────────────────────────────────
    function renderControl() {
        if (hasControl) {
            banner.classList.add('hidden');
            banner.innerHTML = '';
            input.disabled = false;
            sendBtn.disabled = false;
            input.placeholder = 'Message Gently…   ( / commands · @ tools )';
        } else {
            banner.classList.remove('hidden');
            const who = holderLabel || 'another session';
            input.disabled = true;
            sendBtn.disabled = true;
            if (me && me.accounts && !me.authenticated) {
                // Anonymous — viewing is open; sign in to control.
                banner.innerHTML = `<span class="ac-lock">Viewing — sign in to control.</span>`;
                const btn = document.createElement('button');
                btn.className = 'ac-take-control';
                btn.textContent = 'Sign in';
                btn.addEventListener('click', () => { window.location.href = '/login'; });
                banner.appendChild(btn);
                input.placeholder = 'Viewing — sign in to control…';
            } else if (me && me.authenticated && me.can_control === false) {
                // Viewer-role account — watching is all this account can do.
                banner.innerHTML = `<span class="ac-lock">View-only access — you can watch but not control.</span>`;
                input.placeholder = 'View-only access';
            } else {
                banner.innerHTML = `<span class="ac-lock">Control held by ${escapeHtml(who)}</span>`;
                const btn = document.createElement('button');
                btn.className = 'ac-take-control';
                btn.textContent = 'Take control';
                btn.addEventListener('click', () => send({ type: 'take_control' }));
                banner.appendChild(btn);
                input.placeholder = 'Viewing only — take control to drive…';
            }
        }
    }

    function setBusy(busy, source) {
        agentBusy = !!busy;
        busySource = agentBusy ? (source || 'user') : null;
        // Send no longer doubles as Stop — it queues while busy. A separate Stop
        // (shown only for a cancellable user turn) aborts the current turn.
        if (stopBtn) stopBtn.classList.toggle('hidden', !(agentBusy && busySource === 'user'));
        sendBtn.classList.toggle('ac-busy', agentBusy);
        if (agentBusy) {
            input.placeholder = (busySource === 'wake')
                ? 'Gently is acting autonomously — your message will queue'
                : 'Gently is working — your message will queue';
        } else {
            if (hasControl) input.placeholder = 'Message Gently…   ( / commands · @ tools )';
            drainQueue();  // a turn just ended — send the next queued message
        }
    }

    // ── Message queue (type-while-busy) ───────────────────────
    function enqueue(text) { msgQueue.push(text); renderQueue(); }
    function removeQueued(i) {
        if (i >= 0 && i < msgQueue.length) { msgQueue.splice(i, 1); renderQueue(); }
    }
    function clearQueue() { msgQueue = []; renderQueue(); }
    function drainQueue() {
        if (agentBusy || !msgQueue.length) return;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;  // keep queued until reconnect
        const next = msgQueue.shift();
        renderQueue();
        actuallySend(next);
    }
    function renderQueue() {
        if (!queuePanel) return;
        if (!msgQueue.length) { queuePanel.classList.add('hidden'); queuePanel.innerHTML = ''; return; }
        queuePanel.classList.remove('hidden');
        queuePanel.innerHTML = '';
        const head = document.createElement('div');
        head.className = 'ac-queue-head';
        const lbl = document.createElement('span');
        lbl.textContent = `⏳ Queued (${msgQueue.length})`;
        const clear = document.createElement('button');
        clear.className = 'ac-queue-clear';
        clear.textContent = 'Clear all';
        clear.addEventListener('click', clearQueue);
        head.appendChild(lbl);
        head.appendChild(clear);
        queuePanel.appendChild(head);
        msgQueue.forEach((m, i) => {
            const row = document.createElement('div');
            row.className = 'ac-queue-item';
            const span = document.createElement('span');
            span.className = 'ac-queue-text';
            span.textContent = m;
            const x = document.createElement('button');
            x.className = 'ac-queue-remove';
            x.textContent = '✕';
            x.title = 'Remove from queue';
            x.addEventListener('click', () => removeQueued(i));
            row.appendChild(span);
            row.appendChild(x);
            queuePanel.appendChild(row);
        });
    }

    function setConn(ok, label) {
        conn.classList.toggle('ac-conn-ok', ok);
        conn.classList.toggle('ac-conn-bad', !ok);
        conn.textContent = label || (ok ? 'Connected' : 'Reconnecting…');
        if (toggleDot) toggleDot.classList.toggle('ok', ok);
    }

    // ── Transport ─────────────────────────────────────────────
    function send(obj) {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
    }

    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        setConn(false, 'Connecting…');
        ws = new WebSocket(`${proto}//${location.host}/ws/agent`);
        ws.onopen = () => { reconnectDelay = 1000; setConn(true); };
        ws.onclose = () => {
            setConn(false, 'Reconnecting…');
            setBusy(false);
            streaming = false;
            hideActivity();
            setTimeout(connect, reconnectDelay);
            reconnectDelay = Math.min(reconnectDelay * 2, MAX_DELAY);
        };
        ws.onerror = () => {};
        ws.onmessage = (e) => {
            let msg;
            try { msg = JSON.parse(e.data); } catch { return; }
            handle(msg);
        };
    }

    // ── Input handling ────────────────────────────────────────
    function actuallySend(text) {
        if (text.startsWith('/')) {
            addUserMessage(text);                       // commands aren't broadcast; echo locally
            send({ type: 'command', command: text });   // slash commands (e.g. /status)
            // Most commands reply with a single 'command_result' and no stream —
            // do NOT mark the composer busy, or the queue would stick forever.
            // Commands that DO stream (e.g. /wizard) set busy via stream_start.
            return;
        }
        send({ type: 'chat', text });                   // echoed to all via 'user_message'
        // Instant feedback before the first chunk arrives.
        setBusy(true, 'user');
        setActivity('Working…');
    }

    function submit() {
        hideCompletions();
        const text = input.value.trim();
        if (!text) return;
        if (!hasControl) { renderControl(); return; }
        input.value = '';
        autosize();
        // While the agent is busy (a user OR autonomous turn), queue instead of
        // cancelling — Send no longer doubles as Stop.
        if (agentBusy) { enqueue(text); return; }
        actuallySend(text);
    }

    function autosize() {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 140) + 'px';
    }

    function togglePanel(open) {
        panelOpen = (open === undefined) ? !panelOpen : open;
        panel.classList.toggle('open', panelOpen);
        if (toggleBtn) {
            toggleBtn.setAttribute('aria-pressed', panelOpen ? 'true' : 'false');
            toggleBtn.setAttribute('aria-expanded', panelOpen ? 'true' : 'false');
        }
        if (panelOpen) {
            clearBadge();
            if (!ws) connect();
            // Re-pin to the latest content (it may have streamed while closed,
            // where scroll events don't fire to keep stickBottom current).
            setTimeout(() => { input.focus(); jumpToBottom(); }, 50);
        }
        // Opening/closing while docked reflows .app-main — tell viewers to resize.
        if (document.body.classList.contains('chat-docked')) emitLayoutChanged();
    }

    // ── Layout: dock, resize, persistence ─────────────────────
    const CHAT_MIN_W = 320;
    const CHAT_DEFAULT_W = 460;
    // Roomy ceiling: the panel shows agent reasoning, tool calls, approvals and
    // pickers — content that wraps badly in a narrow column — so allow up to
    // ~half the viewport (was min(560, 45vw), which capped power users too low).
    function chatMaxW() { return Math.min(760, Math.round(window.innerWidth * 0.60)); }

    function emitLayoutChanged() {
        // Let the CSS settle, then notify viewers (e.g. the 3D canvas) to resize.
        requestAnimationFrame(() => window.dispatchEvent(new CustomEvent('gently:layout-changed')));
    }

    function curChatWidth() {
        return parseInt(getComputedStyle(document.documentElement).getPropertyValue('--chat-w')) || CHAT_DEFAULT_W;
    }

    function setChatWidth(px, persist) {
        const w = Math.max(CHAT_MIN_W, Math.min(chatMaxW(), Math.round(px)));
        document.documentElement.style.setProperty('--chat-w', w + 'px');
        if (persist) { try { localStorage.setItem('gently-chat-w', String(w)); } catch (_) {} }
        return w;
    }

    function applyDock(docked, persist) {
        document.body.classList.toggle('chat-docked', docked);
        if (pinBtn) {
            pinBtn.setAttribute('aria-pressed', docked ? 'true' : 'false');
            pinBtn.title = docked ? 'Unpin (float over content)' : 'Pin to dock';
        }
        if (persist) { try { localStorage.setItem('gently-chat-docked', docked ? '1' : '0'); } catch (_) {} }
        // Suppress the slide animation across the mode flip, then notify viewers.
        panel.style.transition = 'none';
        requestAnimationFrame(() => { panel.style.transition = ''; emitLayoutChanged(); });
    }

    function togglePin() {
        const docked = !document.body.classList.contains('chat-docked');
        if (docked && !panelOpen) togglePanel(true);  // pinning implies showing
        applyDock(docked, true);
    }

    function setupResize() {
        if (!resizeEl) return;
        let startX = 0, startW = 0, dragging = false, rafId = 0, pid = null;
        const onMove = (e) => {
            if (!dragging) return;
            setChatWidth(startW + (startX - e.clientX), false);  // right panel: drag left = wider
            if (document.body.classList.contains('chat-docked')) {
                if (rafId) cancelAnimationFrame(rafId);
                rafId = requestAnimationFrame(emitLayoutChanged);  // coalesce dock reflow
            }
        };
        const onUp = () => {
            if (!dragging) return;
            dragging = false;
            resizeEl.classList.remove('dragging');
            resizeEl.removeEventListener('pointermove', onMove);
            resizeEl.removeEventListener('pointerup', onUp);
            resizeEl.removeEventListener('pointercancel', onUp);
            if (pid !== null && resizeEl.hasPointerCapture && resizeEl.hasPointerCapture(pid)) {
                try { resizeEl.releasePointerCapture(pid); } catch (_) {}
            }
            pid = null;
            document.body.style.userSelect = '';
            setChatWidth(curChatWidth(), true);
            emitLayoutChanged();
        };
        resizeEl.addEventListener('pointerdown', (e) => {
            if (e.button !== 0) return;  // primary button only
            e.preventDefault();
            dragging = true;
            startX = e.clientX;
            startW = curChatWidth();
            pid = e.pointerId;
            // Capture so move/up/cancel always reach the handle (touch/pen-safe).
            try { resizeEl.setPointerCapture(pid); } catch (_) {}
            resizeEl.classList.add('dragging');
            document.body.style.userSelect = 'none';
            resizeEl.addEventListener('pointermove', onMove);
            resizeEl.addEventListener('pointerup', onUp);
            resizeEl.addEventListener('pointercancel', onUp);
        });
        resizeEl.addEventListener('dblclick', () => { setChatWidth(CHAT_DEFAULT_W, true); emitLayoutChanged(); });
    }

    function restorePrefs() {
        try {
            const w = parseInt(localStorage.getItem('gently-chat-w'));
            if (w) setChatWidth(w, false);
            if (localStorage.getItem('gently-chat-docked') === '1') applyDock(true, false);
        } catch (_) {}
    }

    // Unseen-activity badge on the header toggle — so a closed panel still tells
    // the operator the agent did something (woke, proposed an approval, notified).
    let badgeCount = 0;
    function bumpBadge() {
        if (panelOpen) return;  // they're watching; no badge needed
        badgeCount += 1;
        if (toggleBadge) {
            toggleBadge.textContent = badgeCount > 9 ? '9+' : String(badgeCount);
            toggleBadge.classList.remove('hidden');
        }
    }
    function clearBadge() {
        badgeCount = 0;
        if (toggleBadge) { toggleBadge.classList.add('hidden'); toggleBadge.textContent = ''; }
    }

    // ── Identity ──────────────────────────────────────────────
    function fetchMe() {
        fetch('/api/auth/me').then(r => r.json()).then(m => {
            me = m;
            if (m && m.authenticated) {
                userEl.textContent = m.username;
                userEl.title = `Signed in as ${m.username} (${m.role})`;
                signoutBtn.textContent = 'Sign out';
                signoutBtn.dataset.action = 'logout';
                signoutBtn.style.display = '';
            } else if (m && m.accounts) {
                // Anonymous — viewing is open; sign in to gain control.
                userEl.textContent = 'viewing';
                userEl.title = 'Not signed in — view-only';
                signoutBtn.textContent = 'Sign in';
                signoutBtn.dataset.action = 'login';
                signoutBtn.style.display = '';
            } else {
                // No accounts configured (legacy mode).
                userEl.textContent = '';
                signoutBtn.style.display = 'none';
            }
            renderControl();
        }).catch(() => {});
    }

    // ── Init ──────────────────────────────────────────────────
    function init() {
        panel = document.getElementById('agent-chat');
        log = document.getElementById('agent-chat-log');
        input = document.getElementById('agent-chat-text');
        sendBtn = document.getElementById('agent-chat-send');
        conn = document.getElementById('agent-chat-conn');
        banner = document.getElementById('agent-control-banner');
        closeBtn = document.getElementById('agent-chat-close');
        userEl = document.getElementById('agent-chat-user');
        signoutBtn = document.getElementById('agent-chat-signout');
        toggleBtn = document.getElementById('agent-chat-toggle');
        pinBtn = document.getElementById('agent-chat-pin');
        resizeEl = document.getElementById('agent-chat-resize');
        toggleDot = document.getElementById('agent-chat-toggle-dot');
        toggleBadge = document.getElementById('agent-chat-toggle-badge');
        if (!panel) return;  // markup not present

        restorePrefs();
        if (toggleBtn) toggleBtn.addEventListener('click', () => togglePanel());
        closeBtn.addEventListener('click', () => togglePanel(false));
        if (pinBtn) pinBtn.addEventListener('click', togglePin);
        setupResize();
        // Ctrl/Cmd+J toggles the panel from anywhere.
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && (e.key === 'j' || e.key === 'J')) {
                e.preventDefault();              // suppress browser default (downloads) always
                if (e.repeat) return;            // ignore held-key auto-repeat
                if (document.activeElement === input) return;  // don't toggle while composing
                togglePanel();
            }
        });
        signoutBtn.addEventListener('click', async () => {
            if (signoutBtn.dataset.action === 'login') {
                window.location.href = '/login';
                return;
            }
            try { await fetch('/api/auth/logout', { method: 'POST' }); } catch (_) {}
            window.location.reload();
        });
        fetchMe();

        // Build the autocomplete dropdown inside the composer (positioned above
        // the textarea via CSS).
        const inputWrap = input.parentNode;
        if (inputWrap) {
            acComplete = document.createElement('div');
            acComplete.className = 'ac-complete hidden';
            inputWrap.insertBefore(acComplete, inputWrap.firstChild);

            // Queued-message panel (above the composer) for type-while-busy.
            queuePanel = document.createElement('div');
            queuePanel.className = 'ac-queue hidden';
            if (inputWrap.parentNode) inputWrap.parentNode.insertBefore(queuePanel, inputWrap);

            // Explicit Stop button — shown only during a cancellable user turn.
            stopBtn = document.createElement('button');
            stopBtn.className = 'ac-stop hidden';
            stopBtn.textContent = 'Stop';
            stopBtn.title = 'Stop the current turn';
            stopBtn.addEventListener('click', () => { send({ type: 'cancel' }); setBusy(false); });
            inputWrap.appendChild(stopBtn);

            // Sticky ASK-approval slot — above the queue + composer, never scrolls away.
            pendingSlot = document.createElement('div');
            pendingSlot.className = 'ac-pending hidden';
            if (inputWrap.parentNode) inputWrap.parentNode.insertBefore(pendingSlot, queuePanel);
        }

        // "↓ N new" jump pill + pin-to-bottom scroll tracking.
        jumpPill = document.createElement('button');
        jumpPill.className = 'ac-jump hidden';
        jumpPill.addEventListener('click', jumpToBottom);
        panel.appendChild(jumpPill);
        log.addEventListener('scroll', () => {
            stickBottom = nearBottom();
            if (stickBottom) newCount = 0;
            renderJumpPill();
        });

        sendBtn.addEventListener('click', submit);
        input.addEventListener('input', () => { autosize(); updateCompletions(); });
        // Close the menu shortly after blur (delay lets a mousedown selection land).
        input.addEventListener('blur', () => setTimeout(hideCompletions, 120));
        input.addEventListener('keydown', (e) => {
            // While the completion menu is open it owns the navigation keys.
            if (acItems.length) {
                if (e.key === 'ArrowDown') { e.preventDefault(); moveCompletion(1); return; }
                if (e.key === 'ArrowUp') { e.preventDefault(); moveCompletion(-1); return; }
                if (e.key === 'Tab') { e.preventDefault(); acceptCompletion(acItems[acIdx]); return; }
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); acceptCompletion(acItems[acIdx]); return; }
                if (e.key === 'Escape') { e.preventDefault(); hideCompletions(); return; }
            }
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
            // Escape mirrors Stop: cancel a cancellable (user) turn and clear busy
            // (a cancelled turn emits no stream_end, so clear optimistically).
            if (e.key === 'Escape' && agentBusy && busySource === 'user') {
                e.preventDefault(); send({ type: 'cancel' }); setBusy(false);
            }
        });
    }

    document.addEventListener('DOMContentLoaded', init);

    // Public: programmatically send a message/command (e.g. the Home page's
    // "Start / continue an experiment" button sends '/wizard').
    function runCommand(text) {
        if (!text) return;
        if (!hasControl) { renderControl(); return; }
        actuallySend(text);
    }

    return { togglePanel, runCommand };
})();
