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
    let currentAgentEl = null;    // the agent bubble currently being streamed into
    let thinkingEl = null;

    // DOM refs (resolved in init)
    let fab, panel, log, input, sendBtn, connDot, banner, closeBtn;

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

    // ── Message helpers ───────────────────────────────────────
    function scrollToBottom() {
        log.scrollTop = log.scrollHeight;
    }

    function addBubble(role, htmlOrText, { html = false } = {}) {
        const el = document.createElement('div');
        el.className = `ac-msg ac-${role}`;
        if (html) el.innerHTML = htmlOrText;
        else el.textContent = htmlOrText;
        log.appendChild(el);
        scrollToBottom();
        return el;
    }

    function addSystemLine(text, level = 'info') {
        const el = document.createElement('div');
        el.className = `ac-system ac-level-${level}`;
        el.textContent = text;
        log.appendChild(el);
        scrollToBottom();
        return el;
    }

    function clearThinking() {
        if (thinkingEl && thinkingEl.parentNode) thinkingEl.parentNode.removeChild(thinkingEl);
        thinkingEl = null;
    }

    // ── Protocol handlers ─────────────────────────────────────
    function handle(msg) {
        switch (msg.type) {
            case 'connected':
                reconnectDelay = 1000;
                setConn(true);
                // version / session in the header tooltip
                if (msg.version) connDot.title = `connected · v${msg.version}`;
                break;

            case 'control_status':
                hasControl = !!msg.you_have_control;
                holderLabel = msg.holder_label || null;
                renderControl();
                break;

            case 'stream_start':
                streaming = true;
                clearThinking();
                currentAgentEl = null;  // created lazily on first text
                setBusy(true);
                break;

            case 'thinking':
                if (!thinkingEl) {
                    thinkingEl = document.createElement('div');
                    thinkingEl.className = 'ac-thinking';
                    thinkingEl.textContent = 'thinking…';
                    log.appendChild(thinkingEl);
                    scrollToBottom();
                }
                break;

            case 'text': {
                clearThinking();
                if (!currentAgentEl) {
                    currentAgentEl = addBubble('agent', '', { html: true });
                    currentAgentEl._raw = '';
                }
                currentAgentEl._raw += (msg.text || '');
                currentAgentEl.innerHTML = mdToHtml(currentAgentEl._raw);
                scrollToBottom();
                break;
            }

            case 'tool_start': {
                clearThinking();
                const label = msg.tool_label || msg.tool_name || 'tool';
                const el = document.createElement('div');
                el.className = 'ac-tool ac-tool-running';
                el.dataset.tool = msg.tool_name || '';
                el.innerHTML = `<span class="ac-tool-spin">⚙</span> ${escapeHtml(label)}…`;
                log.appendChild(el);
                scrollToBottom();
                break;
            }

            case 'tool_call': {
                // Mark the most recent running entry for this tool as done.
                const running = [...log.querySelectorAll('.ac-tool-running')]
                    .filter(e => e.dataset.tool === (msg.tool_name || ''));
                const el = running[running.length - 1];
                const label = msg.tool_name || 'tool';
                const dur = msg.duration ? ` · ${msg.duration.toFixed ? msg.duration.toFixed(1) : msg.duration}s` : '';
                const summary = msg.result_summary ? ` — ${escapeHtml(msg.result_summary)}` : '';
                if (el) {
                    el.className = 'ac-tool ac-tool-done';
                    el.innerHTML = `<span class="ac-tool-check">✓</span> ${escapeHtml(label)}${dur}${summary}`;
                } else {
                    addBubble('tool', `✓ ${escapeHtml(label)}${dur}${summary}`, { html: true });
                }
                scrollToBottom();
                break;
            }

            case 'choice_request':
                renderChoice(msg);
                break;

            case 'applied_spec':
                renderSpec(msg.spec || {});
                break;

            case 'stream_end':
                streaming = false;
                clearThinking();
                currentAgentEl = null;
                setBusy(false);
                break;

            case 'command_result':
                if (msg.error) addSystemLine(`${msg.command}: ${msg.error}`, 'error');
                else if (msg.content) addSystemLine(`${msg.command} ✓`, 'info');
                break;

            case 'notification':
                addSystemLine(msg.body ? `${msg.title} — ${msg.body}` : msg.title, msg.level || 'info');
                break;

            case 'error':
                streaming = false;
                setBusy(false);
                addSystemLine(msg.error || 'Unknown error', 'error');
                break;

            case 'ping':
                send({ type: 'pong' });
                break;

            case 'pong':
            case 'state_update':
            case 'browse_result':
                break;  // not surfaced in the chat window (yet)

            default:
                break;  // unknown types ignored (forward-compatible)
        }
    }

    function renderChoice(msg) {
        clearThinking();
        const data = msg.choice_data || {};
        const reqId = msg.request_id || data.request_id || '';
        const wrap = document.createElement('div');
        wrap.className = 'ac-choice';
        const q = document.createElement('div');
        q.className = 'ac-choice-q';
        q.innerHTML = mdToHtml(data.question || 'Choose:');
        wrap.appendChild(q);

        (data.options || []).forEach(opt => {
            const btn = document.createElement('button');
            btn.className = 'ac-choice-opt';
            btn.disabled = !!opt.disabled;
            const desc = opt.description ? `<span class="ac-choice-desc">${escapeHtml(opt.description)}</span>` : '';
            btn.innerHTML = `<span class="ac-choice-label">${escapeHtml(opt.label)}</span>${desc}`;
            btn.addEventListener('click', () => {
                send({ type: 'choice_response', request_id: reqId, selected: opt.id });
                // lock the picker and show the pick
                [...wrap.querySelectorAll('button')].forEach(b => b.disabled = true);
                wrap.classList.add('ac-choice-answered');
                btn.classList.add('ac-choice-picked');
            });
            wrap.appendChild(btn);
        });
        log.appendChild(wrap);
        scrollToBottom();
    }

    function renderSpec(spec) {
        const rows = [];
        const add = (k, v) => { if (v !== undefined && v !== null && v !== '') rows.push([k, v]); };
        add('strain', spec.strain);
        add('temp °C', spec.temperature_c);
        add('slices', spec.num_slices);
        add('exposure ms', spec.exposure_ms);
        add('interval s', spec.interval_s);
        add('stop at', spec.stop_condition);
        if (!rows.length) return;
        const html = '<div class="ac-spec-title">Imaging spec applied</div>' +
            rows.map(([k, v]) => `<div class="ac-spec-row"><span>${escapeHtml(k)}</span><span>${escapeHtml(v)}</span></div>`).join('');
        addBubble('spec', html, { html: true });
    }

    // ── Control / UI state ────────────────────────────────────
    function renderControl() {
        if (hasControl) {
            banner.classList.add('hidden');
            banner.innerHTML = '';
            input.disabled = false;
            sendBtn.disabled = false;
            input.placeholder = 'Message the agent…';
        } else {
            banner.classList.remove('hidden');
            const who = holderLabel || 'another client';
            banner.innerHTML = `<span>🔒 ${escapeHtml(who)} is driving</span>`;
            const btn = document.createElement('button');
            btn.className = 'ac-take-control';
            btn.textContent = 'Take control';
            btn.addEventListener('click', () => send({ type: 'take_control' }));
            banner.appendChild(btn);
            input.disabled = true;
            sendBtn.disabled = true;
            input.placeholder = 'Viewing only — take control to drive…';
        }
    }

    function setBusy(busy) {
        sendBtn.textContent = busy ? 'Stop' : 'Send';
        sendBtn.classList.toggle('ac-busy', busy);
    }

    function setConn(ok) {
        connDot.classList.toggle('ac-conn-ok', ok);
        connDot.classList.toggle('ac-conn-bad', !ok);
        if (!ok) connDot.title = 'reconnecting…';
    }

    // ── Transport ─────────────────────────────────────────────
    function send(obj) {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
    }

    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${proto}//${location.host}/ws/agent`);
        ws.onopen = () => { reconnectDelay = 1000; setConn(true); };
        ws.onclose = () => {
            setConn(false);
            setBusy(false);
            streaming = false;
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
    function submit() {
        if (streaming) { send({ type: 'cancel' }); return; }  // Send doubles as Stop
        const text = input.value.trim();
        if (!text) return;
        if (!hasControl) { renderControl(); return; }
        addBubble('user', text);
        if (text.startsWith('/')) {
            send({ type: 'command', command: text });  // slash commands (e.g. /status)
        } else {
            send({ type: 'chat', text });
        }
        input.value = '';
        autosize();
    }

    function autosize() {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 140) + 'px';
    }

    function togglePanel(open) {
        panelOpen = (open === undefined) ? !panelOpen : open;
        panel.classList.toggle('hidden', !panelOpen);
        fab.classList.toggle('ac-fab-active', panelOpen);
        if (panelOpen) {
            if (!ws) connect();
            setTimeout(() => input.focus(), 50);
        }
    }

    // ── Init ──────────────────────────────────────────────────
    function init() {
        fab = document.getElementById('agent-fab');
        panel = document.getElementById('agent-chat');
        log = document.getElementById('agent-chat-log');
        input = document.getElementById('agent-chat-text');
        sendBtn = document.getElementById('agent-chat-send');
        connDot = document.getElementById('agent-chat-conn');
        banner = document.getElementById('agent-control-banner');
        closeBtn = document.getElementById('agent-chat-close');
        if (!fab || !panel) return;  // markup not present

        fab.addEventListener('click', () => togglePanel());
        closeBtn.addEventListener('click', () => togglePanel(false));
        sendBtn.addEventListener('click', submit);
        input.addEventListener('input', autosize);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
            if (e.key === 'Escape' && streaming) { e.preventDefault(); send({ type: 'cancel' }); }
        });
    }

    document.addEventListener('DOMContentLoaded', init);

    return { togglePanel };
})();
