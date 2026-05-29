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

    // DOM refs (resolved in init)
    let fab, panel, log, input, sendBtn, conn, banner, closeBtn, userEl, signoutBtn;

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

    function scrollToBottom() { log.scrollTop = log.scrollHeight; }

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
        if (role === 'agent') {
            const label = document.createElement('div');
            label.className = 'ac-role';
            label.textContent = 'Gently';
            wrap.appendChild(label);
        }
        const content = document.createElement('div');
        content.className = 'ac-content';
        wrap.appendChild(content);
        log.appendChild(wrap);
        scrollToBottom();
        return content;
    }

    function addUserMessage(text) {
        const content = addTurn('user');
        content.textContent = text;
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
                break;

            case 'control_status':
                hasControl = !!msg.you_have_control;
                holderLabel = msg.holder_label || null;
                renderControl();
                break;

            case 'stream_start':
                streaming = true;
                currentAgentEl = null;  // created lazily on first text
                setBusy(true);
                setActivity('Working…');
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
                scrollToBottom();
                break;
            }

            case 'tool_start': {
                hideActivity();           // the running tool row is the signal now
                currentAgentEl = null;    // text after a tool starts a fresh bubble
                const label = msg.tool_label || msg.tool_name || 'tool';
                const el = document.createElement('div');
                el.className = 'ac-tool ac-tool-running';
                el.dataset.tool = msg.tool_name || '';
                el.innerHTML = `<span class="ac-tool-spin"></span><span class="ac-tool-name">${escapeHtml(label)}</span>`;
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
                const summary = msg.result_summary ? ` — ${escapeHtml(msg.result_summary)}` : '';
                if (el) {
                    el.className = 'ac-tool ac-tool-done';
                    el.innerHTML = `<span class="ac-tool-check">✓</span><span class="ac-tool-name">${escapeHtml(label)}</span><span class="ac-tool-meta">${dur}${summary}</span>`;
                }
                if (streaming) setActivity('Working…');  // agent continues after the tool
                scrollToBottom();
                break;
            }

            case 'choice_request':
                hideActivity();
                renderChoice(msg);
                break;

            case 'applied_spec':
                renderSpec(msg.spec || {});
                break;

            case 'stream_end':
                streaming = false;
                currentAgentEl = null;
                hideActivity();
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
                [...wrap.querySelectorAll('button')].forEach(b => b.disabled = true);
                wrap.classList.add('ac-choice-answered');
                btn.classList.add('ac-choice-picked');
                if (streaming) setActivity('Working…');
            });
            wrap.appendChild(btn);
        });
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

    // ── Control / UI state ────────────────────────────────────
    function renderControl() {
        if (hasControl) {
            banner.classList.add('hidden');
            banner.innerHTML = '';
            input.disabled = false;
            sendBtn.disabled = false;
            input.placeholder = 'Message Gently…';
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

    function setBusy(busy) {
        sendBtn.textContent = busy ? 'Stop' : 'Send';
        sendBtn.classList.toggle('ac-busy', busy);
    }

    function setConn(ok, label) {
        conn.classList.toggle('ac-conn-ok', ok);
        conn.classList.toggle('ac-conn-bad', !ok);
        conn.textContent = label || (ok ? 'Connected' : 'Reconnecting…');
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
    function submit() {
        if (streaming) { send({ type: 'cancel' }); return; }  // Send doubles as Stop
        const text = input.value.trim();
        if (!text) return;
        if (!hasControl) { renderControl(); return; }
        addUserMessage(text);
        if (text.startsWith('/')) {
            send({ type: 'command', command: text });  // slash commands (e.g. /status)
        } else {
            send({ type: 'chat', text });
        }
        // Instant feedback before the first chunk arrives.
        setBusy(true);
        setActivity('Working…');
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
        fab = document.getElementById('agent-fab');
        panel = document.getElementById('agent-chat');
        log = document.getElementById('agent-chat-log');
        input = document.getElementById('agent-chat-text');
        sendBtn = document.getElementById('agent-chat-send');
        conn = document.getElementById('agent-chat-conn');
        banner = document.getElementById('agent-control-banner');
        closeBtn = document.getElementById('agent-chat-close');
        userEl = document.getElementById('agent-chat-user');
        signoutBtn = document.getElementById('agent-chat-signout');
        if (!fab || !panel) return;  // markup not present

        fab.addEventListener('click', () => togglePanel());
        closeBtn.addEventListener('click', () => togglePanel(false));
        signoutBtn.addEventListener('click', async () => {
            if (signoutBtn.dataset.action === 'login') {
                window.location.href = '/login';
                return;
            }
            try { await fetch('/api/auth/logout', { method: 'POST' }); } catch (_) {}
            window.location.reload();
        });
        fetchMe();
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
