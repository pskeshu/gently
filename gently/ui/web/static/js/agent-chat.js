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
    // Commands/messages requested before the socket is open (e.g. from the
    // landing, with the chat panel closed) — flushed in order on ws.onopen.
    let pendingProgrammatic = [];
    const MAX_DELAY = 30000;

    let panelOpen = false;
    let hasControl = true;        // optimistic until the server says otherwise
    const answeredAsks = new Set();  // request_ids answered from EITHER surface (transcript / main stage)
    let holderLabel = null;
    let streaming = false;
    let currentAgentEl = null;    // the agent content element being streamed into
    let activityEl = null;        // the persistent "working…" indicator (reused)
    let me = null;                // { authenticated, username, role, can_control }
    let myConnId = null;          // this connection's id, for labelling own msgs "You"

    // Autocomplete: slash-command + @tool registries (pushed by the server on
    // connect) and the live dropdown state.
    let commands = [];            // [{name, description, aliases, ...}]
    let tools = [];               // [{name, description, params, ...}]
    let acItems = [];             // current completion items shown in the dropdown
    let acIdx = -1;               // highlighted item index
    let autonomousTurn = false;   // true while rendering an autonomous (wake) turn
    let agentBusy = false;        // a turn (user or autonomous) is currently running
    let busySource = null;        // 'user' | 'wake' while busy
    let askPending = false;       // agent is paused waiting for user's ask answer
    let msgQueue = [];            // messages typed while busy, sent on idle
    let queuePanel = null;        // the "⏳ Queued (N)" panel element

    // DOM refs (resolved in init)
    let panel, log, input, sendBtn, conn, banner, closeBtn, userEl, signoutBtn;
    let railBtn, resizeEl, toggleDot, toggleBadge;  // docked-panel chrome
    let pendingSlot = null;       // sticky slot for ASK approval proposals
    let acComplete = null;        // the autocomplete dropdown element

    // ── Safe rendering ────────────────────────────────────────
    function escapeHtml(s) {
        const d = document.createElement('div');
        d.textContent = String(s == null ? '' : s);
        return d.innerHTML;
    }

    // ── Markdown → safe HTML ──────────────────────────────────
    // Block-aware GFM subset: headings, pipe tables, fenced/indented code,
    // ordered/unordered lists, blockquotes, hr, paragraphs; inline bold/italic/
    // strike/code/links. Designed for STREAMING: called repeatedly on a growing
    // prefix, so it must never throw or hang on partial input (half a table, an
    // unclosed fence, a dangling ** or `). All text is escaped before any markup
    // is added — no raw HTML passthrough — and link hrefs are scheme-checked.
    //
    // All regexes are linear (no nested/adjacent unbounded quantifiers over the
    // same class) to avoid catastrophic backtracking on adversarial tool output.
    //
    // Block class names (for styling): ac-md (wrapper), ac-md-h1..ac-md-h6,
    // ac-md-p, ac-md-ul, ac-md-ol, ac-md-li, ac-md-quote, ac-md-hr,
    // ac-md-pre, ac-md-code-block, ac-md-table-wrap, ac-md-table, ac-md-link.

    // escapeHtml() escapes & < > but not quotes; quote-escape for attribute use.
    function escAttr(s) {
        return escapeHtml(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    // Allow only safe link schemes. Reject javascript:/data:/vbscript: and any
    // control chars an attacker might use to smuggle a scheme past the check.
    // Relative URLs (no scheme) and #anchors are allowed.
    function safeHref(raw) {
        // Strip ASCII control chars + whitespace (incl. tab/newline) anywhere in
        // the URL so they can't be used to break the scheme check below.
        const url = String(raw || '').replace(/[\x00-\x20\x7f]/g, '');
        // A scheme is letters/digits/+/-/. followed by ':' before any / ? #.
        const m = url.match(/^([a-zA-Z][a-zA-Z0-9+.\-]*):/);
        if (m) {
            const scheme = m[1].toLowerCase();
            if (scheme !== 'http' && scheme !== 'https' && scheme !== 'mailto') return null;
        }
        return url;
    }

    // Inline spans, applied to ALREADY-ESCAPED text. Order matters: code spans
    // are pulled out first (placeholdered) so their contents aren't re-processed,
    // then links, then emphasis. Bounded quantifiers keep this linear-time.
    function mdInline(escaped) {
        const codes = [];
        // `code` and ``code`` — non-greedy, capped run length, no newlines.
        let s = escaped.replace(/(`{1,2})([^`\n]{0,500}?)\1/g, (_, _t, code) => {
            codes.push(code);
            return 'CODE' + (codes.length - 1) + '';
        });
        // [label](href) — label has no brackets, href no spaces/parens; bounded.
        s = s.replace(/\[([^\]\n]{0,200})\]\(([^()\s]{0,500})\)/g, (m, label, href) => {
            const safe = safeHref(href);
            if (!safe) return label;  // drop a rejected link, keep its text
            return '<a class="ac-md-link" href="' + escAttr(safe) +
                '" target="_blank" rel="noopener noreferrer nofollow">' + label + '</a>';
        });
        // Bold, italic, strikethrough. Each pattern is a single bounded run.
        s = s.replace(/\*\*([^*\n]{1,500}?)\*\*/g, '<strong>$1</strong>');
        s = s.replace(/__([^_\n]{1,500}?)__/g, '<strong>$1</strong>');
        s = s.replace(/(^|[^*])\*([^*\n]{1,500}?)\*/g, '$1<em>$2</em>');
        s = s.replace(/(^|[^_\w])_([^_\n]{1,500}?)_(?![\w])/g, '$1<em>$2</em>');
        s = s.replace(/~~([^~\n]{1,500}?)~~/g, '<del>$1</del>');
        // Restore code spans as real <code> elements.
        s = s.replace(/CODE(\d+)/g, (_, i) => '<code>' + codes[+i] + '</code>');
        return s;
    }

    // Split a GFM table row into escaped, inline-rendered cells. Handles escaped
    // pipes (\|) and ignores the leading/trailing border pipes.
    function tableCells(line) {
        const cells = [];
        let buf = '';
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (ch === '\\' && line[i + 1] === '|') { buf += '|'; i++; continue; }
            if (ch === '|') { cells.push(buf); buf = ''; continue; }
            buf += ch;
        }
        cells.push(buf);
        // Drop empty edge cells produced by leading/trailing border pipes.
        if (cells.length && cells[0].trim() === '') cells.shift();
        if (cells.length && cells[cells.length - 1].trim() === '') cells.pop();
        return cells.map(c => mdInline(escapeHtml(c.trim())));
    }

    // A table delimiter row: |---|:--:|--:| (each cell only - : and spaces, ≥1 -).
    function isTableDivider(line) {
        const t = line.trim().replace(/^\||\|$/g, '');
        if (!t) return false;
        return t.split('|').every(c => /^\s*:?-+:?\s*$/.test(c));
    }

    function mdToHtml(text) {
        const src = String(text == null ? '' : text);
        const lines = src.split('\n');
        const out = [];
        let i = 0;

        // Paragraph buffer: collect consecutive prose lines, flush on a block
        // boundary. Soft line-breaks inside a paragraph become <br>.
        let para = [];
        const flushPara = () => {
            if (!para.length) return;
            const body = para.map(l => mdInline(escapeHtml(l))).join('<br>');
            out.push('<p class="ac-md-p">' + body + '</p>');
            para = [];
        };

        while (i < lines.length) {
            const line = lines[i];
            const trimmed = line.trim();

            // Fenced code block: ``` or ~~~ (optional language). Unterminated
            // fences (mid-stream) consume to end-of-input — never hang.
            const fence = trimmed.match(/^(`{3,}|~{3,})(.*)$/);
            if (fence) {
                flushPara();
                const marker = fence[1][0];
                const minLen = fence[1].length;
                const langRaw = fence[2].trim().split(/\s+/)[0] || '';
                const lang = langRaw.replace(/[^a-zA-Z0-9_+\-.]/g, '').slice(0, 32);
                const code = [];
                i++;
                while (i < lines.length) {
                    const cl = lines[i];
                    const cm = cl.trim();
                    // A closing fence: same marker char, length ≥ opening, nothing else.
                    if (cm[0] === marker && /^(`{3,}|~{3,})\s*$/.test(cm) && cm.replace(/\s+$/, '').length >= minLen) {
                        i++;
                        break;
                    }
                    code.push(cl);
                    i++;
                }
                const langClass = lang ? ' language-' + lang : '';
                out.push('<pre class="ac-md-pre"><code class="ac-md-code-block' + langClass + '">' +
                    escapeHtml(code.join('\n')) + '</code></pre>');
                continue;
            }

            // Blank line: paragraph boundary.
            if (trimmed === '') { flushPara(); i++; continue; }

            // ATX heading: #..###### text.
            const h = trimmed.match(/^(#{1,6})\s+(.*?)\s*#*$/);
            if (h) {
                flushPara();
                const level = h[1].length;
                out.push('<h' + level + ' class="ac-md-h' + level + '">' +
                    mdInline(escapeHtml(h[2])) + '</h' + level + '>');
                i++;
                continue;
            }

            // Horizontal rule: ---, ***, ___ (3+).
            if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
                flushPara();
                out.push('<hr class="ac-md-hr">');
                i++;
                continue;
            }

            // GFM table: a header row followed by a delimiter row. Require the
            // delimiter to confirm it's a table (so a lone pipe line stays prose),
            // which also keeps a half-streamed table as plain text until ready.
            if (trimmed.indexOf('|') !== -1 && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
                flushPara();
                const header = tableCells(line);
                const colCount = header.length;
                i += 2;  // consume header + delimiter
                const bodyRows = [];
                while (i < lines.length) {
                    const rl = lines[i];
                    if (rl.trim() === '' || rl.indexOf('|') === -1) break;
                    bodyRows.push(tableCells(rl));
                    i++;
                }
                let tbl = '<div class="ac-md-table-wrap"><table class="ac-md-table"><thead><tr>';
                for (let c = 0; c < colCount; c++) tbl += '<th>' + (header[c] || '') + '</th>';
                tbl += '</tr></thead><tbody>';
                bodyRows.forEach(row => {
                    tbl += '<tr>';
                    for (let c = 0; c < colCount; c++) tbl += '<td>' + (row[c] || '') + '</td>';
                    tbl += '</tr>';
                });
                tbl += '</tbody></table></div>';
                out.push(tbl);
                continue;
            }

            // Blockquote: one or more leading '>' lines (collected together).
            if (/^>\s?/.test(line)) {
                flushPara();
                const quote = [];
                while (i < lines.length && /^>\s?/.test(lines[i])) {
                    quote.push(lines[i].replace(/^>\s?/, ''));
                    i++;
                }
                const body = quote.map(l => mdInline(escapeHtml(l))).join('<br>');
                out.push('<blockquote class="ac-md-quote">' + body + '</blockquote>');
                continue;
            }

            // Lists: a run of consecutive item lines. Ordered if the first item
            // is "N." / "N)", else unordered (-, *, +). Nesting is rendered flat
            // (depth via a leading-indent class isn't needed for the agent's output).
            const ulMatch = line.match(/^(\s*)[-*+]\s+(.*)$/);
            const olMatch = line.match(/^(\s*)\d{1,9}[.)]\s+(.*)$/);
            if (ulMatch || olMatch) {
                flushPara();
                const ordered = !!olMatch;
                const tag = ordered ? 'ol' : 'ul';
                const cls = ordered ? 'ac-md-ol' : 'ac-md-ul';
                const items = [];
                while (i < lines.length) {
                    const ul = lines[i].match(/^(\s*)[-*+]\s+(.*)$/);
                    const ol = lines[i].match(/^(\s*)\d{1,9}[.)]\s+(.*)$/);
                    const m = ordered ? ol : ul;
                    if (!m) {
                        // A blank line or non-item ends the list.
                        if (lines[i].trim() === '' || (!ul && !ol)) break;
                        break;
                    }
                    items.push(mdInline(escapeHtml(m[2])));
                    i++;
                }
                out.push('<' + tag + ' class="' + cls + '">' +
                    items.map(it => '<li class="ac-md-li">' + it + '</li>').join('') +
                    '</' + tag + '>');
                continue;
            }

            // Default: prose line — accumulate into the paragraph buffer.
            para.push(line);
            i++;
        }
        flushPara();
        return '<div class="ac-md">' + out.join('') + '</div>';
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
        // No per-turn role label: the agent's replies are plain text and the
        // user's sit in a bubble (modern chat convention). Autonomous (wake)
        // turns are still marked by the banner + accent rail, not a label.
        if (role === 'agent' && autonomousTurn) wrap.classList.add('ac-turn-autonomous');
        const content = document.createElement('div');
        content.className = 'ac-content';
        wrap.appendChild(content);
        log.appendChild(wrap);
        scrollToBottom();
        return content;
    }

    /** Normalize an author for display: clean up legacy/anonymous labels. */
    function displayAuthor(author) {
        if (!author) return 'Anonymous';
        // Legacy/per-connection labels ("window 3", "User 5") read as anonymous.
        if (/^(window|user)\s+\d+$/i.test(author)) return 'Anonymous';
        return author;
    }

    function addUserMessage(text, author, authorId) {
        const wrap = document.createElement('div');
        wrap.className = 'ac-turn ac-turn-user';
        // Single shared chat, so every user message is labelled. It's "You" when
        // it's from this connection (authorId match) or — once logged in — from
        // your username (stable across reloads); otherwise the sender's name, or
        // "Anonymous" for an unsigned-in participant. A local echo (no author
        // info) is always you.
        const mine = (!author && !authorId)
            || (authorId && authorId === myConnId)
            || (author && me && me.username && author === me.username);
        const label = document.createElement('div');
        label.className = 'ac-role ac-role-user';
        label.textContent = mine ? 'You' : displayAuthor(author);
        wrap.appendChild(label);
        if (!mine) wrap.classList.add('ac-from-other');
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
                addUserMessage(it.text, it.author, it.author_id);
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
                myConnId = msg.you_id || myConnId;  // for labelling own messages "You"
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
                // The main-stage ask renderer re-renders read-only on control loss.
                if (typeof ClientEventBus !== 'undefined') ClientEventBus.emit('AGENT_CONTROL', { hasControl });
                break;

            case 'history':
                renderHistory(msg.items || []);
                break;

            case 'user_message':
                hideActivity();
                addUserMessage(msg.text, msg.author, msg.author_id);
                break;

            case 'stream_start':
                streaming = true;
                currentAgentEl = null;  // created lazily on first text
                setBusy(true, 'user');
                setActivity('Working…');
                emitActivity('turn_start', { autonomous: false });
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
                emitActivity('turn_start', { autonomous: true });
                break;

            case 'thinking':
                if (streaming) setActivity('Thinking…');
                emitActivity('thinking', { text: msg.text || '' });
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
                emitActivity('text', { text: msg.text || '' });
                break;
            }

            case 'tool_start': {
                hideActivity();           // the running tool row is the signal now
                currentAgentEl = null;    // text after a tool starts a fresh bubble
                // ask_user_choice renders as a choice card via choice_request —
                // skip the noisy spinning tool row for it.
                if (msg.tool_name === 'ask_user_choice') break;
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
                emitActivity('tool_start', { name: msg.tool_name || '', label: label, input: msg.tool_input });
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
                emitActivity('tool_result', {
                    name: msg.tool_name || '', label: label,
                    input: msg.tool_input, duration: msg.duration,
                    summary: summary, full: msg.result_full, is_error: isErr,
                });
                break;
            }

            case 'choice_request':
                hideActivity();
                setAskState(true);  // waiting on user, not working
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
                emitActivity('turn_end');
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
                emitActivity('turn_error', { error: msg.error || 'Unknown error' });
                clearPendingAsks();  // a cancelled/errored turn sends no choice_response
                break;

            case 'ping':
                send({ type: 'pong' });
                break;

            default:
                break;  // pong / state_update / browse_result / unknown — ignored
        }
    }

    // Build an ask card from a choice_data payload. Pure: the caller supplies
    // hasControl + onPick, so the SAME builder renders in the chat transcript
    // and on the main stage (#ask-stage) — one payload, two renderers.
    function buildAskCard(data, opts) {
        opts = opts || {};
        const reqId = opts.reqId || '';
        const isWake = !!opts.isWake;
        const canAct = !!opts.hasControl && !answeredAsks.has(reqId);
        const onPick = opts.onPick || function () {};

        const wrap = document.createElement('div');
        wrap.className = 'ac-choice' + (isWake ? ' ac-choice-wake' : '');
        wrap.dataset.reqId = reqId;
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
            btn.disabled = !!opt.disabled || !canAct;  // observers / already-answered → read-only
            const desc = opt.description ? `<span class="ac-choice-desc">${escapeHtml(opt.description)}</span>` : '';
            btn.innerHTML = `<span class="ac-choice-label">${escapeHtml(opt.label)}</span>${desc}`;
            btn.addEventListener('click', () => onPick(opt.id));
            wrap.appendChild(btn);
        });

        // Free-text escape — the bridge routes an unknown selection to LLM
        // resolution, so the agent's asend always unblocks. (The TUI had this;
        // the web ask cards previously did not.)
        if (canAct) {
            const ow = document.createElement('div');
            ow.className = 'ac-choice-otherwrap';
            const otherBtn = document.createElement('button');
            otherBtn.className = 'ac-choice-opt ac-choice-other';
            otherBtn.innerHTML = '<span class="ac-choice-label">Something else…</span>';
            const form = document.createElement('div');
            form.className = 'ac-choice-otherform hidden';
            const ti = document.createElement('input');
            ti.type = 'text';
            ti.className = 'ac-choice-otherinput';
            ti.placeholder = 'Type your own answer…';
            const go = document.createElement('button');
            go.className = 'ac-choice-othergo';
            go.textContent = '→';
            const submitOther = () => { const v = ti.value.trim(); if (v) onPick(v); };
            otherBtn.addEventListener('click', () => { otherBtn.classList.add('hidden'); form.classList.remove('hidden'); ti.focus(); });
            go.addEventListener('click', submitOther);
            ti.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); submitOther(); } });
            form.appendChild(ti); form.appendChild(go);
            ow.appendChild(otherBtn); ow.appendChild(form);
            wrap.appendChild(ow);
        }

        if (answeredAsks.has(reqId)) wrap.classList.add('ac-choice-answered');
        return wrap;
    }

    // Send the answer ONCE (idempotent across both surfaces), then fire the
    // clear off the CHOICE lifecycle — NOT stream_end (which lands after the
    // answer for in-turn asks, and never for a cancelled turn).
    function answerChoice(reqId, selected) {
        if (!reqId || answeredAsks.has(reqId)) return;
        answeredAsks.add(reqId);
        send({ type: 'choice_response', request_id: reqId, selected });
        setAskState(false);  // resume working-state visuals
        if (streaming) setActivity('Working…');
        if (typeof ClientEventBus !== 'undefined') ClientEventBus.emit('ASK_CLEARED', { request_id: reqId });
    }

    // Disable + mark-answered any transcript / sticky-slot ask card for this
    // request_id ('*' = all). The main stage clears itself via its own handler.
    function markAnswered(reqId) {
        [log, pendingSlot].forEach(scope => {
            if (!scope) return;
            scope.querySelectorAll('.ac-choice').forEach(card => {
                if (reqId !== '*' && card.dataset.reqId !== reqId) return;
                card.querySelectorAll('button').forEach(b => b.disabled = true);
                card.classList.add('ac-choice-answered');
            });
            // Also fade compact pointers (ux_v2 mode — no buttons to disable).
            scope.querySelectorAll('.ac-ask-pointer').forEach(ptr => {
                if (reqId !== '*' && ptr.dataset.reqId !== reqId) return;
                ptr.classList.add('ac-ask-pointer-answered');
            });
        });
        if (pendingSlot) {
            const slotCard = pendingSlot.querySelector('.ac-choice');
            if (reqId === '*' || (slotCard && slotCard.dataset.reqId === reqId)) {
                setTimeout(() => { pendingSlot.classList.add('hidden'); pendingSlot.innerHTML = ''; }, 700);
            }
        }
    }

    // Retire all pending asks (turn cancelled/errored, or socket dropped).
    function clearPendingAsks() {
        if (typeof ClientEventBus !== 'undefined') ClientEventBus.emit('ASK_CLEARED', { request_id: '*' });
        else markAnswered('*');
    }

    function renderChoice(msg) {
        const data = msg.choice_data || {};
        const reqId = msg.request_id || data.request_id || '';
        const isWake = msg.origin === 'wake';
        // Always mirror onto the main stage (AskStage / landing wizard).
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.emit('AGENT_ASK', { request_id: reqId, choice_data: data, origin: msg.origin });
        }
        // Under ux_v2 (#ask-stage present), the main stage owns the full ask UI.
        // Replace the duplicate card in the chat transcript with a compact pointer
        // so the surrounding context (agent reasoning, tool calls) stays readable
        // but the choice buttons aren't shown twice.
        if (document.getElementById('ask-stage')) {
            const ptr = document.createElement('div');
            ptr.className = 'ac-ask-pointer';
            ptr.dataset.reqId = reqId;
            ptr.textContent = '↑ Gently is asking — answer above';
            log.appendChild(ptr);
            scrollToBottom();
            return;
        }
        // v1 / non-ux_v2: render full card as before.
        const card = buildAskCard(data, {
            reqId, isWake, hasControl,
            onPick: (sel) => answerChoice(reqId, sel),
        });
        // ASK approvals pin to the sticky slot above the composer; ordinary
        // choices stay inline in the transcript.
        if (isWake && pendingSlot) {
            pendingSlot.innerHTML = '';
            pendingSlot.appendChild(card);
            pendingSlot.classList.remove('hidden');
            return;
        }
        log.appendChild(card);
        scrollToBottom();
    }

    function renderSpec(spec) {
        const prov = spec.provenance || {};
        const rows = [];
        // (label, value, fieldKey) — fieldKey ties a row to its provenance entry.
        const add = (label, value, key) => {
            if (value === undefined || value === null || value === '') return;
            rows.push({ label, value, src: key ? prov[key] : null });
        };
        add('Strain', spec.strain, 'strain');
        add('Genotype', spec.genotype, 'genotype');
        add('Reporter', spec.reporter, 'reporter');
        add('Channel', spec.laser_wavelength_nm != null ? `${spec.laser_wavelength_nm} nm` : null, 'laser_wavelength_nm');
        add('Temperature', spec.temperature_c != null ? `${spec.temperature_c} °C` : null, 'temperature_c');
        add('Slices', spec.num_slices, 'num_slices');
        add('Exposure', spec.exposure_ms != null ? `${spec.exposure_ms} ms` : null, 'exposure_ms');
        add('Interval', spec.interval_s != null ? `${spec.interval_s} s` : null, 'interval_s');
        add('Stop at', spec.stop_condition, 'stop_condition');
        if (!rows.length) return;
        // A small "where did this come from" tag for inferred values.
        const srcTag = (src) => {
            if (!src || !src.source) return '';
            const where = String(src.source).split(':')[0];
            const conf = src.confidence ? ` · ${src.confidence}` : '';
            const title = escapeHtml(String(src.source) + (src.confidence ? ` (confidence: ${src.confidence})` : ''));
            return ` <span class="ac-spec-src" title="${title}">${escapeHtml(where + conf)}</span>`;
        };
        const el = document.createElement('div');
        el.className = 'ac-spec';
        el.innerHTML = '<div class="ac-spec-title">Imaging spec</div>' +
            rows.map(r => `<div class="ac-spec-row"><span>${escapeHtml(r.label)}</span><span class="ac-spec-val">${escapeHtml(r.value)}${srcTag(r.src)}</span></div>`).join('');
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
            input.placeholder = 'Message Gently…   ( / commands · @ tools )';
        } else {
            banner.classList.remove('hidden');
            const who = holderLabel || 'another session';
            input.disabled = true;
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
        renderComposerButton();  // enable/disable + send/stop mode follow control
    }

    function setBusy(busy, source) {
        agentBusy = !!busy;
        busySource = agentBusy ? (source || 'user') : null;
        if (!agentBusy) askPending = false;  // turn ended — reset ask state too
        renderComposerButton();  // morph send <-> stop
        if (agentBusy) {
            input.placeholder = (busySource === 'wake')
                ? 'Gently is acting autonomously — your message will queue'
                : 'Gently is working — your message will queue';
        } else {
            // Turn ended (completed, errored, or cancelled) — clear the working
            // indicator. Cancel emits no stream_end of its own, so without this
            // the "Working…" dots would spin forever after Stop.
            hideActivity();
            if (hasControl) input.placeholder = 'Message Gently…   ( / commands · @ tools )';
            drainQueue();  // a turn just ended — send the next queued message
        }
    }

    // While an ask is pending the agent is NOT working — it's blocked on user
    // input. Override the "working" UI markers with a calm waiting hint WITHOUT
    // changing agentBusy (queuing semantics stay intact; only visuals change).
    function setAskState(waiting) {
        askPending = waiting;
        if (waiting) {
            if (hasControl) input.placeholder = '↑ Type an answer or pick an option above…';
        } else if (agentBusy) {
            // Restore working-state visuals now the ask has been answered.
            if (hasControl) {
                input.placeholder = (busySource === 'wake')
                    ? 'Gently is acting autonomously — your message will queue'
                    : 'Gently is working — your message will queue';
            }
        }
        renderComposerButton();  // reflect the send/stop mode for the new ask state
    }

    /** Set the composer button to send (up-arrow) or stop (square) mode. */
    function renderComposerButton() {
        if (!sendBtn) return;
        // During a pending ask the agent is blocked on the user — offer Send (to
        // answer the ask), not Stop, even though agentBusy is still set.
        const stopMode = agentBusy && busySource === 'user' && !askPending;
        sendBtn.classList.toggle('is-stop', stopMode);
        if (stopMode) {
            sendBtn.disabled = false;
            sendBtn.setAttribute('aria-label', 'Stop');
            sendBtn.title = 'Stop the current turn';
        } else {
            // Send is enabled only with control and some text to send.
            sendBtn.disabled = !hasControl || input.value.trim() === '';
            sendBtn.setAttribute('aria-label', 'Send message');
            sendBtn.title = 'Send';
        }
    }

    /** Abort the current cancellable turn and clear local busy/indicator state. */
    function cancelTurn() {
        send({ type: 'cancel' });
        setBusy(false);
        clearPendingAsks();
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
        // Feed the shared connection store (agent /ws/agent liveness).
        if (typeof ConnectionStatus !== 'undefined') ConnectionStatus.setAgent(ok);
    }

    // ── Transport ─────────────────────────────────────────────
    function send(obj) {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
    }

    // Flush programmatic sends queued before the socket opened (see runCommand).
    function flushProgrammatic() {
        if (!pendingProgrammatic.length) return;
        pendingProgrammatic.splice(0).forEach(t => actuallySend(t));
    }

    // Mirror the agent stream onto ClientEventBus so the ux_v2 plan wizard can
    // render a tidy activity feed (collapsible tool cards) without a second
    // socket. Additive — the chat-log rendering in handle() is unchanged.
    function emitActivity(kind, extra) {
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.emit('AGENT_ACTIVITY', Object.assign({ kind }, extra || {}));
        }
    }

    function connect() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        setConn(false, 'Connecting…');
        ws = new WebSocket(`${proto}//${location.host}/ws/agent`);
        ws.onopen = () => { reconnectDelay = 1000; setConn(true); flushProgrammatic(); };
        ws.onclose = () => {
            setConn(false, 'Reconnecting…');
            setBusy(false);
            streaming = false;
            hideActivity();
            clearPendingAsks();  // stale asks: clear the stage on socket drop
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
        // Remember collapse state so a reload restores it (defaults to open).
        try { localStorage.setItem('gently-chat-open', panelOpen ? '1' : '0'); } catch (_) {}
        if (railBtn) railBtn.setAttribute('aria-expanded', panelOpen ? 'true' : 'false');
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
        } catch (_) {}
        // The agent panel is always docked — a real column that pushes content,
        // not a float over it. It's open by default; the header Agent toggle /
        // Ctrl+J / × collapse it to width 0 to reclaim space for the viewer.
        document.body.classList.add('chat-docked');
        let open = true;
        try { open = localStorage.getItem('gently-chat-open') !== '0'; } catch (_) {}
        togglePanel(open);
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
        railBtn = document.getElementById('agent-rail-toggle');    // collapsed-rail spark
        resizeEl = document.getElementById('agent-chat-resize');
        // Connection dot + unseen-activity badge now live on the collapsed rail.
        toggleDot = document.getElementById('agent-rail-dot');
        toggleBadge = document.getElementById('agent-rail-badge');
        if (!panel) return;  // markup not present

        restorePrefs();
        // Dual-render: retire a transcript ask card when its ask is answered
        // (from the transcript OR the main stage) or the turn is cancelled.
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('ASK_CLEARED', ({ request_id }) => {
                markAnswered(request_id);
                // If answered from the main stage (AskStage), restore working state.
                if (askPending) setAskState(false);
            });
        }
        if (railBtn) railBtn.addEventListener('click', () => togglePanel(true));
        closeBtn.addEventListener('click', () => togglePanel(false));
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

            // (Stop is no longer a separate button — the composer send button
            // morphs into a stop square while a cancellable turn runs.)

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

        // One button, two roles: stop a running cancellable turn, else send.
        sendBtn.addEventListener('click', () => {
            if (agentBusy && busySource === 'user') cancelTurn();
            else submit();
        });
        input.addEventListener('input', () => { autosize(); updateCompletions(); renderComposerButton(); });
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
                e.preventDefault(); cancelTurn();
            }
        });
        renderComposerButton();  // initial state: disabled until there's text
    }

    document.addEventListener('DOMContentLoaded', init);

    // Public: programmatically send a message/command (e.g. the Home page's
    // "Start / continue an experiment" button sends '/wizard').
    function runCommand(text) {
        if (!text) return;
        if (!hasControl) { renderControl(); return; }
        // Works whether or not the chat panel is open, so the landing can drive
        // the agent (enter plan mode) without foregrounding the chat REPL. If the
        // socket isn't up yet, queue and connect — it flushes on open.
        if (ws && ws.readyState === WebSocket.OPEN) { actuallySend(text); return; }
        pendingProgrammatic.push(text);
        if (!ws) connect();
    }

    return { togglePanel, runCommand, buildAskCard, answerChoice, mdToHtml, hasControl: () => hasControl };
})();
