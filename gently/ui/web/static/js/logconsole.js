/**
 * Process console — the header drawer over the agent and device-layer output.
 *
 * The desktop shell spawns the backend with no console window, so when
 * something misbehaves mid-session there is nowhere to look. This reads the
 * same two streams the operator would have had on a terminal:
 *
 *   Agent         {storage}/logs/gently_*.log        via /api/logs/agent
 *   Device layer  supervisor-captured stdout          via /api/device-layer/log,
 *                 falling back to the layer's own log file when it runs
 *                 externally and the supervisor captured nothing.
 *
 * Polling runs ONLY while the drawer is open. A console that keeps fetching
 * behind a closed panel is exactly the kind of hidden work that made the camera
 * streams fight each other.
 */
const LogConsole = (function () {
    const POLL_MS = 2000;
    const LIMIT = 600;

    let _wired = false, _open = false, _src = 'agent', _follow = true;
    let _timer = null, _inFlight = false, _dirty = false, _lastKey = '';
    const D = {};

    const $ = id => document.getElementById(id);

    function cacheDom() {
        ['logc', 'logc-open', 'logc-scrim', 'logc-close', 'logc-body', 'logc-file',
         'logc-status', 'logc-level', 'logc-follow', 'logc-copy', 'logc-dot']
            .forEach(id => { D[id] = $(id); });
    }

    // ── fetching ──────────────────────────────────────────────────────────
    async function fetchAgent(level) {
        const q = new URLSearchParams({ limit: String(LIMIT) });
        if (level) q.set('level', level);
        const r = await fetch(`/api/logs/agent?${q}`);
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
    }

    async function fetchDevice(level) {
        // Prefer the supervisor's live captured stdout — it exists even before
        // anything has been flushed to a file.
        try {
            const r = await fetch(`/api/device-layer/log?limit=${LIMIT}`);
            if (r.ok) {
                const d = await r.json();
                if (d && Array.isArray(d.lines) && d.lines.length) {
                    return { file: 'device layer (captured stdout)', lines: d.lines };
                }
            }
        } catch (_) { /* fall through to the file */ }
        const q = new URLSearchParams({ limit: String(LIMIT) });
        if (level) q.set('level', level);
        const r2 = await fetch(`/api/logs/device?${q}`);
        if (!r2.ok) throw new Error(`${r2.status}`);
        return r2.json();
    }

    async function refresh() {
        if (!_open) return;
        // A refresh asked for while a poll is in flight must not be dropped —
        // changing the tab or the level filter would otherwise appear to do
        // nothing until the next tick. Remember it and re-run on completion.
        if (_inFlight) { _dirty = true; return; }
        _inFlight = true;
        const level = D['logc-level'] ? D['logc-level'].value : '';
        try {
            const d = _src === 'agent' ? await fetchAgent(level) : await fetchDevice(level);
            render(d);
            setStatus('');
        } catch (e) {
            setStatus(`could not read log (${e.message})`);
        } finally {
            _inFlight = false;
            if (_dirty) { _dirty = false; refresh(); }
        }
    }

    // ── rendering ─────────────────────────────────────────────────────────
    function render(d) {
        const lines = (d && d.lines) || [];
        D['logc-file'].textContent = d && d.file ? d.file : 'no log file yet';
        // Skip the DOM write when nothing changed, so following does not fight
        // a user who has scrolled up to read something.
        const key = `${lines.length}|${lines[lines.length - 1] || ''}`;
        if (key === _lastKey) return;
        _lastKey = key;

        if (!lines.length) {
            // An empty result under a filter is not an empty log — say which.
            const level = D['logc-level'] ? D['logc-level'].value : '';
            if (level) {
                D['logc-body'].textContent =
                    `No ${level.toLowerCase()} lines in the last ${LIMIT}. Set the filter to All to see everything.`;
            } else {
                D['logc-body'].textContent = _src === 'agent'
                    ? 'Nothing logged yet. The agent writes to {storage}/logs/gently_*.log.'
                    : 'Nothing logged yet. Start the device layer from the Devices tab.';
            }
            return;
        }

        const frag = document.createDocumentFragment();
        for (const ln of lines) {
            const row = document.createElement('span');
            row.className = 'logc-line' + severityClass(ln);
            row.textContent = ln;
            frag.appendChild(row);
        }
        D['logc-body'].replaceChildren(frag);

        if (_follow) D['logc-body'].scrollTop = D['logc-body'].scrollHeight;
        flagProblems(lines);
    }

    function severityClass(ln) {
        if (/\b(ERROR|CRITICAL|Traceback)\b/.test(ln)) return ' is-error';
        if (/\bWARNING\b/.test(ln)) return ' is-warn';
        return '';
    }

    // A quiet dot on the header button when the tail contains errors, so a
    // failure that happens while the drawer is closed is still noticed.
    function flagProblems(lines) {
        if (!D['logc-dot']) return;
        const bad = lines.slice(-80).some(ln => /\b(ERROR|CRITICAL|Traceback)\b/.test(ln));
        D['logc-dot'].hidden = !bad;
    }

    function setStatus(msg) {
        if (D['logc-status']) D['logc-status'].textContent = msg || '';
    }

    // ── open / close ──────────────────────────────────────────────────────
    function open() {
        if (_open) return;
        _open = true;
        D['logc'].hidden = false;
        D['logc-open'].setAttribute('aria-expanded', 'true');
        _lastKey = '';
        refresh();
        _timer = setInterval(refresh, POLL_MS);
        D['logc-body'].focus();
    }

    function close() {
        if (!_open) return;
        _open = false;
        D['logc'].hidden = true;
        D['logc-open'].setAttribute('aria-expanded', 'false');
        if (_timer) { clearInterval(_timer); _timer = null; }
    }

    function toggle() { _open ? close() : open(); }

    function selectSource(src) {
        if (src === _src) return;
        _src = src;
        _lastKey = '';
        document.querySelectorAll('.logc-tab').forEach(t =>
            t.classList.toggle('is-active', t.dataset.src === src));
        D['logc-body'].textContent = '';
        refresh();
    }

    function wire() {
        if (_wired) return;
        cacheDom();
        if (!D['logc'] || !D['logc-open']) return;   // page without the header
        _wired = true;

        D['logc-open'].addEventListener('click', toggle);
        D['logc-close'].addEventListener('click', close);
        D['logc-scrim'].addEventListener('click', close);
        document.querySelectorAll('.logc-tab').forEach(t =>
            t.addEventListener('click', () => selectSource(t.dataset.src)));
        D['logc-level'].addEventListener('change', () => { _lastKey = ''; refresh(); });

        D['logc-follow'].addEventListener('click', () => {
            _follow = !_follow;
            D['logc-follow'].setAttribute('aria-pressed', String(_follow));
            D['logc-follow'].classList.toggle('is-on', _follow);
            if (_follow) D['logc-body'].scrollTop = D['logc-body'].scrollHeight;
        });
        D['logc-follow'].classList.add('is-on');

        // Scrolling up is an intent to read — stop yanking the view to the end.
        D['logc-body'].addEventListener('scroll', () => {
            const el = D['logc-body'];
            const atEnd = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
            if (_follow !== atEnd) {
                _follow = atEnd;
                D['logc-follow'].setAttribute('aria-pressed', String(_follow));
                D['logc-follow'].classList.toggle('is-on', _follow);
            }
        });

        D['logc-copy'].addEventListener('click', async () => {
            try {
                await navigator.clipboard.writeText(D['logc-body'].textContent || '');
                setStatus('copied');
                setTimeout(() => setStatus(''), 1500);
            } catch (_) { setStatus('copy blocked by the browser'); }
        });

        document.addEventListener('keydown', e => {
            if (e.key === 'Escape' && _open) { e.preventDefault(); close(); return; }
            // Ctrl/Cmd+` — the shell convention for "show me the console".
            if ((e.ctrlKey || e.metaKey) && e.key === '`') { e.preventDefault(); toggle(); }
        });
    }

    document.addEventListener('DOMContentLoaded', wire);
    return { open, close, toggle };
})();
