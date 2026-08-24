/**
 * The Atrium — a spatial shell for the Gently web UI.
 *
 * Spec: docs/atrium/SPEC.md.  Reference implementation: docs/atrium/canvas-surface.html.
 *
 * A spatial interface in which every capability permanently exists as a framed
 * window; the only thing that changes is attention. An open middle (the BENCH)
 * ringed by fixed structure (the COURTYARD).
 *
 * This adopts the EXISTING .tab-content divs as windows rather than rewriting
 * them. That is deliberate and it is what makes the port cheap: the ten divs
 * are already all in the DOM simultaneously — tabs only hid them with a class.
 * "Everything exists" is not a change we have to make, it is already true. We
 * are deleting the hiding.
 *
 * OFF BY DEFAULT. Enable with ?atrium=1, disable with ?atrium=0; the choice
 * sticks in localStorage. With it off this file registers nothing and the
 * tabbed UI is untouched.
 */
const Atrium = (() => {
    'use strict';

    const FLAG_KEY = 'gently.atrium.enabled';
    const SAVE_KEY = 'gently.atrium.layout.v1';

    /* ── CONFIG — the whole surface is described here (SPEC R9) ─────── */
    const CONFIG = {
        courtyard: {
            pad: 10, gap: 8,
            edges: {
                rail: { side: 'right', size: '300px' },
                deck: { side: 'bottom', size: 'auto', panelWidth: '260px' },
            },
            corners: 'sides',
        },
        // Windows are adopted from #<tab>-content. crit/tol drive SPEC R5.
        windows: [
            { tab: 'home',        title: 'HOME',        x: 40,   y: 200, w: 460, h: 380, crit: 0.3 },
            { tab: 'embryos',     title: 'EMBRYOS',     x: 540,  y: 40,  w: 700, h: 620, crit: 0.8, children: [
                { key: 'default',   title: 'default',   go: () => EmbryosManager.switchView('default') },
                { key: 'board',     title: 'board',     go: () => EmbryosManager.switchView('board') },
                { key: 'filmstrip', title: 'filmstrip', go: () => EmbryosManager.switchView('filmstrip') },
                { key: 'vitals',    title: 'vitals',    go: () => EmbryosManager.switchView('vitals') },
              ] },
            { tab: 'devices',     title: 'DEVICES',     x: 1280, y: 40,  w: 700, h: 620, crit: 0.9, pin: 'rail',
              // a real fact that genuinely goes stale: the scope dropped off
              tol: 20, pressWhen: () => !ConnectionStatus.get().microscopeConnected },
            { tab: 'calibration', title: 'CALIBRATION', x: 540,  y: 700, w: 700, h: 460, crit: 0.85, children: [
                { key: 'profile', title: 'profile', go: () => CalibrationManager.switchView('profile') },
                { key: 'gallery', title: 'gallery', go: () => CalibrationManager.switchView('gallery') },
              ] },
            { tab: 'experiment',  title: 'EXPERIMENT',  x: 1280, y: 700, w: 700, h: 460, crit: 0.6, children: [
                { key: 'overview', title: 'overview', go: () => ExperimentOverview.setView('overview') },
                { key: 'rules',    title: 'rules',    go: () => ExperimentOverview.setView('rules') },
              ] },
            { tab: 'events',      title: 'EVENTS',      x: 40,   y: 620, w: 460, h: 400, crit: 0.4,
              tol: 45, pressWhen: () => !ConnectionStatus.get().gentlyConnected, children: [
                { key: 'log',      title: 'log',      go: () => switchSystemView('log') },
                { key: 'timeline', title: 'timeline', go: () => switchSystemView('timeline') },
                { key: 'summary',  title: 'summary',  go: () => switchSystemView('summary') },
              ] },
            { tab: 'plans',       title: 'PLANS',       x: 2020, y: 40,  w: 620, h: 620, crit: 0.5, children: [
                { key: 'doc',      title: 'document',  click: 'plan-view-switcher' },
                { key: 'graph',    title: 'graph',     click: 'plan-view-switcher' },
                { key: 'board',    title: 'board',     click: 'plan-view-switcher' },
                { key: 'decide',   title: 'decisions', click: 'plan-view-switcher' },
                { key: 'matrix',   title: 'matrix',    click: 'plan-view-switcher' },
                { key: 'timeline', title: 'timeline',  click: 'plan-view-switcher' },
              ] },
            { tab: 'sessions',    title: 'SESSIONS',    x: 2020, y: 700, w: 620, h: 300, crit: 0.3 },
            { tab: 'notebook',    title: 'NOTEBOOK',    x: 40,   y: 1060, w: 460, h: 400, crit: 0.4 },
            { tab: 'gallery',     title: 'GALLERY',     x: 540,  y: 1200, w: 700, h: 400, crit: 0.4 },
        ],
        home: { cx: 940, cy: 560, scale: 0.58 },

        /* R5/R6. urgency = crit x overdue, zero when fresh; the ladder is a
           threshold on that one number, not an authored policy table. The cap
           is the whole safety argument for letting an agent drive this. */
        release: {
            ladder: [
                { at: 0,    channel: 'gauge' },
                { at: 0.85, channel: 'chip' },
                { at: 1.25, channel: 'open' },
                { at: 1.80, channel: 'offer' },
                { at: 2.50, channel: 'seize' },
                { at: 3.30, channel: 'notify' },
                { at: 4.30, channel: 'email' },
            ],
            maxChannel: 'open',      // conservative until an operator has seen it
            overdueCap: 3,
            tickMs: 2000,
        },
        density: 4,
        minScale: 0.12, maxScale: 2.5,
    };

    /* ── surface state ───────────────────────────────────────────────── */
    let vp, board, host, on = false;
    let vx = 0, vy = 0, scale = 1, glideGen = 0, tipDrop = null;
    const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
    const wins = new Map();            // tab -> {el, frame, cfg}
    const TRAIL = [];
    let travelling = false;

    /* ── R1: the bench is a transform ────────────────────────────────── */
    function apply() {
        board.style.transform = `translate(${vx}px,${vy}px) scale(${scale})`;
        const m = document.getElementById('atr-masked');
        if (m) {
            m.style.setProperty('--atr-gs', 80 * scale + 'px');
            m.style.setProperty('--atr-gx', vx + 'px');
            m.style.setProperty('--atr-gy', vy + 'px');
        }
        const z = document.getElementById('atr-zoom');
        if (z) z.textContent = Math.round(scale * 100) + '%';
        // A CSS transform fires neither resize nor ResizeObserver, so anything
        // holding a backing store has to be told. (SPEC R10 / migration cat. f)
        window.dispatchEvent(new CustomEvent('atrium:transform', { detail: { scale, vx, vy } }));
        if (tipDrop) { clearTimeout(tipDrop); }
        tipDrop = setTimeout(dropStaleTooltips, 60);   // coalesce across a glide
    }

    const stopGlide = () => { glideGen++; };

    function glide(tx, ty, ts) {
        const gen = ++glideGen, sx = vx, sy = vy, ss = scale, t0 = performance.now(), D = 420;
        (function step(now) {
            if (gen !== glideGen) return;                       // superseded
            const k = Math.min(1, (now - t0) / D), e = 1 - Math.pow(1 - k, 3);
            vx = sx + (tx - sx) * e; vy = sy + (ty - sy) * e; scale = ss + (ts - ss) * e;
            apply();
            if (k < 1) requestAnimationFrame(step);
        })(performance.now());
    }

    function zoomAt(cx, cy, k) {
        stopGlide();
        const ns = clamp(scale * k, CONFIG.minScale, CONFIG.maxScale);
        vx = cx - (cx - vx) * (ns / scale);
        vy = cy - (cy - vy) * (ns / scale);
        scale = ns; apply();
    }

    /* ── framing ─────────────────────────────────────────────────────── */
    const onBench = () => [...board.children].filter(e => e.classList.contains('atr-win'));

    function frameOn(els, pad = 90) {
        els = els.filter(e => e && !e.dataset.slot);
        if (!els.length) return;
        const x1 = Math.min(...els.map(e => e.offsetLeft));
        const y1 = Math.min(...els.map(e => e.offsetTop));
        const x2 = Math.max(...els.map(e => e.offsetLeft + e.offsetWidth));
        const y2 = Math.max(...els.map(e => e.offsetTop + e.offsetHeight));
        const ts = clamp(Math.min(innerWidth / (x2 - x1 + pad * 2), innerHeight / (y2 - y1 + pad * 2)),
                         CONFIG.minScale, CONFIG.maxScale);
        glide((innerWidth - (x2 - x1) * ts) / 2 - x1 * ts,
              (innerHeight - (y2 - y1) * ts) / 2 - y1 * ts, ts);
    }

    /* ── R2: attention travels; nothing is created or destroyed ──────── */
    function attend(tab, opts = {}) {
        if (typeof tab === 'string' && tab.includes(':')) {   // R8: a child is a destination
            const [parent, kid] = tab.split(':');
            const pw = wins.get(parent);
            if (!pw) return false;
            attend(parent, opts);
            return activateChild(pw.frame, kid);
        }
        const w = wins.get(tab);
        if (!w) return false;
        if (!travelling) {
            if (TRAIL[TRAIL.length - 1] !== tab) TRAIL.push(tab);
            if (TRAIL.length > 40) TRAIL.shift();
            paintTrail();
        }
        document.body.classList.add('atr-focused');
        wins.forEach((v, t) => {
            const hit = t === tab;
            v.frame.classList.toggle('atr-attend', hit);
            if (hit) {
                // R4 says travelling opens what you land on — but a pinned
                // window cannot be worked in at rail width, so travel brings
                // it home to the bench first.
                if (v.frame.dataset.slot) unpin(v.frame);
                fold(v.frame, false);
            }
        });
        // keep the legacy chokepoint honest: .active and TAB_CHANGED still fire
        document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
        if (typeof state !== 'undefined') state.tab = tab;
        if (typeof ClientEventBus !== 'undefined') ClientEventBus.emit('TAB_CHANGED', tab);
        lazyInit(tab);
        if (!opts.noGlide) frameOn([w.frame], 110);
        return true;
    }

    function bench() {
        document.body.classList.remove('atr-focused');
        wins.forEach(v => v.frame.classList.remove('atr-attend'));
        frameOn(onBench(), 70);
    }

    function goHome(anim = true) {
        const { cx, cy, scale: hs } = CONFIG.home;
        const tx = innerWidth / 2 - cx * hs, ty = innerHeight / 2 - cy * hs;
        document.body.classList.remove('atr-focused');
        wins.forEach(v => v.frame.classList.remove('atr-attend'));
        if (anim) glide(tx, ty, hs); else { stopGlide(); vx = tx; vy = ty; scale = hs; apply(); }
    }

    function back() {
        if (TRAIL.length < 2) return false;
        TRAIL.pop();
        travelling = true; attend(TRAIL[TRAIL.length - 1]); travelling = false;
        paintTrail(); return true;
    }

    function paintTrail() {
        const b = document.getElementById('atr-back');
        if (b) b.disabled = TRAIL.length < 2;
    }

    /* The tab shell lazily initialised some panels on first show. Attention
       is now the trigger, so the same hooks fire on first travel. */
    const inited = new Set();
    function lazyInit(tab) {
        if (inited.has(tab)) return;
        inited.add(tab);
        try {
            if (tab === 'home' && typeof HomeApp !== 'undefined') HomeApp.init();
            if (tab === 'calibration' && typeof renderCalibrationGallery === 'function') renderCalibrationGallery();
            if (tab === 'events' && typeof renderEventsTable === 'function') renderEventsTable();
            if (tab === 'plans' && typeof CampaignsApp !== 'undefined') CampaignsApp.init();
            if (tab === 'embryos' && typeof EmbryosManager !== 'undefined') EmbryosManager.clearDetectionBadge?.();
        } catch (e) {
            console.warn('[atrium] lazy init failed for', tab, e);
        }
    }

    /* ── R4: folded is a rendering, not a hidden state ───────────────── */
    const FOLDED_H = 32;
    function fold(frame, on_) {
        const was = frame.classList.contains('atr-folded');
        frame.classList.toggle('atr-folded', on_);
        frame.style.height = (on_ ? FOLDED_H : +frame.dataset.h) + 'px';
        // Unfolding gives a window a box it did not have. occupancy3d.js and
        // projection-viewer.js already listen for gently:layout-changed to
        // re-measure, so reuse that convention rather than inventing one.
        if (was && !on_) setTimeout(() => window.dispatchEvent(new Event('gently:layout-changed')), 280);
    }
    const salience = f => (+f.dataset.crit || 0);
    function setDensity(n) {
        const ranked = onBench().sort((a, b) => salience(b) - salience(a));
        ranked.forEach((f, i) => fold(f, i >= n));
        const l = document.getElementById('atr-density-n');
        if (l) l.textContent = n;
        save();
    }

    /* ── R4 corollary: a folded window is a live gauge, not a blank ──── */
    function gauge(tab, key, fmt) {
        const w = wins.get(tab);
        const peek = w && w.frame.querySelector('.atr-peek');
        if (!peek) return;
        SharedState.on(key, v => { peek.textContent = fmt(v); });   // sticky: paints now
    }

    /* ── compatibility: the transform containing-block trap ──────────
       A CSS transform makes an ancestor the containing block for
       position:fixed descendants. So a "full-screen" overlay created INSIDE a
       window is not viewport-fixed at all — it is scaled by the bench, offset
       by the pan, and then clipped by the window's overflow:hidden. Measured
       in situ: a fixed 50px box at (0,0) rendered at (20,53) and 33px wide.

       Every overlay in the app today happens to be parented to <body> or to
       #app-main, so nothing is broken right now. That is luck, not design —
       the next modal written inside a panel breaks silently and ONLY under the
       Atrium, which is the worst kind of bug to inherit. So: portal it out and
       say so loudly. This is a detector as much as a fix. */
    let portalObserver = null;
    function portalFixedOverlays() {
        const escape = el => {
            if (!(el instanceof HTMLElement) || !el.closest('#atr-board')) return;
            if (getComputedStyle(el).position !== 'fixed') return;
            console.warn('[atrium] portalled a position:fixed overlay out of the bench —',
                el.id || el.className,
                '\n  A transformed ancestor is its containing block, so inside a window it',
                '\n  would be scaled, offset and clipped. Parent overlays to <body>.');
            document.body.appendChild(el);
        };
        portalObserver = new MutationObserver(muts => {
            for (const m of muts) {
                m.addedNodes.forEach(n => {
                    escape(n);
                    if (n instanceof HTMLElement) n.querySelectorAll('*').forEach(escape);
                });
            }
        });
        portalObserver.observe(board, { childList: true, subtree: true });
        // anything already trapped at enable() time
        board.querySelectorAll('*').forEach(escape);
    }

    /* A tooltip anchored to a window is stale the moment the bench moves. Its
       own dismissal triggers are mouseleave and a capturing scroll listener,
       and a transform change is NEITHER — so it would hang in space while its
       anchor slides away. Cheaper to drop them here than to teach app.js's
       Tooltips module about the bench. */
    function dropStaleTooltips() {
        document.querySelectorAll('.tooltip').forEach(t => t.remove());
    }

    /* ── R5: pressure ────────────────────────────────────────────────
       A window presses when its own predicate says a fact is outstanding.
       urgency = crit x overdue. Zero when fresh — importance amplifies
       urgency, it does not manufacture it. Adding a source is one line:
       give a window `tol` and `pressWhen`. */
    const RELEASES = [];
    let pressTimer = null;

    function urgency(f) {
        const tol = +f.dataset.tol || 0;
        if (!tol || !f._since) return 0;                 // not pressing
        const overdue = Math.min((Date.now() - f._since) / 1000 / tol, CONFIG.release.overdueCap);
        return +((+f.dataset.crit || 0) * Math.max(0, overdue)).toFixed(3);
    }
    function channelFor(u) {
        const L = CONFIG.release.ladder;
        const cap = L.findIndex(r => r.channel === CONFIG.release.maxChannel);
        let i = 0; while (i + 1 < L.length && u >= L[i + 1].at) i++;
        return Math.min(i, cap < 0 ? L.length - 1 : cap);
    }

    /* R6: release at the lowest channel that will still be seen in time.
       A window only climbs — one release per rung, so nothing spams. */
    function pressTick() {
        wins.forEach(({ frame: f, cfg }) => {
            if (!cfg.pressWhen) return;
            let pressing = false;
            try { pressing = !!cfg.pressWhen(); } catch (_) { pressing = false; }
            if (!pressing) {                              // resolved is not refreshed
                if (f._since) { f._since = 0; f._rung = 0; f.classList.remove('atr-calling'); }
                return;
            }
            if (!f._since) f._since = Date.now();
            const u = urgency(f), rung = channelFor(u);
            if (rung <= (f._rung || 0)) return;
            f._rung = rung;
            const ch = CONFIG.release.ladder[rung].channel;
            RELEASES.unshift({ t: new Date().toLocaleTimeString(), tab: cfg.tab, u: u.toFixed(2), ch });
            if (ch === 'chip') f.classList.add('atr-calling');
            if (ch === 'open') {
                // The courtyard is for gauges. A rail is 300px and DEVICES wants
                // 553px, so unfolding a pinned window IN PLACE just cramps it —
                // measured: devices-container overflowing its rail by 262px.
                // Working on something means bringing it to the bench.
                if (f.dataset.slot) unpin(f);
                fold(f, false);
                f.classList.add('atr-calling');
            }
            if (ch === 'offer' || ch === 'seize') attend(cfg.tab);
            paintReleases();
        });
    }
    function paintReleases() {
        const el = document.getElementById('atr-releases');
        if (!el) return;
        el.textContent = RELEASES.length ? `${RELEASES[0].ch} · ${RELEASES[0].tab}` : '';
        el.title = RELEASES.slice(0, 8).map(r => `${r.t}  ${r.ch}  ${r.tab}  u=${r.u}`).join('\n')
                 || 'nothing released';
        el.hidden = !RELEASES.length;
    }

    /* ── R9: the courtyard is generated from config ──────────────────── */
    function buildCourtyard(c) {
        host.querySelectorAll('.atr-win').forEach(f => unpin(f));      // evacuate first
        const chrome = host.querySelector('#atr-chrome');
        if (chrome) document.body.appendChild(chrome);
        host.innerHTML = '';
        const at = side => Object.keys(c.edges).find(k => c.edges[k].side === side);
        const [L, R, T, B] = ['left', 'right', 'top', 'bottom'].map(at);
        const size = k => (k ? c.edges[k].size : '0px'), cell = k => k || '.';
        const sides = c.corners === 'sides';
        const rows = sides
            ? [[cell(L), cell(T), cell(R)], [cell(L), 'free', cell(R)], [cell(L), cell(B), cell(R)]]
            : [[cell(T), cell(T), cell(T)], [cell(L), 'free', cell(R)], [cell(B), cell(B), cell(B)]];
        Object.assign(host.style, {
            padding: c.pad + 'px', gap: c.gap + 'px',
            gridTemplateColumns: `${size(L)} 1fr ${size(R)}`,
            gridTemplateRows: `${size(T)} 1fr ${size(B)}`,
            gridTemplateAreas: rows.map(r => `"${r.join(' ')}"`).join(' '),
        });
        for (const [name, e] of Object.entries(c.edges)) {
            const d = document.createElement('div');
            d.className = 'atr-slot ' + (e.side === 'left' || e.side === 'right' ? 'vert' : 'horz');
            d.id = 'atr-slot-' + name; d.style.gridArea = name;
            if (e.panelWidth) d.style.setProperty('--atr-pw', e.panelWidth);
            host.appendChild(d);
        }
        if (chrome) host.appendChild(chrome);
    }

    function pin(frame, slot) {
        const s = document.getElementById('atr-slot-' + slot);
        if (!s) return unpin(frame);                       // edge no longer exists
        if (!frame.dataset.ox) { frame.dataset.ox = frame.offsetLeft; frame.dataset.oy = frame.offsetTop; }
        frame.dataset.slot = slot; s.appendChild(frame); fold(frame, true);
    }
    function unpin(frame) {
        if (!frame.dataset.slot) return;
        frame.style.left = frame.dataset.ox + 'px';
        frame.style.top = frame.dataset.oy + 'px';
        delete frame.dataset.slot; board.appendChild(frame);
    }

    /* ── adoption: wrap each existing .tab-content as a window ───────── */
    function adopt(cfg) {
        const content = document.getElementById(cfg.tab + '-content');
        if (!content) return null;
        const f = document.createElement('div');
        f.className = 'atr-win'; f.id = 'atr-' + cfg.tab;
        f.dataset.h = cfg.h; f.dataset.crit = cfg.crit ?? 0.5; f.dataset.tab = cfg.tab;
        if (cfg.tol) f.dataset.tol = cfg.tol;
        f._since = 0; f._rung = 0;
        f.style.cssText = `left:${cfg.x}px;top:${cfg.y}px;width:${cfg.w}px;height:${cfg.h}px`;
        f.innerHTML = `<div class="atr-head"><button class="atr-fold" title="fold / open">▼</button>`
                    + `<b>${cfg.title}</b><span class="atr-peek"></span></div>`
                    + (cfg.children ? `<div class="atr-kids"></div>` : '')
                    + `<div class="atr-body"></div>`;
        f.querySelector('.atr-body').appendChild(content);   // MOVE, never clone
        content.classList.add('active');                      // it is always live now
        f.querySelector('.atr-fold').onclick = ev => {
            ev.stopPropagation(); fold(f, !f.classList.contains('atr-folded')); save();
        };
        const head = f.querySelector('.atr-head');
        head.addEventListener('pointerdown', e => {
            if (e.target.tagName === 'BUTTON') return;
            head.setPointerCapture(e.pointerId);
            const ox = e.clientX, oy = e.clientY, px = f.offsetLeft, py = f.offsetTop;
            const mv = m => {                                 // /scale: track the cursor at any zoom
                f.style.left = px + (m.clientX - ox) / scale + 'px';
                f.style.top = py + (m.clientY - oy) / scale + 'px';
            };
            head.addEventListener('pointermove', mv);
            head.addEventListener('pointerup', () => { head.removeEventListener('pointermove', mv); save(); }, { once: true });
        });
        if (cfg.children) {
            f._kids = cfg.children;
            f._active = cfg.children[0].key;
            paintKids(f);
        }
        board.appendChild(f);
        return f;
    }

    /* R8: a child is a full destination, not a click-path. The strip calls the
       panel's OWN switcher, so no panel logic changes — the win is that
       attend('embryos:vitals') addresses it from anywhere on the surface. */
    function paintKids(f) {
        const strip = f.querySelector('.atr-kids');
        if (!strip) return;
        strip.innerHTML = f._kids.map(k =>
            `<span class="atr-kid ${k.key === f._active ? 'on' : ''}" data-kid="${k.key}">${k.title}</span>`
        ).join('');
        strip.querySelectorAll('[data-kid]').forEach(n => n.onclick = () => activateChild(f, n.dataset.kid));
    }

    function activateChild(f, key) {
        const k = (f._kids || []).find(c => c.key === key);
        if (!k) return false;
        f._active = key;
        paintKids(f);
        try {
            if (k.click) {
                // Drive the panel's own control. Every switcher in this app is
                // initViewSwitcher(id, ...) delegating on [data-view], so a
                // click reaches even a module-private handler — campaigns.js
                // keeps switchPlanView inside an IIFE and it is unreachable
                // any other way.
                const btn = f.querySelector(`#${k.click} [data-view="${k.key}"]`);
                if (btn) btn.click();
                else return false;
            } else if (k.go) {
                k.go();
            }
        } catch (e) {
            console.warn('[atrium] child', key, 'unavailable yet —', e.message);
        }
        return true;
    }

    /* ── persistence ─────────────────────────────────────────────────── */
    function save() {
        try {
            localStorage.setItem(SAVE_KEY, JSON.stringify({
                density: +(document.getElementById('atr-density')?.value || CONFIG.density),
                pos: Object.fromEntries([...wins].map(([t, w]) => [t, {
                    x: +(w.frame.dataset.ox ?? w.frame.offsetLeft),
                    y: +(w.frame.dataset.oy ?? w.frame.offsetTop),
                }])),
            }));
        } catch (_) { /* private mode */ }
    }
    function restore() {
        let d; try { d = JSON.parse(localStorage.getItem(SAVE_KEY) || 'null'); } catch (_) { return false; }
        if (!d) return false;
        for (const [t, p] of Object.entries(d.pos || {})) {
            const w = wins.get(t);
            if (w && !w.frame.dataset.slot) { w.frame.style.left = p.x + 'px'; w.frame.style.top = p.y + 'px'; }
        }
        if (d.density) { const el = document.getElementById('atr-density'); if (el) el.value = d.density; setDensity(d.density); }
        return true;
    }

    /* ── chrome: lives in the FREE cell, so no edge can collide with it ─ */
    function buildChrome() {
        const c = document.createElement('div');
        c.id = 'atr-chrome';
        c.innerHTML = `
          <div id="atr-tl">
            <div id="atr-hint">Atrium · drag to pan · wheel to zoom · Backspace back
              <a href="#" id="atr-off">exit</a></div>
            <div id="atr-chips"></div>
          </div>
          <div id="atr-br">
            <span id="atr-releases" hidden></span>
            <span id="atr-zoom">100%</span>
            <label>density <input type="range" id="atr-density" min="1" max="10"
              value="${CONFIG.density}"><b id="atr-density-n">${CONFIG.density}</b></label>
          </div>`;
        host.appendChild(c);
        const chips = c.querySelector('#atr-chips');
        // R2: any set of addressable things renders as chips that move attention
        chips.innerHTML = `<button class="atr-btn" id="atr-back" title="back" disabled>&larr;</button>`
            + `<button class="atr-btn" data-atr-go="__home">⌂ start</button>`
            + `<button class="atr-btn" data-atr-go="__bench">all</button>`
            + CONFIG.windows.filter(w => document.getElementById(w.tab + '-content'))
                .map(w => `<button class="atr-btn" data-atr-go="${w.tab}">${w.title.toLowerCase()}</button>`).join('');
        chips.querySelectorAll('[data-atr-go]').forEach(b => b.onclick = () => {
            const d = b.dataset.atrGo;
            d === '__home' ? goHome() : d === '__bench' ? bench() : attend(d);
        });
        c.querySelector('#atr-back').onclick = back;
        c.querySelector('#atr-density').oninput = e => setDensity(+e.target.value);
        c.querySelector('#atr-off').onclick = e => { e.preventDefault(); disable(); };
    }

    /* ── enable / disable ────────────────────────────────────────────── */
    function enable() {
        if (on) return;
        const shell = document.getElementById('tab-content') || document.body;

        vp = document.createElement('div'); vp.id = 'atr-viewport';
        const masked = document.createElement('div'); masked.id = 'atr-masked';
        board = document.createElement('div'); board.id = 'atr-board';
        host = document.createElement('div'); host.id = 'atr-pinned';
        masked.appendChild(board); vp.appendChild(masked);
        document.body.appendChild(vp); document.body.appendChild(host);
        document.body.classList.add('atrium-on');

        buildCourtyard(CONFIG.courtyard);
        buildChrome();

        for (const cfg of CONFIG.windows) {
            const f = adopt(cfg);
            if (f) wins.set(cfg.tab, { frame: f, cfg });
        }
        for (const cfg of CONFIG.windows) {
            if (cfg.pin && wins.has(cfg.tab)) pin(wins.get(cfg.tab).frame, cfg.pin);
        }

        gauge('embryos', 'selectedEmbryoId', v => (v ? '◆ ' + v : ''));
        gauge('devices', 'stageXY', v => (v ? `${Math.round(v.x)}, ${Math.round(v.y)} µm` : ''));

        vp.addEventListener('wheel', e => {
            if (e.target.closest('.atr-body')) return;      // frames own their own zoom
            e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0015));
        }, { passive: false });

        vp.addEventListener('pointerdown', e => {
            if (e.target.closest('.atr-win')) return;
            stopGlide(); vp.setPointerCapture(e.pointerId); vp.classList.add('panning');
            const sx = e.clientX - vx, sy = e.clientY - vy;
            const mv = m => { vx = m.clientX - sx; vy = m.clientY - sy; apply(); };
            vp.addEventListener('pointermove', mv);
            vp.addEventListener('pointerup', () => {
                vp.removeEventListener('pointermove', mv); vp.classList.remove('panning');
            }, { once: true });
        });

        addEventListener('keydown', onKey);
        // R2 corollary: switchTab keeps working, it just means "travel" now.
        if (typeof window.switchTab === 'function' && !window.__atrSwitchTab) {
            window.__atrSwitchTab = window.switchTab;
            window.switchTab = t => (on ? attend(t) : window.__atrSwitchTab(t));
        }

        on = true;
        pressTimer = setInterval(pressTick, CONFIG.release.tickMs);
        portalFixedOverlays();
        setDensity(CONFIG.density);
        restore();
        goHome(false);
        try { localStorage.setItem(FLAG_KEY, '1'); } catch (_) {}
        console.info('[atrium] on —', wins.size, 'windows adopted. ?atrium=0 to leave.');
    }

    function disable() {
        try { localStorage.setItem(FLAG_KEY, '0'); } catch (_) {}
        location.href = location.pathname + '?atrium=0';
    }

    function onKey(e) {
        if (e.metaKey || e.ctrlKey || e.altKey) return;
        const t = e.target;
        if (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(t.tagName)) return;
        if (e.key === 'Backspace') { e.preventDefault(); back(); return; }
        if (e.key === '0') { e.preventDefault(); goHome(); return; }
        if (e.key === '`') { e.preventDefault(); bench(); return; }
        const dir = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
        if (dir) { e.preventDefault(); stepDir(dir); }
    }

    /* nearest window that way, punishing lateral drift so a row reads as a row */
    function stepDir([dx, dy]) {
        const cur = board.querySelector('.atr-win.atr-attend') || onBench()[0];
        if (!cur) return;
        const cx = cur.offsetLeft + cur.offsetWidth / 2, cy = cur.offsetTop + cur.offsetHeight / 2;
        let best = null, cost = Infinity;
        for (const f of onBench()) {
            if (f === cur) continue;
            const x = f.offsetLeft + f.offsetWidth / 2, y = f.offsetTop + f.offsetHeight / 2;
            const along = (x - cx) * dx + (y - cy) * dy, lat = Math.abs((x - cx) * dy - (y - cy) * dx);
            if (along <= 8) continue;
            const c = along + lat * 2;
            if (c < cost) { cost = c; best = f; }
        }
        if (best) attend(best.dataset.tab);
    }

    function wanted() {
        const q = new URLSearchParams(location.search).get('atrium');
        if (q === '1') return true;
        if (q === '0') return false;
        try { return localStorage.getItem(FLAG_KEY) === '1'; } catch (_) { return false; }
    }

    function init() {
        if (!wanted()) return;
        if (document.readyState === 'loading') addEventListener('DOMContentLoaded', enable, { once: true });
        else enable();
    }

    return { init, enable, disable, attend, bench, home: goHome, back, setDensity,
             urgency, channelFor, pressTick, RELEASES, dropStaleTooltips,
             fold, pin, unpin, buildCourtyard, CONFIG,
             get on() { return on; }, get windows() { return wins; }, _trail: TRAIL };
})();

Atrium.init();
