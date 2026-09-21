/**
 * Temperature chart — the water trace against its setpoint, on the Devices tab.
 *
 * What it plots and why it looks like this:
 *
 * - **One series.** The water temperature is the data; the setpoint is a
 *   reference, not a peer, so it is a dashed hairline in ink rather than a
 *   second coloured line. One series needs no legend — the heading names it.
 * - **The scale never magnifies noise.** The old chart fitted the y-axis to
 *   min/max of the data, so ±0.05 °C of sensor jitter filled the plot and a
 *   rock-steady bath looked like a rollercoaster. The window is now at least
 *   MIN_SPAN_C wide and centred on the setpoint, so "flat" looks flat and a
 *   real excursion is the only thing that moves.
 * - **Time is the x-axis.** Points were previously spaced by index, so a gap in
 *   sampling (a device-layer restart) silently compressed into a normal step.
 *   Position now comes from the timestamp.
 * - **The tolerance band is the judgement.** ±TOL_C around the setpoint, drawn
 *   as a wash: inside it, the bath is where you put it. Outside, the end marker
 *   and its label take the drift colour AND say how far off — never colour alone.
 * - **Hover and keyboard read the same thing.** A crosshair snaps to the nearest
 *   sample; arrow keys walk it. Values lead, labels follow.
 *
 * Colours are Gently's own steps, checked against the chart surface in both
 * themes (single-hue series, status used one at a time with a label).
 */
const TemperatureGraph = (() => {
    const SVGNS = "http://www.w3.org/2000/svg";
    const MAX_POINTS = 600;      // rolling ~10 min @ 1 Hz
    const MIN_SPAN_C = 1.0;      // never zoom tighter than this (see above)
    const TOL_C = 0.3;           // "at setpoint" band, ± °C
    const H = 150;

    let _root = null;
    let _samples = [];
    let _session = "current";
    let _hoverIdx = null;        // crosshair position, or null for "live end"

    // The chart sits inside a fold. The fold, not this module, decides whether
    // the plot is on screen; this module only decides whether there is anything
    // to plot. Keeping those separate is why `hidden` here and the collapsed
    // class over there never fight.
    const OPEN_KEY = "gently.tempchart.open";

    function wrap() { return document.getElementById("devices-tempwrap"); }

    function setWrapVisible(on) {
        const w = wrap();
        if (w) w.hidden = !on;
    }

    function setCollapsed(collapsed) {
        const w = wrap();
        const btn = document.getElementById("devices-tempwrap-toggle");
        if (!w || !btn) return;
        w.classList.toggle("is-collapsed", collapsed);
        btn.setAttribute("aria-expanded", String(!collapsed));
        btn.title = collapsed ? "Show the last hour" : "Hide the chart";
        // Re-render on expand: the SVG is sized from clientWidth, which is 0
        // while the fold is shut.
        if (!collapsed) render();
    }

    function initFold() {
        const btn = document.getElementById("devices-tempwrap-toggle");
        if (!btn || btn.dataset.wired === "1") return;
        btn.dataset.wired = "1";
        let open = false;
        try { open = localStorage.getItem(OPEN_KEY) === "1"; } catch (e) { /* collapsed */ }
        setCollapsed(!open);
        btn.addEventListener("click", () => {
            const nowCollapsed = !wrap().classList.contains("is-collapsed");
            setCollapsed(nowCollapsed);
            try { localStorage.setItem(OPEN_KEY, nowCollapsed ? "0" : "1"); } catch (e) { /* not fatal */ }
        });
    }

    // ── small helpers ────────────────────────────────────────────────────
    const num = v => (v == null || !Number.isFinite(Number(v)) ? null : Number(v));
    const ms = s => { const t = Date.parse(s && s.t); return Number.isFinite(t) ? t : null; };

    function lastSetpoint() {
        for (let i = _samples.length - 1; i >= 0; i--) {
            const sp = num(_samples[i].setpoint_c);
            if (sp != null) return sp;
        }
        return null;
    }

    function fmtAgo(msAgo) {
        const s = Math.round(msAgo / 1000);
        if (s < 5) return "now";
        if (s < 90) return `${s}s ago`;
        const m = Math.round(s / 60);
        return m < 60 ? `${m}m ago` : `${Math.round(m / 60)}h ago`;
    }

    function el(name, attrs, cls) {
        const e = document.createElementNS(SVGNS, name);
        if (cls) e.setAttribute("class", cls);
        for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
        return e;
    }

    function renderHeadline() {
        const out = document.getElementById("devices-tempwrap-now");
        if (!out) return;
        const last = _samples[_samples.length - 1];
        const w = last ? num(last.water_c) : null;
        const sp = lastSetpoint();
        if (w == null) { out.textContent = ""; return; }
        const off = sp != null ? w - sp : null;
        out.textContent = w.toFixed(1) + "°"
            + (off == null ? ""
                : Math.abs(off) <= TOL_C ? " · at setpoint"
                    : ` · ${off > 0 ? "+" : "−"}${Math.abs(off).toFixed(1)}° off ${sp.toFixed(1)}°`);
        out.classList.toggle("is-drifting", off != null && Math.abs(off) > TOL_C);
    }

    function init(container, sessionId) {
        ClientEventBus.off("TEMPERATURE_UPDATE", onEvent);
        _root = container;
        _session = sessionId || "current";
        _samples = [];
        _hoverIdx = null;
        initFold();
        backfill();
        ClientEventBus.on("TEMPERATURE_UPDATE", onEvent);
    }

    async function backfill() {
        try {
            const r = await fetch(`/api/temperature/${_session}/history`);
            if (!r.ok) { renderEmpty(); return; }
            const body = await r.json();
            // Adopt the resolved session_id (e.g. 'current' → real id) so that
            // subsequent event filtering is consistent.
            _session = body.session_id || _session;
            _samples = (body.samples || []).slice(-MAX_POINTS);
            render();
        } catch (e) {
            console.warn('[TemperatureGraph] backfill error:', e);
            renderEmpty();
        }
    }

    function onEvent(data) {
        // data = {session_id, sample: {t, water_c, setpoint_c, state}}
        if (!data || !data.sample) return;
        _samples.push(data.sample);
        if (_samples.length > MAX_POINTS) _samples.shift();
        render();
    }

    // Presence-driven (docs/architecture/PANELS.md rule 6). This used to render
    // "No temperature data yet" — a permanent label reporting that nothing had
    // happened, charging the operator ~60 px of vertical space for it. On a
    // 1080p screen that pushed the Light and Marking panels below a scrollbar,
    // reported from the rig. A section with nothing to say takes no room.
    function renderEmpty() {
        if (!_root) return;
        _root.innerHTML = '';
        _root.hidden = true;
        setWrapVisible(false);
    }

    function render() {
        if (!_root) return;
        if (!_samples.length) { renderEmpty(); return; }
        _root.hidden = false;
        setWrapVisible(true);
        renderHeadline();

        const pts = _samples
            .map(s => ({ t: ms(s), w: num(s.water_c), sp: num(s.setpoint_c), state: s.state }))
            .filter(p => p.t != null && p.w != null);
        if (!pts.length) { renderEmpty(); return; }

        const W = _root.clientWidth || 480;
        const pad = { top: 14, right: 74, bottom: 20, left: 40 };
        const plotW = Math.max(10, W - pad.left - pad.right);
        const plotH = H - pad.top - pad.bottom;

        // ── scales ───────────────────────────────────────────────────────
        const t0 = pts[0].t;
        const t1 = pts[pts.length - 1].t;
        const tSpan = Math.max(1000, t1 - t0);
        const sx = t => pad.left + ((t - t0) / tSpan) * plotW;

        const sp = lastSetpoint();
        const ws = pts.map(p => p.w);
        let lo = Math.min(...ws);
        let hi = Math.max(...ws);
        if (sp != null) { lo = Math.min(lo, sp - TOL_C); hi = Math.max(hi, sp + TOL_C); }
        const centre = sp != null ? sp : (lo + hi) / 2;
        // A window that never zooms tighter than MIN_SPAN_C, centred on the
        // setpoint so "a bit high" and "a bit low" read symmetrically.
        let half = Math.max(MIN_SPAN_C / 2, Math.abs(hi - centre), Math.abs(centre - lo)) * 1.12;
        const yLo = centre - half;
        const yHi = centre + half;
        const sy = v => pad.top + plotH - ((v - yLo) / (yHi - yLo)) * plotH;

        const svg = el("svg", {
            viewBox: `0 0 ${W} ${H}`, width: "100%", height: H,
            role: "img",
            "aria-label": `Water temperature over the last ${fmtAgo(t1 - t0)}`
                + (sp != null ? `, setpoint ${sp.toFixed(1)} degrees` : ""),
        }, "temp-chart");

        // ── the setpoint's tolerance band + line (reference, not a series) ─
        if (sp != null) {
            svg.appendChild(el("rect", {
                x: pad.left, y: sy(sp + TOL_C),
                width: plotW, height: Math.max(1, sy(sp - TOL_C) - sy(sp + TOL_C)),
            }, "temp-band"));
            svg.appendChild(el("line", {
                x1: pad.left, x2: pad.left + plotW, y1: sy(sp), y2: sy(sp),
            }, "temp-setpoint-line"));
            // The reference label rides the LEFT end of its line. At the right
            // it sat on top of the end-of-series label every time the bath was
            // actually at setpoint — which is most of the time, and exactly
            // when both labels say the same number.
            const spLbl = el("text", { x: pad.left + 4, y: Math.max(pad.top + 8, sy(sp + TOL_C) - 4) }, "temp-ref-label");
            spLbl.textContent = `setpoint ${sp.toFixed(1)}°`;
            svg.appendChild(spLbl);
        }

        // ── y gridlines: the window edges and the middle, nothing more ────
        for (const v of [yLo, centre, yHi]) {
            if (v !== centre || sp == null) {
                svg.appendChild(el("line", {
                    x1: pad.left, x2: pad.left + plotW, y1: sy(v), y2: sy(v),
                }, "temp-grid-line"));
            }
            const lbl = el("text", { x: pad.left - 6, y: sy(v) + 3 }, "temp-grid-label");
            lbl.textContent = v.toFixed(1);
            svg.appendChild(lbl);
        }

        // ── x ticks: oldest and newest, in elapsed terms ──────────────────
        const now = el("text", { x: pad.left + plotW, y: H - 5 }, "temp-grid-label temp-x-end");
        now.textContent = "now";
        svg.appendChild(now);
        const then = el("text", { x: pad.left, y: H - 5 }, "temp-grid-label temp-x-start");
        then.textContent = fmtAgo(t1 - t0);
        svg.appendChild(then);

        // ── the series ───────────────────────────────────────────────────
        const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.t).toFixed(1)},${sy(p.w).toFixed(1)}`).join(" ");
        svg.appendChild(el("path", { d, fill: "none" }, "temp-water-line"));

        // ── the end of the line: the current reading, directly labelled ──
        const last = pts[pts.length - 1];
        const drifting = sp != null && Math.abs(last.w - sp) > TOL_C;
        const endCls = drifting ? " is-drifting" : "";
        svg.appendChild(el("circle", {
            cx: sx(last.t), cy: sy(last.w), r: 4,
        }, "temp-end-dot" + endCls));
        const endLbl = el("text", { x: Math.min(sx(last.t) + 10, W - 4), y: sy(last.w) + 4 },
            "temp-end-label" + endCls);
        endLbl.textContent = `${last.w.toFixed(1)}°`;
        svg.appendChild(endLbl);

        // ── crosshair + hit layer (hover and keyboard read the same) ─────
        const cross = el("line", { x1: 0, x2: 0, y1: pad.top, y2: pad.top + plotH }, "temp-cross");
        cross.setAttribute("visibility", "hidden");
        svg.appendChild(cross);
        const crossDot = el("circle", { r: 4, cx: 0, cy: 0 }, "temp-cross-dot");
        crossDot.setAttribute("visibility", "hidden");
        svg.appendChild(crossDot);

        const hit = el("rect", {
            x: pad.left, y: pad.top, width: plotW, height: plotH,
            fill: "transparent", tabindex: "0",
            role: "application",
            "aria-label": "Temperature trace — use arrow keys to read values",
        }, "temp-hit");
        svg.appendChild(hit);

        const tip = document.createElement("div");
        tip.className = "temp-tip";
        tip.hidden = true;

        function showAt(i) {
            const p = pts[Math.max(0, Math.min(pts.length - 1, i))];
            if (!p) return;
            const x = sx(p.t), y = sy(p.w);
            cross.setAttribute("x1", x); cross.setAttribute("x2", x);
            cross.setAttribute("visibility", "visible");
            crossDot.setAttribute("cx", x); crossDot.setAttribute("cy", y);
            crossDot.setAttribute("visibility", "visible");
            const off = p.sp != null ? p.w - p.sp : (sp != null ? p.w - sp : null);
            tip.hidden = false;
            tip.textContent = "";
            const v = document.createElement("strong");
            v.textContent = `${p.w.toFixed(2)}°`;
            const meta = document.createElement("span");
            meta.textContent = ` ${fmtAgo(t1 - p.t)}`
                + (off != null ? ` · ${off >= 0 ? "+" : "−"}${Math.abs(off).toFixed(2)}° vs set` : "");
            tip.appendChild(v); tip.appendChild(meta);
            const left = Math.max(0, Math.min((x / W) * _root.clientWidth - 60, _root.clientWidth - 130));
            tip.style.left = left + "px";
            _hoverIdx = i;
        }
        function hideCross() {
            cross.setAttribute("visibility", "hidden");
            crossDot.setAttribute("visibility", "hidden");
            tip.hidden = true;
            _hoverIdx = null;
        }
        function nearest(clientX) {
            const box = svg.getBoundingClientRect();
            const xInView = ((clientX - box.left) / box.width) * W;
            let best = 0, bestD = Infinity;
            pts.forEach((p, i) => {
                const dd = Math.abs(sx(p.t) - xInView);
                if (dd < bestD) { bestD = dd; best = i; }
            });
            return best;
        }

        hit.addEventListener("pointermove", e => showAt(nearest(e.clientX)));
        hit.addEventListener("pointerleave", hideCross);
        hit.addEventListener("focus", () => showAt(_hoverIdx == null ? pts.length - 1 : _hoverIdx));
        hit.addEventListener("blur", hideCross);
        hit.addEventListener("keydown", e => {
            const i = _hoverIdx == null ? pts.length - 1 : _hoverIdx;
            if (e.key === "ArrowLeft") { e.preventDefault(); showAt(i - 1); }
            else if (e.key === "ArrowRight") { e.preventDefault(); showAt(i + 1); }
            else if (e.key === "Home") { e.preventDefault(); showAt(0); }
            else if (e.key === "End") { e.preventDefault(); showAt(pts.length - 1); }
            else if (e.key === "Escape") { hideCross(); }
        });

        _root.innerHTML = "";
        _root.appendChild(svg);
        _root.appendChild(tip);
    }

    function dispose() {
        ClientEventBus.off("TEMPERATURE_UPDATE", onEvent);
        _root = null;
        _samples = [];
    }

    // Exposed for testing / forced refresh from devices.js
    return { init, dispose, _render: render, _samples: () => _samples };
})();

window.TemperatureGraph = TemperatureGraph;
