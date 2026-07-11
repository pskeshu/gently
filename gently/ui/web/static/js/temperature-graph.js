/**
 * Temperature Graph — hand-rolled SVG line chart for the Devices tab.
 *
 * Shows the water-temp trace (solid) and stepped setpoint line (dashed),
 * backfilled from /api/temperature/{session}/history and then updated live
 * via TEMPERATURE_UPDATE events from the ClientEventBus.
 *
 * No external dependencies. Calm empty state; never renders mock data.
 *
 * Usage:
 *   TemperatureGraph.init(containerEl, 'current');
 *   // The component self-subscribes; no extra subscription needed in the caller.
 *   TemperatureGraph.dispose();  // on teardown
 */
const TemperatureGraph = (() => {
    const SVGNS = "http://www.w3.org/2000/svg";
    const MAX_POINTS = 600;  // rolling ~10 min @ 1 Hz

    let _root = null;
    let _samples = [];
    let _session = "current";

    function init(container, sessionId) {
        ClientEventBus.off("TEMPERATURE_UPDATE", onEvent);
        _root = container;
        _session = sessionId || "current";
        _samples = [];
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

    function renderEmpty() {
        if (!_root) return;
        _root.innerHTML = '<div class="temp-graph-empty">No temperature data yet</div>';
    }

    function render() {
        if (!_root) return;
        if (!_samples.length) { renderEmpty(); return; }

        const W = _root.clientWidth || 480;
        const H = 160;
        const pad = { top: 12, right: 16, bottom: 20, left: 28 };

        const ws  = _samples.map(s => s.water_c).filter(v => v != null);
        const sps = _samples.map(s => s.setpoint_c).filter(v => v != null);

        if (!ws.length) { renderEmpty(); return; }

        const allVals = [...ws, ...sps];
        const lo = Math.min(...allVals) - 0.5;
        const hi = Math.max(...allVals) + 0.5;
        const range = Math.max(0.001, hi - lo);
        const plotW = W - pad.left - pad.right;
        const plotH = H - pad.top - pad.bottom;
        const n = _samples.length;

        const sx = i => pad.left + (i / Math.max(1, n - 1)) * plotW;
        const sy = v => pad.top + plotH - ((v - lo) / range) * plotH;

        const svg = document.createElementNS(SVGNS, "svg");
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute("width", "100%");
        svg.setAttribute("aria-hidden", "true");

        // Y-axis gridlines (3 levels)
        const gridG = document.createElementNS(SVGNS, "g");
        gridG.setAttribute("class", "temp-graph-grid");
        for (let k = 0; k <= 2; k++) {
            const v = lo + (k / 2) * (hi - lo);
            const y = sy(v);
            const gridLine = document.createElementNS(SVGNS, "line");
            gridLine.setAttribute("x1", pad.left);
            gridLine.setAttribute("x2", W - pad.right);
            gridLine.setAttribute("y1", y);
            gridLine.setAttribute("y2", y);
            gridLine.setAttribute("class", "temp-grid-line");
            gridG.appendChild(gridLine);

            const lbl = document.createElementNS(SVGNS, "text");
            lbl.setAttribute("x", pad.left - 3);
            lbl.setAttribute("y", y);
            lbl.setAttribute("class", "temp-grid-label");
            lbl.textContent = v.toFixed(1);
            gridG.appendChild(lbl);
        }
        svg.appendChild(gridG);

        // Helper: build a polyline from a point-string array
        const makeLine = (pts, cls) => {
            if (!pts.length) return;
            const p = document.createElementNS(SVGNS, "polyline");
            p.setAttribute("points", pts.join(" "));
            p.setAttribute("class", cls);
            p.setAttribute("fill", "none");
            svg.appendChild(p);
        };

        // Water temp — skip null samples (gap rather than crash)
        const waterPts = [];
        _samples.forEach((s, i) => {
            if (s.water_c != null) waterPts.push(`${sx(i).toFixed(1)},${sy(s.water_c).toFixed(1)}`);
        });
        makeLine(waterPts, "temp-water");

        // Setpoint — stepped: carry the last known setpoint forward
        const spPts = [];
        let lastSP = null;
        _samples.forEach((s, i) => {
            const sp = s.setpoint_c != null ? s.setpoint_c : lastSP;
            if (sp != null) {
                spPts.push(`${sx(i).toFixed(1)},${sy(sp).toFixed(1)}`);
                lastSP = sp;
            }
        });
        makeLine(spPts, "temp-setpoint");

        // Current readout line (last sample)
        const last = _samples[_samples.length - 1];
        const wStr  = last.water_c    != null ? `${last.water_c.toFixed(1)} °C`   : "—";
        const spStr = last.setpoint_c != null ? `${last.setpoint_c.toFixed(1)} °C` : "—";
        const stStr = last.state ? ` · ${last.state}` : "";

        const readout = document.createElement("div");
        readout.className = "temp-graph-readout";
        readout.title = "Water temperature → setpoint (state)";
        readout.textContent = `${wStr} → ${spStr}${stStr}`;

        _root.innerHTML = "";
        _root.appendChild(readout);
        _root.appendChild(svg);
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
