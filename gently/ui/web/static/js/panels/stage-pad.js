/**
 * The XY stage, from the bottom camera's pane.
 *
 *     StagePad.mount('op-stage-host');
 *
 * Four buttons and a step. Each press asks the server to move by that much
 * from where the stage is (POST /api/devices/stage/jog); the server refuses
 * while the sample is at the objective and keeps the target inside the XY
 * envelope. The buttons are named by stage axis, not by screen direction:
 * which way +X moves in the picture depends on how the camera is mounted,
 * and a pad that says "left" and moves the picture right is worse than one
 * that says "−X". Watch the image; after one press you know.
 *
 * Arrow keys work while the pad has focus; Shift multiplies the step by ten.
 * The readout follows the stage, driven from here or from the joystick.
 */
const StagePad = (() => {
    'use strict';

    const STEPS = [10, 50, 200, 1000];
    let host = null;
    let step = 200;
    let pos = { x: null, y: null };
    let busy = false;
    let lastSaid = '';
    let timer = null;

    const $ = id => document.getElementById(id);
    const fmt = v => (v == null || !Number.isFinite(v)) ? '—' : Math.round(v).toLocaleString();
    const toast = (m, bad) => { if (typeof showGentlyToast === 'function') showGentlyToast(m, null, null, bad ? 8000 : 2500, bad ? 'error' : undefined); };

    function locked() {
        const cam = $('op-cam-bottom');
        return !!(cam && cam.classList.contains('is-locked'));
    }

    function render() {
        if (!host) return;
        const off = busy || locked();
        host.innerHTML = `
            <div class="sp" tabindex="0" aria-label="XY stage pad: arrow keys move the stage by the chosen step, Shift for ten times">
                <div class="sp-row sp-head">
                    <span class="sp-pos" title="Where the stage is, µm">x <b class="op-num">${fmt(pos.x)}</b> &nbsp; y <b class="op-num">${fmt(pos.y)}</b> <i>µm</i></span>
                    ${locked() ? '<span class="sp-locked">XY locked — sample at objective</span>' : ''}
                </div>
                <div class="sp-grid">
                    <span></span>
                    <button class="op-nbtn sp-btn" type="button" data-dx="0" data-dy="1" title="Move +Y by ${step} µm" ${off ? 'disabled' : ''}>▲ +Y</button>
                    <span></span>
                    <button class="op-nbtn sp-btn" type="button" data-dx="-1" data-dy="0" title="Move −X by ${step} µm" ${off ? 'disabled' : ''}>◀ −X</button>
                    <span class="sp-mid">${busy ? '…' : lastSaid}</span>
                    <button class="op-nbtn sp-btn" type="button" data-dx="1" data-dy="0" title="Move +X by ${step} µm" ${off ? 'disabled' : ''}>+X ▶</button>
                    <span></span>
                    <button class="op-nbtn sp-btn" type="button" data-dx="0" data-dy="-1" title="Move −Y by ${step} µm" ${off ? 'disabled' : ''}>▼ −Y</button>
                    <span></span>
                </div>
                <div class="sp-row sp-steps">
                    <span class="op-cap">step</span>
                    ${STEPS.map(v => `<button class="op-nbtn sp-step${v === step ? ' is-on' : ''}" type="button" data-step="${v}">${v >= 1000 ? (v / 1000) + ' mm' : v + ' µm'}</button>`).join('')}
                </div>
                <div class="op-cap sp-note">Named by stage axis: which way +X moves in the picture depends on the camera's mounting. Arrow keys move too; Shift = ×10.</div>
            </div>`;
        host.querySelectorAll('.sp-btn').forEach(b => b.addEventListener('click', () => jog(Number(b.dataset.dx) * step, Number(b.dataset.dy) * step)));
        host.querySelectorAll('.sp-step').forEach(b => b.addEventListener('click', () => { step = Number(b.dataset.step); render(); }));
        const pad = host.querySelector('.sp');
        if (pad) pad.addEventListener('keydown', onKey);
    }

    function onKey(ev) {
        const dir = { ArrowUp: [0, 1], ArrowDown: [0, -1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] }[ev.key];
        if (!dir) return;
        ev.preventDefault();
        const k = ev.shiftKey ? 10 : 1;
        jog(dir[0] * step * k, dir[1] * step * k);
    }

    async function jog(dx, dy) {
        if (busy || locked() || (!dx && !dy)) return;
        busy = true;
        lastSaid = '';
        render();
        try {
            const r = await fetch('/api/devices/stage/jog', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ dx, dy }),
            });
            const d = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
            if (Number.isFinite(d.x) && Number.isFinite(d.y)) pos = { x: d.x, y: d.y };
            lastSaid = d.clamped ? (d.moved ? 'at the edge' : 'at the edge — no move') : '';
            if (d.clamped) toast(d.moved ? 'Stopped at the edge of the XY envelope.' : 'Already at the edge of the XY envelope.');
        } catch (e) {
            lastSaid = '';
            toast(`The stage did not move: ${e.message}`, true);
        } finally {
            busy = false;
            render();
        }
    }

    async function refresh() {
        if (!host || !host.offsetParent) return;   // pane not showing: don't poll
        try {
            const r = await fetch('/api/devices/stage/envelope');
            if (!r.ok) return;
            const d = await r.json();
            const x = d.x != null ? d.x : (d.position || {}).x;
            const y = d.y != null ? d.y : (d.position || {}).y;
            if (Number.isFinite(Number(x)) && Number.isFinite(Number(y))) {
                const changed = pos.x !== Number(x) || pos.y !== Number(y);
                pos = { x: Number(x), y: Number(y) };
                if (changed && !busy) render();
            }
        } catch (_) { /* the rig is away; the readout keeps its last number */ }
    }

    function mount(hostId) {
        host = $(hostId);
        if (!host) return;
        render();
        refresh();
        clearInterval(timer);
        timer = setInterval(refresh, 1500);
        // The lock banner comes and goes with the F-drive; follow it.
        const cam = $('op-cam-bottom');
        if (cam && typeof MutationObserver !== 'undefined') {
            new MutationObserver(() => render()).observe(cam, { attributes: true, attributeFilter: ['class'] });
        }
    }

    return { mount, jog, refresh, STEPS };
})();

if (typeof module !== 'undefined') module.exports = StagePad;
