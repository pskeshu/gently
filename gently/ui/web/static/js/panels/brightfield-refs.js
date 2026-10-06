/**
 * Dark and flat-field references for the brightfield (DIC overview) frames.
 *
 *     BrightfieldRefs.mount('op-bfref-host');
 *
 * Lives under the DIC overview fields of the Acquisition pane and reads the
 * light, LED brightness and exposure from them, so the references are taken
 * for exactly the frames the plan will take. Two steps, in order:
 *
 *   1. Dark — the room light is cycled on and off (its state has no
 *      read-back), the LED closed, one frame taken.
 *   2. Flat — the operator drives the stage so no embryo is in view, then
 *      several frames are averaged under the plan's light.
 *
 * Both are filed in the session (calibration/brightfield/<record>/) and every
 * overview frame taken afterwards names them. Nothing here starts a run.
 */
const BrightfieldRefs = (() => {
    'use strict';

    let host = null;
    let state = { records: [], match: null, busy: false };
    let record = null;        // the record the dark opened, for the flat to join
    let last = { dark: null, flat: null };
    let working = null;       // 'dark' | 'flat' while a capture runs
    let timer = null;

    const $ = id => document.getElementById(id);
    const esc = s => (typeof escapeHtml === 'function') ? escapeHtml(String(s == null ? '' : s)) : String(s == null ? '' : s);

    function spec() {
        const light = ($('op-plan-dic-light') || {}).value || 'room';
        const pct = ($('op-plan-dic-led') || {}).value;
        const exp = ($('op-plan-dic-exposure') || {}).value;
        return {
            light,
            led_intensity_pct: light === 'led' && pct !== '' && pct != null ? Number(pct) : null,
            exposure_ms: exp !== '' && exp != null ? Number(exp) : null,
        };
    }

    function specText(s) {
        const bits = [s.light === 'room' ? 'room light' : s.light === 'led' ? `LED${s.led_intensity_pct != null ? ` ${s.led_intensity_pct}%` : ''}` : 'light as it is'];
        bits.push(s.exposure_ms != null ? `${s.exposure_ms} ms` : "camera's exposure");
        return bits.join(' · ');
    }

    async function refresh() {
        if (!host) return;
        const s = spec();
        const q = new URLSearchParams();
        q.set('light', s.light);
        if (s.led_intensity_pct != null) q.set('led_intensity_pct', String(s.led_intensity_pct));
        if (s.exposure_ms != null) q.set('exposure_ms', String(s.exposure_ms));
        try {
            const r = await fetch(`/api/brightfield/references?${q}`);
            if (r.ok) state = await r.json();
            else state = { records: [], match: null, busy: false, error: `HTTP ${r.status}` };
        } catch (e) {
            state = { records: [], match: null, busy: false, error: String(e) };
        }
        render();
    }

    function fmtStats(st) {
        if (!st) return '';
        const sat = st.saturated_fraction != null && st.saturated_fraction > 0.001 ? ` · <b class="bf-warn">${(st.saturated_fraction * 100).toFixed(2)}% saturated</b>` : '';
        return `mean ${st.mean} · sd ${st.std} · max ${st.max}${sat}`;
    }

    function render() {
        if (!host) return;
        const s = spec();
        const m = state.match_record;
        const have = m ? `<span class="bf-ok">✓ references taken ${esc((m.flat && m.flat.taken_at) || (m.dark && m.dark.taken_at) || '')}</span>`
            : `<span class="bf-missing">✕ none yet for ${esc(specText(s))}</span>`;
        const checks = m && m.checks ? m.checks : (last.flat && last.flat.checks) || (last.dark && last.dark.checks) || {};
        const warn = [];
        if (checks.dark_is_dark === false) warn.push('The dark is not dark: it is more than half as bright as the flat. The room light did not go off — cycle it again, or check the room.');
        if (checks.flat_unsaturated === false) warn.push('The flat is saturated in places. Lower the LED or the exposure and retake.');
        const img = (k) => last[k] && last[k].thumbnail ? `<img class="bf-thumb" src="data:image/png;base64,${last[k].thumbnail}" alt="${k} reference">` : '';
        host.innerHTML = `
            <div class="bf-refs">
                <div class="bf-head"><span class="bf-title">Dark and flat references</span> ${have}</div>
                <div class="bf-cap">Taken once per session for these frames (${esc(specText(s))}). Every overview frame then names them, so the analysis downstream can correct: (frame − dark) / (flat − dark).</div>
                ${state.error ? `<div class="bf-warn">${esc(state.error)}</div>` : ''}
                <div class="bf-steps">
                    <div class="bf-step">
                        <div class="bf-step-head"><b>1. Dark</b> <span class="bf-cap">the room light is switched on, then off, so its state is known; the LED is closed; one frame.</span></div>
                        <div class="bf-row">
                            <button class="op-btn" type="button" data-bf="dark" ${working ? 'disabled' : ''}>${working === 'dark' ? 'Taking the dark… (about 5 s)' : last.dark ? 'Retake dark' : 'Take dark'}</button>
                            ${last.dark ? `<span class="bf-cap">${fmtStats(last.dark.stats)}</span>` : ''}
                        </div>
                        ${img('dark')}
                    </div>
                    <div class="bf-step">
                        <div class="bf-step-head"><b>2. Flat</b> <span class="bf-cap">first drive the stage so <u>no embryo</u> is in the bottom camera's view; then five frames are averaged under the plan's light.</span></div>
                        <div class="bf-row">
                            <button class="op-btn" type="button" data-bf="flat" ${working || !last.dark && !record ? 'disabled' : ''} title="${!last.dark && !record ? 'Take the dark first' : ''}">${working === 'flat' ? 'Taking the flat…' : last.flat ? 'Retake flat' : 'Take flat'}</button>
                            ${last.flat ? `<span class="bf-cap">${fmtStats(last.flat.stats)} · ${last.flat.frames} frames</span>` : ''}
                        </div>
                        ${img('flat')}
                    </div>
                </div>
                ${warn.map(w => `<div class="bf-warn">! ${esc(w)}</div>`).join('')}
                ${state.records && state.records.length ? `<div class="bf-cap">${state.records.length} record${state.records.length !== 1 ? 's' : ''} in this session · calibration/brightfield/</div>` : ''}
            </div>`;
        host.querySelectorAll('[data-bf]').forEach(b => b.addEventListener('click', () => take(b.dataset.bf)));
    }

    async function take(kind) {
        if (working) return;
        const s = spec();
        if (kind === 'flat') {
            const ok = confirm('Is the bottom camera\'s view clear of embryos?\n\nDrive the stage to an empty part of the dish first. The flat is the empty field under the plan\'s light.');
            if (!ok) return;
        }
        working = kind;
        render();
        try {
            const body = Object.assign({}, s, record ? { record } : {});
            if (kind === 'flat') body.frames = 5;
            const r = await fetch(`/api/brightfield/references/${kind}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
            });
            const d = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(d.detail || `HTTP ${r.status}`);
            last[kind] = d;
            record = d.record;
            if (kind === 'dark') last.flat = null;   // a new dark starts a new pair
            if (typeof showGentlyToast === 'function') {
                showGentlyToast(kind === 'dark' ? 'Dark reference filed.' : 'Flat reference filed. The overview frames will name it.');
            }
        } catch (e) {
            if (typeof showGentlyToast === 'function') showGentlyToast(`The ${kind} could not be taken: ${e.message}`, null, null, 10000, 'error');
            else alert(`The ${kind} could not be taken: ${e.message}`);
        } finally {
            working = null;
            await refresh();
        }
    }

    function mount(hostId) {
        host = $(hostId);
        if (!host) return;
        ['op-plan-dic-light', 'op-plan-dic-led', 'op-plan-dic-exposure'].forEach(id => {
            const el = $(id);
            if (el) el.addEventListener('change', () => { clearTimeout(timer); timer = setTimeout(refresh, 150); });
        });
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('TIMELAPSE_STATE', (st) => {
                // A new session: what was taken belongs to the old one.
                if (st && st.session_id && state.session_id && st.session_id !== state.session_id) {
                    record = null; last = { dark: null, flat: null };
                    refresh();
                }
            });
        }
        refresh();
    }

    return { mount, refresh, spec };
})();

if (typeof module !== 'undefined') module.exports = BrightfieldRefs;
