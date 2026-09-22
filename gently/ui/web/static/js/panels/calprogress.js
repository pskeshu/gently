/**
 * Calibration progress — what the calibration is looking at, while it looks.
 *
 *     CalProgressPanel.mount('op-cal-progress');
 *     CalProgressPanel.begin('embryo_1');      // the pane pressed Calibrate
 *     CalProgressPanel.finish(true, '101.2 um/deg');
 *
 * WHY
 *
 * Calibration is the longest thing the Devices tab can start: sixty to eighty
 * exposures, a minute or more, every one of them a frame Claude is being asked
 * to judge. The pane showed a disabled button for the whole of it. Kesavan, on
 * the walkthrough: "we have the calibration button, but it shows no in place
 * update on what is going on... not even images preview of calibration or
 * whatever." A run that shows nothing is indistinguishable from a run that has
 * hung, and when it ends badly there is no way to see WHICH frames were bad.
 *
 * Nothing new had to be measured. Every one of those frames is already pushed
 * to the viz server (`agent.push_viz`) and already broadcast to every client as
 * a `type: 'image'` message — the v1 gallery renders them after the fact. The
 * Operate pane simply never listened. This panel listens, and the metadata
 * riding along with each frame (galvo, piezo, visible, feature_score, sweep,
 * galvo_name, r_squared) is enough to narrate the run without the backend
 * emitting a single extra byte.
 *
 * WHAT IT DOES NOT OWN
 *
 * The run. operate.js starts the calibration and reports its result; this
 * panel only watches the frames go by. It therefore also narrates a
 * calibration the AGENT started from chat, which the pane never knew about —
 * a frame arriving with no run in progress opens one.
 */
const CalProgressPanel = (() => {
    'use strict';

    // Frames kept in the strip. The whole run is 60-80 exposures; holding
    // every base64 PNG in the DOM is real memory, and the tail is what tells
    // you where the run is now.
    const MAX_FRAMES = 40;
    // A run is over when the frames stop. The agent's calibration has no
    // "finished" signal on this path, so quiet for this long ends it.
    const IDLE_MS = 20000;

    const TYPES = new Set([
        'presence_check',
        'edge_detection', 'focus_sweep', 'focus_plot', 'focus_montage', 'calibration_summary',
    ]);

    let _host = null;
    let _frames = [];
    let _shown = -1;      // index being displayed; -1 = follow the newest
    let _running = false;
    let _embryo = null;
    // A batch walks the slide, so its frames come from several embryos and the
    // phase line has to say which one is under the objective right now.
    let _walk = false;
    let _idle = null;
    let _bound = false;

    const $ = id => document.getElementById(id);
    const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
        c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const num = (v, digits) => (typeof v === 'number' && isFinite(v) ? v.toFixed(digits) : null);

    // ── what a frame is, in words ──────────────────────────────────────────
    // Each phase names itself from the frame's own metadata, so the line reads
    // as the run's commentary rather than a progress bar with no content.
    function phaseOf(img) {
        const m = img.metadata || {};
        switch (img.data_type) {
            case 'presence_check': return 'Checking there is something there';
            case 'edge_detection': return 'Finding the embryo edges';
            case 'focus_sweep': {
                const side = m.galvo_name ? `${m.galvo_name} ` : '';
                const kind = { sparse: 'coarse', dense: 'dense', fine: 'fine' }[m.sweep] || '';
                return `Focus sweep · ${side}${kind}`.trim();
            }
            case 'focus_plot': return `Fitting the ${m.galvo_name || ''} focus curve`.trim();
            case 'focus_montage': return 'Choosing the focus slice';
            case 'calibration_summary': return 'Calibrated';
            default: return 'Calibrating';
        }
    }

    function captionOf(img) {
        const m = img.metadata || {};
        const bits = [];
        switch (img.data_type) {
            // The probe frame carries Claude's own words, which are the whole
            // point of it: "empty field" and "out of focus" get different fixes.
            case 'presence_check': {
                const g = num(m.galvo, 3);
                if (g !== null) bits.push(`galvo ${g}°`);
                bits.push(m.visible ? 'something here' : 'nothing here');
                if (m.description) bits.push(String(m.description).slice(0, 80));
                break;
            }
            case 'edge_detection': {
                const g = num(m.galvo, 3);
                if (g !== null) bits.push(`galvo ${g}°`);
                if (m.visible === true) bits.push('embryo visible');
                else if (m.visible === false) bits.push('nothing there');
                if (typeof m.feature_score === 'number') bits.push(`features ${m.feature_score}/10`);
                break;
            }
            case 'focus_sweep': {
                const p = num(m.piezo, 1), s = num(m.score, 2);
                if (m.galvo_name) bits.push(m.galvo_name);
                if (p !== null) bits.push(`piezo ${p} µm`);
                if (s !== null) bits.push(`sharpness ${s}`);
                if (m.peak_detected) bits.push('peak found');
                break;
            }
            case 'focus_plot': {
                const b = num(m.best_piezo, 1), r = num(m.r_squared, 3);
                if (m.galvo_name) bits.push(m.galvo_name);
                if (b !== null) bits.push(`best piezo ${b} µm`);
                if (r !== null) bits.push(`R² ${r}`);
                break;
            }
            case 'focus_montage': {
                if (m.pick != null) bits.push(`picked ${m.pick}`);
                if (m.method) bits.push(String(m.method));
                if (m.reasoning) bits.push(String(m.reasoning));
                break;
            }
            case 'calibration_summary': {
                const s = num(m.slope, 1);
                if (s !== null) bits.push(`${s} µm/deg`);
                const rt = num(m.r_squared_top, 3), rb = num(m.r_squared_bottom, 3);
                if (rt !== null && rb !== null) bits.push(`R² ${rt} top / ${rb} bottom`);
                break;
            }
            default: break;
        }
        return bits.join(' · ');
    }

    // A frame worth keeping when the strip overflows. The plots, the montage
    // and the summary are conclusions, not exposures — they are what you want
    // to still be looking at when the run ends.
    const isConclusion = img =>
        img.data_type === 'focus_plot' || img.data_type === 'calibration_summary' ||
        img.data_type === 'focus_montage';

    function render() {
        if (!_host) return;
        const wrap = _host.querySelector('.cp');
        if (!wrap) return;
        if (!_frames.length) { wrap.hidden = true; return; }
        wrap.hidden = false;

        const idx = _shown >= 0 && _shown < _frames.length ? _shown : _frames.length - 1;
        const cur = _frames[idx];

        // The phase is the RUN's phase (the newest frame), even while you are
        // looking back at an earlier one — otherwise scrubbing the strip makes
        // it look as though the run went backwards.
        const latest = _frames[_frames.length - 1];
        const who = _walk ? (latest.metadata || {}).embryo_id : null;
        wrap.querySelector('.cp-phase').textContent =
            phaseOf(latest) + (who ? ` · ${who}` : '');
        wrap.querySelector('.cp-count').textContent =
            `${_frames.length} frame${_frames.length === 1 ? '' : 's'}`;
        wrap.querySelector('.cp-dot').hidden = !_running;

        const img = wrap.querySelector('.cp-img');
        if (cur.base64_png) {
            img.src = `data:image/png;base64,${cur.base64_png}`;
            img.hidden = false;
        } else {
            img.hidden = true;
        }
        wrap.querySelector('.cp-cap').textContent = captionOf(cur);

        const strip = wrap.querySelector('.cp-strip');
        strip.innerHTML = _frames.map((f, i) => {
            const label = esc(captionOf(f) || phaseOf(f));
            const cls = `cp-thumb${i === idx ? ' is-on' : ''}${isConclusion(f) ? ' is-key' : ''}`;
            const pic = f.base64_png
                ? `<img src="data:image/png;base64,${f.base64_png}" alt="">` : '';
            return `<button class="${cls}" type="button" data-i="${i}"
                     title="${label}" aria-label="${label}">${pic}</button>`;
        }).join('');
        const on = strip.querySelector('.cp-thumb.is-on');
        if (_shown < 0) strip.scrollLeft = strip.scrollWidth;
        else if (on && on.scrollIntoView) on.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }

    function armIdle() {
        if (_idle) clearTimeout(_idle);
        _idle = setTimeout(() => { _running = false; render(); }, IDLE_MS);
    }

    function onImage(data) {
        if (!data || !TYPES.has(data.data_type)) return;
        const emb = (data.metadata || {}).embryo_id;
        // A frame for a different embryo belongs to a different run.
        if (_running && _embryo && emb && emb !== _embryo) return;
        if (!_running) {
            // Nobody pressed Calibrate here — the agent is calibrating. Show it,
            // and do not pin it to one embryo: the agent may be walking the
            // slide too, and dropping the other embryos' frames would make a
            // batch look like it stalled after the first one.
            _frames = [];
            _shown = -1;
            _running = true;
            _embryo = null;
            _walk = true;
        }
        _frames.push(data);
        if (_frames.length > MAX_FRAMES) {
            // Drop the oldest EXPOSURE, never a conclusion: the findings are
            // the part of the run you still want when it ends.
            const drop = _frames.findIndex(f => !isConclusion(f));
            _frames.splice(drop < 0 ? 0 : drop, 1);
            if (_shown > 0) _shown -= 1;
        }
        armIdle();
        render();
    }

    function begin(embryoId) {
        _frames = [];
        _shown = -1;
        _running = true;
        _embryo = embryoId || null;
        // No id means "whatever the run visits" — a batch, not one embryo.
        _walk = !embryoId;
        armIdle();
        if (!_host) return;
        const wrap = _host.querySelector('.cp');
        if (!wrap) return;
        // Nothing has arrived yet, but the run HAS started: say so rather than
        // leaving the pane blank until the first exposure comes back.
        wrap.hidden = false;
        wrap.querySelector('.cp-phase').textContent = 'Starting the calibration';
        wrap.querySelector('.cp-count').textContent = 'no frames yet';
        wrap.querySelector('.cp-dot').hidden = false;
        wrap.querySelector('.cp-img').hidden = true;
        wrap.querySelector('.cp-cap').textContent = '';
        wrap.querySelector('.cp-strip').innerHTML = '';
    }

    function finish(ok, text) {
        _running = false;
        if (_idle) { clearTimeout(_idle); _idle = null; }
        if (!_host) return;
        const wrap = _host.querySelector('.cp');
        if (!wrap) return;
        if (!_frames.length) {
            // A run that produced no frames has nothing to show; the pane's
            // own result line already says what happened.
            wrap.hidden = true;
            return;
        }
        render();
        wrap.querySelector('.cp-dot').hidden = true;
        wrap.querySelector('.cp-phase').textContent = ok ? 'Calibrated' : 'Calibration failed';
        // The result replaces the caption only while the panel is still
        // following the run. If the operator has scrubbed back to a frame,
        // that frame's own caption is what belongs under it — the phase line
        // above already says how the run ended.
        if (text && _shown < 0) wrap.querySelector('.cp-cap').textContent = text;
    }

    function mount(hostId) {
        _host = $(hostId);
        if (!_host) return;
        _host.innerHTML = `
          <div class="cp" hidden>
            <div class="cp-head">
              <span class="cp-dot" aria-hidden="true" hidden></span>
              <span class="cp-phase"></span>
              <span class="cp-count"></span>
            </div>
            <div class="cp-stage"><img class="cp-img" alt="Calibration frame" hidden></div>
            <div class="cp-cap"></div>
            <div class="cp-strip"></div>
          </div>`;
        _host.querySelector('.cp-strip').addEventListener('click', e => {
            const b = e.target.closest('.cp-thumb');
            if (!b) return;
            const i = parseInt(b.dataset.i, 10);
            // Clicking the newest frame goes back to following the run.
            _shown = (i === _frames.length - 1) ? -1 : i;
            render();
        });
        if (!_bound && typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('IMAGE_RECEIVED', onImage);
            _bound = true;
        }
        render();
    }

    return { mount, begin, finish, phaseOf, captionOf, _onImage: onImage };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = CalProgressPanel;
