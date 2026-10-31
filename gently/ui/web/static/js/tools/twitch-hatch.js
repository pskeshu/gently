/**
 * Twitching to hatching — a tool pane over a session's brightfield frames.
 *
 * Pick a field, scrub its frames, click an embryo on the frame it first
 * twitches (a row), then at the frame it hatches press the row's "hatch here".
 * Minutes and temperature over the interval come back from the server, which
 * reads them off the frames' clocks and the temperature log.
 *
 * Positions are fractions of the frame (0..1), so the same click lands on the
 * full TIFF and on any thumbnail of it.
 */
const TwitchHatchTool = {
    sid: null, data: null, field: 0, idx: 0,
    history: [],   // rows as they were before each save; Ctrl+Z walks it back

    async mount(el, sid) {
        if (this.sid !== sid) this.history = [];
        this.sid = sid; this.el = el;
        document.removeEventListener('keydown', this._onKey);
        this._onKey = ev => {
            if (!document.body.contains(el)) return;
            if (/^(INPUT|TEXTAREA|SELECT)$/.test(ev.target.tagName) && ev.target.type !== 'range') return;
            if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === 'z') {
                ev.preventDefault();
                if (this.history.length) this.save(this.history.pop(), { undo: true });
                return;
            }
            // Arrows walk the frames while the pane is on screen.
            const steps = { ArrowLeft: -1, ArrowRight: 1, Home: 'first', End: 'last' };
            if (!(ev.key in steps) || ev.altKey || ev.ctrlKey || ev.metaKey) return;
            const r = el.getBoundingClientRect();
            if (r.bottom < 0 || r.top > window.innerHeight) return;
            ev.preventDefault();
            this.step(steps[ev.key]);
        };
        document.addEventListener('keydown', this._onKey);
        this.data = await fetch(`/api/sessions/${encodeURIComponent(sid)}/tools/twitch_hatch`).then(r => r.json());
        const fields = this.fields();
        if (!fields.includes(this.field)) this.field = fields[0] ?? 0;
        this.render();
    },

    fields() { return [...new Set((this.data.frames || []).map(f => f.field ?? 0))].sort((a, b) => a - b); },
    frames() { return (this.data.frames || []).filter(f => (f.field ?? 0) === this.field).sort((a, b) => (a.frame ?? 0) - (b.frame ?? 0)); },
    frame() { return this.frames()[Math.min(this.idx, this.frames().length - 1)]; },
    url(stem) { return `/api/sessions/${encodeURIComponent(this.sid)}/snapshot/${stem}.png`; },
    // The frame as its numbers, averaged 2×2: a 2048² frame is 2 MB and the
    // pane shows it at ~720 px, so nothing visible is lost and a hover reads
    // the value. A click to mark still lands in fractions of the frame.
    u16(stem) { return `/api/sessions/${encodeURIComponent(this.sid)}/snapshot/${stem}.u16?scale=2`; },
    hms(iso) { const d = new Date(iso); return isNaN(d) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); },

    render() {
        const frames = this.frames();
        if (!frames.length) { this.el.innerHTML = '<div class="empty-tab">No brightfield frames in this session</div>'; return; }
        const cur = this.frame();
        const report = this.data.report || [];
        const here = report.map((r, i) => [r, i]).filter(([r]) => r.field === this.field);
        this.el.innerHTML = `
          <div class="th-bar">
            <span class="th-fields">${this.fields().map(f => `<button class="session-edit-btn ${f === this.field ? 'is-on' : ''}" data-field="${f}">field ${f}</button>`).join('')}</span>
            <button type="button" class="th-step" data-step="-1" title="Previous frame (←)" aria-label="Previous frame">‹</button>
            <input class="th-slider" type="range" min="0" max="${frames.length - 1}" value="${this.idx}" aria-label="Frame">
            <button type="button" class="th-step" data-step="1" title="Next frame (→)" aria-label="Next frame">›</button>
            <span class="th-when">${this.whenText()}</span>
          </div>
          <div class="th-stage">
            <div class="th-img" title="Click an embryo on the frame it first twitches"></div>
            ${here.map(([r, i]) => `<span class="th-pin ${r.hatch_frame == null ? 'is-open' : ''}" style="left:${r.x * 100}%;top:${r.y * 100}%" title="#${i + 1}">${i + 1}</span>`).join('')}
          </div>
          <table class="th-table">
            <thead><tr><th>#</th><th>field</th><th>twitch</th><th>hatch</th><th>minutes</th><th>°C</th><th>vs Sulston (${this.data.sulston_min} min)</th><th>by</th><th></th></tr></thead>
            <tbody>${!report.length ? '<tr><td colspan="9" class="th-empty">No embryo marked yet. Click an embryo on the frame where it first twitches; then move to the frame it hatches and press its hatch button.</td></tr>' : ''}${report.map((r, i) => `
              <tr>
                <td>${i + 1}</td><td>${r.field}</td>
                <td><a href="#" data-go="${r.field}:${r.twitch_frame}">f${r.twitch_frame}</a> ${r.twitch_at ? this.hms(r.twitch_at) : ''}</td>
                <td>${r.hatch_frame != null
                    ? `<a href="#" data-go="${r.field}:${r.hatch_frame}">f${r.hatch_frame}</a> ${r.hatch_at ? this.hms(r.hatch_at) : ''}`
                    : `<button class="session-edit-btn" data-hatch="${i}" ${r.field !== this.field ? 'disabled' : ''}>hatch here (f${cur.frame})</button>`}</td>
                <td>${r.minutes ?? ''}</td><td>${r.temp_c ?? ''}</td>
                <td>${r.vs_sulston_min == null ? '' : (r.vs_sulston_min > 0 ? '+' : '') + r.vs_sulston_min}</td>
                <td>${typeof Who !== 'undefined' ? Who.chip(r.by, { when: r.at }) : ''}</td>
                <td><button class="session-edit-btn" data-del="${i}" title="Remove">×</button></td>
              </tr>`).join('')}
            </tbody>
          </table>
          ${report.length ? `<div class="th-foot"><a href="/api/sessions/${encodeURIComponent(this.sid)}/tools/twitch_hatch" target="_blank" rel="noopener">report as JSON</a></div>` : ''}`;
        this.wire();
    },

    // A scrub changes the picture, its caption and the "hatch here" labels; the
    // rest of the pane stands, so the viewer is not rebuilt per step.
    scrub() {
        const cur = this.frame(), el = this.el;
        this.showFrame(cur);
        el.querySelector('.th-when').textContent = this.whenText();
        const slider = el.querySelector('.th-slider');
        if (slider && +slider.value !== this.idx) slider.value = this.idx;
        el.querySelectorAll('[data-hatch]').forEach(b => { b.textContent = `hatch here (f${cur.frame})`; });
        el.querySelectorAll('.th-step').forEach(b => { b.disabled = +b.dataset.step < 0 ? this.idx <= 0 : this.idx >= this.frames().length - 1; });
    },

    whenText() {
        const cur = this.frame(), n = this.frames().length;
        return `frame ${cur.frame ?? ''} · ${this.idx + 1} of ${n} · ${this.hms(cur.captured_at)}`;
    },

    // One frame forward or back; the arrows, the keyboard and Home/End all land here.
    step(d) {
        const n = this.frames().length;
        if (!n) return;
        const idx = Math.max(0, Math.min(n - 1, d === 'first' ? 0 : d === 'last' ? n - 1 : this.idx + d));
        if (idx === this.idx) return;
        this.idx = idx;
        this.scrub();
    },

    // The viewer under the pins. One per pane; a frame change is a load into
    // it, and a later load wins over a slower earlier one, so a sweep across
    // the slider shows the frames it can and lands on the last.
    showFrame(frame) {
        const host = this.el.querySelector('.th-img');
        if (!host) return;
        if (!this.view || this.view.host !== host) {
            if (this.view) this.view.destroy();
            this.view = U16View.create(host, { bar: true });
        }
        const keep = this.view.data ? [this.view.lo, this.view.hi, this.view.gamma] : null;
        this.view.load(this.u16(frame.stem)).then(meta => {
            // The contrast a person set holds across frames of one field.
            if (meta && keep) this.view.setRange(keep[0], keep[1], keep[2]);
        }).catch(() => {});
    },

    wire() {
        const el = this.el;
        el.querySelectorAll('[data-field]').forEach(b => b.onclick = () => { this.field = +b.dataset.field; this.idx = 0; this.render(); });
        const slider = el.querySelector('.th-slider');
        slider.oninput = () => { this.idx = +slider.value; this.scrub(); };
        el.querySelectorAll('.th-step').forEach(b => { b.onclick = () => this.step(+b.dataset.step); });
        this.scrub();   // the arrows' disabled state at the ends
        this.showFrame(this.frame());
        el.querySelector('.th-img').onclick = ev => {
            const img = this.view ? this.view.canvas : ev.currentTarget, r = img.getBoundingClientRect();
            if (ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom) return;
            this.save([...this.data.rows, {
                field: this.field, x: (ev.clientX - r.left) / r.width, y: (ev.clientY - r.top) / r.height,
                twitch_frame: this.frame().frame, hatch_frame: null,
            }]);
        };
        el.querySelectorAll('[data-hatch]').forEach(b => b.onclick = () => {
            const rows = this.data.rows.map((r, i) => i === +b.dataset.hatch ? { ...r, hatch_frame: this.frame().frame } : r);
            this.save(rows);
        });
        el.querySelectorAll('[data-del]').forEach(b => b.onclick = () => this.save(this.data.rows.filter((_, i) => i !== +b.dataset.del)));
        el.querySelectorAll('[data-go]').forEach(a => a.onclick = ev => {
            ev.preventDefault();
            const [f, n] = a.dataset.go.split(':').map(Number);
            this.field = f; this.idx = Math.max(0, this.frames().findIndex(x => x.frame === n)); this.render();
        });
    },

    async save(rows, { undo = false } = {}) {
        if (!undo) this.history.push(this.data.rows);
        const r = await fetch(`/api/sessions/${encodeURIComponent(this.sid)}/tools/twitch_hatch`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rows }),
        });
        if (!r.ok) { const d = await r.json().catch(() => ({})); const m = d.detail || `HTTP ${r.status}`; if (typeof showGentlyToast === 'function') showGentlyToast(m, null, null, 6000, 'error'); else console.error(m); return; }
        await this.mount(this.el, this.sid);
    },
};

window.GentlyTools = Object.assign(window.GentlyTools || {}, { twitch_hatch: TwitchHatchTool });
