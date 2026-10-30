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
            if (!(ev.ctrlKey || ev.metaKey) || ev.key.toLowerCase() !== 'z' || !document.body.contains(el)) return;
            if (/^(INPUT|TEXTAREA)$/.test(ev.target.tagName) && ev.target.type !== 'range') return;
            ev.preventDefault();
            if (this.history.length) this.save(this.history.pop(), { undo: true });
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
            <input class="th-slider" type="range" min="0" max="${frames.length - 1}" value="${this.idx}">
            <span class="th-when">frame ${cur.frame ?? ''} · ${this.hms(cur.captured_at)}</span>
          </div>
          <div class="th-stage">
            <img class="th-img" src="${this.url(cur.stem)}?max=1024" draggable="false" title="Click an embryo on the frame it first twitches">
            ${here.map(([r, i]) => `<span class="th-pin ${r.hatch_frame == null ? 'is-open' : ''}" style="left:${r.x * 100}%;top:${r.y * 100}%" title="#${i + 1}">${i + 1}</span>`).join('')}
          </div>
          <table class="th-table">
            <thead><tr><th>#</th><th>field</th><th>twitch</th><th>hatch</th><th>minutes</th><th>°C</th><th>vs Sulston (${this.data.sulston_min} min)</th><th>by</th><th></th></tr></thead>
            <tbody>${report.map((r, i) => `
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
    // rest of the pane stands, so the image element is not rebuilt per step.
    scrub() {
        const cur = this.frame(), el = this.el;
        this.setFrame(el.querySelector('.th-img'), `${this.url(cur.stem)}?max=1024`);
        el.querySelector('.th-when').textContent = `frame ${cur.frame ?? ''} · ${this.hms(cur.captured_at)}`;
        el.querySelectorAll('[data-hatch]').forEach(b => { b.textContent = `hatch here (f${cur.frame})`; });
    },

    // A slider sweep asks the server for the frames it can show, one at a time.
    setFrame(img, src) {
        if (!img._gated) {
            img._gated = true;
            const landed = () => { img.dataset.loading = ''; const w = img.dataset.want; if (w) { img.dataset.want = ''; this.setFrame(img, w); } };
            img.addEventListener('load', landed); img.addEventListener('error', landed);
        }
        if (img.dataset.loading) { img.dataset.want = src; return; }
        img.dataset.loading = '1'; img.src = src;
    },

    wire() {
        const el = this.el;
        el.querySelectorAll('[data-field]').forEach(b => b.onclick = () => { this.field = +b.dataset.field; this.idx = 0; this.render(); });
        const slider = el.querySelector('.th-slider');
        slider.oninput = () => { this.idx = +slider.value; this.scrub(); };
        el.querySelector('.th-img').onclick = ev => {
            const img = ev.currentTarget, r = img.getBoundingClientRect();
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
