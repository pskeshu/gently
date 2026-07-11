/**
 * ContextSurface (ux_v2): renders the agent's "mind" as a calm, always-visible
 * panel — open questions (uncertainty), watchpoints (attention), expectations
 * (beliefs) — read from /api/context and refreshed live on the CONTEXT_UPDATED
 * event (the store emits it; the server broadcasts it to /ws; no polling).
 *
 * The control holder can resolve items inline (answer a question, resolve a
 * watchpoint, confirm an expectation); observers see it read-only. No-ops
 * unless #context-surface is present (flag off → v1 untouched).
 */
const ContextSurface = (() => {
    let el = null, loading = false;

    const esc = (s) => (typeof escapeHtml === 'function')
        ? escapeHtml(String(s == null ? '' : s)) : String(s == null ? '' : s);
    const hasControl = () =>
        (typeof AgentChat !== 'undefined' && AgentChat.hasControl) ? AgentChat.hasControl() : true;

    async function fetchAndRender() {
        if (!el || loading) return;
        loading = true;
        try {
            const [ctx, nb] = await Promise.all([
                fetch('/api/context').then(r => r.json()).catch(() => ({})),
                fetch('/api/notebook/notes?limit=5').then(r => r.json()).catch(() => ({})),
            ]);
            render(ctx || {}, (nb && nb.notes) || []);
        } catch (e) { /* keep last render */ }
        finally { loading = false; }
    }

    function section(label, html) { return html ? `<div class="cx-lens"><div class="cx-lens-h">${label}</div>${html}</div>` : ''; }

    function render(data, notes) {
        if (!el) return;
        notes = notes || [];
        const hc = hasControl();
        const questions = data.questions || [], watchpoints = data.watchpoints || [], expectations = data.expectations || [];
        el.classList.remove('hidden');
        if (!questions.length && !watchpoints.length && !expectations.length && !notes.length) {
            // Show an empty-state rather than vanishing, so the surface is
            // discoverable before the agent has formed any beliefs.
            el.innerHTML = '<div class="cx-title">Agent’s view</div>' +
                '<div class="cx-empty">Nothing yet — the agent’s notes, expectations, and open questions appear here as it works.</div>';
            return;
        }

        const qHtml = questions.map(it => `
            <div class="cx-item" data-kind="questions" data-id="${esc(it.id)}">
                <span class="cx-dot cx-q"></span>
                <span class="cx-text">${esc(it.content)}</span>
                ${hc ? '<button class="cx-act" data-act="answer">Answer</button>' : ''}
                ${hc ? '<div class="cx-answer hidden"><input class="cx-answer-input" placeholder="Answer / resolve…"><button class="cx-answer-go">→</button></div>' : ''}
            </div>`).join('');
        const wHtml = watchpoints.map(it => `
            <div class="cx-item" data-kind="watchpoints" data-id="${esc(it.id)}">
                <span class="cx-dot cx-w"></span>
                <span class="cx-text">${esc(it.target)}${it.condition ? ' — ' + esc(it.condition) : ''}</span>
                ${hc ? '<button class="cx-act" data-act="resolve">Resolve</button>' : ''}
            </div>`).join('');
        const eHtml = expectations.map(it => `
            <div class="cx-item" data-kind="expectations" data-id="${esc(it.id)}">
                <span class="cx-dot cx-e"></span>
                <span class="cx-text">${esc(it.target)}${it.prediction ? ': ' + esc(it.prediction) : ''}</span>
                ${hc ? '<button class="cx-act" data-act="confirm" title="Mark confirmed">✓</button>' : ''}
            </div>`).join('');

        // kind → existing cx-dot color: observation=blue, finding=green, question=amber
        const dotFor = (k) => k === 'finding' ? 'cx-e' : (k === 'question' ? 'cx-q' : 'cx-w');
        const nHtml = notes.map(n => `
            <div class="cx-item cx-note" data-note="1">
                <span class="cx-dot ${dotFor(n.kind)}"></span>
                <span class="cx-text">${esc(n.title || n.body)}</span>
            </div>`).join('');

        el.innerHTML = '<div class="cx-title">Agent’s view</div>' +
            section('Open questions', qHtml) + section('Watching', wHtml) +
            section('Expectations', eHtml) + section('From the notebook', nHtml);
        wire();
    }

    function wire() {
        el.querySelectorAll('.cx-item').forEach(item => {
            const kind = item.dataset.kind, id = item.dataset.id;
            const actBtn = item.querySelector('.cx-act');
            if (!actBtn) return;
            const act = actBtn.dataset.act;
            if (act === 'answer') {
                const box = item.querySelector('.cx-answer');
                const input = item.querySelector('.cx-answer-input');
                const submit = () => resolve(kind, id, { resolution: input.value.trim() });
                actBtn.addEventListener('click', () => { box.classList.toggle('hidden'); if (!box.classList.contains('hidden')) input.focus(); });
                item.querySelector('.cx-answer-go').addEventListener('click', submit);
                input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); submit(); } });
            } else if (act === 'resolve') {
                actBtn.addEventListener('click', () => resolve(kind, id, {}));
            } else if (act === 'confirm') {
                actBtn.addEventListener('click', () => resolve(kind, id, { status: 'confirmed' }));
            }
        });
        el.querySelectorAll('.cx-note').forEach(row => {
            row.style.cursor = 'pointer';
            row.addEventListener('click', () => {
                if (typeof switchTab === 'function') switchTab('notebook');
            });
        });
    }

    async function resolve(kind, id, body) {
        try {
            await fetch(`/api/context/${kind}/${encodeURIComponent(id)}/resolve`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
            });
        } catch (e) { /* ignore; surface stays as-is */ }
        fetchAndRender();  // CONTEXT_UPDATED also re-fetches for every client
    }

    function init() {
        el = document.getElementById('context-surface');
        if (!el) return;  // ux_v2 off → no-op
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('CONTEXT_UPDATED', () => fetchAndRender());
            ClientEventBus.on('AGENT_CONTROL', () => fetchAndRender());  // re-render with/without resolve controls
        }
        fetchAndRender();
    }

    document.addEventListener('DOMContentLoaded', init);
    return { refresh: fetchAndRender };
})();
