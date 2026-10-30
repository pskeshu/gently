/**
 * Inbox — the asks addressed to you: lines on sessions and projects that
 * name you with @. A count beside your name in the header, the list behind
 * it, and a toast when a new one lands while you are here. Read from
 * /api/inbox, which derives it from the notebook; "seen" is the one thing
 * it writes. Opening an ask goes to the session (or project) it was
 * written on, where the answer is written: a line, or the marks themselves.
 * The asks are one list in time order: an unopened one carries a dot and a
 * heavier line, an opened one sits where it was in plain weight, an answered
 * one says so. Under them sit your own lines, "By you": where each was
 * written and, for an ask of yours, who has answered it.
 */
const Inbox = (() => {
    let data = { me: null, asks: [], new: [], open: [], done: [], mine: [] };
    let tab = 'for';          // which tab the popover shows: 'for' (you) or 'by' (you)
    let known = null;        // ids in `new` at the last load, so a toast is for a fresh one
    let pill = null, pop = null, timer = null;

    async function load() {
        let d;
        try { d = await (await fetch('/api/inbox')).json(); } catch (e) { return; }
        if (!d || !Array.isArray(d.asks)) return;
        data = d;
        const unseen = data.asks.filter(i => !i.seen);
        const ids = new Set(unseen.map(i => i.note.id));
        if (known && data.me) {
            const fresh = unseen.find(i => !known.has(i.note.id));
            if (fresh && typeof showGentlyToast === 'function') {
                showGentlyToast(`${Who.name(fresh.note.by)}${place(fresh, ', on ')}: ${fresh.note.body}`, 'Open', () => open(fresh), 8000);
            }
        }
        known = ids;
        draw();
    }

    function schedule() {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => { timer = null; load(); }, 400);
    }

    function place(i, lead) {
        const name = i.session ? (i.session.name || String(i.session.id).slice(0, 8)) : (i.project ? i.project.name : '');
        return name ? `${lead || ''}${name}` : '';
    }

    // The count sits after your name. Presence redraws its container on every
    // update, so the pill is put back each time.
    function draw() {
        const you = document.querySelector('.presence-you');
        if (!you) return;
        if (!pill) {
            pill = document.createElement('button');
            pill.type = 'button';
            pill.className = 'inbox-pill';
            pill.addEventListener('click', e => { e.stopPropagation(); toggle(); });
        }
        const n = data.asks.filter(i => !i.seen).length;
        // The count is what you have not opened. Always drawn: with nothing
        // unopened it is a quiet 0, and behind it are the list and your own lines.
        pill.textContent = String(n);
        pill.classList.toggle('has-new', n > 0);
        pill.classList.toggle('is-empty', !n);
        pill.title = n ? `${n} unopened for you` : 'Nothing unopened · the list and your own lines';
        if (pill.previousElementSibling !== you) you.insertAdjacentElement('afterend', pill);
    }

    function toggle() { if (pop) close(); else show(); }
    function close() { if (pop) pop.remove(); pop = null; document.removeEventListener('click', onDoc); }
    function onDoc(e) { if (pop && !pop.contains(e.target)) close(); }

    // One ask: unopened carries a dot and a heavier line; opened is plain;
    // answered says so. The place is where it was written.
    function item(i) {
        const where = place(i, 'on ');
        const cls = (i.seen ? 'is-seen' : 'is-new') + (i.answered ? ' is-answered' : '');
        return `<button type="button" class="presence-pop-item inbox-item ${cls}" data-open="${escapeHtml(i.note.id)}" title="${i.seen ? 'Opened' : 'Not opened yet'}${i.answered ? ' · answered' : ''}">
            <span class="inbox-who rd-head"><i class="inbox-dot" aria-hidden="true"></i>${Who.head(i.note.by)}<span class="rd-when">${escapeHtml(formatDate(i.note.created_at))}</span>${where ? `<span class="rd-place">${escapeHtml(where)}</span>` : ''}${i.answered ? '<span class="inbox-state is-answered">answered</span>' : ''}</span>
            <span class="inbox-line">${Mentions.html(i.note.body, i.note.mentions)}</span>
        </button>`;
    }

    // One of your own lines: when and where, and for an ask, who has answered.
    function mineItem(i) {
        const where = place(i, 'on ');
        const asked = (i.note.mentions || []);
        const answered = i.answered_by || [];
        const names = ids => ids.map(p => `<b class="rd-mention" style="--who:${escapeHtml(Who.color(p))}">${escapeHtml(Who.name(p))}</b>`).join(', ');
        const state = !asked.length ? '' : answered.length
            ? `<span class="inbox-state is-answered">answered by ${names(answered)}</span>`
            : `<span class="inbox-state is-waiting">waiting on ${names(asked)}</span>`;
        return `<button type="button" class="presence-pop-item inbox-item is-mine" data-open="${escapeHtml(i.note.id)}" data-mine="1">
            <span class="inbox-who rd-head"><span class="rd-when">${escapeHtml(formatDate(i.note.created_at))}</span>${where ? `<span class="rd-place">${escapeHtml(where)}</span>` : ''}${state}</span>
            <span class="inbox-line">${Mentions.html(i.note.body, i.note.mentions)}</span>
        </button>`;
    }

    function show() {
        document.querySelectorAll('.presence-pop').forEach(p => p.remove());
        pop = document.createElement('div');
        pop.className = 'presence-pop inbox-pop';
        paint(pop);
        (document.getElementById('presence-container') || document.body).appendChild(pop);
        setTimeout(() => document.addEventListener('click', onDoc), 0);
    }

    // Two tabs: the asks for you, with how many are unopened, and your own
    // lines. Switching stays inside the popover; the tab is remembered.
    function paint(pop) {
        const asks = data.asks || [], mine = data.mine || [];
        const unopened = asks.filter(i => !i.seen).length;
        const tabs = `<div class="inbox-tabs" role="tablist">
            <button type="button" role="tab" class="inbox-tab${tab === 'for' ? ' on' : ''}" data-tab="for" aria-selected="${tab === 'for'}">For you${unopened ? ` <b class="inbox-tab-n">${unopened}</b>` : ''}</button>
            <button type="button" role="tab" class="inbox-tab${tab === 'by' ? ' on' : ''}" data-tab="by" aria-selected="${tab === 'by'}">By you</button>
        </div>`;
        const body = tab === 'for'
            ? (asks.length ? asks.slice(0, 30).map(item).join('') : '<div class="inbox-none">Nothing for you yet.</div>')
            : (mine.length ? mine.slice(0, 20).map(mineItem).join('') : '<div class="inbox-none">Nothing written yet. A line on a session or a project lands here.</div>');
        pop.innerHTML = tabs + `<div class="inbox-list">${body}</div>`;
        pop.querySelectorAll('.inbox-tab').forEach(b => b.addEventListener('click', e => { e.stopPropagation(); tab = b.dataset.tab; paint(pop); }));
        const all = [...asks, ...mine];
        pop.querySelectorAll('[data-open]').forEach(b => b.addEventListener('click', () => {
            const i = all.find(x => x.note.id === b.dataset.open);
            if (i) open(i, { mine: !!b.dataset.mine });
        }));
    }

    // Opening an ask: it is seen, and you are taken to where it was written.
    // Opening one of your own lines just goes there.
    async function open(i, opts = {}) {
        close();
        if (!opts.mine) {
            try {
                await fetch('/api/inbox/seen', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids: [i.note.id] }) });
            } catch (e) { /* the page still opens */ }
        }
        if (i.session && i.session.id) {
            goTab('sessions');
            if (typeof ReviewApp !== 'undefined' && ReviewApp.loadSession) ReviewApp.loadSession(i.session.id);
        } else if (i.project && i.project.id) {
            goTab('plans');
            if (typeof openCampaign === 'function') openCampaign(i.project.id);
        }
        load();
    }

    function goTab(tab) {
        const nav = document.querySelector(`.v2-nav-item[data-tab="${tab}"]`);
        if (nav) nav.click();
        else if (typeof switchTab === 'function') switchTab(tab);
    }

    function init() {
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.on('CONTEXT_UPDATED', d => { if (!d || !d.kind || d.kind === 'note') schedule(); });
            ClientEventBus.on('PRESENCE_UPDATE', () => { draw(); schedule(); });
        }
        const start = () => (typeof Who !== 'undefined' && Who.load ? Who.load() : Promise.resolve()).then(load);
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
        else start();
    }

    init();
    return { load, open };
})();
