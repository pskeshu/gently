/**
 * Inbox — the asks addressed to you: lines on sessions and projects that
 * name you with @. A count beside your name in the header, the list behind
 * it, and a toast when a new one lands while you are here. Read from
 * /api/inbox, which derives it from the notebook; "seen" is the one thing
 * it writes. Opening an ask goes to the session (or project) it was
 * written on, where the answer is written: a line, or the marks themselves.
 */
const Inbox = (() => {
    let data = { me: null, new: [], open: [], done: [] };
    let known = null;        // ids in `new` at the last load, so a toast is for a fresh one
    let pill = null, pop = null, timer = null;

    async function load() {
        let d;
        try { d = await (await fetch('/api/inbox')).json(); } catch (e) { return; }
        if (!d || !Array.isArray(d.new)) return;
        data = d;
        const ids = new Set(data.new.map(i => i.note.id));
        if (known && data.me) {
            const fresh = data.new.find(i => !known.has(i.note.id));
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
        const n = data.new.length, o = data.open.length;
        pill.textContent = String(n || o || '');
        pill.classList.toggle('has-new', n > 0);
        pill.classList.toggle('is-empty', !n && !o);
        pill.title = n ? `${n} new for you` : (o ? `${o} open for you` : 'Nothing for you');
        if (pill.previousElementSibling !== you) you.insertAdjacentElement('afterend', pill);
    }

    function toggle() { if (pop) close(); else show(); }
    function close() { if (pop) pop.remove(); pop = null; document.removeEventListener('click', onDoc); }
    function onDoc(e) { if (pop && !pop.contains(e.target)) close(); }

    function item(i, cls) {
        const where = place(i, 'on ');
        return `<button type="button" class="presence-pop-item inbox-item ${cls}" data-open="${escapeHtml(i.note.id)}">
            <span class="inbox-who">${Who.chip(i.note.by, { when: i.note.created_at })}${where ? ` <span class="pj-muted">${escapeHtml(where)}</span>` : ''}</span>
            <span class="inbox-line">${Mentions.html(i.note.body, i.note.mentions)}</span>
        </button>`;
    }

    function show() {
        document.querySelectorAll('.presence-pop').forEach(p => p.remove());
        pop = document.createElement('div');
        pop.className = 'presence-pop inbox-pop';
        const section = (title, items, cls) => items.length
            ? `<div class="presence-pop-title">${title}</div>` + items.map(i => item(i, cls)).join('') : '';
        pop.innerHTML = section('New', data.new, 'is-new') + section('Open', data.open, '') + section('Done', data.done.slice(0, 5), 'is-done')
            || '<div class="presence-pop-title">Nothing for you</div>';
        const all = [...data.new, ...data.open, ...data.done];
        pop.querySelectorAll('[data-open]').forEach(b => b.addEventListener('click', () => {
            const i = all.find(x => x.note.id === b.dataset.open);
            if (i) open(i);
        }));
        (document.getElementById('presence-container') || document.body).appendChild(pop);
        setTimeout(() => document.addEventListener('click', onDoc), 0);
    }

    // Opening an ask: it is seen, and you are taken to where it was written.
    async function open(i) {
        close();
        try {
            await fetch('/api/inbox/seen', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids: [i.note.id] }) });
        } catch (e) { /* the page still opens */ }
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
