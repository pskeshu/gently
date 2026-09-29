/**
 * SettingsStore — the browser's preferences, for everything that reads them.
 *
 * Three layers, the later winning: what the code ships with, this rig's
 * defaults (set from Settings with "Save as rig defaults"), and what this
 * browser has chosen.
 *
 * The middle layer used to reach nothing. The old Settings page fetched the
 * rig's defaults and layered them into its own form, but the dashboard read
 * localStorage alone, so a fresh browser on a rig with saved defaults showed
 * the shipped ones. The views read through here now, and hear
 * SETTINGS_CHANGED when anything moves.
 *
 * Loaded before every module that reads a preference. No dependencies.
 */
const SettingsStore = (function () {
    'use strict';

    const KEY = 'gently-dashboard-config';
    let _rig = {};
    let _rigLoaded = false;

    const isObj = v => v && typeof v === 'object' && !Array.isArray(v);

    function deepMerge(target, source) {
        const out = Object.assign({}, target);
        Object.keys(source || {}).forEach(k => {
            out[k] = isObj(source[k]) ? deepMerge(isObj(target[k]) ? target[k] : {}, source[k]) : source[k];
        });
        return out;
    }

    /** What this browser has chosen, and nothing else. */
    function local() {
        try {
            const v = JSON.parse(localStorage.getItem(KEY) || '{}');
            return isObj(v) ? v : {};
        } catch (_) { return {}; }
    }

    function dig(obj, path) {
        return String(path).split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
    }

    function put(obj, path, value) {
        const keys = String(path).split('.');
        let o = obj;
        keys.slice(0, -1).forEach(k => { if (!isObj(o[k])) o[k] = {}; o = o[k]; });
        o[keys[keys.length - 1]] = value;
        return obj;
    }

    function emit(detail) {
        if (typeof ClientEventBus !== 'undefined') ClientEventBus.emit('SETTINGS_CHANGED', detail || {});
    }

    /** shipped < rig < this browser */
    function merged(shipped) {
        return deepMerge(deepMerge(shipped || {}, _rig), local());
    }

    /** One preference. `fallback` is what the code ships with. */
    function get(path, fallback) {
        const mine = dig(local(), path);
        if (mine !== undefined) return mine;
        const rigs = dig(_rig, path);
        return rigs !== undefined ? rigs : fallback;
    }

    function set(path, value) {
        try {
            localStorage.setItem(KEY, JSON.stringify(put(local(), path, value)));
        } catch (_) { return false; }  // private mode: the choice lasts the page
        emit({ path, value });
        return true;
    }

    function replaceAll(prefs) {
        try { localStorage.setItem(KEY, JSON.stringify(isObj(prefs) ? prefs : {})); } catch (_) { return false; }
        emit({ path: null });
        return true;
    }

    function clear() {
        try { localStorage.removeItem(KEY); } catch (_) { /* nothing to clear */ }
        emit({ path: null });
    }

    function setRigDefaults(defaults) {
        _rig = isObj(defaults) ? defaults : {};
        _rigLoaded = true;
        emit({ path: null, source: 'rig' });
    }

    async function loadRigDefaults() {
        if (_rigLoaded) return _rig;
        try {
            const res = await fetch('/api/config/dashboard-defaults');
            const d = res.ok ? await res.json() : {};
            setRigDefaults(d);
        } catch (_) { _rigLoaded = true; }
        return _rig;
    }

    return {
        KEY, deepMerge, dig, local, merged, get, set, replaceAll, clear,
        setRigDefaults, loadRigDefaults, rigDefaults: () => _rig,
    };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = SettingsStore;
