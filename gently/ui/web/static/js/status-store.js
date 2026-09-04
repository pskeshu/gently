/**
 * ConnectionStatus — the single source of truth for connection liveness.
 *
 * Fixes the "three disagreeing indicators" bug where the header pill, the home
 * landing line, and the agent dock each computed connection state from their
 * own signal at their own time (home.js read state.connected ONCE at tab init,
 * before the /ws handshake, and never corrected — showing "Offline" while the
 * header showed "Online").
 *
 * Three genuinely distinct signals (kept separate, not flattened):
 *   - gentlyConnected     : the main /ws telemetry socket          (websocket.js)
 *   - microscopeConnected : the /api/device-status health poll     (app.js)
 *   - agentConnected      : the /ws/agent chat socket              (agent-chat.js)
 *
 * Writers call set*(); readers subscribe(). The store is STICKY: subscribe()
 * immediately replays the current snapshot to the new subscriber, so a late
 * subscriber can never miss the initial state. Events only fire on real change.
 */
const ConnectionStatus = (() => {
    const s = { gentlyConnected: false, microscopeConnected: false, agentConnected: false };

    function emit() {
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.emit('CONNECTION_STATUS', { ...s });
        }
    }

    function set(key, val) {
        val = !!val;
        if (s[key] === val) return;   // only emit on actual change
        s[key] = val;
        emit();
    }

    return {
        setGently(v) { set('gentlyConnected', v); },
        setMicroscope(v) { set('microscopeConnected', v); },
        setAgent(v) { set('agentConnected', v); },
        get() { return { ...s }; },

        /**
         * Subscribe to status changes AND immediately receive the current
         * snapshot (sticky replay). This is the guard against the original bug:
         * a subscriber that registers after the first emit still renders from
         * the correct current state instead of a stale default.
         */
        subscribe(handler) {
            if (typeof ClientEventBus !== 'undefined') {
                ClientEventBus.on('CONNECTION_STATUS', handler);
            }
            try { handler({ ...s }); } catch (e) { console.error('ConnectionStatus subscriber error', e); }
        }
    };
})();

/**
 * SharedState — the same semantics as ConnectionStatus above, for the values
 * that are NOT booleans and NOT about connectivity.
 *
 * Deliberately a sibling rather than an extension: ConnectionStatus coerces
 * with !!val and is named for connection liveness, so putting a stage position
 * in it would be a naming lie and would touch its five existing consumers for
 * no reason.
 *
 * It exists because the audit in docs/atrium/MIGRATION.md found the same value
 * maintained in several places at once, with no notification between them:
 *
 *   stageXY           4 independent copies (devices.js, operate.js,
 *                     occupancy3d.js, marking.js) fed by 3 unshared handlers.
 *                     Two pinned surfaces can display contradictory positions.
 *   selectedEmbryoId  4 copies (embryos.js, operate.js, gallery.js, devices.js)
 *                     with 7 writers in embryos.js alone.
 *   sessionId         5 separate derivations, one of which reads it out of the
 *                     DOM (app.js).
 *   agentBusy         a four-flag machine in agent-chat.js, hand-propagated to
 *                     five readers across eight files.
 *
 * Under tabs this was survivable because only one panel was visible. The Atrium
 * makes every window live at once (SPEC.md R2), so every state change has to
 * fan out to all of them — which is exactly what a sticky store does and what
 * manual invalidation cannot.
 *
 * STICKY: subscribe() replays the current value immediately, so a window that
 * mounts late cannot miss the state it needs. That is the property the original
 * "three disagreeing indicators" bug turned on.
 */
const SharedState = (() => {
    const s = {
        stageXY: null,          // {x, y} in um, or null if unknown
        selectedEmbryoId: null,
        sessionId: null,
        agentBusy: false,
        hasControl: false,
        // Read-back light state, owned by panels/light.js. One value, so every
        // mounted Light panel agrees — see docs/architecture/PANELS.md rule 2.
        light: null,
    };
    const subs = new Map();     // key -> Set<handler>

    const same = (a, b) => a === b
        || (a && b && typeof a === 'object' && typeof b === 'object'
            && JSON.stringify(a) === JSON.stringify(b));

    function set(key, val) {
        if (!(key in s)) { console.warn('SharedState: unknown key', key); return; }
        if (same(s[key], val)) return;           // only emit on real change
        s[key] = val;
        (subs.get(key) || []).forEach(h => {
            try { h(val, key); } catch (e) { console.error('SharedState subscriber error', key, e); }
        });
        if (typeof ClientEventBus !== 'undefined') {
            ClientEventBus.emit('SHARED_STATE', { key, value: val });
        }
    }

    return {
        set,
        get(key) { return key ? s[key] : { ...s }; },

        /** Subscribe to one key and immediately receive its current value. */
        on(key, handler) {
            if (!subs.has(key)) subs.set(key, new Set());
            subs.get(key).add(handler);
            try { handler(s[key], key); } catch (e) { console.error('SharedState subscriber error', key, e); }
            return () => subs.get(key).delete(handler);
        },
    };
})();
