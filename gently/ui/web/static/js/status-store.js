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
