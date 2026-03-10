/**
 * Client-side Event Bus for Gently Visualization
 * Decouples WebSocket message routing from UI managers.
 *
 * Usage:
 *   ClientEventBus.on('VOLUME_ACQUIRED', (data) => { ... });
 *   ClientEventBus.emit('VOLUME_ACQUIRED', data);
 */
const ClientEventBus = {
    _listeners: {},

    on(eventType, handler) {
        if (!this._listeners[eventType]) {
            this._listeners[eventType] = [];
        }
        this._listeners[eventType].push(handler);
    },

    off(eventType, handler) {
        const list = this._listeners[eventType];
        if (!list) return;
        this._listeners[eventType] = list.filter(h => h !== handler);
    },

    emit(eventType, data) {
        const handlers = this._listeners[eventType];
        if (handlers) {
            for (const handler of handlers) {
                try {
                    handler(data);
                } catch (err) {
                    console.error(`EventBus handler error for ${eventType}:`, err);
                }
            }
        }
        // Also fire wildcard listeners
        const wildcardHandlers = this._listeners['*'];
        if (wildcardHandlers) {
            for (const handler of wildcardHandlers) {
                try {
                    handler(eventType, data);
                } catch (err) {
                    console.error(`EventBus wildcard handler error:`, err);
                }
            }
        }
    }
};
