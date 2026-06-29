/**
 * Main application state and initialization for Gently Visualization
 */

// State
const state = {
    ws: null,
    connected: false,
    tab: TABS.HOME,  // Default to the Home landing tab
    snapshots: [],
    calibration: [],
    embryos: [],
    volumes3d: [],
    currentImage: null,
    current3dVolume: null,  // Currently displayed 3D volume
    currentZ: 0,
    // Events tab state
    allEvents: [],
    eventTypeFilter: '',
    eventSourceFilter: '',
    eventSources: new Set()
};

// Data type classification
const CALIBRATION_TYPES = ['focus_sweep', 'focus_plot', 'edge_detection', 'calibration_summary',
                           'focus_snap', 'focus_coarse', 'focus_curve', 'focus_assess'];
const ANALYSIS_TYPES = ['segmentation', 'detection', 'classification', 'tracking',
                        'roi_detection', 'cropped_roi', 'vision_prepared', 'timeline', 'cv_visualization'];
const VOLUME_TYPES = ['volume', 'volume_projection', 'z_stack', 'timelapse'];

// Statusbar (cached DOM refs)
let _statusLeft, _statusRight, _timelapseText;
function updateStatusbar() {
    if (!_statusLeft) _statusLeft = document.getElementById('status-left');
    if (!_statusRight) _statusRight = document.getElementById('status-right');
    if (!_timelapseText) _timelapseText = document.getElementById('timelapse-status-text');
    if (!_statusLeft) return;

    // Plans and Sessions tabs manage their own statusbar content
    if (state.tab === TABS.PLANS || state.tab === TABS.SESSIONS) return;

    const embryoCount = state.embryos?.length || 0;
    const imageCount = state.snapshots?.length || 0;
    const eventCount = state.allEvents?.length || 0;
    _statusLeft.textContent = `${embryoCount} embryo${embryoCount !== 1 ? 's' : ''} \u00B7 ${imageCount} image${imageCount !== 1 ? 's' : ''} \u00B7 ${eventCount} event${eventCount !== 1 ? 's' : ''}`;
    if (_statusRight && _timelapseText) _statusRight.textContent = _timelapseText.textContent;
}

// UI Update functions
function updateMainCount() {
    const el = document.getElementById('main-count');
    if (el) el.textContent = state.snapshots.length;
}

function updateCalibrationCount() {
    const el = document.getElementById('calibration-count');
    if (el) el.textContent = state.calibration.length + state.volumes3d.length;
}

function switchTab(tabName) {
    if (!tabName) return;
    state.tab = tabName;
    // ux_v2 grouped rail mirrors the active tab off this single chokepoint.
    if (typeof ClientEventBus !== 'undefined') ClientEventBus.emit('TAB_CHANGED', tabName);

    // Update tab styling
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    const activeTab = document.querySelector(`.tab[data-tab="${tabName}"]`);
    if (activeTab) activeTab.classList.add('active');

    // Show/hide content
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const content = document.getElementById(`${tabName}-content`);
    if (content) content.classList.add('active');

    // Lazy-init Home landing tab
    if (tabName === TABS.HOME && typeof HomeApp !== 'undefined') HomeApp.init();

    // Render galleries
    if (tabName === TABS.CALIBRATION) renderCalibrationGallery();
    if (tabName === TABS.EVENTS) renderEventsTable();

    // Clear detection badge when viewing Embryos tab
    if (tabName === TABS.EMBRYOS && typeof EmbryosManager !== 'undefined') {
        EmbryosManager.clearDetectionBadge();
    }

    // Lazy-init Plans tab
    if (tabName === TABS.PLANS && typeof CampaignsApp !== 'undefined') {
        CampaignsApp.init();
    }

    // Lazy-init Sessions tab
    if (tabName === TABS.SESSIONS && typeof ReviewApp !== 'undefined') {
        ReviewApp.init();
    }

    // Lazy-init Experiment tab (mockup with stubbed data)
    if (tabName === TABS.EXPERIMENT && typeof ExperimentOverview !== 'undefined') {
        ExperimentOverview.init();
    }

    // Lazy-init Notebook tab
    if (tabName === TABS.NOTEBOOK && typeof NotebookApp !== 'undefined') {
        NotebookApp.init();
    }

    // Lazy-init Gallery tab
    if (tabName === TABS.GALLERY && typeof GalleryTab !== 'undefined') {
        GalleryTab.init();
    }

    // Update statusbar for context
    updateStatusbar();
}

/**
 * Copy session ID to clipboard
 */
function copySessionId() {
    const sessionLink = document.getElementById('session-id-link');
    const copyBtn = document.getElementById('session-copy-btn');

    if (!sessionLink || !sessionLink.textContent) return;

    const sessionId = sessionLink.textContent;

    function onCopied() {
        copyBtn.classList.add('copied');
        copyBtn.title = 'Copied!';
        setTimeout(() => {
            copyBtn.classList.remove('copied');
            copyBtn.title = 'Copy session ID';
        }, 1500);
    }

    // clipboard API requires secure context; fall back for plain HTTP
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(sessionId).then(onCopied).catch(err => {
            console.error('Failed to copy session ID:', err);
        });
    } else {
        const ta = document.createElement('textarea');
        ta.value = sessionId;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        ta.remove();
        onCopied();
    }
}

/**
 * Tooltip System - Shows helpful hints on hover
 */
const Tooltips = {
    current: null,
    timeout: null,

    init() {
        // Use event delegation for better performance
        document.addEventListener('mouseenter', (e) => {
            if (!e.target || !e.target.closest) return;
            const target = e.target.closest('[data-tooltip]');
            if (target) this.show(target);
        }, true);

        document.addEventListener('mouseleave', (e) => {
            if (!e.target || !e.target.closest) return;
            const target = e.target.closest('[data-tooltip]');
            if (target) this.hide();
        }, true);

        // Hide on scroll
        document.addEventListener('scroll', () => this.hide(), true);
    },

    show(target) {
        const text = target.dataset.tooltip;
        if (!text) return;

        // Small delay before showing
        this.timeout = setTimeout(() => {
            // Remove any existing tooltip
            this.hide();

            // Create tooltip element
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = text;
            document.body.appendChild(tooltip);

            // Position below the target element
            const rect = target.getBoundingClientRect();
            const tooltipRect = tooltip.getBoundingClientRect();

            // Default position below, centered
            let top = rect.bottom + 8;
            let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);

            // Keep within viewport bounds
            if (left < 8) left = 8;
            if (left + tooltipRect.width > window.innerWidth - 8) {
                left = window.innerWidth - tooltipRect.width - 8;
            }

            // If below would go off-screen, show above
            if (top + tooltipRect.height > window.innerHeight - 8) {
                top = rect.top - tooltipRect.height - 8;
                tooltip.classList.add('tooltip-above');
            }

            tooltip.style.top = `${top}px`;
            tooltip.style.left = `${left}px`;

            this.current = tooltip;
        }, 150);  // 150ms delay - faster for better responsiveness
    },

    hide() {
        if (this.timeout) {
            clearTimeout(this.timeout);
            this.timeout = null;
        }
        if (this.current) {
            this.current.remove();
            this.current = null;
        }
    }
};

/**
 * Presence Manager - Collaborative presence like Google Docs
 */
const PresenceManager = {
    clientId: null,
    name: null,
    clients: [],

    // Animal names for anonymous users
    ANIMALS: [
        'Koala', 'Penguin', 'Fox', 'Owl', 'Panda', 'Tiger', 'Dolphin',
        'Eagle', 'Bear', 'Wolf', 'Rabbit', 'Deer', 'Otter', 'Falcon',
        'Hedgehog', 'Badger', 'Lynx', 'Seal', 'Raven', 'Crane', 'Gecko',
        'Meerkat', 'Lemur', 'Toucan', 'Sloth', 'Jaguar', 'Pelican', 'Moose'
    ],

    init() {
        // Load or generate client ID
        this.clientId = localStorage.getItem('gently-client-id');
        if (!this.clientId) {
            this.clientId = this.generateId();
            localStorage.setItem('gently-client-id', this.clientId);
        }

        // Load saved name or generate anonymous name
        this.name = localStorage.getItem('gently-user-name');
        if (!this.name) {
            this.name = this.getAnonymousName();
        }

        // Subscribe to presence updates via event bus
        ClientEventBus.on('PRESENCE_UPDATE', (clients) => this.handlePresenceUpdate(clients));
    },

    generateId() {
        return 'xxxx-xxxx'.replace(/x/g, () =>
            Math.floor(Math.random() * 16).toString(16)
        );
    },

    getAnonymousName() {
        // Use client ID to pick a consistent animal
        const hash = this.clientId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
        const animal = this.ANIMALS[hash % this.ANIMALS.length];
        return `Anonymous ${animal}`;
    },

    sendJoin() {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: 'join',
                client_id: this.clientId,
                name: this.name
            }));
        }
    },

    handlePresenceUpdate(clients) {
        this.clients = clients;
        this.render();
    },

    render() {
        const container = document.getElementById('presence-container');
        if (!container) return;

        // Clear existing
        container.innerHTML = '';

        // Sort: put "you" first
        const sorted = [...this.clients].sort((a, b) => {
            if (a.is_you) return -1;
            if (b.is_you) return 1;
            return 0;
        });

        // Render avatars (max 5 visible, then +N)
        const maxVisible = 5;
        const visible = sorted.slice(0, maxVisible);
        const overflow = sorted.length - maxVisible;

        visible.forEach(client => {
            const avatar = document.createElement('div');
            avatar.className = 'presence-avatar' + (client.is_you ? ' is-you' : '');
            avatar.style.backgroundColor = client.color;
            avatar.textContent = this.getInitials(client.name);
            avatar.setAttribute('data-tooltip', client.is_you ? `${client.name} (you)` : client.name);

            // Click on your own avatar to change name
            if (client.is_you) {
                avatar.style.cursor = 'pointer';
                avatar.addEventListener('click', () => this.showNamePrompt());
            }

            container.appendChild(avatar);
        });

        // Show overflow count
        if (overflow > 0) {
            const more = document.createElement('div');
            more.className = 'presence-overflow';
            more.textContent = `+${overflow}`;
            more.setAttribute('data-tooltip', `${overflow} more viewer${overflow > 1 ? 's' : ''}`);
            container.appendChild(more);
        }
    },

    getInitials(name) {
        if (!name) return '?';
        // For "Anonymous X", use animal initial
        if (name.startsWith('Anonymous ')) {
            return name.split(' ')[1]?.[0] || 'A';
        }
        // Otherwise use first letter of each word (max 2)
        const words = name.trim().split(/\s+/);
        if (words.length === 1) {
            return words[0][0].toUpperCase();
        }
        return (words[0][0] + words[1][0]).toUpperCase();
    },

    setName(name) {
        if (!name || name.trim() === '') return;

        this.name = name.trim();
        localStorage.setItem('gently-user-name', this.name);

        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({
                type: 'set_name',
                name: this.name
            }));
        }
    },

    showNamePrompt() {
        const current = this.name;
        const isAnonymous = current.startsWith('Anonymous ');

        const newName = prompt(
            'Enter your display name (or leave blank for anonymous):',
            isAnonymous ? '' : current
        );

        if (newName === null) return; // Cancelled

        if (newName.trim() === '') {
            // Reset to anonymous
            localStorage.removeItem('gently-user-name');
            this.name = this.getAnonymousName();
            this.setName(this.name);
        } else {
            this.setName(newName);
        }
    }
};

/**
 * Theme Manager - Dark/Light mode toggle
 */
const ThemeManager = {
    storageKey: 'gently-theme',

    init() {
        // Load saved theme or default to light (toggle handled by _header.html)
        const savedTheme = localStorage.getItem(this.storageKey) || 'light';
        this.setTheme(savedTheme);
    },

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(this.storageKey, theme);
    },

    toggle() {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        this.setTheme(next);
    },

    getTheme() {
        return document.documentElement.getAttribute('data-theme') || 'dark';
    }
};

/**
 * Keyboard Shortcuts Handler
 */
const KeyboardShortcuts = {
    enabled: true,

    shortcuts: {
        '1': () => switchTab(TABS.EMBRYOS),     // Embryos
        '2': () => switchTab(TABS.EVENTS),      // System
        '3': () => switchTab('main'),            // Live View
        '4': () => switchTab(TABS.CALIBRATION),  // Calibration
        '5': () => switchTab(TABS.DEVICES),      // Devices
        '6': () => switchTab(TABS.EXPERIMENT),   // Experiment
        'ArrowUp': () => KeyboardShortcuts.adjustZSlider(1),
        'ArrowDown': () => KeyboardShortcuts.adjustZSlider(-1),
        '?': () => KeyboardShortcuts.showHelp(),
        't': () => ThemeManager.toggle(),
        'p': () => { if (state.tab === TABS.CALIBRATION) CalibrationManager.switchView('profile'); },
        'g': () => { if (state.tab === TABS.CALIBRATION) CalibrationManager.switchView('gallery'); },
    },

    init() {
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
    },

    handleKeyDown(e) {
        // Ignore if typing in input or lightbox is open
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (typeof Lightbox !== 'undefined' && Lightbox.isOpen) return;

        const handler = this.shortcuts[e.key];
        if (handler) {
            e.preventDefault();
            handler();
        }
    },

    adjustZSlider(direction) {
        if (state.tab !== 'main' || !state.current3dVolume) return;

        const slider = document.getElementById('z-slider');
        if (!slider) return;

        const newVal = parseInt(slider.value) + direction;
        if (newVal >= parseInt(slider.min) && newVal <= parseInt(slider.max)) {
            slider.value = newVal;
            slider.dispatchEvent(new Event('input'));
        }
    },

    showHelp() {
        // Create help modal if it doesn't exist
        let modal = document.getElementById('shortcuts-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'shortcuts-modal';
            modal.className = 'shortcuts-modal';
            modal.innerHTML = `
                <div class="shortcuts-content">
                    <div class="shortcuts-header">
                        <h3>Keyboard Shortcuts</h3>
                        <button class="shortcuts-close" onclick="KeyboardShortcuts.hideHelp()">&times;</button>
                    </div>
                    <div class="shortcuts-body">
                        <div class="shortcut-group">
                            <h4>Navigation</h4>
                            <div class="shortcut"><kbd>1</kbd> Embryos tab</div>
                            <div class="shortcut"><kbd>2</kbd> System tab</div>
                            <div class="shortcut"><kbd>3</kbd> Live View tab</div>
                            <div class="shortcut"><kbd>4</kbd> Calibration tab</div>
                            <div class="shortcut"><kbd>5</kbd> Devices tab</div>
                        </div>
                        <div class="shortcut-group">
                            <h4>3D Volume</h4>
                            <div class="shortcut"><kbd>&uarr;</kbd> <kbd>&darr;</kbd> Navigate Z-slices</div>
                        </div>
                        <div class="shortcut-group">
                            <h4>Lightbox</h4>
                            <div class="shortcut"><kbd>&larr;</kbd> <kbd>&rarr;</kbd> Prev/Next image</div>
                            <div class="shortcut"><kbd>Esc</kbd> Close lightbox</div>
                            <div class="shortcut"><kbd>+</kbd> <kbd>-</kbd> Zoom in/out</div>
                            <div class="shortcut"><kbd>0</kbd> Reset zoom</div>
                        </div>
                        <div class="shortcut-group">
                            <h4>Events</h4>
                            <div class="shortcut"><kbd>Ctrl</kbd>+<kbd>F</kbd> Search events</div>
                        </div>
                        <div class="shortcut-group">
                            <h4>General</h4>
                            <div class="shortcut"><kbd>t</kbd> Toggle dark/light theme</div>
                            <div class="shortcut"><kbd>?</kbd> Show this help</div>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);

            // Close on backdrop click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) this.hideHelp();
            });
        }

        modal.classList.add('active');
    },

    hideHelp() {
        const modal = document.getElementById('shortcuts-modal');
        if (modal) modal.classList.remove('active');
    }
};

// Event listeners
// ==========================================
// Connection Status Popover
// ==========================================

function toggleStatusPopover(event) {
    const popover = document.getElementById('status-popover');
    const wrapper = document.getElementById('status-button')?.closest('.status-button-wrapper');
    toggleDropdown(popover, event);
    // Sync the wrapper's 'open' class for chevron rotation CSS
    if (wrapper) {
        wrapper.classList.toggle('open', popover && !popover.classList.contains('hidden'));
    }
}

let _microscopeConnected = false;

function fetchDeviceStatus() {
    fetch('/api/device-status')
        .then(r => r.json())
        .then(data => {
            _microscopeConnected = data.microscope;
            ConnectionStatus.setMicroscope(data.microscope);
        })
        .catch(() => {
            // Transient poll failure: keep the last-known badge. The next
            // successful poll re-renders via the store if the value changed
            // (writing '--' here could stick, since the store only re-renders
            // on an actual change, not on an unchanged success).
        });
}

function _setBadge(id, isOn, onText, offText) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = isOn ? onText : offText;
    el.classList.toggle('online', isOn);
    el.classList.toggle('offline', !isOn);
}

function updateGentlyStatus(connected) {
    // Feed the single source of truth; the header re-renders via the
    // ConnectionStatus subscriber (renderConnectionUI).
    ConnectionStatus.setGently(connected);
}

// Single renderer for the header connection UI, driven by a ConnectionStatus
// snapshot. Subscribed once at startup, so the pill, both popover badges, and
// the dot always reflect the same shared state.
function renderConnectionUI(s) {
    _setBadge('status-gently-badge', s.gentlyConnected, 'Online', 'Offline');
    _setBadge('status-microscope-badge', s.microscopeConnected, 'Online', 'Offline');
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    if (!dot || !text) return;

    dot.classList.remove('connected', 'partial');
    if (s.gentlyConnected && s.microscopeConnected) {
        dot.classList.add('connected');
        text.textContent = 'Connected';
    } else if (s.gentlyConnected) {
        dot.classList.add('partial');
        text.textContent = 'Online';
    } else {
        text.textContent = 'Offline';
    }
}

// Back-compat shim: any legacy caller re-renders from the current snapshot.
function updateTopLevelDot() {
    renderConnectionUI(ConnectionStatus.get());
}

document.addEventListener('DOMContentLoaded', () => {
    // Initialize presence manager (before WebSocket so ID is ready)
    PresenceManager.init();

    // Initialize tooltip system
    Tooltips.init();

    // Initialize theme manager
    ThemeManager.init();

    // Initialize keyboard shortcuts
    KeyboardShortcuts.init();

    // Tab click handlers
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });

    // Z-slider event listener (Live View)
    const zSlider = document.getElementById('z-slider');
    if (zSlider) {
        zSlider.addEventListener('input', (e) => {
            if (!state.current3dVolume) return;
            state.currentZ = parseInt(e.target.value);
            updateZSliderDisplay();
            loadZSlice(state.current3dVolume.uid, state.currentZ);
        });
    }

    // Initialize events tab
    initEventsTab();

    // Sync wrapper 'open' class when toggleDropdown closes the popover via outside click
    document.addEventListener('click', () => {
        const popover = document.getElementById('status-popover');
        const wrapper = document.getElementById('status-button')?.closest('.status-button-wrapper');
        if (wrapper && popover) {
            wrapper.classList.toggle('open', !popover.classList.contains('hidden'));
        }
    });

    // Connection status: one source of truth, three writers (this /ws, the
    // device-status poll, and the agent /ws/agent). Subscribe the header
    // renderer BEFORE connecting so the first handshake renders correctly.
    ConnectionStatus.subscribe(renderConnectionUI);

    // Start WebSocket connection
    connectWebSocket();

    // Fetch device status periodically
    fetchDeviceStatus();
    setInterval(fetchDeviceStatus, 15000);

    // Initial statusbar update (subsequent updates are event-driven from websocket.js)
    setTimeout(updateStatusbar, 500);

    // Handle hash-based tab routing (e.g., /#plans, /#sessions, /#plans:campaignId)
    const hash = window.location.hash.slice(1); // remove #
    if (hash) {
        const [tab, param] = hash.split(':');
        if (tab === TABS.HOME || tab === TABS.PLANS || tab === TABS.SESSIONS || tab === TABS.EMBRYOS || tab === TABS.CALIBRATION || tab === TABS.EVENTS || tab === TABS.EXPERIMENT || tab === TABS.NOTEBOOK || tab === TABS.GALLERY) {
            switchTab(tab);
            if (tab === TABS.PLANS && param && typeof openCampaign === 'function') {
                setTimeout(() => openCampaign(param), 200);
            }
        }
        // Clean up the hash
        history.replaceState(null, '', '/');
    }
});
