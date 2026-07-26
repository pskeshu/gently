/* Session replay recorder — rrweb capture + semantic action log.
 *
 * Design contract (docs/superpowers/specs/2026-07-13-session-replay-design.md):
 * the app always wins over the recording. Everything here is wrapped so that any
 * failure degrades or disables the recorder silently — it must never throw into
 * the page, block the main thread, or surface errors to the UI.
 *
 * The client is deliberately blind: it does not know the gently session id.
 * It POSTs batches to /replay/ingest and the server files them under the
 * active session (or an unassigned bucket). Removal = delete this file and
 * its template include.
 */
(function () {
  "use strict";

  if (window.__gentlyReplay) return; // double-include guard
  var STATE = {
    tab: Math.random().toString(16).slice(2, 10),
    rrwebBuf: [],
    actionBuf: [],
    dropped: 0,
    failures: 0,
    disabled: false,
    stopFn: null,
    flushTimer: null,
  };
  window.__gentlyReplay = STATE;

  var INGEST_URL = "/replay/ingest";
  var FLUSH_MS = 4000; // short interval: bounds loss at tab close / app quit
  // Count-based cap, NOT bytes: sizing would mean JSON.stringify on every
  // rrweb event on the main thread (mutation-storm hot path). Serialization
  // happens exactly once, at flush. 20k events ≫ one flush interval's worth.
  var MAX_BUF_EVENTS = 20000; // drop-oldest beyond this (app > data)
  var MAX_FAILURES = 5; // consecutive ingest failures before self-disable
  var CHECKOUT_MS = 5 * 60 * 1000; // periodic full snapshots: cheap seeking later
  // Live camera <img> gets base64 data-URI src swaps at frame rate — recording
  // it would add ~100KB+ per frame on the main thread. Blocked from capture;
  // the bus-summary action records that frames were flowing instead.
  var BLOCK_SELECTOR = "#op-img-bottom, #op-img-spim";
  // Machine-driven, high-churn regions: the live map re-renders at stage-poll
  // rate, the 3D occupancy canvas animates, the temperature graph redraws on
  // every reading. In 'balanced' fidelity these are blocked from the DOM stream
  // (a placeholder box replays in their place) — they're the bulk of the volume.
  var HIGH_CHURN =
    "#devices-map-svg, #occ3d-container, #occ3d-minimap, #devices-temp-graph";

  // Recording fidelity, most-specific first: a ?replay=<level> URL override (for
  // an ad-hoc high/low-fidelity capture), else the server default stamped on the
  // recorder's <script> tag, else 'balanced'.
  function readFidelity() {
    try {
      var q = new URLSearchParams(location.search).get("replay");
      if (q && /^(full|balanced|actions|off)$/.test(q)) return q;
    } catch (e) {}
    try {
      var s = document.querySelector("script[data-gently-replay-fidelity]");
      var v = s && s.getAttribute("data-gently-replay-fidelity");
      if (v && /^(full|balanced|actions|off)$/.test(v)) return v;
    } catch (e) {}
    return "balanced";
  }

  function disable(reason) {
    if (STATE.disabled) return;
    STATE.disabled = true;
    try {
      if (STATE.stopFn) STATE.stopFn();
    } catch (e) {}
    try {
      if (STATE.flushTimer) clearTimeout(STATE.flushTimer);
    } catch (e) {}
    STATE.rrwebBuf = [];
    STATE.actionBuf = [];
    try {
      // One console line for postmortem of the postmortem tool; never a UI surface.
      console.warn("[gently-replay] recording disabled:", reason);
    } catch (e) {}
  }

  function pushEvent(buf, ev) {
    buf.push(ev);
    if (STATE.rrwebBuf.length > MAX_BUF_EVENTS) {
      // Drop oldest rrweb events (bulkiest stream); record the gap honestly.
      STATE.dropped += STATE.rrwebBuf.length - MAX_BUF_EVENTS;
      STATE.rrwebBuf.splice(0, STATE.rrwebBuf.length - MAX_BUF_EVENTS);
    }
  }

  function takeBatch() {
    // Fold the ClientEventBus counters into the action stream: one compact
    // record per batch saying what was flowing (frames, temperature, tokens)
    // without recording each high-frequency event.
    var counts = STATE.busCounts;
    var types = counts ? Object.keys(counts) : [];
    if (types.length) {
      STATE.actionBuf.push({
        t: new Date().toISOString(),
        action: "bus-summary",
        route: String(location.pathname),
        params: counts,
      });
      STATE.busCounts = {};
    }
    if (!STATE.rrwebBuf.length && !STATE.actionBuf.length) return null;
    var batch = {
      tab: STATE.tab,
      ts: Date.now(),
      url: String(location.pathname + location.search),
      rrweb: STATE.rrwebBuf,
      actions: STATE.actionBuf,
    };
    if (STATE.dropped) {
      batch.gap = { dropped: STATE.dropped };
      STATE.dropped = 0;
    }
    STATE.rrwebBuf = [];
    STATE.actionBuf = [];
    return batch;
  }

  // Stay well under the server's 32MB ingest cap. A single flush that exceeds it
  // is 413'd and the WHOLE batch is lost — rrweb frames AND the clicks/actions
  // riding with it — so we split the bulky rrweb stream instead of gambling a
  // giant POST. (The buffer is capped by event COUNT, not bytes, so batch size
  // is otherwise unbounded: a big full-snapshot or mutation storm can blow it.)
  var SAFE_MAX_BYTES = 8 * 1024 * 1024;

  // Send a batch, recursively halving its rrweb array until each POST body is
  // under SAFE_MAX_BYTES. Actions/gap ride with the FIRST chunk only, so nothing
  // is dropped and nothing is duplicated.
  function sendBatch(batch, depth) {
    var body;
    try {
      body = JSON.stringify(batch);
    } catch (e) {
      return Promise.resolve();
    }
    var rrweb = batch.rrweb || [];
    if (body.length <= SAFE_MAX_BYTES || rrweb.length <= 1 || depth > 12) {
      return fetch(INGEST_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
        // no keepalive on periodic flushes: keepalive caps bodies at ~64KB
      })
        .then(function (r) {
          if (r.ok) {
            STATE.failures = 0;
          } else {
            onFailure("HTTP " + r.status);
          }
        })
        .catch(function (e) {
          onFailure(e && e.message);
        });
    }
    // Too big — halve the rrweb stream. First half carries the actions + gap.
    var mid = Math.floor(rrweb.length / 2);
    var first = {
      tab: batch.tab, ts: batch.ts, url: batch.url,
      rrweb: rrweb.slice(0, mid), actions: batch.actions, gap: batch.gap,
    };
    var second = {
      tab: batch.tab, ts: batch.ts, url: batch.url,
      rrweb: rrweb.slice(mid), actions: [],
    };
    return sendBatch(first, depth + 1).then(function () {
      return sendBatch(second, depth + 1);
    });
  }

  function flush() {
    if (STATE.disabled) return;
    var batch = takeBatch();
    if (!batch) return scheduleFlush();
    sendBatch(batch, 0).then(scheduleFlush);
  }

  function onFailure(why) {
    STATE.failures += 1;
    if (STATE.failures >= MAX_FAILURES) disable("ingest failing (" + why + ")");
  }

  function scheduleFlush() {
    if (STATE.disabled) return;
    var backoff = Math.min(STATE.failures * 2000, 20000);
    STATE.flushTimer = setTimeout(function () {
      // WebKit (Linux desktop shell) has no requestIdleCallback — fall back.
      if (typeof requestIdleCallback === "function") {
        requestIdleCallback(flush, { timeout: 1500 });
      } else {
        flush();
      }
    }, FLUSH_MS + backoff);
  }

  function finalFlush() {
    if (STATE.disabled) return;
    var batch = takeBatch();
    if (!batch) return;
    batch.final = true;
    try {
      var body = JSON.stringify(batch);
      // sendBeacon survives page teardown but has a ~64KB quota; fall back to
      // keepalive fetch, then accept the loss (spec: lose ≤ one batch).
      var ok = false;
      if (navigator.sendBeacon) {
        ok = navigator.sendBeacon(
          INGEST_URL,
          new Blob([body], { type: "application/json" })
        );
      }
      if (!ok && body.length < 60000) {
        fetch(INGEST_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: body,
          keepalive: true,
        }).catch(function () {});
      }
    } catch (e) {}
  }

  /* ---- semantic action log (layer 1) ---- */

  function describeTarget(el) {
    var d = { tag: el.tagName ? el.tagName.toLowerCase() : "?" };
    if (el.id) d.id = el.id;
    if (el.name) d.name = el.name;
    // The UI already carries a semantic vocabulary in data-* attributes
    // (data-tab, data-view, data-bz, data-landing, …) — harvest it wholesale.
    try {
      var ds = el.dataset;
      var keys = ds ? Object.keys(ds) : [];
      if (keys.length) {
        d.data = {};
        for (var i = 0; i < Math.min(keys.length, 8); i++) {
          d.data[keys[i]] = String(ds[keys[i]]).slice(0, 60);
        }
      }
    } catch (e) {}
    var label = "";
    try {
      label = (el.getAttribute("aria-label") || el.title || el.textContent || "")
        .trim()
        .replace(/\s+/g, " ")
        .slice(0, 60);
    } catch (e) {}
    if (label) d.label = label;
    return d;
  }

  // Action name, best-first: explicit data-action; the existing semantic
  // data-* vocabulary (tab:home, view:board, bz:-10 …); a stable id; else a
  // class+text fallback (covers the few legacy inline-onclick buttons).
  var NAMING_DATA_KEYS = ["tab", "view", "landing", "subtab", "mode", "nbKind", "bz", "fd", "gv", "pz", "goTab", "step", "screen"];
  function actionName(el) {
    var v = el.getAttribute && el.getAttribute("data-action");
    if (v) return v;
    var ds = el.dataset || {};
    for (var i = 0; i < NAMING_DATA_KEYS.length; i++) {
      var k = NAMING_DATA_KEYS[i];
      if (ds[k] !== undefined) {
        return k.replace(/[A-Z]/g, function (c) { return "-" + c.toLowerCase(); }) + ":" + ds[k];
      }
    }
    if (el.id) return el.tagName.toLowerCase() + "#" + el.id;
    var cls = (el.className && String(el.className).split(/\s+/)[0]) || "";
    var text = "";
    try {
      text = (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 24);
    } catch (e) {}
    return "click:" + (cls || el.tagName.toLowerCase()) + (text ? ":" + text : "");
  }

  function logAction(action, el, extra) {
    if (STATE.disabled) return;
    var ev = {
      t: new Date().toISOString(),
      action: action,
      route: String(location.pathname),
    };
    if (el) ev.target = describeTarget(el);
    if (extra) ev.params = extra;
    pushEvent(STATE.actionBuf, ev);
  }

  function onClick(e) {
    try {
      var el = e.target && e.target.closest
        ? e.target.closest("[data-action], button, a, input[type=button], input[type=submit], [role=button], select, summary")
        : null;
      if (!el) return;
      logAction(actionName(el), el);
    } catch (err) {}
  }

  // Count ClientEventBus traffic by type (frames, temperature, agent tokens…)
  // without recording each event — one bus-summary action per flush batch.
  function hookEventBus() {
    try {
      // event-bus.js declares `const ClientEventBus` — a lexical global that
      // is NOT a window property, so reference the bare identifier.
      var bus =
        typeof ClientEventBus !== "undefined" ? ClientEventBus : window.ClientEventBus;
      if (!bus || typeof bus.emit !== "function" || bus.__gentlyReplayWrapped) return;
      var emit = bus.emit;
      bus.emit = function (type) {
        try {
          if (!STATE.disabled) {
            var c = STATE.busCounts || (STATE.busCounts = {});
            c[type] = (c[type] || 0) + 1;
          }
        } catch (e) {}
        return emit.apply(this, arguments);
      };
      bus.__gentlyReplayWrapped = true;
    } catch (e) {}
  }

  function onSubmit(e) {
    try {
      var f = e.target;
      logAction(
        f.getAttribute && f.getAttribute("data-action")
          ? f.getAttribute("data-action")
          : "submit",
        f,
        { form: f.id || f.name || undefined }
      );
    } catch (err) {}
  }

  function hookNavigation() {
    var emitRoute = function () {
      logAction("navigate", null, { to: String(location.pathname + location.hash) });
    };
    try {
      var push = history.pushState;
      history.pushState = function () {
        push.apply(this, arguments);
        emitRoute();
      };
      var replace = history.replaceState;
      history.replaceState = function () {
        replace.apply(this, arguments);
        emitRoute();
      };
    } catch (e) {}
    window.addEventListener("popstate", emitRoute);
    window.addEventListener("hashchange", emitRoute);
  }

  // A blockSelector that matches nothing does not error — it silently stops
  // blocking. The failure mode is severe and invisible: rename the camera <img>
  // and rrweb starts capturing a base64 data-URI `src` swap at frame rate on the
  // main thread. So check once at boot that every selector still binds.
  //
  // Self-calibrating, because this recorder also runs on launch/login/settings
  // where none of these elements exist: only report when SOME selector in the
  // list matched, which is what identifies the page as the one the list belongs
  // to. Blind spot: every selector rotting simultaneously reads as "wrong page"
  // and stays silent — they live in different subsystems, so that is far less
  // likely than the false positives a page sentinel would produce.
  function auditSelectors(selectorList) {
    var sels = String(selectorList)
      .split(",")
      .map(function (s) { return s.trim(); })
      .filter(Boolean);
    var missing = [];
    var hit = 0;
    sels.forEach(function (s) {
      var found = false;
      try {
        found = !!document.querySelector(s);
      } catch (e) {
        // An invalid selector never matches, so rrweb would not block it either.
        missing.push(s + " (invalid)");
        return;
      }
      if (found) hit++;
      else missing.push(s);
    });
    if (!hit || !missing.length) return null; // wrong page, or all good
    return missing;
  }

  /* ---- boot ---- */

  function start() {
    var fidelity = readFidelity();
    if (fidelity === "off") return disable("fidelity=off");
    STATE.fidelity = fidelity;

    // 'actions' records the semantic log only — no rrweb DOM stream (tiny).
    // 'full' keeps every mutation; 'balanced' additionally blocks the
    // machine-driven high-churn regions (the bulk of the volume).
    if (fidelity !== "actions") {
      if (!window.rrweb || typeof window.rrweb.record !== "function") {
        return disable("rrweb not loaded");
      }
      var block = fidelity === "full" ? BLOCK_SELECTOR : BLOCK_SELECTOR + ", " + HIGH_CHURN;
      try {
        STATE.stopFn = window.rrweb.record({
          emit: function (event) {
            pushEvent(STATE.rrwebBuf, event);
          },
          recordCanvas: false, // stored microscope pixels replay by reference (file store)
          blockSelector: block,
          checkoutEveryNms: CHECKOUT_MS,
          slimDOMOptions: true,
          // 120ms mousemove still gives a smooth pointer trail on replay; 50ms
          // measurably taxed the A/B's action latencies for no postmortem value.
          sampling: { mousemove: 120, scroll: 300, media: 800, input: "last" },
        });
      } catch (e) {
        return disable("rrweb.record failed: " + (e && e.message));
      }
      STATE.blockMisses = auditSelectors(block);
      if (STATE.blockMisses) {
        console.warn(
          "[gently-replay] blockSelector no longer matches — high-churn regions " +
            "may now be recorded at frame rate: " +
            STATE.blockMisses.join(", ")
        );
      }
    }
    document.addEventListener("click", onClick, true);
    document.addEventListener("submit", onSubmit, true);
    hookNavigation();
    hookEventBus();
    logAction("page-load", null, {
      url: String(location.href),
      viewport: window.innerWidth + "x" + window.innerHeight,
      fidelity: fidelity,
      // Present only when a block selector has rotted, so a postmortem explains
      // an unexpectedly huge or janky recording instead of leaving it a mystery.
      blockSelectorMisses: STATE.blockMisses || undefined,
    });
    window.addEventListener("pagehide", finalFlush);
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "hidden") finalFlush();
    });
    scheduleFlush();
  }

  // Start on DOMContentLoaded: the initial full-DOM snapshot is synchronous
  // and belongs inside page load, where it amortizes into load time. (An
  // idle-start variant was A/B-tested and rejected — the deferred snapshot
  // landed inside the user's first interactions instead, which is worse.)
  try {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", start);
    } else {
      start();
    }
  } catch (e) {
    disable("init failed: " + (e && e.message));
  }
})();
