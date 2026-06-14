"""Tests for the logging->EventBus bridge."""

from __future__ import annotations

import logging

import pytest

from gently.core.event_bus import EventBus, EventType
from gently.core.log_bridge import (
    _NEVER_BRIDGE,
    LogToBusHandler,
    configure_log_bridge,
)


@pytest.fixture
def bus_with_capture():
    bus = EventBus()
    seen = []
    bus.subscribe(EventType.LOG_RECORD, lambda ev: seen.append(ev))
    return bus, seen


@pytest.fixture
def isolated_logger(request):
    """Fresh logger per test; cleaned up after.

    Using uniquely-named loggers under the gently.* namespace ensures we
    don't tangle with any handlers installed by other tests in the suite.
    """
    name = f"gently.test_logbridge.{request.node.name}"
    lgr = logging.getLogger(name)
    lgr.setLevel(logging.DEBUG)
    # Don't propagate to root; we don't want pytest's caplog to swallow it
    # before our handler sees it.
    original_propagate = lgr.propagate
    lgr.propagate = False
    yield lgr
    # Teardown: strip handlers we added
    for h in list(lgr.handlers):
        lgr.removeHandler(h)
    lgr.propagate = original_propagate


def test_handler_publishes_each_record(bus_with_capture, isolated_logger):
    """Every log call below threshold becomes a LOG_RECORD event."""
    bus, seen = bus_with_capture
    isolated_logger.addHandler(LogToBusHandler(bus, level=logging.DEBUG))

    isolated_logger.debug("debug msg")
    isolated_logger.info("info msg")
    isolated_logger.warning("warn msg")
    isolated_logger.error("error msg")

    assert len(seen) == 4
    levels = [ev.data["level_name"] for ev in seen]
    assert levels == ["DEBUG", "INFO", "WARNING", "ERROR"]
    msgs = [ev.data["message"] for ev in seen]
    assert msgs == ["debug msg", "info msg", "warn msg", "error msg"]


def test_handler_respects_level_threshold(bus_with_capture, isolated_logger):
    """Records below the handler level are dropped."""
    bus, seen = bus_with_capture
    isolated_logger.addHandler(LogToBusHandler(bus, level=logging.WARNING))

    isolated_logger.debug("nope")
    isolated_logger.info("also nope")
    isolated_logger.warning("yes")
    isolated_logger.error("yes")

    assert [ev.data["level_name"] for ev in seen] == ["WARNING", "ERROR"]


def test_exc_text_captured_on_exception(bus_with_capture, isolated_logger):
    """logger.exception() includes the formatted traceback in payload."""
    bus, seen = bus_with_capture
    isolated_logger.addHandler(LogToBusHandler(bus, level=logging.DEBUG))

    try:
        raise RuntimeError("simulated")
    except RuntimeError:
        isolated_logger.exception("blew up")

    assert len(seen) == 1
    payload = seen[0].data
    assert payload["level_name"] == "ERROR"
    assert payload["message"] == "blew up"
    assert payload.get("exc_text") and "RuntimeError" in payload["exc_text"]
    assert "simulated" in payload["exc_text"]


def test_reentry_guard_prevents_infinite_loop(bus_with_capture, isolated_logger):
    """A bus subscriber that itself logs must NOT spawn cascading events.

    Without the guard, every subscriber-emitted log would republish as
    another LOG_RECORD, spawn another subscriber call, ... ad infinitum.
    """
    bus, seen = bus_with_capture
    isolated_logger.addHandler(LogToBusHandler(bus, level=logging.DEBUG))

    # Subscriber that re-logs on every event it sees.
    def loud(ev):
        isolated_logger.debug("subscriber-internal log")

    bus.subscribe(EventType.LOG_RECORD, loud)

    isolated_logger.info("trigger")
    # Exactly one event — the original. The subscriber's log was
    # suppressed by the re-entry guard.
    assert len(seen) == 1
    assert seen[0].data["message"] == "trigger"


def test_handler_skips_bridge_internals(bus_with_capture):
    """Records from the bridge's own loggers must never be republished."""
    bus, seen = bus_with_capture
    h = LogToBusHandler(bus, level=logging.DEBUG)

    for blocked in _NEVER_BRIDGE:
        lgr = logging.getLogger(blocked)
        lgr.setLevel(logging.DEBUG)
        lgr.addHandler(h)
        try:
            lgr.info("from %s", blocked)
        finally:
            lgr.removeHandler(h)

    assert seen == []  # nothing republished


def test_configure_log_bridge_off_returns_none(bus_with_capture, monkeypatch):
    """GENTLY_LOG_BUS=off disables the bridge entirely."""
    bus, seen = bus_with_capture
    monkeypatch.setenv("GENTLY_LOG_BUS", "off")
    h = configure_log_bridge(bus=bus, loggers=["gently.cfg_off_test"])
    assert h is None
    logging.getLogger("gently.cfg_off_test").info("should not appear")
    assert seen == []


def test_configure_log_bridge_attaches_handler(bus_with_capture, monkeypatch):
    """Default-on path attaches a handler that publishes records."""
    bus, seen = bus_with_capture
    monkeypatch.setenv("GENTLY_LOG_BUS", "on")
    monkeypatch.setenv("GENTLY_LOG_BUS_LEVEL", "INFO")
    monkeypatch.delenv("GENTLY_LOG_BUS_INCLUDE_THIRDPARTY", raising=False)

    target_name = "gently.cfg_attach_test"
    target = logging.getLogger(target_name)
    target.setLevel(logging.DEBUG)  # otherwise inherits root WARNING

    h = configure_log_bridge(bus=bus, loggers=[target_name])
    assert h is not None
    try:
        target.info("hi")
        assert any(ev.data["message"] == "hi" for ev in seen)
    finally:
        target.removeHandler(h)


def test_configure_log_bridge_is_idempotent(bus_with_capture, monkeypatch):
    """Calling configure twice doesn't double-attach."""
    bus, seen = bus_with_capture
    monkeypatch.setenv("GENTLY_LOG_BUS", "on")

    target = "gently.idem_test"
    logging.getLogger(target).setLevel(logging.DEBUG)
    h1 = configure_log_bridge(bus=bus, loggers=[target])
    h2 = configure_log_bridge(bus=bus, loggers=[target])
    assert h1 is not None
    try:
        logging.getLogger(target).warning("once")
        # If both handlers were attached we'd see two events for the same
        # record. One event = no double-attach.
        warn_events = [ev for ev in seen if ev.data["message"] == "once"]
        assert len(warn_events) == 1
    finally:
        for h in (h1, h2):
            if h is not None:
                try:
                    logging.getLogger(target).removeHandler(h)
                except Exception:
                    pass


def test_third_party_excluded_by_default(bus_with_capture, monkeypatch):
    """uvicorn / aiohttp / bluesky etc. don't get bridged unless opted in."""
    bus, seen = bus_with_capture
    monkeypatch.setenv("GENTLY_LOG_BUS", "on")
    monkeypatch.delenv("GENTLY_LOG_BUS_INCLUDE_THIRDPARTY", raising=False)
    # Default behaviour: loggers list omits third-party. We pass None so
    # the function picks its own default.
    logging.getLogger("gently").setLevel(logging.DEBUG)
    h = configure_log_bridge(bus=bus, loggers=None)
    assert h is not None
    try:
        logging.getLogger("aiohttp.access").info("noisy 1")
        logging.getLogger("bluesky").info("noisy 2")
        assert seen == []
        # But gently.* still works
        logging.getLogger("gently.proves_attached").info("kept")
        assert any(ev.data["message"] == "kept" for ev in seen)
    finally:
        for n in ("gently", "gently_perception"):
            try:
                logging.getLogger(n).removeHandler(h)
            except Exception:
                pass


def test_third_party_included_when_opted_in(bus_with_capture, monkeypatch):
    """GENTLY_LOG_BUS_INCLUDE_THIRDPARTY=1 brings the noisy loggers in."""
    bus, seen = bus_with_capture
    monkeypatch.setenv("GENTLY_LOG_BUS", "on")
    monkeypatch.setenv("GENTLY_LOG_BUS_INCLUDE_THIRDPARTY", "1")
    logging.getLogger("aiohttp").setLevel(logging.DEBUG)
    logging.getLogger("aiohttp.access").setLevel(logging.DEBUG)

    h = configure_log_bridge(bus=bus, loggers=None)
    assert h is not None
    try:
        logging.getLogger("aiohttp.access").info("now visible")
        assert any(ev.data["message"] == "now visible" for ev in seen)
    finally:
        # Strip handler from every logger we might have attached to —
        # belt and braces, since the function defaults to a long list.
        for name in list(logging.Logger.manager.loggerDict.keys()) + [
            "gently",
            "gently_perception",
        ]:
            try:
                logging.getLogger(name).removeHandler(h)
            except Exception:
                pass
