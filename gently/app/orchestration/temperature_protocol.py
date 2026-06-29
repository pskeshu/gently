import asyncio
import logging

from gently.app.orchestration.exclusive import BurstAcquisition
from gently.core.event_bus import EventType

logger = logging.getLogger(__name__)


async def wait_for_temperature_lock(client, timeout_s: float, poll_s: float = 2.0) -> bool:
    """Poll the controller until it reports a locked state, or timeout. Substring 'LOCKED'."""
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    while True:
        try:
            resp = await client.get_temperature()
        except Exception as exc:
            logger.warning("wait_for_temperature_lock poll failed: %s", exc)
            resp = {}
        if "LOCKED" in str(resp.get("state", "")):
            return True
        if loop.time() - t0 >= timeout_s:
            return False
        await asyncio.sleep(poll_s)


async def run_temp_change_burst_protocol(
    orchestrator,
    embryo_id,
    target_setpoint_c,
    *,
    frames=60,
    mode="1hz",
    num_slices=1,
    bursts_before=1,
    bursts_after=1,
    lock_timeout_s=600.0,
    poll_s=2.0,
    burst_runner=None,
    tactic_id=None,
):
    """Scripted temp-change burst protocol: brightfield before/during/after a setpoint change."""
    client = orchestrator.client
    if burst_runner is None:

        async def burst_runner(b):
            await b.run(orchestrator)

    async def one_burst(phase):
        b = BurstAcquisition(
            embryo_id,
            frames=frames,
            mode=mode,
            num_slices=num_slices,
            temperature_provider=getattr(orchestrator, "_temperature_provider", None),
            laser_config="ALL OFF",
        )
        b._phase = phase
        await burst_runner(b)

    locked = False
    error = None
    cancelled = False
    try:
        await client.set_laser_config("ALL OFF")
        await client.set_led("Open")
        orchestrator._emit_event(
            EventType.TEMP_PROTOCOL_STARTED,
            {
                "embryo_id": embryo_id,
                "target_setpoint_c": target_setpoint_c,
                "frames": frames,
                "bursts_before": bursts_before,
                "bursts_after": bursts_after,
                "tactic_id": tactic_id,
            },
        )
        for _ in range(bursts_before):
            await one_burst("before")
        await client.set_temperature(target_setpoint_c)
        orchestrator._emit_event(
            EventType.TEMPERATURE_SETPOINT_CHANGED,
            {"embryo_id": embryo_id, "to": target_setpoint_c},
        )
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        while True:
            await one_burst("during")
            try:
                st = str((await client.get_temperature()).get("state", ""))
            except Exception:
                st = ""
            if "LOCKED" in st:
                locked = True
                break
            if loop.time() - t0 >= lock_timeout_s:
                break
        for _ in range(bursts_after):
            await one_burst("after")
    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:
        error = str(exc)
        logger.exception("temp-change burst protocol failed")
    finally:
        orchestrator._emit_event(
            EventType.TEMP_PROTOCOL_COMPLETED,
            {
                "embryo_id": embryo_id,
                "locked": locked,
                "cancelled": cancelled,
                "error": error,
                "tactic_id": tactic_id,
            },
        )
    return {"locked": locked, "cancelled": cancelled, "error": error}
