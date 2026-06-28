import asyncio, logging
logger = logging.getLogger(__name__)

async def wait_for_temperature_lock(client, timeout_s: float, poll_s: float = 2.0) -> bool:
    """Poll the controller until it reports a locked state, or timeout. Substring 'LOCKED'."""
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    while True:
        try:
            resp = await client.get_temperature()
        except Exception as exc:
            logger.warning("wait_for_temperature_lock poll failed: %s", exc); resp = {}
        if "LOCKED" in str(resp.get("state", "")):
            return True
        if loop.time() - t0 >= timeout_s:
            return False
        await asyncio.sleep(poll_s)
