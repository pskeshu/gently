#!/usr/bin/env python3
"""
Temporary web GUI to play with the SwitchBot Bot that switches the diSPIM room
light (on for bottom-camera/brightfield imaging, off otherwise).

This is a TEST TOOL, not part of the production device layer. It drives the Bot
directly over BLE using the same command protocol as
``gently.hardware.switchbot.SwitchBot`` (same command bytes + GATT UUIDs), but
over a single *persistent* connection so the buttons feel snappy and the morse
blinker is fast — the device-layer class is connect-per-command (~1-2 s each),
which is fine for a plan step but hopeless for blinking.

Features: ON / OFF / PRESS buttons, and a morse-code blinker (blinks the real
room light + mirrors the pattern on screen). The Bot is a mechanical switch
pusher, so each toggle is a ~0.5-1 s servo move — morse is deliberately slow.

Run:
    .venv/bin/python diagnostics/switchbot_webgui.py
    # then open http://127.0.0.1:8765

    .venv/bin/python diagnostics/switchbot_webgui.py --address EC:6F:04:06:5B:23 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Reuse the device-layer device's protocol definitions (single source of truth).
from gently.hardware.switchbot import _COMMANDS, _CTRL_CHAR

logger = logging.getLogger("switchbot_webgui")

DEFAULT_ADDRESS = "EC:6F:04:06:5B:23"

# ITU morse, letters + digits. Unsupported characters are skipped.
MORSE = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
}


class Bot:
    """A single persistent BLE connection to the Bot, with serialized access."""

    def __init__(self, address: str):
        self.address = address
        self._client: Any = None
        self._lock = asyncio.Lock()
        self._morse_task: asyncio.Task | None = None
        self.state = "unknown"
        self.busy = False

    async def _ensure(self):
        from bleak import BleakClient

        if self._client is not None and self._client.is_connected:
            return
        self._client = BleakClient(self.address, timeout=20)
        await self._client.connect()
        logger.info("connected to %s", self.address)

    async def _write(self, action: str):
        """Write one command, reconnecting once if the link dropped."""
        from bleak.exc import BleakError

        for attempt in (1, 2):
            try:
                await self._ensure()
                await self._client.write_gatt_char(_CTRL_CHAR, _COMMANDS[action], response=True)
                if action in ("on", "off"):
                    self.state = action
                return
            except (BleakError, OSError, asyncio.TimeoutError) as exc:
                logger.warning("write %s attempt %d failed: %s", action, attempt, exc)
                self._client = None  # force reconnect
                if attempt == 2:
                    raise

    async def _cancel_morse(self):
        task = self._morse_task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._morse_task = None

    async def command(self, action: str) -> str:
        """ON/OFF/PRESS. Interrupts any running morse (manual override)."""
        await self._cancel_morse()
        async with self._lock:
            await self._write(action)
        return self.state

    def schedule(self, text: str, unit: float):
        """Build an on/off timeline [(state, seconds), ...] for a message."""
        seq = [("off", round(unit, 3))]  # settle to a known state first
        for ch in text.upper():
            if ch == " ":
                seq.append(("off", round(unit * 7, 3)))
                continue
            code = MORSE.get(ch)
            if not code:
                continue
            for sym in code:
                seq.append(("on", round(unit * (3 if sym == "-" else 1), 3)))
                seq.append(("off", round(unit, 3)))  # intra-letter gap
            st, _ = seq[-1]
            seq[-1] = (st, round(unit * 3, 3))  # upgrade to inter-letter gap
        return seq

    async def start_morse(self, text: str, unit: float):
        await self._cancel_morse()
        seq = self.schedule(text, unit)
        if len(seq) <= 1:
            return None
        restore = self.state
        self._morse_task = asyncio.create_task(self._play(seq, restore))
        return seq

    async def _play(self, seq, restore: str):
        async with self._lock:
            self.busy = True
            try:
                for state, dur in seq:
                    await self._write(state)
                    await asyncio.sleep(dur)
                await self._write(restore if restore in ("on", "off") else "off")
            finally:
                self.busy = False

    async def stop(self):
        await self._cancel_morse()
        async with self._lock:
            await self._write("off")
        return self.state

    async def disconnect(self):
        await self._cancel_morse()
        if self._client is not None and self._client.is_connected:
            await self._client.disconnect()


BOT: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if BOT is not None:
        await BOT.disconnect()


app = FastAPI(lifespan=lifespan)


class MorseReq(BaseModel):
    text: str
    unit: float = 1.5


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE.replace("__ADDRESS__", BOT.address)


@app.get("/status")
async def status():
    return {"state": BOT.state, "busy": BOT.busy, "address": BOT.address}


@app.post("/cmd/{action}")
async def cmd(action: str):
    if action not in _COMMANDS:
        return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)
    try:
        state = await BOT.command(action)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    return {"state": state}


@app.post("/morse")
async def morse(req: MorseReq):
    unit = max(0.3, min(4.0, req.unit))
    text = req.text[:40]
    try:
        seq = await BOT.start_morse(text, unit)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    if seq is None:
        return JSONResponse({"error": "nothing sendable in that text"}, status_code=400)
    seconds = round(sum(d for _, d in seq), 1)
    return {"schedule": seq, "unit": unit, "seconds": seconds}


@app.post("/stop")
async def stop():
    try:
        state = await BOT.stop()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    return {"state": state}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>diSPIM Room Light</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif;
         background:#0d1117; color:#e6edf3; display:flex; min-height:100vh;
         align-items:center; justify-content:center; }
  .card { width:min(440px,92vw); padding:28px; background:#161b22;
          border:1px solid #30363d; border-radius:16px; text-align:center; }
  h1 { margin:0 0 2px; font-size:20px; }
  .sub { color:#8b949e; font-size:12px; margin-bottom:18px; font-variant-numeric:tabular-nums; }
  .bulb { width:96px; height:96px; border-radius:50%; margin:6px auto 4px;
          background:#21262d; border:1px solid #30363d; transition:.12s;
          box-shadow:inset 0 0 12px #0008; }
  .bulb.on { background:#ffd24a; border-color:#ffe08a;
             box-shadow:0 0 36px 8px #ffd24a99, inset 0 0 12px #fff6; }
  .state { font-size:13px; color:#8b949e; height:18px; margin-bottom:14px;
           text-transform:uppercase; letter-spacing:.08em; }
  .row { display:flex; gap:8px; justify-content:center; margin-bottom:16px; }
  button { flex:1; padding:12px 0; font-size:14px; font-weight:600; cursor:pointer;
           border-radius:10px; border:1px solid #30363d; background:#21262d; color:#e6edf3; }
  button:hover { border-color:#58a6ff; }
  button:active { transform:translateY(1px); }
  button.on  { background:#1f6feb33; border-color:#1f6feb; }
  button.off { background:#30363d; }
  button.stop{ background:#f8514933; border-color:#f85149; flex:0 0 84px; }
  .morse { border-top:1px solid #30363d; padding-top:16px; }
  input[type=text]{ width:100%; box-sizing:border-box; padding:10px; margin-bottom:10px;
           background:#0d1117; color:#e6edf3; border:1px solid #30363d; border-radius:8px;
           font-size:15px; text-transform:uppercase; letter-spacing:.1em; }
  .speed { display:flex; align-items:center; gap:8px; font-size:12px;
           color:#8b949e; margin-bottom:12px; }
  .speed input { flex:1; }
  .mrow { display:flex; gap:8px; }
  .status { margin-top:14px; font-size:12px; color:#8b949e; min-height:16px; }
</style></head><body>
<div class="card">
  <h1>diSPIM Room Light</h1>
  <div class="sub">SwitchBot Bot &middot; __ADDRESS__</div>
  <div id="bulb" class="bulb"></div>
  <div id="state" class="state">&mdash;</div>
  <div class="row">
    <button class="on"  onclick="cmd('on')">ON</button>
    <button class="off" onclick="cmd('off')">OFF</button>
    <button onclick="cmd('press')">PRESS</button>
  </div>
  <div class="morse">
    <input id="msg" type="text" maxlength="40" value="SOS" placeholder="message">
    <div class="speed">
      <span>fast</span>
      <input id="speed" type="range" min="0.5" max="3" step="0.1" value="1.5" oninput="sv()">
      <span>slow</span>
      <span id="speedval" style="width:34px">0.7s</span>
    </div>
    <div class="mrow">
      <button onclick="sendMorse()">Send Morse</button>
      <button class="stop" onclick="stop()">Stop</button>
    </div>
  </div>
  <div id="status" class="status"></div>
</div>
<script>
const $ = id => document.getElementById(id);
let timers = [];
function setBulb(on){ $('bulb').classList.toggle('on', on); }
function setState(s){ $('state').textContent = s;
  if(s==='on')setBulb(true); else if(s==='off')setBulb(false); }
function clearTimers(){ timers.forEach(clearTimeout); timers = []; }
function sv(){ $('speedval').textContent = (+$('speed').value).toFixed(1)+'s'; }

async function cmd(a){
  clearTimers();
  $('status').textContent = '…';
  try {
    const r = await fetch('/cmd/'+a, {method:'POST'});
    const j = await r.json();
    if(j.error){ $('status').textContent = '⚠ '+j.error; return; }
    if(a==='press'){ setBulb(true); setTimeout(()=>setBulb(false), 350); }
    else setState(j.state);
    $('status').textContent = 'ok';
  } catch(e){ $('status').textContent = '⚠ '+e; }
}

async function sendMorse(){
  clearTimers();
  const text = $('msg').value, unit = +$('speed').value;
  $('status').textContent = 'sending…';
  try {
    const r = await fetch('/morse', {method:'POST', headers:{'content-type':'application/json'},
                                     body: JSON.stringify({text, unit})});
    const j = await r.json();
    if(j.error){ $('status').textContent = '⚠ '+j.error; return; }
    $('status').textContent = 'blinking "'+text+'" — '+j.seconds+'s';
    let t = 0;
    for(const [st,dur] of j.schedule){
      timers.push(setTimeout(()=>setBulb(st==='on'), t*1000));
      t += dur;
    }
    timers.push(setTimeout(()=>{ $('status').textContent='done'; }, t*1000));
  } catch(e){ $('status').textContent = '⚠ '+e; }
}

async function stop(){
  clearTimers();
  await fetch('/stop', {method:'POST'});
  $('status').textContent = 'stopped';
  refresh();
}
async function refresh(){
  try { const j = await (await fetch('/status')).json(); setState(j.state); } catch(e){}
}
sv(); refresh();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Temporary SwitchBot room-light web GUI")
    ap.add_argument("--address", default=DEFAULT_ADDRESS, help="Bot BLE MAC address")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="bind host (default: localhost only)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    global BOT
    BOT = Bot(args.address)
    print(f"\n  diSPIM Room Light GUI  →  http://{args.host}:{args.port}\n  Bot: {args.address}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
