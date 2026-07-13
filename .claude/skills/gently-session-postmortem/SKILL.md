---
name: gently-session-postmortem
description: Use when analyzing how a user actually operated the gently UI — replaying a recorded session, walking the semantic action log, correlating UI actions with agent/device behavior at the same timestamps, or rendering what the screen showed at a specific moment.
---

# Gently session postmortem

Every gently session records its UI interactions (always-on, rrweb-based;
kill switch `GENTLY_REPLAY=0`). You are the intended reader: work text-first
through the action log, and render pixels only for the moments that matter.

## Where the artifacts live

Per session, under the storage root (`D:\Gently3` on the Windows production
machine; on Linux dev runs the default is the *literal* directory `D:/Gently3`
under the cwd the server launched from, unless `GENTLY_STORAGE_PATH` is set):

```
sessions/{folder}/ui-replay/
  actions.jsonl       # semantic action log — READ THIS FIRST
  rrweb-{tab}.jsonl   # full DOM event stream, one file per browser tab
  meta.yaml           # tabs seen, first_seen, user agents
ui-replay/unassigned-{YYYYMMDD}/   # batches that arrived with no active session
```

Session id → folder mapping: `sessions/_index.yaml`.

## The walk

1. **Read `actions.jsonl`** (one JSON object per line). Kinds you'll see:
   - `page-load` — with url + viewport; a new `tab` id per browser tab/reload.
   - `tab:embryos`, `view:board`, `bz:-10`, `button#op-detect`,
     `click:marking-action-btn:Done` — clicks, named from the UI's semantic
     `data-*` vocabulary, element ids, or class+text fallback. `target` carries
     tag/id/dataset/label.
   - `submit`, `navigate` — forms and SPA route/hash changes.
   - `bus-summary` — per-flush counts of ClientEventBus traffic
     (`BOTTOM_CAMERA_FRAME×240` means live frames were streaming; token
     streaming shows as high `AGENT_*` counts). This is how you know what the
     *system* was doing between clicks.
   - `gap` — the recorder dropped events under pressure (params say how many).
2. **Join with the rest of the file store on timestamps.** Everything else is
   already on disk: agent logs (`logs/gently_*.log`), perception traces and
   `predictions.jsonl` per embryo, `timelapse.yaml`, events. See the
   `gently-debugging` skill for that map.
   **Clock semantics:** `actions.jsonl` `t` is the *browser's* clock as UTC
   ISO; rrweb `timestamp` is epoch ms; server logs print server-local time.
   Convert everything to epoch before joining, and expect modest client/server
   skew (same machine ⇒ sub-second; remote browser ⇒ whatever their clock is).
3. **Render the moments that matter** (needs the repo venv with Playwright):

   ```bash
   python tools/session_replay/render_frame.py \
     --session 81865db3 --t 2026-07-13T07:05:38Z --out /tmp/frame.png \
     [--tab 04dce1bd] [--url http://localhost:8080] [--storage PATH]
   ```

   `--t` takes an ms offset, `52s`, `mm:ss`, or an absolute ISO timestamp
   (paste straight from an action's `t`). `--url` pointing at a *running*
   gently makes stored images inside the frame resolve; without it you get
   structure + inlined CSS only. Default tab is the largest stream.
4. **Human scrubbing**: `http://localhost:8080/replay` lists recordings;
   `/replay/{session_id}` plays one (tab picker, speed, click an action to
   seek to it).

## Caveats

- The live camera `<img id="op-cam-img">` is deliberately blocked from
  capture (base64 frame storms) — replays show a placeholder box there. What
  the camera saw lives in the file store (volumes/projections); whether it
  was streaming is in `bus-summary`.
- A session's recording can span multiple `rrweb-{tab}` files (reloads, second
  windows). `meta.yaml` + `page-load` actions give the timeline of tabs.
- The final ≤4s batch of a tab can be missing if the browser/app closed
  uncleanly (see spec: quit-flush race).
- Recording only exists for sessions run with `settings.ui.replay` on
  (default on; `GENTLY_REPLAY=0` disables).
