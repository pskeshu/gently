# Reviewing a brightfield time series in the Embryos tab — interaction study

Date: 2026-10-06. Companion prototype: `2026-10-06-embryos-timeline-prototype.html` (same folder; open it in a browser from the repo so the `main.css` link resolves).

## What exists today

- `gently/ui/web/static/js/embryos.js` — `renderDicStrip` shows the **last 12** thumbnails (`?max=256`) above the cards, scrolled to the right edge; a click opens `openDicViewer`, a modal with one full frame, prev/next buttons and `←`/`→`/`Esc`. The film view (`_filmDicRow`) puts the overview as the first row of the filmstrip, one cell per frame, honouring `skipInterval`. `renderFilmstripView` auto-follows the right edge if the user was within 24 px of it — an implicit "follow newest" that already exists.
- `gently/ui/web/static/js/review.js` — `wireScrub`: `mousemove` across one thumbnail maps x to timepoint index and swaps the `<img src>`; `mouseleave` snaps back to the latest. Caption `t{n} · i/N`. No drag, no keys, no play. This is the gesture the user liked.
- `stage-pad.js` — arrow keys act only while the pad has focus; `Shift` = ×10. Buttons named by what they do, not by screen direction.
- `gently/app/brightfield.py` — dark and flat per session, keyed on (light, LED %, exposure); every frame names the record that applies; `corrected = (frame - dark) / (flat - dark) * mean(flat - dark)`.

Data shape: 50–200 frames, 2048² 16-bit, 5–10 min apart over 8–16 h, served as PNG (`?max=N` for thumbnails), live arrival over the websocket with a base64 thumbnail.

## Survey: what each tool does, and what to borrow

| Tool | Pattern (gesture) | Fit for a wall display, mouse + occasional touch |
|---|---|---|
| **Fiji/ImageJ** hyperstack | One slider per dimension; `←`/`→` (or `<`/`>`) step; `\` starts/stops animation; `Alt+\` opens Animation Options (fps, first/last frame, **Loop Back and Forth** so the series does not jump from last to first); `Space`+drag pans; `+`/`-` zoom, `5` = 100%. | Step keys and ping-pong loop are worth copying. `\` is unguessable; use Space. |
| **napari** dims slider | Play button at the slider's left end; **right-click** it for fps (default 10), direction, loop mode (once / loop / back-and-forth). Current index shown beside the slider. | Play affordance on the bar itself is good. Right-click menus are hostile to touch — expose fps/loop as visible buttons. |
| **Micro-Manager** viewer | Per-axis sliders with an "animate this axis" toggle on each axis label and a single FPS field; handles new time points arriving while animating. | The "new frames keep arriving while you look" case is exactly ours. |
| **OMERO.iviewer** | Z/T sliders; playback once advanced 4 steps/s, now waits for each plane to render before advancing (0.14.0). Several images open at once with **sync Z/T/zoom** across them. | "Advance when the frame is ready, not on a timer" is the right rule for 2048² PNGs over a LAN. Sync'd side-by-side is the model for comparison. |
| **OMERO.figure** | Each panel is a mini viewer with its own T index; labels from metadata (timestamp, elapsed) are **dynamic**, bound to the current T. | Per-frame label bound to metadata, not typed by hand. |
| **Incucyte** VesselView | Whole-vessel bird's-eye; scroll through scan times; slide-show with a transition-time setting; overlay metrics on the image. | Our one frame *is* the bird's-eye of the dish; "slide-show with interval" = play with fps. Metrics overlaid on the frame (per-embryo stage call) is the long-term direction. |
| **BioTek Gen5** | Kinetic image review via movie maker with **automated image alignment** before playback. | Alignment is overkill for a fixed stage; the drift we want to *see*, not hide. Not borrowed. |
| **Zeiss ZEN** (Time Series, ZEN Connect) | Time series defined by interval/duration/count; Connect is a sample-centric overlay of multiscale images with metadata. | Nothing timeline-specific worth copying; the metadata-first stance supports showing exposure/light/refs per frame. |
| **Frame.io** | `Space`/`K` play-pause; `J`/`L` shuttle at 2×/4×/8× (press again to go faster); `←`/`→` one frame; `Shift+←/→` ten frames; `Ctrl+L` loop; `T` fit, `+`/`-` zoom, Cmd+scroll fluid zoom; hover a thumbnail to scrub it. Speed indicator flashes on the video when J/K/L is used. | Arrow + Shift+arrow is the pattern. Shuttle (J/L) is editor muscle memory; a lab user will not know it — skip, keep fps buttons. |
| **YouTube** | Hover the progress bar → **storyboard thumbnail above the pointer** (one sprite sheet, cropped per position; no per-frame requests). `,`/`.` step a frame while paused; `Home`/`End`; `0–9` jump to 10 % marks; `<`/`>` speed. | Hover-preview-above-the-bar is the central borrow. Sprite sheet is how to make 200 previews cheap — see Performance. |
| **Observable / d3 brush** | Drag on an overview to select a range; drag the selection to move it; handles to resize; `Space` locks size while dragging; `Alt` grows from centre. Focus+context: brush the overview, a detail view zooms to it. | A range brush is the right tool if the user ever needs "play only 14:00–16:00". Not for v1; one scrub cursor is enough for 200 frames. |
| **Apple Photos Live Photo** | Edit → a filmstrip slider of all frames; drag the handle to scrub; release and "Make Key Photo". Press-and-hold the photo plays it. | Press-and-hold-to-play on the big image is a lovely touch gesture and cheap: `pointerdown` on the image starts play, `pointerup` stops. Worth adding for the wall display. |
| **Google Earth Timelapse** | Year slider along the bottom; play/pause FAB kept next to the timeline; speed control at the slider's right end; map gets all the real estate, controls stay visible. | Layout model: image first, one bar, controls hugging the bar. Matches a wall display read from two metres. |

## Recommendations

### 1. Timeline control

**Both, but not equal:** a dense frame-indexed bar is the primary control; thumbnails appear only as the hover preview above it (YouTube) and as the existing filmstrip row for people who want to see the whole series at once. A strip of 200 thumbnails at a legible size is 200 × 64 px = 12 800 px — eleven screens of horizontal scrolling. Today's "last 12" strip hides 94 % of a long run; the bar hides none.

200 frames in 1200 px is 6 px per frame: large enough to hit with a mouse (the slot, not a tick), too small for a fingertip, so the bar is **44 px tall** and scrubbing is a continuous drag — a finger does not need to land on a slot, it slides to it. Draw per-frame ticks only when the slot is ≥ 4 px (they vanish for very long runs; nothing else changes).

Marks, by colour from `main.css`: current frame `--accent` (3 px, full height); newest `--accent-green` at the right end, always; pointer `--text-muted`; warnings `--accent-amber` as a short foot under the slot, so several adjacent warnings read as a band; a pause in the run `--text-muted` as a 2 px seam between the frames it separates. No labels on the bar — the hover preview carries `n · HH:MM`.

**Frame-indexed, not time-proportional.** Reasons: (a) the series is a *sequence of looks*, each one a round of the run; the user steps "one back", not "six minutes back"; (b) a time-proportional bar makes a 2-hour pause a dead 1/8 of the width where there is nothing to scrub to, and compresses the live run's frames when the pause was long; (c) frame-indexing lets the bar grow by a constant slot per arriving frame, so the live edge is stable. Mark the pause as a seam and say "paused 40 m before this frame" in the metadata row — the information survives, the control stays uniform. If a time axis is ever wanted, draw it as a thin ruler *under* the bar with HH labels, warped to frame index.

### 2. Scrub, play, step

- **Pointer**: hover shows the preview and a pointer mark without moving the main frame (so a wandering mouse on the wall display does not change what everyone is looking at); **press-and-drag scrubs** the main frame, with pointer capture so the drag can leave the bar. This differs from `review.js wireScrub`, where hover alone changes the image — right for a card-sized thumbnail, wrong for the one big frame a room is watching. `touch-action: none` on the bar so a finger drag is not eaten by page scroll.
- **Keys**: `←`/`→` one frame, `Shift` ×10 (Frame.io; matches `stage-pad.js`'s Shift = ×10), `,`/`.` as aliases (YouTube), `Home`/`End`, `Space` play/pause, `N` newest, `C` corrected. Document keys on-screen in one muted line, as the stage pad does. Arrow keys should act whenever the Embryos tab is showing and no input is focused — not only when the modal is open as today.
- **Play**: fps buttons 2/5/10/20 (napari default 10 is right for 200 frames: 20 s per pass); loop off by default; offer ImageJ's back-and-forth as the loop mode rather than wrap — the drift between last and first frame is exactly the jump it exists to avoid. Advance **on image load, not on a timer** (OMERO's lesson): the interval is a *minimum*, and a frame that has not arrived delays the next tick rather than being skipped. A fast path for scrubbing is the `?max=512` image; on pause and on arrival at a frame, swap in the full PNG (see 6).
- **Follow newest**: on by default and whenever the tab opens; any user navigation (drag, key, play) disengages it; the `Newest` button (and `N`) re-engages it. Show the state as a small `● FOLLOWING NEWEST` tag in the image corner so a glance from across the room says whether the picture is live. This replaces the 24 px right-edge heuristic in `renderFilmstripView` with an explicit state, which also stops the "I scrolled slightly and lost the live feed" failure.
- **Touch**: press-and-hold on the big frame plays while held (Live Photo); release stops and stays on that frame. Zero UI cost.

### 3. Comparison

Build **A/B swipe** on the single large frame, not side-by-side. The question being asked is "has anything moved / grown since X" — a swipe divider on the same pixels shows a 20 px drift as a visible step at the divider; two images side by side force the eye to saccade and compare positions from memory, which fails for small drift. Side-by-side is only better when the frames are *not* registered (different positions), which is not our case: one camera, one dish, fixed stage.

Two presets cover the real use: **first vs now** (development and drift) and **raw vs corrected** (did the flat help, is dust on the sensor or in the dish). Implementation is one extra `<img>` clipped by `clip-path: inset(0 0 0 X%)` and a draggable divider; the raw/corrected variant needs a `?corrected=1` query on the frame endpoint (the formula in `brightfield.py` is a few numpy lines; serve it, do not reimplement in JS). An "A" frame is picked by `Shift+click` on the bar or a "pin as A" button; default A is frame 1.

### 4. Image handling

- **Zoom/pan**: wheel zooms about the pointer, drag pans, double-click toggles fit / 1:1, `0` resets (Frame.io `T`/`Cmd+0`, ImageJ `4`/`5`). Pure CSS `transform: translate() scale()` on the image element; ~40 lines. Zoom persists across frames and across the A/B pair so a region can be watched through time — this is the whole reason to have it.
- **1:1** matters because the full frame is 2048² shown at ~900 px: a 2.3× downsample hides the detail a user zooms for. On zoom > 1, load the full PNG if the `?max` one is showing.
- **Difference / motion view**: *worth it, as a toggle, cheap version only*. Draw current and previous into a canvas, `globalCompositeOperation = 'difference'`, boost contrast. It makes a hatching embryo or a stage slip pop out of a grey field, which is what a wall display is for. Skip anything registered, temporal-filtered or coloured-by-direction. ~25 lines; drop it if the canvas path for 2048² proves slow on the rig's PC (measure first — a 2048² `drawImage` + difference is ~10 ms on anything from the last decade).

### 5. What to show about each frame

One muted row under the image, tabular numerals, in this order: `frame n/N` · clock time · `elapsed Hh MMm` since frame 1 · round · stage x, y µm · exposure ms · light (room / LED %) · `raw` or `corrected · dark 20ms, flat led-35pct` naming the reference records (the yaml stem in `brightfield.py`) · warnings in `--accent-amber` ("paused 40 m before this frame", "light was on at exposure", "stage off target by 310 µm"). Elapsed is the number a developmental biologist reads; clock time is the one a technician reads; show both. Everything comes from frame metadata the backend already writes; nothing is computed in the UI except elapsed and the pause gap.

### 6. Performance

- **Three sizes**: 128 px for the hover preview (ideally one sprite sheet per session, `/api/dic/sprite.png`, regenerated when a frame arrives — 200 × 128² RGBA is 13 MB raw, ~1.5 MB PNG, one request; YouTube's trick), 512 px for scrubbing and playback, full PNG only when paused or zoomed. The websocket's base64 thumb seeds the small cache for the newest frame.
- **Preload neighbours**: on settle at frame *i*, prefetch 512 px for *i±1..3* and full for *i*. During play, prefetch *i+1..+fps/2*.
- **Cancel in-flight**: `fetch` with an `AbortController` per slot, abort when the scrub passes it; or simpler, a single `Image` for the main frame and only commit `src` on `requestAnimationFrame` so 100 pointer events per second produce ~60 loads at most, and only the latest is awaited (compare a request id before painting — the `img.dataset.t` guard in `wireScrub` is the right idea).
- **Memory cap**: `Map` as an LRU, ~150 × 512 px (≈ 150 MB decoded) and ~6 full frames (≈ 100 MB). Evict the farthest from the current index first, never the newest. Decoded images, not bytes, are what cost memory; cap counts, not kilobytes.
- Draw the bar on one `<canvas>`, not 200 DOM nodes; redraw on state change only.

## Not recommended now

Range brush (d3), J/K/L shuttle, image registration before playback, a time-proportional axis, per-frame DOM thumbnails on the bar. Each has a case; none has one yet.

## Prototype

`2026-10-06-embryos-timeline-prototype.html` shows: large frame; frame-indexed bar with hover preview (`n · HH:MM`), pointer-drag scrub, warning feet, pause seam, newest/current/pointer marks; `←`/`→`/`Shift`/`Home`/`End`/`,`/`.`; `Space` play/pause with fps and loop; `Newest` button re-engaging follow; a fake raw/corrected toggle (removes the vignette and sensor dust); a fake frame arriving every 4 s. Not in the prototype: A/B swipe, zoom/pan, difference view, image loading (frames are procedural canvases).

## Sources

- ImageJ shortcuts and Animation Options: https://imagej.net/ij/docs/shortcuts.html , https://imagej.net/ij/docs/menus/image.html , https://forum.image.sc/t/option-to-play-a-stack-through-only-once-no-looping/32871
- napari viewer and preferences: https://napari.org/stable/tutorials/fundamentals/viewer.html , https://napari.org/stable/guides/preferences.html
- Micro-Manager 2.0 user guide: https://micro-manager.org/Version_2.0_Users_Guide , https://micro-manager.org/apidoc/mmstudio/2.0.0/org/micromanager/display/DataViewer.html
- OMERO.iviewer: https://omero-guides.readthedocs.io/en/latest/iviewer/docs/iviewer_viewing.html , https://forum.image.sc/t/omero-iviewer-step-size-for-t-z-auto-play/103449
- OMERO.figure: https://omero-guides.readthedocs.io/en/latest/figure/docs/omero_figure.html
- Incucyte manual: https://www.sartorius.com/download/1087348/incucyte-live-cell-analysis-systems-user-manual-en-l-8000-04-1--data.pdf
- BioTek Gen5: https://cqls.oregonstate.edu/sites/cqls.oregonstate.edu/files/5321045_rev_r_gen5_gettingstartedguide.pdf
- Zeiss ZEN Time Series / ZEN Connect: https://knowledge.zeiss.com/rms/en/zen-core/toolkits-modules/acquisition-toolkits/time-series , https://asset-downloads.zeiss.com/catalogs/download/mic/f8f5f37b-85dc-45d3-bcd0-c60517774a9b/EN_product-information_ZEN_Connect.pdf
- Frame.io shortcuts and player: https://help.frame.io/en/articles/9105337-keyboard-shortcuts , https://help.frame.io/en/articles/9105311-player-page-features
- YouTube shortcuts and storyboard previews: https://support.google.com/youtube/answer/7631406 , https://dev.to/masonwritescode/build-scrub-bar-thumbnail-previews-with-ffmpeg-and-a-webvtt-sprite-3ei2
- d3-brush: https://d3js.org/d3-brush
- Apple Live Photos: https://support.apple.com/en-ie/HT207310
- Google Earth Timelapse redesign: https://medium.com/google-design/redesigning-google-earth-timelapse-135a963cc35
