"""Dark and flat-field references for brightfield (bottom-camera) imaging.

A brightfield frame is only as good as what is divided out of it. Two
references, taken once per session, at the exposure the run will use:

- the **dark**: the camera with nothing lit. The room light has no read-back
  (a SwitchBot button pusher: its "state" is the last command sent, and the
  GUI has shown "off" with the light on), so the dark CYCLES it — on, settle,
  off, settle — so that the state is known, closes the LED, and only then
  snaps. No other source is touched: a brightfield run drives no laser.
- the **flat**: the empty field under the run's light, several frames
  averaged. The operator drives the stage clear of the embryos first (the
  map may be stale, so this is theirs to do); the panel asks them to.

Both land in ``<session>/calibration/brightfield/<stamp>/`` with a
``brightfield.yaml`` saying when, under what light and exposure, how many
frames, and what the images measured — and every overview frame that is
taken afterwards names the record that applies to it, so whoever analyses
the frames downstream need not guess. Correction, for the record:

    corrected = (frame - dark) / (flat - dark) * mean(flat - dark)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

FOLDER = ("calibration", "brightfield")
RECORD = "brightfield.yaml"


@dataclass(frozen=True)
class ReferenceSpec:
    """What a reference is good for: the light and exposure of the frames."""

    light: str = "room"  # room | led | none
    led_intensity_pct: int | None = None
    exposure_ms: float | None = None

    def key(self) -> tuple:
        pct = (
            int(self.led_intensity_pct) if self.light == "led" and self.led_intensity_pct else None
        )
        return (self.light, pct, _exp(self.exposure_ms))

    def flat_name(self) -> str:
        light = self.light + (
            f"-{int(self.led_intensity_pct)}pct"
            if self.light == "led" and self.led_intensity_pct
            else ""
        )
        return f"flat_{light}_{_exp_name(self.exposure_ms)}.tif"

    def dark_name(self) -> str:
        return f"dark_{_exp_name(self.exposure_ms)}.tif"


def _exp(ms: float | None) -> float | None:
    return None if ms is None else round(float(ms), 3)


def _exp_name(ms: float | None) -> str:
    if ms is None:
        return "cameraexposure"
    v = float(ms)
    return f"{v:g}ms".replace(".", "p")


def _real(image: Any) -> bool:
    """The client's empty-capture placeholder is a 100x100 of zeros."""
    return image is not None and getattr(image, "ndim", 0) == 2 and tuple(image.shape) != (100, 100)


def stats(image: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(image)
    full = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else None
    out: dict[str, Any] = {
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p99_9": round(float(np.percentile(arr, 99.9)), 1),
        "shape": [int(s) for s in arr.shape],
        "dtype": str(arr.dtype),
    }
    if full:
        out["saturated_fraction"] = round(float((arr >= full).mean()), 6)
    return out


# ── taking them ─────────────────────────────────────────────────────────────


async def take_dark(
    client: Any, exposure_ms: float | None, *, settle_s: float = 1.0, cycle: bool = True
) -> dict[str, Any]:
    """The camera with nothing lit. Cycles the room light so its state is
    known, closes the LED, snaps. Returns {image, stats, steps}."""
    steps: list[str] = []

    async def say(step: str) -> None:
        steps.append(step)
        logger.info("dark reference: %s", step)

    res = await client.set_led("Closed")
    if isinstance(res, dict) and res.get("success") is False:
        raise RuntimeError(f"the LED would not close: {res.get('error') or res}")
    await say("LED closed")
    if cycle:
        res = await client.set_room_light("on")
        if isinstance(res, dict) and res.get("success") is False:
            raise RuntimeError(f"the room light did not answer: {res.get('error') or res}")
        await say("room light commanded on")
        await asyncio.sleep(settle_s)
    res = await client.set_room_light("off")
    if isinstance(res, dict) and res.get("success") is False:
        raise RuntimeError(f"the room light did not answer: {res.get('error') or res}")
    await say("room light commanded off")
    await asyncio.sleep(settle_s)
    await say(f"settled {settle_s:g} s")
    result = await client.capture_bottom_image(use_led=False, exposure_ms=exposure_ms)
    image = (result or {}).get("image")
    if not _real(image):
        raise RuntimeError("the camera returned no frame")
    await say("frame captured")
    arr = np.asarray(image)
    return {"image": arr, "stats": stats(arr), "steps": steps}


async def take_flat(
    client: Any, spec: ReferenceSpec, *, frames: int = 5, settle_s: float = 1.0
) -> dict[str, Any]:
    """The empty field under the run's light: ``frames`` frames, averaged.
    The light this call lit, it puts out. Returns {image, stats, steps,
    frames}. The operator has already driven the stage clear."""
    steps: list[str] = []
    frames = max(1, int(frames))
    lit: str | None = None
    try:
        if spec.light == "room":
            res = await client.set_room_light("on")
            if isinstance(res, dict) and res.get("success") is False:
                raise RuntimeError(f"the room light did not come on: {res.get('error') or res}")
            lit = "room"
            steps.append("room light commanded on")
            await asyncio.sleep(settle_s)
        elif spec.light == "led":
            if spec.led_intensity_pct is not None:
                res = await client.set_led_intensity(int(spec.led_intensity_pct))
                if isinstance(res, dict) and res.get("success") is False:
                    raise RuntimeError(
                        f"the LED would not take {spec.led_intensity_pct}%: "
                        f"{res.get('error') or res}"
                    )
                steps.append(f"LED set to {int(spec.led_intensity_pct)}%")
            res = await client.set_led("Open")
            if isinstance(res, dict) and res.get("success") is False:
                raise RuntimeError(f"the LED would not open: {res.get('error') or res}")
            lit = "led"
            steps.append("LED opened")
            await asyncio.sleep(settle_s)
        else:
            steps.append("light left as it is")
        acc: np.ndarray | None = None
        dtype = None
        for i in range(frames):
            result = await client.capture_bottom_image(use_led=False, exposure_ms=spec.exposure_ms)
            image = (result or {}).get("image")
            if not _real(image):
                raise RuntimeError(f"the camera returned no frame ({i + 1} of {frames})")
            arr = np.asarray(image)
            dtype = arr.dtype
            acc = arr.astype(np.float64) if acc is None else acc + arr
        steps.append(f"{frames} frame{'s' if frames != 1 else ''} captured")
    finally:
        if lit == "room":
            try:
                await client.set_room_light("off")
                steps.append("room light commanded off")
            except Exception as exc:
                logger.warning("flat reference: the room light did not go off: %s", exc)
        elif lit == "led":
            try:
                await client.set_led("Closed")
                steps.append("LED closed")
            except Exception as exc:
                logger.warning("flat reference: the LED did not close: %s", exc)
    assert acc is not None and dtype is not None
    mean = acc / frames
    if np.issubdtype(dtype, np.integer):
        image_out = np.clip(np.rint(mean), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    else:
        image_out = mean.astype(dtype)
    return {"image": image_out, "stats": stats(image_out), "steps": steps, "frames": frames}


# ── filing them ─────────────────────────────────────────────────────────────


def references_dir(store: Any, session_id: str) -> Path | None:
    sd = store._session_dir(session_id)
    return None if sd is None else Path(sd).joinpath(*FOLDER)


def open_record(store: Any, session_id: str, spec: ReferenceSpec) -> Path:
    """A new folder for one set of references, named for when it began."""
    base = references_dir(store, session_id)
    if base is None:
        raise FileNotFoundError(f"Session not found: {session_id}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = base / stamp
    n = 1
    while folder.exists():
        n += 1
        folder = base / f"{stamp}_{n}"
    folder.mkdir(parents=True)
    _write(
        folder,
        {
            "record": folder.name,
            "session_id": session_id,
            "spec": asdict(spec),
            "started_at": _now(),
        },
    )
    return folder


def file_image(
    folder: Path, kind: str, image: np.ndarray, spec: ReferenceSpec, taken: dict[str, Any]
) -> dict:
    """Write the dark or the flat into ``folder`` and say so in its record.
    Returns the record as written."""
    import tifffile

    name = spec.dark_name() if kind == "dark" else spec.flat_name()
    tifffile.imwrite(str(folder / name), image)
    doc = read_record(folder) or {"record": folder.name}
    doc.setdefault("spec", asdict(spec))
    entry = {
        "file": name,
        "taken_at": _now(),
        "stats": taken.get("stats"),
        "steps": taken.get("steps"),
    }
    if kind == "flat":
        entry["frames_averaged"] = taken.get("frames")
    doc[kind] = entry
    doc["checks"] = checks(doc)
    _write(folder, doc)
    return doc


def checks(doc: dict) -> dict[str, Any]:
    """What the two images say about each other. A dark that is as bright
    as the flat is not a dark: the light did not go off."""
    out: dict[str, Any] = {}
    dark = (doc.get("dark") or {}).get("stats") or {}
    flat = (doc.get("flat") or {}).get("stats") or {}
    if dark and flat and flat.get("mean"):
        ratio = float(dark["mean"]) / float(flat["mean"])
        out["dark_to_flat_mean_ratio"] = round(ratio, 3)
        out["dark_is_dark"] = ratio < 0.5
    if flat:
        sat = float(flat.get("saturated_fraction") or 0.0)
        out["flat_saturated_fraction"] = sat
        out["flat_unsaturated"] = sat < 0.001
    if dark and flat:
        out["complete"] = True
    return out


def read_record(folder: Path) -> dict | None:
    path = Path(folder) / RECORD
    if not path.is_file():
        return None
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return doc if isinstance(doc, dict) else None


def list_records(store: Any, session_id: str) -> list[dict]:
    """Every reference record in the session, oldest first, with ``folder``."""
    base = references_dir(store, session_id)
    if base is None or not base.is_dir():
        return []
    out = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        doc = read_record(entry)
        if doc is None:
            continue
        doc["folder"] = str(entry)
        doc["relative"] = str(Path(*FOLDER) / entry.name)
        out.append(doc)
    return out


def matching(records: list[dict], spec: ReferenceSpec) -> dict | None:
    """The newest complete record (dark AND flat) taken for ``spec``."""
    want = spec.key()
    for doc in reversed(records):
        got = doc.get("spec") or {}
        have = ReferenceSpec(
            light=got.get("light", "room"),
            led_intensity_pct=got.get("led_intensity_pct"),
            exposure_ms=got.get("exposure_ms"),
        ).key()
        if have == want and doc.get("dark") and doc.get("flat"):
            return doc
    return None


def spec_of_frame(meta: dict) -> ReferenceSpec:
    """The light and exposure a filed frame was taken under, from its metadata."""
    light = str(meta.get("light") or "room")
    pct = meta.get("led_intensity_pct") if light == "led" else None
    return ReferenceSpec(
        light=light,
        led_intensity_pct=int(pct) if pct else None,
        exposure_ms=meta.get("exposure_ms"),
    )


def references_for_frame(meta: dict, records: list[dict]) -> dict | None:
    """What a frame should be corrected with: what it names, else the newest
    complete record taken for its own light and exposure — so references
    taken after the run reach the frames taken before them."""
    named = meta.get("references") or {}
    if named.get("dark") and named.get("flat"):
        return dict(named)
    return for_frame(matching(records, spec_of_frame(meta)))


def resolve_reference_paths(session_dir: Path, refs: dict | None) -> tuple[Path, Path] | None:
    """The dark and flat files on disk for a frame's references, both present."""
    if not refs or not refs.get("dark") or not refs.get("flat"):
        return None
    dark = Path(session_dir) / str(refs["dark"])
    flat = Path(session_dir) / str(refs["flat"])
    if dark.is_file() and flat.is_file():
        return dark, flat
    return None


def correct(frame: np.ndarray, dark: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """(frame − dark) / (flat − dark) · mean(flat − dark), in the frame's dtype.
    Shapes must agree; a flat that equals the dark anywhere leaves that pixel
    at the frame's own value rather than dividing by zero."""
    f = np.asarray(frame, dtype=np.float64)
    d = np.asarray(dark, dtype=np.float64)
    g = np.asarray(flat, dtype=np.float64) - d
    scale = float(g.mean()) if g.size else 1.0
    safe = np.where(np.abs(g) < 1e-9, 1.0, g)
    out = np.where(np.abs(g) < 1e-9, f, (f - d) / safe * scale)
    if np.issubdtype(np.asarray(frame).dtype, np.integer):
        info = np.iinfo(np.asarray(frame).dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(np.asarray(frame).dtype)
    return out.astype(np.asarray(frame).dtype)


def for_frame(record: dict | None) -> dict | None:
    """What a frame's metadata carries: enough to find the files from the
    session folder, wherever the session folder goes."""
    if not record:
        return None
    rel = record.get("relative") or str(Path(*FOLDER) / str(record.get("record")))
    return {
        "record": record.get("record"),
        "dark": f"{rel}/{(record.get('dark') or {}).get('file')}" if record.get("dark") else None,
        "flat": f"{rel}/{(record.get('flat') or {}).get('file')}" if record.get("flat") else None,
    }


def _write(folder: Path, doc: dict) -> None:
    (Path(folder) / RECORD).write_text(
        yaml.safe_dump(doc, sort_keys=False, default_flow_style=False), encoding="utf-8"
    )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
