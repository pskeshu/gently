"""Export a session the way a biologist wants to find it again.

"a nice export method that can organize and store the data in a neat manner
that is easily usable in fiji … volume files or snapshot files that are
sorted by timepoint or something instead of uid in filename … put inside
folder such that it is easier to find the original imprints of the
experiment etc, or stored with metadata files"

The session folder is laid out for the software: one folder per embryo id,
snapshots named by uuid, state in YAML and JSONL. The export is laid out for
the person: one folder per embryo named by what they called it, every file
named so a plain sort is time order, and beside the images the plain-text
record of what the experiment was — the plan, the run, the stage calls, the
events, the temperature, what was said — plus a README that points back at
the originals. Files are copies, never links, so editing an export in Fiji
cannot touch the session.

    <dest>/<session folder name>/
        README.txt
        embryos.csv  stage_calls.csv  events.csv  temperature.csv
        metadata/    session.yaml acquisition.yaml timelapse.yaml events.jsonl
                     temperature.jsonl conversation.json
        <label>/     volumes/<label>_t0001.tif …   volumes.csv
                     projections/<label>_t0001.jpg …
                     embryo.yaml  calibration/  timelapse.mp4
        dic/         dic_f0001_20261004-213000.tif …   dic.csv

Fiji opens a volumes/ folder with File › Import › Image Sequence…, in order.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

Progress = Callable[[int, int, str], None]

_EVENT_TEXT = {
    "ACQUISITION_STARTED",
    "ACQUISITION_COMPLETED",
    "ACQUISITION_STOPPED",
    "ACQUISITION_FAILED",
    "SESSION_RESTORED",
    "TRIGGER_FIRED",
    "BURST_START",
    "BURST_COMPLETE",
    "POWER_RAMP_STEP",
    "HATCHING_DETECTED",
    "DETECTION_TRIGGERED",
    "EMBRYO_TERMINATED",
    "EMBRYO_SKIPPED",
    "OPERATOR_REMOVED_EMBRYO",
    "TEMPERATURE_SETPOINT_CHANGED",
    "TEMP_PROTOCOL_COMPLETED",
    "ERROR_OCCURRED",
    "WARNING_ISSUED",
}


def label_for(embryo: dict[str, Any]) -> str:
    """The folder and file prefix an embryo gets: what it was called, made
    safe for a filename, with its id so two "A"s cannot collide."""
    eid = str(embryo.get("embryo_id") or "embryo")
    nick = str(embryo.get("nickname") or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", nick).strip("-.")
    return f"{safe}_{eid}" if safe and safe.lower() != eid.lower() else eid


def _stamp(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(str(iso)).strftime("%Y%m%d-%H%M%S")
    except ValueError:
        return re.sub(r"[^0-9]", "", str(iso))[:14]


def _copy(src: Path, dst: Path, step: Callable[[str], None]) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    step(dst.name)
    return True


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})


def plan_lines(plan: dict | None) -> list[str]:
    """The acquisition plan in sentences, for the README."""
    if not plan:
        return ["No acquisition plan was recorded for this session."]
    out: list[str] = []
    if plan.get("interval_seconds"):
        out.append(f"Interval: every {plan['interval_seconds']} s")
    if plan.get("volumes") is False:
        out.append("Volumes: off (brightfield only)")
    else:
        vol = []
        if plan.get("num_slices") is not None:
            vol.append(f"{plan['num_slices']} slices")
        if plan.get("exposure_ms") is not None:
            vol.append(f"{plan['exposure_ms']} ms exposure")
        if vol:
            out.append("Volumes: " + ", ".join(vol))
        lasers = []
        if plan.get("laser_config"):
            lasers.append(str(plan["laser_config"]).replace("_", " "))
        for wl, pct in (plan.get("laser_powers") or {}).items():
            lasers.append(f"{wl} nm at {pct}%")
        if lasers:
            out.append("Laser: " + ", ".join(lasers))
    stop = plan.get("stop_condition") or {}
    if stop:
        kind = stop.get("kind") or stop.get("condition_type") or stop.get("type")
        out.append(f"Stop: {kind}" + (f" {stop['value']}" if stop.get("value") is not None else ""))
    dic = plan.get("dic") or {}
    if dic.get("enabled"):
        bits = []
        if dic.get("every_seconds"):
            bits.append(f"every {dic['every_seconds']} s")
        if dic.get("light"):
            bits.append(
                f"{dic['light']}"
                + (
                    f" {dic['led_intensity_pct']}%"
                    if dic.get("led_intensity_pct") is not None
                    else ""
                )
            )
        if dic.get("exposure_ms") is not None:
            bits.append(f"{dic['exposure_ms']} ms")
        out.append("DIC overview: " + ", ".join(bits))
    else:
        out.append("DIC overview: off")
    return out


def export_session(
    store: Any,
    session_id: str,
    dest: Path | None = None,
    progress: Progress | None = None,
) -> Path:
    """Write the export of ``session_id`` under ``dest`` (default
    ``<root>/exports``) and return its folder. Re-exporting overwrites the
    same folder. ``progress(done, total, what)`` is called per file."""
    sd = store._session_dir(session_id)
    if sd is None or not Path(sd).exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    sd = Path(sd)
    out = Path(dest) if dest else Path(store.root) / "exports"
    out = out / sd.name
    out.mkdir(parents=True, exist_ok=True)

    info = store.get_session(session_id) or {}
    embryos = store.list_embryos(session_id) or []
    plan = store.get_acquisition_plan(session_id)
    volumes = store.list_volumes(session_id) or []
    snapshots = store.list_snapshots(session_id, "dic") or []
    predictions = store.get_predictions(session_id) or []

    # Everything that will be copied, counted first so progress means something.
    total = (
        len(volumes)
        + len(snapshots)
        + sum(
            len(store.list_projection_timepoints(session_id, e["embryo_id"]) or []) for e in embryos
        )
        + 6
    )
    done = 0

    def step(what: str) -> None:
        nonlocal done
        done += 1
        if progress:
            progress(done, total, what)

    # --- the record, as it was kept -----------------------------------------
    meta_dir = out / "metadata"
    for name in (
        "session.yaml",
        "acquisition.yaml",
        "timelapse.yaml",
        "events.jsonl",
        "temperature.jsonl",
        "conversation.json",
        "decisions.jsonl",
    ):
        _copy(sd / name, meta_dir / name, lambda _n: None)

    # --- the record, as a person reads it ----------------------------------
    by_embryo_preds: dict[str, list[dict]] = {}
    for p in predictions:
        by_embryo_preds.setdefault(str(p.get("embryo_id")), []).append(p)

    checkpoint: dict = {}
    try:
        doc = yaml.safe_load((sd / "timelapse.yaml").read_text(encoding="utf-8"))
        checkpoint = doc if isinstance(doc, dict) else {}
    except (OSError, yaml.YAMLError):
        checkpoint = {}
    rows_ck = checkpoint.get("embryos") or {}

    embryo_rows = []
    for e in embryos:
        eid = e["embryo_id"]
        pos = e.get("position_coarse") or {}
        preds = by_embryo_preds.get(eid, [])
        ck = rows_ck.get(eid) or {}
        embryo_rows.append(
            {
                "label": label_for(e),
                "embryo_id": eid,
                "nickname": e.get("nickname"),
                "role": e.get("role"),
                "strain": e.get("strain"),
                "x_um": pos.get("x", e.get("position_x")),
                "y_um": pos.get("y", e.get("position_y")),
                "timepoints": len([v for v in volumes if v["embryo_id"] == eid]),
                "last_stage": preds[-1].get("predicted_stage") if preds else "",
                "complete": ck.get("is_complete"),
                "completion_reason": ck.get("completion_reason"),
                "total_exposure_ms": ck.get("total_exposure_ms"),
            }
        )
    _write_csv(
        out / "embryos.csv",
        embryo_rows,
        [
            "label",
            "embryo_id",
            "nickname",
            "role",
            "strain",
            "x_um",
            "y_um",
            "timepoints",
            "last_stage",
            "complete",
            "completion_reason",
            "total_exposure_ms",
        ],
    )
    step("embryos.csv")

    labels = {r["embryo_id"]: r["label"] for r in embryo_rows}
    _write_csv(
        out / "stage_calls.csv",
        [
            {
                "label": labels.get(str(p.get("embryo_id")), p.get("embryo_id")),
                "embryo_id": p.get("embryo_id"),
                "timepoint": p.get("timepoint"),
                "stage": p.get("predicted_stage"),
                "confidence": p.get("confidence"),
                "transitional": p.get("is_transitional"),
                "reasoning": p.get("reasoning"),
            }
            for p in sorted(
                predictions, key=lambda p: (str(p.get("embryo_id")), p.get("timepoint") or 0)
            )
        ],
        ["label", "embryo_id", "timepoint", "stage", "confidence", "transitional", "reasoning"],
    )
    step("stage_calls.csv")

    events = [
        {
            "time": r.get("timestamp"),
            "type": r.get("event_type") or r.get("type"),
            "data": json.dumps(r.get("data"), default=str) if r.get("data") is not None else "",
        }
        for r in _read_jsonl(sd / "events.jsonl")
        if (r.get("event_type") or r.get("type")) in _EVENT_TEXT
    ]
    _write_csv(out / "events.csv", events, ["time", "type", "data"])
    step("events.csv")

    _write_csv(
        out / "temperature.csv",
        [
            {
                "time": r.get("t"),
                "water_c": r.get("water_c"),
                "setpoint_c": r.get("setpoint_c"),
                "state": r.get("state"),
            }
            for r in _read_jsonl(sd / "temperature.jsonl")
        ],
        ["time", "water_c", "setpoint_c", "state"],
    )
    step("temperature.csv")

    # --- the images, one folder per embryo, in time order --------------------
    for e in embryos:
        eid = e["embryo_id"]
        label = labels[eid]
        edir = out / label
        src_e = sd / "embryos" / eid
        _copy(src_e / "embryo.yaml", edir / "embryo.yaml", lambda _n: None)
        _copy(src_e / "timelapse.mp4", edir / "timelapse.mp4", lambda _n: None)
        if (src_e / "calibration").is_dir():
            shutil.copytree(src_e / "calibration", edir / "calibration", dirs_exist_ok=True)

        vol_rows = []
        for v in sorted(
            (v for v in volumes if v["embryo_id"] == eid), key=lambda v: v["timepoint"]
        ):
            tp = int(v["timepoint"])
            name = f"{label}_t{tp:04d}.tif"
            _copy(Path(v["file_path"]), edir / "volumes" / name, step)
            meta = v.get("metadata") or {}
            shape = v.get("shape") or []
            vol_rows.append(
                {
                    "file": f"volumes/{name}",
                    "timepoint": tp,
                    "acquired_at": v.get("acquired_at"),
                    "z": shape[0] if len(shape) == 3 else "",
                    "y": shape[-2] if len(shape) >= 2 else "",
                    "x": shape[-1] if len(shape) >= 2 else "",
                    "dtype": v.get("dtype"),
                    "num_slices": meta.get("num_slices"),
                    "exposure_ms": meta.get("exposure_ms"),
                    "interval_seconds": meta.get("interval_seconds"),
                    "laser_488_pct": meta.get("laser_power_488_pct"),
                    "laser_561_pct": meta.get("laser_power_561_pct"),
                    "laser_405_pct": meta.get("laser_power_405_pct"),
                    "laser_637_pct": meta.get("laser_power_637_pct"),
                    "acquisition_mode": meta.get("acquisition_mode"),
                }
            )
        if vol_rows:
            _write_csv(
                edir / "volumes.csv",
                vol_rows,
                [
                    "file",
                    "timepoint",
                    "acquired_at",
                    "z",
                    "y",
                    "x",
                    "dtype",
                    "num_slices",
                    "exposure_ms",
                    "interval_seconds",
                    "laser_488_pct",
                    "laser_561_pct",
                    "laser_405_pct",
                    "laser_637_pct",
                    "acquisition_mode",
                ],
            )
        for tp in store.list_projection_timepoints(session_id, eid) or []:
            src = store.get_projection_path(session_id, eid, tp)
            if src is not None:
                _copy(Path(src), edir / "projections" / f"{label}_t{int(tp):04d}.jpg", step)

    # --- the DIC overview, by frame then time, not by uuid ------------------
    dic_rows = []
    for i, rec in enumerate(
        sorted(
            snapshots,
            key=lambda r: ((r.get("metadata") or {}).get("frame") or 0, r.get("captured_at") or ""),
        ),
        start=1,
    ):
        meta = rec.get("metadata") or {}
        frame = int(meta.get("frame") or i)
        when = meta.get("captured_at") or rec.get("captured_at")
        name = f"dic_f{frame:04d}_{_stamp(when)}.tif" if when else f"dic_f{frame:04d}.tif"
        if _copy(Path(rec.get("file_path") or ""), out / "dic" / name, step):
            pos = meta.get("position") or {}
            dic_rows.append(
                {
                    "file": f"dic/{name}",
                    "frame": frame,
                    "round": meta.get("round"),
                    "captured_at": when,
                    "x_um": pos.get("x"),
                    "y_um": pos.get("y"),
                    "exposure_ms": meta.get("exposure_ms"),
                    "light": meta.get("light"),
                    "led_intensity_pct": meta.get("led_intensity_pct"),
                    "width": rec.get("width"),
                    "height": rec.get("height"),
                }
            )
    if dic_rows:
        _write_csv(
            out / "dic" / "dic.csv",
            dic_rows,
            [
                "file",
                "frame",
                "round",
                "captured_at",
                "x_um",
                "y_um",
                "exposure_ms",
                "light",
                "led_intensity_pct",
                "width",
                "height",
            ],
        )
    step("dic.csv")

    # --- the README: what this is, and where the originals are --------------
    name = info.get("name") or sd.name
    lines = [
        f"{name}",
        "=" * len(str(name)),
        "",
        f"Gently session {session_id}, created {info.get('created_at', '')}.",
        f"Exported {datetime.now().isoformat(timespec='seconds')}.",
        f"Originals: {sd}",
        "",
        "These are copies, not links: editing or saving anything here cannot touch the session.",
        "",
    ]
    if info.get("description"):
        lines += [str(info["description"]), ""]
    lines += ["Acquisition plan", "----------------", *plan_lines(plan), ""]
    if checkpoint:
        lines += [
            "Run",
            "---",
            f"Status at last checkpoint: {checkpoint.get('status')}",
            f"Started: {checkpoint.get('started_at')}   Last saved: {checkpoint.get('saved_at')}",
            f"Rounds: {checkpoint.get('current_round')}"
            f"   Timepoints: {checkpoint.get('total_timepoints')}",
            "",
        ]
    lines += ["Embryos", "-------"]
    for r in embryo_rows:
        lines.append(
            f"  {r['label']}/   role {r['role']}, {r['timepoints']} timepoints"
            + (f", last stage {r['last_stage']}" if r["last_stage"] else "")
            + (", complete" if r["complete"] else "")
        )
    lines += [
        "",
        "What is where",
        "-------------",
        "  <embryo>/volumes/<embryo>_t0001.tif ...   one 3D TIFF per timepoint,"
        " sorted = time order",
        "  <embryo>/volumes.csv                      when each was taken and with what settings",
        "  <embryo>/projections/                     the per-timepoint JPEG projections",
        "  <embryo>/calibration/                     the calibration runs (frames, plots, fit)",
        "  dic/dic_f0001_<date-time>.tif ...         the DIC overview frames, in order;"
        " dic/dic.csv says when and where",
        "  embryos.csv, stage_calls.csv, events.csv, temperature.csv",
        "  metadata/                                 the session's own files, as kept"
        " (YAML/JSONL/JSON)",
        "",
        "Fiji",
        "----",
        "  A time series: File > Import > Image Sequence..., choose an <embryo>/volumes folder.",
        "  One volume: File > Open on a single .tif (it is a Z stack).",
        "  The DIC overview as a movie: Image Sequence on the dic/ folder.",
        "",
    ]
    (out / "README.txt").write_text("\n".join(lines), encoding="utf-8")
    step("README.txt")
    logger.info("Exported session %s to %s (%d files)", session_id, out, done)
    return out
