"""A volume from side B: the B trio under one name, and the card told which side.

The ASIdiSPIM plugin's per-side preparation writes SPIMFirstSide on the galvo
and arms the side's own piezo and camera. Here that is a second compound
device, ``volume_scanner_b``, built from scanner_b, piezo_b and camera_b, which
a plan takes by name; the scanner's volume configure writes the side.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

# Patch the hardware stack the way the side-param test does, by loading it.
_spec = importlib.util.spec_from_file_location(
    "lightsheet_side_param", Path(__file__).with_name("test_lightsheet_side_param.py")
)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

from gently.hardware.dispim.devices.acquisition import DiSPIMVolumeScanner  # noqa: E402
from gently.hardware.dispim.devices.scanner import DiSPIMScanner  # noqa: E402


def _volume_scanner(side: str) -> tuple[DiSPIMVolumeScanner, MagicMock]:
    scanner, camera, piezo, lasers = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    vs = DiSPIMVolumeScanner(
        scanner=scanner,
        camera=camera,
        piezo=piezo,
        laser_control=lasers,
        core=MagicMock(),
        name=f"volume_scanner{'_b' if side == 'B' else ''}",
        side=side,
    )
    return vs, scanner


def test_a_side_b_volume_scanner_configures_its_scanner_for_side_b():
    vs, scanner = _volume_scanner("B")
    vs.configure(
        num_slices=40,
        exposure_ms=10.0,
        galvo_amplitude=0.5,
        galvo_center=0.0,
        piezo_amplitude=25.0,
        piezo_center=50.0,
    )
    kwargs = scanner.configure_for_volume_acquisition.call_args.kwargs
    assert kwargs["first_side"] == "B"
    assert kwargs["num_slices"] == 40
    assert vs.name == "volume_scanner_b" and vs.side == "B"


def test_side_a_is_the_default_and_unchanged():
    vs, scanner = _volume_scanner("A")
    vs.configure(
        num_slices=50,
        exposure_ms=10.0,
        galvo_amplitude=0.5,
        galvo_center=0.0,
        piezo_amplitude=25.0,
        piezo_center=50.0,
    )
    assert scanner.configure_for_volume_acquisition.call_args.kwargs["first_side"] == "A"


def test_the_scanner_writes_spim_first_side_to_the_card():
    core = MagicMock()
    sc = DiSPIMScanner(name="Scanner:CD:33", core=core)
    sc.configure_for_volume_acquisition(
        galvo_amplitude=0.5, galvo_center=0.0, num_slices=40, first_side="B"
    )
    writes = {(c.args[0], c.args[1]): c.args[2] for c in core.setProperty.call_args_list}
    assert writes[("Scanner:CD:33", "SPIMFirstSide")] == "B"
    assert writes[("Scanner:CD:33", "SPIMNumSides")] == 1
    assert writes[("Scanner:CD:33", "SPIMNumSlices")] == 40


def test_the_factory_builds_the_b_pair_from_the_b_trio(monkeypatch):
    from gently.hardware.dispim import device_factory as df

    class Core:
        def getLoadedDevices(self):
            return ["HamCam1", "HamCam2", "Scanner:AB:33", "Scanner:CD:33",
                    "PiezoStage:P:34", "PiezoStage:Q:35", "Multi Camera"]  # fmt: skip

        def __getattr__(self, name):  # every other core call is a no-op
            return lambda *a, **k: None

    # Device classes are stand-ins; only the wiring is under test.
    stand_ins = (
        "DiSPIMScanner",
        "DiSPIMPiezo",
        "DiSPIMXYStage",
        "DiSPIMZstage",
        "DiSPIMFDrive",
        "DiSPIMLED",
        "DiSPIMLaserControl",
        "DiSPIMBottomCamera",
        "DiSPIMLightSheetSnap",
    )
    for cls in stand_ins:
        if hasattr(df, cls):
            monkeypatch.setattr(df, cls, lambda *a, **k: MagicMock())
    devices = df.create_devices_from_mmcore(Core(), {})
    assert "volume_scanner_b" in devices and "lightsheet_snap_b" in devices
    assert devices["volume_scanner_b"].side == "B"
    assert devices["volume_scanner_b"].name == "volume_scanner_b"
    assert devices["volume_scanner"].side == "A"


def test_the_client_names_the_b_device_for_side_b():
    import asyncio

    from gently.hardware.dispim.client import DiSPIMMicroscope

    client = DiSPIMMicroscope.__new__(DiSPIMMicroscope)
    seen: dict = {}

    async def fake_submit(name, kwargs, timeout):
        seen.update(kwargs)
        return {"success": False, "error": "no rig"}

    client._submit_plan_and_wait = fake_submit  # type: ignore[method-assign]
    asyncio.run(client.acquire_volume(num_slices=12, side="B"))
    assert seen["volume_scanner"] == "volume_scanner_b"
    asyncio.run(client.acquire_volume(num_slices=12))
    assert seen["volume_scanner"] == "volume_scanner"
