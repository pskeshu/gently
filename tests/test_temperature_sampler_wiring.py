"""Wiring guard: asserts that agent.py contains the TemperatureSampler lifecycle hooks.

This is an intentionally lightweight source-text guard — a full agent boot is a
heavy integration concern verified end-to-end in a later task.  DeviceStateMonitor's
own wiring is likewise not unit-tested at this level.
"""


def test_agent_initializes_temperature_sampler_attribute():
    """agent.py must declare the attribute *and* construct the sampler.

    Uses find_spec to locate the source file without executing the module
    (agent.py has heavy runtime deps like anthropic that aren't present in the
    test environment).
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("gently.app.agent")
    assert spec is not None and spec.origin is not None, (
        "Could not locate gently/app/agent.py via importlib"
    )
    text = Path(spec.origin).read_text(encoding="utf-8")
    assert "temperature_sampler" in text, "agent.py has no 'temperature_sampler' attribute"
    assert "TemperatureSampler(" in text, "agent.py never constructs a TemperatureSampler"
