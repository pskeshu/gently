"""Piezo-galvo calibration, one measurement at a time.

Not to be confused with ``gently.app.calibration`` — that is dark/flat
per-pixel correction and shares nothing but the word.
"""

METHODS = ("legacy", "scan")
DEFAULT_METHOD = "legacy"
