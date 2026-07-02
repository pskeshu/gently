"""Vendored third-party SDKs bundled with gently.

Vendor-supplied packages that are NOT published to PyPI, copied in so the device
layer always has them regardless of the machine's environment.

- ``acuitynano_precision_thermalizer_serial`` — USB serial transport for the
  ACUITYnano Precision Thermal Controller.

The MQTT transport (``acuitynano_precision_thermalizer_api``) is deliberately
NOT bundled here: it embeds broker credentials, which don't belong in the repo.
Install it on the device-layer machine if you use ``backend: mqtt``.

``gently.hardware.temperature`` imports these via ``_load_vendor()``, preferring
a system-installed copy of the same package name (so an official vendor update
can override), and falling back to the bundled copy here.
"""
