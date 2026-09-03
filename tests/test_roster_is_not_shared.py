"""No two modules may hold the same embryo roster array.

`operate.js` and `devices.js` both bind `_embryos` from the `EMBRYOS_UPDATE`
payload. Both used to assign `payload.embryos` directly, so they shared one
mutable array object: a `push` in devices.js appeared in operate.js's roster as
a row with no id, no coordinates and no marker, whose `×` built
`/api/embryos/` and came back 405. Observed at 05:15-07:30 in the 2026-08-07
walkthrough.

`.slice()` at each binding is the whole fix — each module then owns its copy,
and the element writes that remain are copy-on-write into their own slot.

ponytail: a source test, because these are IIFEs with no export surface and no
DOM in CI. It checks the binding, not the behaviour. If the roster ever moves
behind a real module boundary, delete this and test the boundary instead.
"""

from __future__ import annotations

import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"

# `_embryos = <something>.embryos` with no defensive copy.
BY_REFERENCE = re.compile(r"_embryos\s*=\s*.*\.embryos\s*(?!\s*\.\s*slice)(?::|\s|;|\))")


def test_each_module_copies_the_roster() -> None:
    offenders: list[str] = []
    for name in ("operate.js", "devices.js"):
        path = JS / name
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "_embryos" not in line or ".embryos" not in line:
                continue
            if ".embryos.slice()" in line:
                continue
            if BY_REFERENCE.search(line):
                offenders.append(f"{name}:{n}: {line.strip()}")

    assert not offenders, (
        "the roster is bound by reference — the other module can mutate it:\n  "
        + "\n  ".join(offenders)
    )


def test_a_row_without_an_id_cannot_issue_a_delete() -> None:
    src = (JS / "operate.js").read_text(encoding="utf-8")
    body = src[src.index("async function deleteEmbryo(id)") :]
    guard = body[: body.index("fetch(")]
    assert "if (!id)" in guard, (
        "deleteEmbryo no longer refuses an empty id — it will build "
        "`/api/embryos/`, 307 to a GET-only route, and report 405"
    )


if __name__ == "__main__":
    test_each_module_copies_the_roster()
    test_a_row_without_an_id_cannot_issue_a_delete()
    print("ok")
