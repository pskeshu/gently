"""Role-scoped tactic target resolver.

``resolve_scope_embryos`` maps a tactic ``scope`` dict + a roster of embryo
dicts to a list of embryo IDs that the tactic should operate on.

Scope modes
-----------
``global``
    All embryo IDs in the roster.
``embryos``
    Explicit list from ``scope['embryo_ids']``.
``role``
    IDs of embryos whose ``role`` field matches ``scope['role']``.
missing/unknown
    Empty list (safe default — never errors).

Embryo dict shape
-----------------
The resolver expects the shape produced by ``/api/embryos/positions``:
each dict must have an ``'embryo_id'`` key (not just ``'id'``).
"""


def resolve_scope_embryos(scope: dict | None, embryos: list[dict]) -> list[str]:
    """Return the embryo IDs that match *scope* from *embryos*.

    Parameters
    ----------
    scope:
        Tactic scope dict, e.g. ``{"mode": "role", "role": "test"}``.
        ``None`` is treated as an unknown scope → returns ``[]``.
    embryos:
        List of embryo dicts, each with at minimum ``embryo_id`` and ``role``
        keys.  Any dict lacking ``embryo_id`` is silently skipped.

    Returns
    -------
    list[str]
        Matched embryo IDs.  Never raises.
    """
    if not scope:
        return []

    mode = scope.get("mode")

    if mode == "global":
        return [e["embryo_id"] for e in embryos if "embryo_id" in e]

    if mode == "embryos":
        return list(scope.get("embryo_ids") or [])

    if mode == "role":
        target_role = scope.get("role")
        return [
            e["embryo_id"] for e in embryos if "embryo_id" in e and e.get("role") == target_role
        ]

    return []
