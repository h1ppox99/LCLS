"""io -- reusable readers for the local LCLS xppl1016922 copy."""
from __future__ import annotations


def scan_shots(run: int, max_events=None, prefer_psana: bool = True):
    """Per-shot scalar table for `run`, from psana if it is installed and from
    the pure-Python XTC reader if it is not.

    Two readers exist because psana ships only for linux-64/osx-64 and part of
    this project is developed on osx-arm64, where every claim needing per-shot
    scalars would otherwise be blocked on the package manager rather than on the
    data. `io.read_xtc` stays authoritative -- it decodes through psana's own
    types and works for frames as well; `io.xtc_raw` decodes the scalar structs
    directly and stops there. Both return the same `ShotMeta`.
    """
    if prefer_psana:
        try:
            from automask.io.read_xtc import scan_shots as _psana_scan
            return _psana_scan(run, max_events)
        except ImportError:
            pass
    from automask.io.xtc_raw import scan_shots as _raw_scan
    return _raw_scan(run, max_events)
