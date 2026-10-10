"""SAVING A .nc SO THAT IT CAN NEVER BE LEFT HALF-WRITTEN.

# ! WATCH OUT -- a .nc cut off mid-write keeps its final name and arviz opens it anyway.

Produces: <destination>.nc and its sidecar <destination>.nc.sha256
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_idata_atomic(idata, destination, script: str = "", *, build_mmm_kwargs: dict):
    """Saves a .nc so that it can NEVER be left half-written under its final name.
    `build_mmm_kwargs` is MANDATORY: it is the specification the model is rebuilt with, and it goes
    both to `idata.posterior.attrs` and to the sidecar.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    idata.posterior.attrs["build_mmm_kwargs"] = json.dumps(build_mmm_kwargs)
    idata.to_netcdf(tmp)
    os.replace(tmp, destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(json.dumps({
        "_generated": "do not edit by hand", "file": destination.name, "sha256": _sha256(destination),
        "bytes": destination.stat().st_size, "written": datetime.now().isoformat(timespec="seconds"),
        "script": script, "build_mmm_kwargs": build_mmm_kwargs,
    }, indent=2), encoding="utf-8")
    return destination


def verify_idata(destination, with_hash: bool = True):
    """(ok, reason). Check BEFORE using a .nc that was not just written."""
    destination = Path(destination)
    if not destination.is_file():
        return False, f"does not exist: {destination}"
    if destination.with_suffix(destination.suffix + ".partial").exists():
        return False, ("there is a .partial next to it: an earlier write was interrupted. The .nc "
                       "that is there may be from an OLD run, not from the interrupted one.")
    side = destination.with_suffix(destination.suffix + ".sha256")
    if not side.is_file():
        return True, "no sidecar: it cannot be verified, but there is no reason for suspicion either."
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"unreadable sidecar: {exc}"
    if destination.stat().st_size != meta.get("bytes"):
        return False, f"size {destination.stat().st_size} != {meta.get('bytes')} of the sidecar"
    if with_hash and _sha256(destination) != meta.get("sha256"):
        return False, "the sha256 does NOT match the sidecar's: the file changed or got corrupted"
    return True, f"ok (written {meta.get('written')} by {meta.get('script') or '?'})"
