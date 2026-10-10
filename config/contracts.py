"""The registry of regimes: name -> module with its contract.
"""

from config import log_regime
from config.mmm_dgp_config import REGIMES

CONTRACTS = {log_regime.NAME: log_regime}
assert tuple(CONTRACTS) == REGIMES, (tuple(CONTRACTS), REGIMES)


def contract(regime: str):
    """The module with everything specific to `regime`. KeyError with the list if it does not exist."""
    if regime not in CONTRACTS:
        raise KeyError(f"unknown regime {regime!r}; known: {list(CONTRACTS)}")
    return CONTRACTS[regime]


class WrongRegimeError(RuntimeError):
    """A script of one regime received a dataset (or a .nc) of the other."""


class InconsistentDatasetError(RuntimeError):
    """The dataset on disk does not declare its regime, or its CSV is not the one its JSON declares."""


def content_sha256(path) -> str:
    """sha256 of the file CONTENT, with line endings normalized to LF. That way the hash does not
    depend on whether git left the CSV with CRLF (Windows, core.autocrlf) or with LF (a Linux clone).
    """
    import hashlib
    from pathlib import Path as _Path
    return hashlib.sha256(_Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


_REGIME_CACHE: dict = {}


def dataset_regime(json_path=None, csv_path=None) -> str:
    """The regime of the CURRENT dataset, read from the truth JSON and checked against its CSV. It
    fails (it does not guess) if the JSON does not declare a regime, if it declares an unknown one, if
    it does not record the hash of its CSV, or if the CSV on disk is not that one.
    """
    import json
    from pathlib import Path as _Path
    from config.dataset_sync import DATASET_CSV, TRUTH_JSON
    jp = _Path(json_path) if json_path is not None else TRUTH_JSON
    cp = _Path(csv_path) if csv_path is not None else DATASET_CSV
    try:
        key = (str(jp), jp.stat().st_mtime_ns, jp.stat().st_size, str(cp), cp.stat().st_mtime_ns, cp.stat().st_size)
    except OSError as exc:
        raise InconsistentDatasetError(f"there is no generated dataset: {exc}. Run the generator "
                                       f"with --regime log.") from None
    if key in _REGIME_CACHE:
        return _REGIME_CACHE[key]
    meta = json.loads(jp.read_text(encoding="utf-8")).get("metadata", {})
    regime = meta.get("regime")
    if regime is None:
        raise InconsistentDatasetError(f"{jp.name} does not declare its regime: "
                                       f"regenerate it with the generator.")
    if regime not in CONTRACTS:
        raise InconsistentDatasetError(f"{jp.name} declares the regime {regime!r}, which does not exist; "
                                       f"known: {list(CONTRACTS)}")
    expected = meta.get("csv_sha256")
    if expected is None:
        raise InconsistentDatasetError(f"{jp.name} does not record the hash of its CSV (it was written by "
                                       f"a version of the generator older than that check): regenerate it "
                                       f"with step 1 of the README.")
    if content_sha256(cp) != expected:
        raise InconsistentDatasetError(
            f"{cp.name} is NOT the CSV that {jp.name} declares (different content sha256). Someone "
            f"changed one without the other. Regenerate the dataset with step 1 of the README.")
    _REGIME_CACHE[key] = regime
    return regime


def current_contract():
    """The contract of the regime of the dataset on disk."""
    return contract(dataset_regime())


def idata_regime(idata) -> str:
    """The regime of a .nc, read from its own content: the target scale (constant_data.target_scale)
    falls in the TARGET_SCALE_RANGE of exactly one regime.
    """
    import numpy as _np
    ts = float(_np.asarray(idata.constant_data["target_scale"]).reshape(-1)[0])
    candidates = [n for n, c in CONTRACTS.items() if c.TARGET_SCALE_RANGE[0] <= ts < c.TARGET_SCALE_RANGE[1]]
    if len(candidates) != 1:
        raise InconsistentDatasetError(f"target_scale={ts:.6g} does not identify a regime ({candidates}).")
    return candidates[0]


def require_regime(expected: str, who, found: str | None = None) -> None:
    """Fails if the current dataset (or `found`) is not of the `expected` regime, saying what was
    expected, what was found and how to change it.
    """
    from pathlib import Path as _Path
    if found is None:
        found = dataset_regime()
    if found != expected:
        who = _Path(str(who)).name
        raise WrongRegimeError(
            f"\n{who} belongs to the '{expected}' regime and the current dataset is of the '{found}' regime "
            f"(read from synthetic_data/true_parameters.json).\n"
            f"It does not run on the other regime: it would give meaningless numbers instead of an error.\n"
            f"Regenerate the dataset with step 1 of the README.\n")
