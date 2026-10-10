"""Is the CSV on disk today the same one a given .nc was fitted with?
"""

from pathlib import Path

from config.mmm_dgp_config import CANONICAL_GEO, CANONICAL_TRADE
from config.contracts import current_contract

_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = _ROOT / "synthetic_data" / "dataset.csv"
# the truth JSON DECLARES the regime and the hash of its CSV (config.contracts.dataset_regime).
TRUTH_JSON = _ROOT / "synthetic_data" / "true_parameters.json"

GENERATOR_ARTIFACTS: frozenset[str] = frozenset({
    "synthetic_data/dataset.csv",
    "synthetic_data/true_parameters.json",
    "synthetic_data/media_share.csv",
    # the readable manifest of the configuration. It travels with the dataset, and the generator is
    # deterministic by seed: regenerating does not dirty the tree.
    "synthetic_data/dataset_manifest.md",
})

# Cache by (paths + mtimes): within one process the verdict does not change unless one of the two
# files is touched, and opening the .nc for every figure of the batch would be wasted time.
_CACHE: dict = {}


def check_sync(idata_path, csv_path=None) -> tuple[bool, str]:
    """Is today's CSV the same one this .nc was fitted with? Returns `(out_of_sync, reason)`. The
    reason says HOW the answer was reached, which matters as much as the answer: "content verified"
    is not the same as "could not verify and fell back to the mtime".
    """
    idata_path = Path(idata_path)
    csv_path = Path(csv_path) if csv_path is not None else DATASET_CSV
    try:
        key = (str(idata_path), str(csv_path),
               idata_path.stat().st_mtime, csv_path.stat().st_mtime)
    except OSError:
        return False, "not verifiable (the .nc or the CSV is missing)"
    if key not in _CACHE:
        _CACHE[key] = _compare(idata_path, csv_path)
    return _CACHE[key]


def _compare(idata_path: Path, csv_path: Path) -> tuple[bool, str]:
    import numpy as np
    import pandas as pd
    import xarray as xr

    try:
        # only the fit_data group: much cheaper than loading the whole InferenceData
        with xr.open_dataset(idata_path, group="fit_data") as ds:
            fit = ds.to_dataframe()
        has_geo = "geo" in (fit.index.names or [])
        fit = fit.reset_index(drop=True)
    except (OSError, KeyError, ValueError):
        # old .nc without fit_data -> falls back to the mtime, the only thing available. The reason
        # says so explicitly, so that nobody reads "verified" where it says "guessed".
        newer = csv_path.stat().st_mtime > idata_path.stat().st_mtime
        return newer, ("mtime only (the .nc does not store fit_data): "
                       + ("CSV is newer" if newer else "CSV is not newer"))

    try:
        csv = pd.read_csv(csv_path, parse_dates=["date"])
        current = (csv[(csv["geo"] == CANONICAL_GEO) & (csv["trade_channel"] == CANONICAL_TRADE)]
                   .sort_values("date").reset_index(drop=True))
    except (OSError, KeyError, ValueError):
        return False, "not verifiable (could not read the CSV)"

    if has_geo:
        # this repo fits one single slice (geo x trade): a .nc with a geo dimension did not come from here
        return True, "content: the .nc has a geo dimension and this repo fits one single slice"

    if len(current) != len(fit):
        return True, f"content: rows {len(current)} vs {len(fit)} of the fit"

    if "date" in fit.columns and "date" in current.columns:
        if not (pd.to_datetime(fit["date"]).to_numpy()
                == pd.to_datetime(current["date"]).to_numpy()).all():
            return True, "content: the dates differ from the fit"

    equal = _equal   # the same tolerance on both paths

    # (1) Columns that are as-is on both sides.
    direct = [c for c in fit.columns if c in current.columns and c != "date"]
    for col in direct:
        if not equal(fit[col], current[col]):
            return True, f"content: column {col} differs from the fit"

    derived = current_contract().FIT_DATA_DERIVED   # regime READ from the dataset
    verified = list(direct)
    for col, (source, f, centered) in derived.items():
        if col not in fit.columns or source not in current.columns:
            continue
        a = np.asarray(fit[col], dtype=float)
        b = f(np.asarray(current[source], dtype=float))
        if centered:
            a, b = np.diff(a), np.diff(b)
        if not equal(a, b):
            return True, f"content: {col} differs from {source} of the current CSV"
        verified.append(col)

    # t_norm_c is determined by the dates and the length, already verified above.
    missing = [c for c in fit.columns
               if c not in verified and c not in ("date", "t_norm_c")]
    detail = f"{len(verified)}/{len(fit.columns) - 1} columns x {len(fit)} rows"
    if missing:
        return False, f"content partially verified ({detail}; not compared: {','.join(missing)})"
    return False, f"content verified ({detail} + dates)"


def _equal(a, b) -> bool:
    import numpy as np
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       rtol=1e-9, atol=1e-12, equal_nan=True)
