"""Ground-truth reader for the validation scripts.
"""

import json
from pathlib import Path

from config.mmm_dgp_config import REGIMES, normalize_label, split_label

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_JSON_PATH = ROOT / "synthetic_data" / "true_parameters.json"
_DEFAULT_CSV_PATH = ROOT / "synthetic_data" / "dataset.csv"

_cache: dict | None = None


def load_true_params(json_path: Path | None = None) -> dict:
    """Loads (and caches in memory) the full true-parameters JSON. The cache is per process: if you need
    to reread the JSON after regenerating the dataset within the same process, set _cache = None, or
    simply reimport the module.
    """
    global _cache
    path = json_path or _DEFAULT_JSON_PATH
    if _cache is None or json_path is not None:
        with open(path) as f:
            data = json.load(f)
        if json_path is None:
            from config.contracts import dataset_regime
            dataset_regime()     # the LIVE dataset has to be consistent (CSV = the JSON's)
            _cache = data
        return data
    return _cache


def get_true_L(channel: str, geo: str, trade: str, json_path: Path | None = None) -> float:
    """Effective L (composed ceiling: L_national * geo_multiplier * media_attenuation, WITHOUT
    geo_size) for (channel, geo, trade).
    """
    data = load_true_params(json_path)
    key = f"{channel}|{geo}|{trade}"
    return data["effective_params"]["media"][key]["L_effective"]


def get_channel_params(channel: str, json_path: Path | None = None) -> dict:
    """adstock_alpha, lam, x50_half_saturation for a channel (constant across geo/trade -- they do not
    vary by geo, see the generator's docstring).
    """
    data = load_true_params(json_path)
    for key, vals in data["effective_params"]["media"].items():
        if key.split("|")[0] == channel:
            return {k: v for k, v in vals.items() if k != "L_effective"}
    raise KeyError(f"Unknown channel: {channel!r}")


def true_contributions(dfm, geo: str | None = None, trade: str | None = None,
                       json_path: Path | None = None) -> dict:
    """True c_k(t) of each channel in the (geo, trade) slice: L_effective * satfrac(adstock(x, alpha),
    lam).
    """
    import numpy as np
    from config.mmm_dgp_config import ADSTOCK_NORMALIZE, CHANNEL_COLUMNS, CANONICAL_GEO, L_MAX, CANONICAL_TRADE
    from synthetic_data.adstock_dgp import geometric_adstock_aligned
    geo, trade = geo or CANONICAL_GEO, trade or CANONICAL_TRADE
    out = {}
    for col in CHANNEL_COLUMNS:
        ch = col.replace("_impressions_000", "")
        p = get_channel_params(ch, json_path)
        ad = geometric_adstock_aligned(dfm[col].values, p["adstock_alpha"],
                                       l_max=L_MAX, normalize=ADSTOCK_NORMALIZE)
        L = get_true_L(ch, geo, trade, json_path)
        x = ad / 1.0
        out[ch] = L * (1 - np.exp(-p["lam"] * x)) / (1 + np.exp(-p["lam"] * x))
    return out


def get_demand_params(geo: str, trade: str, json_path: Path | None = None) -> dict:
    """Truth of the controls for (geo, trade). In the log regime the keys are gamma_true, eta_true and
    promo_lift (elasticities and lift).
    """
    data = load_true_params(json_path)
    key = f"{geo}|{trade}"
    return data["effective_params"]["demand"][key]
def get_dataset_label(json_path: Path | None = None) -> str | None:
    """The dataset variant label (e.g. "log__no_noise"), read from metadata.dataset_label.
    """
    data = load_true_params(json_path)
    label = data.get("metadata", {}).get("dataset_label")
    # always in the canonical form <regime>__<noise>. A label without a regime prefix belongs to the
    # log regime (see config.split_label).
    return normalize_label(label) if label else None


def _extract_label_from_filename(path: Path) -> str | None:
    """The label is encoded in the name of the .nc by config/idata_paths.py
    (3a_posterior.nc -> 3a_posterior__log__no_noise.nc).
    """
    parts = Path(path).stem.split("__")
    if len(parts) >= 3 and parts[-2] in REGIMES:  # trivial: parts of the file name
        return f"{parts[-2]}__{parts[-1]}"
    if len(parts) >= 2:  # trivial: parts of the file name
        return normalize_label(parts[-1])
    return None


def warn_if_idata_stale(idata_path: Path, csv_path: Path | None = None, json_path: Path | None = None) -> bool:
    """Two independent ways in which comparing this .nc against the truth is invalid, neither of them
    caught by any Python error (everything loads fine): the CSV is not the one the .nc was fitted with,
    or the .nc belongs to another dataset variant.
    """
    idata_path = Path(idata_path)
    csv_path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV_PATH
    if not idata_path.exists() or not csv_path.exists():
        return False

    problem = False

    from config.dataset_sync import check_sync
    out_of_sync, reason = check_sync(idata_path, csv_path)
    if out_of_sync:
        _bar = "!" * 78  # cosmetic
        print(
            f"\n{_bar}\n"
            f"WARNING: {csv_path.name} is NOT the dataset {idata_path.name} was fitted "
            f"with.\n"
            f"Reason: {reason}.\n"
            f"The saved posterior belongs to ANOTHER version of the data -- this "
            f"comparison against the CURRENT ground truth is not valid. Refit the model "
            f"before trusting these numbers.\n"
            f"{_bar}\n"
        )
        problem = True

    try:
        current_label = get_dataset_label(json_path)
    except (FileNotFoundError, KeyError):
        current_label = None
    idata_label = _extract_label_from_filename(idata_path)
    if current_label is not None and idata_label is not None and idata_label != current_label:
        print(
            f"\n{'!' * 78}\n"  # cosmetic
            f"WARNING: {idata_path.name} belongs to the dataset variant "
            f"'{idata_label}', but the CURRENT dataset is '{current_label}'.\n"
            f"You are comparing a posterior fitted on ANOTHER dataset (another noise "
            f"level, or another variant) against the ground truth of this one. The numbers will "
            f"look plausible and they will be wrong -- there is no Python error that "
            f"catches it for you.\n"
            f"{'!' * 78}\n"  # cosmetic
        )
        problem = True

    return problem
