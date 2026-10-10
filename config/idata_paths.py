"""Central paths of the .nc files (InferenceData) that the model scripts produce and consume.
"""

from pathlib import Path

from config.mmm_dgp_config import normalize_label, split_label

ROOT = Path(__file__).resolve().parents[1]

# name -> {"base": path WITHOUT the dataset_label suffix, "producer": which script generates it
# (for the require_idata error message)}
IDATA_REGISTRY = {
    "linear_trend": {
        "base": ROOT / "model/3_fit/3a_posterior.nc",
        "producer": "model/3_fit/3a_fit.py",
    },
}


def idata_path(name: str, dataset_label: str, slice_: str | None = None) -> Path:
    """Builds the final path by adding the dataset_label: 3a_posterior.nc ->
    3a_posterior__<label>.nc.
    """
    if name not in IDATA_REGISTRY:
        raise KeyError(f"'{name}' is not in IDATA_REGISTRY. Valid keys: {list(IDATA_REGISTRY)}")
    base = IDATA_REGISTRY[name]["base"]
    label = normalize_label(dataset_label)
    _allowed = IDATA_REGISTRY[name].get("regimes")
    if _allowed is not None and split_label(label)[0] not in _allowed:
        from config.contracts import WrongRegimeError
        raise WrongRegimeError(
            f"the model '{name}' only exists in the regime {list(_allowed)} and the dataset is "
            f"'{label}'. Regenerate the dataset with step 1 of the README.")
    prefix = f"{base.stem}__{slice_}" if slice_ else base.stem
    new = base.with_name(f"{prefix}__{label}{base.suffix}")
    regime, rest = split_label(label)
    if regime == "log" and not new.exists():
        legacy = base.with_name(f"{prefix}__{rest}{base.suffix}")
        if legacy.exists():
            return legacy
    return new


def require_idata(name: str, dataset_label: str, slice_: str | None = None) -> Path:
    """Like idata_path(), but checks that the file exists. If not, the error message says WHICH
    SCRIPT generates it -- not just "the file does not exist", which forces you to go look in the code
    for who produces it.
    """
    path = idata_path(name, dataset_label, slice_=slice_)
    if not path.exists():
        producer = IDATA_REGISTRY[name]["producer"]
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            f"Generate it by running: {producer}\n"
            f"(with the current dataset -- dataset_label='{dataset_label}', "
            f"see metadata.dataset_label in synthetic_data/true_parameters.json)"
        )
    return path
