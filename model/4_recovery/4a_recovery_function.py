"""Reusable validation function -- the main tool of the rerun cycle. Run on its own, it prints the
recovery table of a .nc.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synthetic_data.true_params_reader import get_true_L, get_channel_params, get_demand_params, warn_if_idata_stale
from config.diagnostics import FOREST_ERROR_THRESHOLD  # noqa: E402
from config.mmm_dgp_config import CHANNEL_COLUMNS, CANONICAL_GEO, CANONICAL_TRADE, HDI_PROB
from config.contracts import contract, require_regime, dataset_regime, idata_regime


class TargetScaleNotFoundError(RuntimeError):
    """The idata does not carry target_scale/channel_scale in constant_data -- no default value is
    assumed (it FAILS with a clear message instead of assuming).
    """


def _resolve_scales(idata) -> tuple[float, dict[str, float]]:
    """Reads target_scale (scalar) and channel_scale (by channel) FROM THE IDATA ITSELF -- never from a
    constant hardcoded in this module.
    """
    if not hasattr(idata, "constant_data"):
        raise TargetScaleNotFoundError(
            "The idata has no 'constant_data' group. This idata was not produced by "
            "pymc-marketing's MMM.fit()/MMM.sample_prior_predictive() (which record "
            "target_scale and channel_scale as pm.Data automatically), or it was saved without "
            "that group. The target_scale cannot be resolved safely -- "
            "fix the sampling script so that it keeps constant_data on "
            "idata.to_netcdf()."
        )
    cd = idata.constant_data
    if "target_scale" not in cd:
        raise TargetScaleNotFoundError(
            "idata.constant_data has no 'target_scale'. Available variables: "
            f"{list(cd.data_vars)}. No value is assumed."
        )
    target_scale = float(np.asarray(cd["target_scale"]).reshape(-1)[0])

    channel_scale = {}
    if "channel_scale" in cd:
        arr = cd["channel_scale"]
        channels = arr.coords["channel"].values if "channel" in arr.coords else None
        values = np.asarray(arr).reshape(-1)
        if channels is None:
            raise TargetScaleNotFoundError(
                "idata.constant_data['channel_scale'] has no 'channel' coordinate -- "
                "each value cannot be tied to its channel without assuming an order."
            )
        channel_scale = {str(c): float(v) for c, v in zip(channels, values)}
    return target_scale, channel_scale


def _channel_name(ch_col: str) -> str:
    return ch_col.replace("_impressions_000", "")


# what changes between regimes -- which controls there are, how their truth is converted and in which
# range the target_scale of a .nc falls -- lives in config/<regime>_regime.py (CONTROLS, control_truth,
# TARGET_SCALE_RANGE). One single validate_recovery.


def _nc_contract(idata):
    """The contract of the regime of the .nc -- READ from its content (target_scale) -- requiring that it
    be the same as the current dataset's, whose truth it is going to be compared against.
    """
    nc_regime = idata_regime(idata)
    require_regime(nc_regime, "validate_recovery (the .nc belongs to the " + nc_regime + " regime)",
                   found=dataset_regime())
    return contract(nc_regime)


def validate_recovery(
    idata,
    geo: str = CANONICAL_GEO,
    trade: str = CANONICAL_TRADE,
    channel_columns: list[str] | None = None,
    control_columns: list[str] | None = None,
    hdi_prob: float = HDI_PROB,
    idata_path_for_staleness_check: Path | str | None = None,
) -> pd.DataFrame:
    """Parameter recovery table, sorted by descending relative bias. It applies the scale conversions
    automatically (lam *= channel_scale, L/gamma/eta/promo_lift /= target_scale), read FROM THE IDATA
    ITSELF -- see _resolve_scales.
    """
    if idata_path_for_staleness_check is not None:
        stale = warn_if_idata_stale(Path(idata_path_for_staleness_check))
        if stale:
            print("(the table below is computed anyway, but with the warning above unresolved)")

    if channel_columns is None:
        channel_columns = CHANNEL_COLUMNS  # single source in config/mmm_dgp_config.py
    C = _nc_contract(idata)   # regime READ from the .nc, and it has to be the dataset's
    if control_columns is None:
        control_columns = list(C.CONTROLS)

    target_scale, channel_scale = _resolve_scales(idata)

    posterior = idata.posterior
    rows = []

    # --- adstock_alpha: dimensionless, no scale conversion ---------
    if "adstock_alpha" in posterior:
        for ch_col in channel_columns:
            ch = _channel_name(ch_col)
            true_params = get_channel_params(ch)
            true_val = true_params["adstock_alpha"]
            draws = posterior["adstock_alpha"].sel(channel=ch_col) if "channel" in posterior["adstock_alpha"].dims else posterior["adstock_alpha"]
            draws = _sel_geo(draws, geo)
            _append_row(rows, f"adstock_alpha[{ch}]", true_val, draws, hdi_prob)

    # --- saturation_lam: lam_posterior = lam_dgp * channel_scale --------
    if "saturation_lam" in posterior:
        for ch_col in channel_columns:
            ch = _channel_name(ch_col)
            true_params = get_channel_params(ch)
            if ch_col not in channel_scale:
                raise TargetScaleNotFoundError(
                    f"channel_scale has no entry for {ch_col!r} -- available: {list(channel_scale)}"
                )
            true_val_scaled = true_params["lam"] * channel_scale[ch_col]
            draws = _sel_geo(posterior["saturation_lam"].sel(channel=ch_col), geo)
            _append_row(rows, f"saturation_lam[{ch}]", true_val_scaled, draws, hdi_prob)

    # --- saturation_beta (L): beta_posterior = L_dgp / target_scale -----
    if "saturation_beta" in posterior:
        for ch_col in channel_columns:
            ch = _channel_name(ch_col)
            true_L = get_true_L(ch, geo, trade)
            true_val_scaled = true_L / target_scale
            draws = _sel_geo(posterior["saturation_beta"].sel(channel=ch_col), geo)
            _append_row(rows, f"saturation_beta[{ch}] (L)", true_val_scaled, draws, hdi_prob)

    # --- gamma_control: gamma_posterior = gamma_dgp / target_scale, EXCEPT promo_flag, whose truth is
    # log(1 + lift) (config.log_regime.control_truth) -------------------------
    if "gamma_control" in posterior:
        demand = get_demand_params(geo, trade)
        control_true = C.control_truth(demand)
        for ctrl in control_columns:
            if ctrl not in control_true:
                continue  # e.g. t_norm_c: it has no closed ground truth in the DGP
            true_val_scaled = control_true[ctrl] / target_scale
            draws = _sel_geo(posterior["gamma_control"].sel(control=ctrl), geo)
            _append_row(rows, f"gamma_control[{ctrl}]", true_val_scaled, draws, hdi_prob)

    df = pd.DataFrame(rows).sort_values("relative_bias_pct", ascending=False, key=abs).reset_index(drop=True)

    # --- check: posterior correlation eta-vs-promo_lift -
    _price_col = C.CONTROLS[0]   # log_price_c
    if "gamma_control" in posterior and _price_col in control_columns and "promo_flag" in control_columns:
        eta_draws = _sel_geo(posterior["gamma_control"].sel(control=_price_col), geo).values.flatten()
        promo_draws = _sel_geo(posterior["gamma_control"].sel(control="promo_flag"), geo).values.flatten()
        corr_posterior = float(np.corrcoef(eta_draws, promo_draws)[0, 1])
        if corr_posterior < FOREST_ERROR_THRESHOLD:
            print(f"\n{'!' * 78}")  # cosmetic
            print(f"ALARM: POSTERIOR correlation eta(price) <-> promo_lift = "
                  f"{corr_posterior:+.3f}, below {FOREST_ERROR_THRESHOLD:+.2f}.")
            print("The design collinearity is biting: the two parameters are "
                  "compensating each other.")
            print(f"{'!' * 78}")  # cosmetic
        else:
            print(f"  [no alarm] posterior corr. eta<->promo = {corr_posterior:+.3f} "
                  f"(alarm if < {FOREST_ERROR_THRESHOLD:+.2f})")

    return df


def _sel_geo(draws, geo: str):
    """Selects a geo if the variable HAS a geo dimension; otherwise, no-op."""
    return draws.sel(geo=geo) if "geo" in getattr(draws, "dims", ()) else draws


def _append_row(rows, name, true_val, draws, hdi_prob):
    draws_flat = np.asarray(draws).flatten()
    post_mean = float(draws_flat.mean())
    rel_bias = 100 * (post_mean - true_val) / true_val if true_val != 0 else np.nan  # trivial: unit
    hdi = az.hdi(draws_flat, hdi_prob=hdi_prob)
    lo, hi = float(hdi[0]), float(hdi[1])
    covers = lo <= true_val <= hi
    rows.append({
        "parameter": name,
        "true_value": true_val,
        "posterior_mean": post_mean,
        "relative_bias_pct": rel_bias,
        f"hdi{int(hdi_prob*100)}_lo": lo,  # trivial: unit
        f"hdi{int(hdi_prob*100)}_hi": hi,  # trivial: unit
        "covers_hdi": covers,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validates an idata against the generator's ground truth.")
    parser.add_argument("idata_path", type=str, help="Path to the .nc to validate.")
    parser.add_argument("--geo", default=CANONICAL_GEO)
    # CANONICAL_TRADE, not the literal: a hand-written label gets out of sync with config/.
    parser.add_argument("--trade", default=CANONICAL_TRADE)
    args = parser.parse_args()

    idata = az.from_netcdf(args.idata_path)
    df = validate_recovery(idata, geo=args.geo, trade=args.trade, idata_path_for_staleness_check=args.idata_path)
    pd.set_option("display.width", 200)  # cosmetic
    pd.set_option("display.max_columns", 20)  # cosmetic
    print("\n" + "=" * 100)  # cosmetic
    print(f"RECOVERY TABLE -- {args.idata_path} (geo={args.geo}, trade={args.trade})")
    print("=" * 100)  # cosmetic
    print(df.round(4).to_string(index=False))  # cosmetic
    n_fail = (~df["covers_hdi"]).sum()
    print(f"\n{n_fail}/{len(df)} parameters with the true value OUTSIDE the HDI.")
