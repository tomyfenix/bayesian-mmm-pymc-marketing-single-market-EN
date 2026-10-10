"""BUSINESS reading primitives on an already fitted posterior. The five figures 5a1-5e1 share the same
counterfactuals, so they live here only once.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.mmm_dgp_config import (CHANNEL_COLUMNS, L_MAX, ADSTOCK_NORMALIZE,
                                   CANONICAL_GEO, CANONICAL_TRADE, HDI_PROB, true_demand,
                                   PRICE_BASE)
from config.budget import N_DRAWS, SWEEP_DRAWS_SEED, MARGINAL_ROAS_EPS

# This whole module exponentiates log-points, which only makes sense in the log regime: in another
# regime it would give numbers without an error. It stops on import, reading the dataset's regime.
from config.contracts import require_regime  # noqa: E402
require_regime("log", "viz/business.py (figures 5a1-5e1)")

ECONOMIC_CONTROLS = ["log_price_c", "promo_flag", "log_acv_c"]


def short_channel(col: str) -> str:
    return col.replace("_impressions_000", "")


# --------------------------------------------------------------------------- data
def canonical_slice() -> pd.DataFrame:
    df = pd.read_csv(_ROOT / "synthetic_data" / "dataset.csv", parse_dates=["date"])
    return (df[(df.geo == CANONICAL_GEO) & (df.trade_channel == CANONICAL_TRADE)]
            .sort_values("date").reset_index(drop=True))


def centering_reference(dfm: pd.DataFrame, column: str) -> float:
    """The value log_<column>_c is centered on. It is the GEOMETRIC mean of the period,
    exp(mean(log(x))) -- NOT the arithmetic one (verified: for price it gives 104.1395 against 105.0680
    for the arithmetic one).
    """
    return float(np.exp(np.log(dfm[column].values).mean()))


# --------------------------------------------------------------- contributions (posterior)
def log_contributions(idata):
    """(chain, draw, date, channel) in LOG-POINTS."""
    return idata.posterior["channel_contribution_original_scale"]


def posterior_sales(idata):
    """Fitted sales in units, per draw: exp(log S)."""
    return np.exp(idata.posterior["y_original_scale"])


def loss_by_channel(idata) -> np.ndarray:
    """Fraction of the period's sales lost when switching EACH channel off separately. Returns
    (n_draws, n_channels).
    """
    c = log_contributions(idata).stack(m=("chain", "draw")).transpose("m", "date", "channel").values
    S = np.exp(idata.posterior["y_original_scale"].stack(m=("chain", "draw"))
               .transpose("m", "date").values)
    tot = S.sum(axis=1, keepdims=True)
    return np.stack([1 - (S * np.exp(-c[:, :, j])).sum(axis=1) / tot[:, 0]
                     for j in range(c.shape[2])], axis=1)


def joint_loss(idata) -> np.ndarray:
    """Fraction lost when switching the FOUR channels off at once. (n_draws,)"""
    c = log_contributions(idata).stack(m=("chain", "draw")).transpose("m", "date", "channel").values
    S = np.exp(idata.posterior["y_original_scale"].stack(m=("chain", "draw"))
               .transpose("m", "date").values)
    return 1 - (S * np.exp(-c.sum(axis=2))).sum(axis=1) / S.sum(axis=1)  # trivial: array axis


def log_components(idata) -> dict:
    """Every additive term of the model, in log-points, per draw and week. The list has to be COMPLETE
    or the change decomposition does not close: 4 channels + 3 economic controls + trend (if present)
    + seasonality + intercept.
    """
    post = idata.posterior
    out = {}
    c = post["channel_contribution_original_scale"]
    for col in CHANNEL_COLUMNS:
        out[short_channel(col)] = c.sel(channel=col)
    ctrl = post["control_contribution_original_scale"]
    for name in [str(x) for x in ctrl.coords["control"].values]:
        out[name] = ctrl.sel(control=name)
    out["seasonality"] = post["yearly_seasonality_contribution_original_scale"]
    total = sum(out.values())
    out["intercept"] = post["y_original_scale"] - total
    return out


def group_trend_seasonality(comp: dict) -> dict:
    """Merges `t_norm_c` and `seasonality` into a single component. WHY: the DGP does NOT have a separate
    trend term -- it puts it INSIDE the seasonal expression, `log(1 + seasonality + trend)`
    (config.mmm_dgp_config.true_demand).
    """
    out = {k: v for k, v in comp.items() if k not in ("t_norm_c", "seasonality")}
    pieces = [comp[k] for k in ("t_norm_c", "seasonality") if k in comp]
    if pieces:
        out["trend + seasonality"] = sum(pieces)
    return out


def log_change_decomposition(idata, mask_0, mask_1) -> pd.DataFrame:
    """EXACT decomposition of the change in log(sales) between two periods. Returns a DataFrame with the
    mean and the HDI94 by component, plus the total and the residual.
    """
    import arviz as az
    comp = group_trend_seasonality(log_components(idata))
    rows = []
    for name, da in comp.items():
        d = (da.isel(date=mask_1).mean("date") - da.isel(date=mask_0).mean("date"))
        v = d.values.ravel()
        lo, hi = az.hdi(v, hdi_prob=HDI_PROB)
        rows.append({"component": name, "delta_log": float(v.mean()),
                     "hdi_lo": float(lo), "hdi_hi": float(hi)})
    y = idata.posterior["y_original_scale"]
    tot = (y.isel(date=mask_1).mean("date") - y.isel(date=mask_0).mean("date")).values.ravel()
    df = pd.DataFrame(rows)
    df.attrs["total_log"] = float(tot.mean())
    df.attrs["residual_log"] = float(tot.mean() - df["delta_log"].sum())
    # In %: g_k = exp(delta_k)-1. The factors MULTIPLY exactly but ADD UP only to first order; the
    # missing part is the cross-effect and it is reported, never silently prorated.
    df["delta_pct"] = np.exp(df["delta_log"]) - 1
    G = np.exp(df.attrs["total_log"]) - 1
    df.attrs["total_pct"] = G
    df.attrs["cross_effect_pct"] = G - df["delta_pct"].sum()
    return df


# ------------------------------------------------------------------- truth of the DGP
def true_log_contributions(dfm: pd.DataFrame) -> dict:
    """True c_k(t) in log-points: L_effective * satfrac(adstock(x, alpha_true), lam_true)."""
    from synthetic_data.true_params_reader import true_contributions
    return true_contributions(dfm, CANONICAL_GEO, CANONICAL_TRADE)


def true_log_components(dfm: pd.DataFrame) -> dict:
    """The same components as log_components(), but from the generator."""
    import json
    TP = json.loads((_ROOT / "synthetic_data" / "true_parameters.json").read_text(encoding="utf-8"))
    tp = TP["trade_params"][CANONICAL_TRADE]
    n = len(dfm)
    t = np.arange(n)
    woy = dfm["date"].dt.isocalendar().week.to_numpy().astype(float)
    out = dict(true_log_contributions(dfm))
    out["log_price_c"] = tp["price_elasticity"] * np.log(dfm["price"].values / PRICE_BASE)
    out["promo_flag"] = np.log(1 + dfm["promo_flag"].values * tp["promo_lift"])
    out["log_acv_c"] = tp["distribution_elasticity"] * np.log(dfm["acv"].values / 100.0)  # trivial: unit
    seasonality, trend = true_demand(woy, t)   # the SAME function the generator uses
    out["seasonality"] = np.log(1 + seasonality + trend)
    out["intercept"] = np.full(n, np.log(tp["base_sales"] * TP["geo_size"][CANONICAL_GEO]))
    out["t_norm_c"] = np.zeros(n)   # the DGP has no separate trend term
    return out


def true_log_change_decomposition(dfm: pd.DataFrame, m0, m1) -> pd.DataFrame:
    comp = group_trend_seasonality(true_log_components(dfm))
    rows = [{"component": k, "delta_log": float(v[m1].mean() - v[m0].mean())}
            for k, v in comp.items()]
    return pd.DataFrame(rows)


def true_loss_by_channel(dfm: pd.DataFrame) -> dict:
    """Same counterfactual definition as loss_by_channel(), on the DGP."""
    truth = true_log_contributions(dfm)
    comp = true_log_components(dfm)
    logS = sum(v for k, v in comp.items() if k != "t_norm_c")
    S = np.exp(logS)
    out = {ch: float(1 - (S * np.exp(-c)).sum() / S.sum()) for ch, c in truth.items()}
    all_channels = sum(truth.values())
    out["_joint"] = float(1 - (S * np.exp(-all_channels)).sum() / S.sum())
    return out


# ------------------------------------------------------- budget sweep and ROAS
def _saturate(x_scaled, lam, beta):
    """The SAME formula as pymc-marketing's LogisticSaturation, the one 4f1 uses. The scale conversion is
    not reimplemented: lam and beta are read from validate_recovery(), which already returns them on
    the scale where this formula is correct (same criterion as 4f1).
    """
    return beta * (1 - np.exp(-lam * x_scaled)) / (1 + np.exp(-lam * x_scaled))


def channel_sweep(idata, dfm, channel_col, multipliers, n_draws=N_DRAWS, seed=SWEEP_DRAWS_SEED):
    """Total revenue of the period when scaling the impressions of ONE channel by each multiplier.
    Returns (mult, revenue[draw, mult]) in monetary units.
    """
    from synthetic_data.adstock_dgp import geometric_adstock_aligned
    import arviz as az
    post = idata.posterior
    ch = short_channel(channel_col)
    cscale = float(idata.constant_data["channel_scale"].sel(channel=channel_col).values)


    # target_scale: without it the rebuilt curve stays on the model's INTERNAL scale and not in
    # log-points, and the ROAS comes out ~12x smaller (it was caught because the posterior ROAS was 1.40
    # against 18.67 for the truth). The check below would catch it on its own.
    tscale = float(idata.constant_data["target_scale"].values)

    rng = np.random.default_rng(seed)
    lam = post["saturation_lam"].sel(channel=channel_col).stack(m=("chain", "draw")).values
    bet = post["saturation_beta"].sel(channel=channel_col).stack(m=("chain", "draw")).values
    c_k = (post["channel_contribution_original_scale"].sel(channel=channel_col)
           .stack(m=("chain", "draw")).transpose("m", "date").values)
    S = np.exp(post["y_original_scale"].stack(m=("chain", "draw")).transpose("m", "date").values)
    alpha = post["adstock_alpha"].sel(channel=channel_col).stack(m=("chain", "draw")).values
    idx = rng.choice(lam.shape[0], size=min(n_draws, lam.shape[0]), replace=False)
    lam, bet, c_k, S, alpha = lam[idx], bet[idx], c_k[idx], S[idx], alpha[idx]

    x_raw = dfm[channel_col].values
    ad = np.stack([geometric_adstock_aligned(x_raw, float(a), l_max=L_MAX,
                                             normalize=ADSTOCK_NORMALIZE) for a in alpha])

    price = dfm["price"].values
    # sales with the channel switched OFF, per draw and week: the base the curve is rebuilt on
    S_without = S * np.exp(-c_k)

    def contribution(m):
        # The adstock is LINEAR: scaling the impressions by m scales the adstocked series by m exactly,
        # with no assumption about the filter's gain.
        x_sc = (ad * m) / cscale
        return _saturate(x_sc, lam[:, None], bet[:, None]) * tscale

    # CHECK: at m=1 the rebuilt curve has to give EXACTLY the contribution the model itself stored. If
    # not, the scale conversion is wrong and everything that follows (ROAS, sweep, 5d1, 5e1) measures
    # something else. It is always verified, never assumed.
    err = np.abs(contribution(1.0) - c_k).max()
    if err > 1e-6 * max(1.0, np.abs(c_k).max()):  # trivial: floating-point tolerance
        raise AssertionError(
            f"channel_sweep({ch}): the rebuilt curve does not reproduce "
            f"channel_contribution_original_scale at m=1 (max|diff|={err:.3e}). "
            "Check the scale conversion before trusting the ROAS or the sweep.")

    out = np.empty((len(idx), len(multipliers)))
    for i_m, m in enumerate(multipliers):
        out[:, i_m] = ((S_without * np.exp(contribution(m))) * price).sum(axis=1)
    return np.asarray(multipliers, dtype=float), out


def average_and_marginal_roas(idata, dfm, channel_col, eps=MARGINAL_ROAS_EPS, n_draws=N_DRAWS, seed=SWEEP_DRAWS_SEED):
    """Average and marginal ROAS of the channel, per draw. average = incremental revenue of having the
    channel on / spend of the channel; marginal = d(revenue)/d(spend) at the observed level, by finite
    difference over the SAME sweep as 5e1 (m = 1 and m = 1+eps).
    """
    ch = short_channel(channel_col)
    spend = float(dfm[f"{ch}_spend"].sum())
    mults, rev = channel_sweep(idata, dfm, channel_col, [0.0, 1.0, 1.0 + eps], n_draws=n_draws,
                               seed=seed)
    rev_0, rev_1, rev_e = rev[:, 0], rev[:, 1], rev[:, 2]
    average = (rev_1 - rev_0) / spend
    marginal = (rev_e - rev_1) / (eps * spend)
    return average, marginal, spend


def true_roas(dfm, channel_col, eps=MARGINAL_ROAS_EPS):
    """The same two ROAS, computed on the DGP."""
    from synthetic_data.adstock_dgp import geometric_adstock_aligned
    from synthetic_data.true_params_reader import get_true_L, get_channel_params
    ch = short_channel(channel_col)
    p = get_channel_params(ch)
    L = get_true_L(ch, CANONICAL_GEO, CANONICAL_TRADE)
    ad = geometric_adstock_aligned(dfm[channel_col].values, p["adstock_alpha"],
                                   l_max=L_MAX, normalize=ADSTOCK_NORMALIZE)
    comp = true_log_components(dfm)
    logS = sum(v for k, v in comp.items() if k != "t_norm_c")
    S_without = np.exp(logS - comp[ch])
    price = dfm["price"].values
    spend = float(dfm[f"{ch}_spend"].sum())

    def revenue(m):
        z = p["lam"] * ad * m
        c = L * (1 - np.exp(-z)) / (1 + np.exp(-z))
        return float(((S_without * np.exp(c)) * price).sum())

    i0, i1, ie = revenue(0.0), revenue(1.0), revenue(1.0 + eps)
    return (i1 - i0) / spend, (ie - i1) / (eps * spend), spend


def true_sweep(dfm, channel_col, multipliers):
    from synthetic_data.adstock_dgp import geometric_adstock_aligned
    from synthetic_data.true_params_reader import get_true_L, get_channel_params
    ch = short_channel(channel_col)
    p = get_channel_params(ch)
    L = get_true_L(ch, CANONICAL_GEO, CANONICAL_TRADE)
    ad = geometric_adstock_aligned(dfm[channel_col].values, p["adstock_alpha"],
                                   l_max=L_MAX, normalize=ADSTOCK_NORMALIZE)
    comp = true_log_components(dfm)
    logS = sum(v for k, v in comp.items() if k != "t_norm_c")
    S_without = np.exp(logS - comp[ch])
    price = dfm["price"].values
    out = []
    for m in multipliers:
        z = p["lam"] * ad * m
        c = L * (1 - np.exp(-z)) / (1 + np.exp(-z))
        out.append(float(((S_without * np.exp(c)) * price).sum()))
    return np.array(out)
