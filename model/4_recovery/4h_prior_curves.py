"""The response curves the priors imply, channel by channel, sampled from the prior. It is one of the
expensive figures: sampling the prior is almost all of the time.

Produces: model/4_recovery/plots/4h1_prior_curves__<dataset_label>.png
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.mmm_dgp_config import adstock_mode  # noqa: E402
from config.diagnostics import SAMPLING_SEED, N_PRIOR_SAMPLES, FIGURE_SUBSAMPLE_SEED, N_DRAWS_BANDS  # noqa: E402
from config.mmm_dgp_config import FOURIER_MU_PRIOR  # noqa: E402
from config.contracts import require_regime  # noqa: E402
require_regime("log", __file__)  # this script belongs to the log regime: with another dataset it stops here
from config.mmm_dgp_config import (L_MAX, CHANNEL_COLUMNS, adstock_alpha_prior_arrays, ADSTOCK_NORMALIZE,
                                   saturation_lam_prior, intercept_prior, check_prior_scale,
                                   YEARLY_SEASONALITY)
from config.contracts import contract
from config.idata_paths import require_idata
from config.log_regime import fit_gamma_control, fit_priors
from synthetic_data.adstock_dgp import geometric_adstock_aligned
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.canonical import validate_recovery
from viz.intervals import pointwise_hdi
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, PRIOR_COLOR, apply_style, channel_color, channel_short_name
from viz.labels import PRIOR, TRUTH, TRUE_VALUE, CONTRACTION_SHORT, ADSTOCKED_IMPRESSIONS, LAG_WEEKS
from viz.io import figure_path, save_figure

apply_style()
SEED = SAMPLING_SEED
CONTROL_COLUMNS = ["log_price_c", "promo_flag", "log_acv_c", "t_norm_c"]

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
table = validate_recovery(idata, geo="Geo_Centro", trade="Retail").set_index("parameter")
channel_scale = {
    str(c): float(v) for c, v in zip(idata.constant_data["channel_scale"].coords["channel"].values,
                                      idata.constant_data["channel_scale"].values)
}


def build_mmm():
    """The 3a_ model, rebuilt to sample its prior. Every prior reads config/ (fit_priors and
    fit_gamma_control), and check_prior_scale() checks them against their scale as in 3a_.
    """
    df = pd.read_csv(ROOT / "synthetic_data" / "dataset.csv", parse_dates=["date"])
    df_model = (
        df.query("geo == 'Geo_Centro' and trade_channel == 'Retail'")
        .sort_values("date").reset_index(drop=True)
    )
    df_model["log_price"] = np.log(df_model["price"])
    df_model["log_acv"] = np.log(df_model["acv"])
    df_model["log_price_c"] = df_model["log_price"] - df_model["log_price"].mean()
    df_model["log_acv_c"] = df_model["log_acv"] - df_model["log_acv"].mean()
    N = len(df_model)
    t_norm = np.arange(N) / N
    df_model["t_norm_c"] = t_norm - t_norm.mean()
    df_model["y_log_sales"] = np.log(df_model["sales"])
    X = df_model[["date"] + CHANNEL_COLUMNS + CONTROL_COLUMNS]
    y = df_model["y_log_sales"]
    TS = float(np.maximum(np.abs(y).max(), 1.0))
    pri = fit_priors(TS, CHANNEL_COLUMNS)   # the priors of 3a_, from config/

    _alphas, _betas = adstock_alpha_prior_arrays(CHANNEL_COLUMNS)
    _alpha_a = xr.DataArray(_alphas, dims="channel", coords={"channel": CHANNEL_COLUMNS})
    _alpha_b = xr.DataArray(_betas, dims="channel", coords={"channel": CHANNEL_COLUMNS})
    adstock = GeometricAdstock(l_max=L_MAX, mode=adstock_mode(), priors={"alpha": Prior("Beta", alpha=_alpha_a, beta=_alpha_b, dims="channel")})
    beta_sigma = xr.DataArray(pri["beta_sigma"], dims="channel", coords={"channel": CHANNEL_COLUMNS})
    saturation = LogisticSaturation(priors={
        "lam": saturation_lam_prior(),   # one single place in config/
        "beta": Prior("HalfNormal", sigma=beta_sigma, dims="channel"),
    })
    _mu, _sg, _lo, _hi = fit_gamma_control(TS, CONTROL_COLUMNS)   # the same priors as 3a_, from config/
    gamma_mu = xr.DataArray(_mu, dims="control", coords={"control": CONTROL_COLUMNS})
    gamma_sigma = xr.DataArray(_sg, dims="control", coords={"control": CONTROL_COLUMNS})
    gamma_lower = xr.DataArray(_lo, dims="control", coords={"control": CONTROL_COLUMNS})
    gamma_upper = xr.DataArray(_hi, dims="control", coords={"control": CONTROL_COLUMNS})
    gamma_control_prior = Prior("TruncatedNormal", mu=gamma_mu, sigma=gamma_sigma, lower=gamma_lower, upper=gamma_upper, dims="control")
    likelihood_prior = Prior("Normal", sigma=Prior("HalfNormal", sigma=pri["likelihood_sigma"], dims=()), dims=("date",))
    gamma_fourier_prior = Prior("Laplace", mu=FOURIER_MU_PRIOR, b=pri["fourier_b"], dims="fourier_mode")
    mmm = MMM(
        date_column="date", channel_columns=CHANNEL_COLUMNS, control_columns=CONTROL_COLUMNS,
        adstock=adstock, saturation=saturation, yearly_seasonality=YEARLY_SEASONALITY, time_varying_intercept=False,
        model_config={
            "intercept": intercept_prior(),
            "gamma_control": gamma_control_prior, "gamma_fourier": gamma_fourier_prior,
            "likelihood": likelihood_prior,
        },
    )
    mmm.build_model(X, y)
    check_prior_scale(mmm, contract("log"), {"target_scale": TS})   # the same check as 3a_
    return mmm, X, y


mmm_prior, X, y = build_mmm()
mmm_prior.sample_prior_predictive(X, y, samples=N_PRIOR_SAMPLES, random_seed=SEED, extend_idata=True)
prior = mmm_prior.idata.prior

df = pd.read_csv(ROOT / "synthetic_data" / "dataset.csv", parse_dates=["date"])
df_model = df.query("geo == 'Geo_Centro' and trade_channel == 'Retail'").sort_values("date")

lags = np.arange(L_MAX)
rng = np.random.default_rng(FIGURE_SUBSAMPLE_SEED)
N_SUB = N_DRAWS_BANDS


def saturation_fn(x_scaled, lam, beta):
    return beta * (1 - np.exp(-lam * x_scaled)) / (1 + np.exp(-lam * x_scaled))


def weights(alpha):
    w = alpha ** lags
    return w / w.sum()


fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))  # cosmetic

for col, ch_col in enumerate(CHANNEL_COLUMNS):
    ch = channel_short_name(ch_col)
    cscale = channel_scale[ch_col]

    # --- row 0: saturation from the prior -
    ax = axes[0, col]
    alpha_mean_post = float(table.loc[f"adstock_alpha[{ch}]", "posterior_mean"])
    post_adstock = geometric_adstock_aligned(df_model[ch_col].values, alpha_mean_post, l_max=L_MAX, normalize=ADSTOCK_NORMALIZE)
    pmax = post_adstock.max()
    x_grid_raw = np.linspace(0, pmax * 1.15, 200)  # cosmetic
    x_grid_scaled = x_grid_raw / cscale

    lam_prior = prior["saturation_lam"].sel(channel=ch_col).values.ravel()
    beta_prior = prior["saturation_beta"].sel(channel=ch_col).values.ravel()
    idx = rng.choice(len(lam_prior), size=min(N_SUB, len(lam_prior)), replace=False)
    curves = np.array([saturation_fn(x_grid_scaled, lam_prior[i], beta_prior[i]) for i in idx])
    curve_lo, curve_hi = pointwise_hdi(curves)
    for i in idx[:80]:  # cloud of individual curves, besides the band
        ax.plot(x_grid_raw, saturation_fn(x_grid_scaled, lam_prior[i], beta_prior[i]), color=PRIOR_COLOR, lw=0.4, alpha=0.15)  # cosmetic
    ax.fill_between(x_grid_raw, curve_lo, curve_hi, color=PRIOR_COLOR, alpha=0.15)  # cosmetic

    true_lam = float(table.loc[f"saturation_lam[{ch}]", "true_value"])
    true_beta = float(table.loc[f"saturation_beta[{ch}] (L)", "true_value"])
    ax.plot(x_grid_raw, saturation_fn(x_grid_scaled, true_lam, true_beta), color=INK, lw=1.8, ls="--", label="true curve")  # cosmetic
    ax.set_title(f"{ch} — saturation ({PRIOR})", fontsize=10, color=INK)  # cosmetic
    ax.set_xlabel(ADSTOCKED_IMPRESSIONS)
    if col == 0:
        ax.set_ylabel(f"contribution ({PRIOR})")
        ax.legend(fontsize=7.5, frameon=False)  # cosmetic
    ax.grid(lw=0.5, color=GRID)  # cosmetic
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    # --- row 1: adstock decay from the prior -
    ax2 = axes[1, col]
    alpha_prior_draws = prior["adstock_alpha"].sel(channel=ch_col).values.ravel()
    idx2 = rng.choice(len(alpha_prior_draws), size=min(N_SUB, len(alpha_prior_draws)), replace=False)
    for i in idx2[:80]:
        ax2.plot(lags, weights(alpha_prior_draws[i]), color=channel_color(ch_col), lw=0.5, alpha=0.15)  # cosmetic
    w_draws = np.array([weights(a) for a in alpha_prior_draws[idx2]])
    w_lo, w_hi = pointwise_hdi(w_draws)
    ax2.fill_between(lags, w_lo, w_hi, color=channel_color(ch_col), alpha=0.20)  # cosmetic

    alpha_true = float(table.loc[f"adstock_alpha[{ch}]", "true_value"])
    ax2.plot(lags, weights(alpha_true), color=INK, lw=1.8, ls="--", marker="x", markersize=5, label=TRUE_VALUE)  # cosmetic
    ax2.set_title(f"{ch} — adstock decay ({PRIOR})", fontsize=10, color=INK)  # cosmetic
    ax2.set_xlabel(LAG_WEEKS)
    if col == 0:
        ax2.set_ylabel(f"normalised weight ({PRIOR})")
    ax2.grid(lw=0.5, color=GRID)  # cosmetic
    for side in ["top", "right"]:
        ax2.spines[side].set_visible(False)

fig.suptitle(
    f"4h1 — Saturation and adstock-decay curves SAMPLED FROM THE {PRIOR.upper()} (before seeing data),\n"
    f"with the {TRUTH} overlaid — compare the width of this cloud against 4f1/4g1 (posterior): that difference is the {CONTRACTION_SHORT}",
    fontsize=12.5, x=0.02, ha="left",  # cosmetic
)
fig.tight_layout(rect=[0, 0, 1, 0.93])  # cosmetic

_path = figure_path(ROOT, "4_recovery", "4h1_prior_curves", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)
print(f"Plot saved to {_path.relative_to(ROOT)}")
