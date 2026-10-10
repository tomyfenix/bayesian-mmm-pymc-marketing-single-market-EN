"""Contraction against bias, in four quadrants: which parameters the model learned from the data and
which stayed at the prior. It samples the prior, which is almost all of its run time.

Produces: model/4_recovery/plots/4e1_contraction_vs_bias__<dataset_label>.png
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
from config.diagnostics import SAMPLING_SEED, N_PRIOR_SAMPLES, QUADRANT_BIAS_THRESHOLD  # noqa: E402
from config.mmm_dgp_config import FOURIER_MU_PRIOR  # noqa: E402
from config.contracts import require_regime  # noqa: E402
require_regime("log", __file__)  # this script belongs to the log regime: with another dataset it stops here
from config.mmm_dgp_config import (L_MAX, CHANNEL_COLUMNS, adstock_alpha_prior_arrays, contraction_sd,
                                   saturation_lam_prior, intercept_prior, check_prior_scale,
                                   YEARLY_SEASONALITY)
from config.contracts import contract
from config.idata_paths import require_idata
from config.log_regime import fit_gamma_control, fit_priors
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.canonical import validate_recovery
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, MISS_COLOR, apply_style, channel_color, channel_short_name
from viz.labels import BIAS, REL_BIAS, CONTRACTION, CHANNEL, MISSES_HDI
from viz.io import figure_path, save_figure

apply_style()
SEED = SAMPLING_SEED
CONTROL_COLUMNS = ["log_price_c", "promo_flag", "log_acv_c", "t_norm_c"]

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
post = idata.posterior


def build_mmm():
    """The 3a_ model, rebuilt only to sample its prior and compute contraction_sd(). Every prior reads
    config/ (fit_priors and fit_gamma_control), and check_prior_scale() checks them against their scale
    as in 3a_.
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

table = validate_recovery(idata, geo="Geo_Centro", trade="Retail")
table = table.set_index("parameter")

GRID_ORDER = (
    [f"adstock_alpha[{c.replace('_impressions_000', '')}]" for c in CHANNEL_COLUMNS]
    + [f"saturation_lam[{c.replace('_impressions_000', '')}]" for c in CHANNEL_COLUMNS]
    + [f"saturation_beta[{c.replace('_impressions_000', '')}] (L)" for c in CHANNEL_COLUMNS]
    + ["gamma_control[log_price_c]", "gamma_control[promo_flag]", "gamma_control[log_acv_c]"]
)
SHORT_LABEL = {
    "gamma_control[log_price_c]": "eta", "gamma_control[promo_flag]": "promo", "gamma_control[log_acv_c]": "gamma(ACV)",
}


def _parse(name):
    base = name.split("[")[0]
    inner = name.split("[", 1)[1].split("]")[0]
    coord = f"{inner}_impressions_000" if base in ("adstock_alpha", "saturation_lam", "saturation_beta") else inner
    return base, coord


fig, ax = plt.subplots(figsize=(9, 8))  # cosmetic
BIAS_THRESHOLD = QUADRANT_BIAS_THRESHOLD
ax.axvline(BIAS_THRESHOLD, color=INK_MUTED, lw=1.0, ls=":")

xs_plot, ys_plot = [], []
for name in GRID_ORDER:
    base, coord = _parse(name)
    dim = "control" if base == "gamma_control" else "channel"
    prior_vals = prior[base].sel({dim: coord}).values.ravel()
    post_vals = post[base].sel({dim: coord}).values.ravel()
    contr = contraction_sd(prior_vals, post_vals) * 100  # trivial: unit
    r = table.loc[name]
    abs_bias = abs(float(r["relative_bias_pct"]))
    covers = bool(r["covers_hdi"])
    ch = next((c for c in CHANNEL_COLUMNS if channel_short_name(c) in name), None)
    color = channel_color(ch) if ch else INK_SECONDARY
    marker = "o" if covers else "X"
    ax.scatter(abs_bias, contr, s=90, color=color, marker=marker, edgecolor=(MISS_COLOR if not covers else "white"),  # cosmetic
               linewidth=1.6 if not covers else 0.8, zorder=3)  # cosmetic
    fam_short = {"adstock_alpha": "α", "saturation_lam": "λ", "saturation_beta": "L"}.get(base, "")
    ax.annotate(f"{fam_short}[{channel_short_name(coord) if dim=='channel' else SHORT_LABEL.get(name, coord)}]",
                (abs_bias, contr), fontsize=7.5, color=INK_SECONDARY, xytext=(4, 4), textcoords="offset points")  # cosmetic
    xs_plot.append(abs_bias)
    ys_plot.append(contr)

ax.set_xlabel(f"|{REL_BIAS}| (%)")
ax.set_ylabel(f"{CONTRACTION}, SD convention (%)")
ax.set_title(
    f"4e1 — Contraction (SD) vs. |{BIAS}| — quadrants of the 2×2 cross\n"
    f"dotted line: the {BIAS} threshold ({BIAS_THRESHOLD:.0f}%); ✕ = {MISSES_HDI}, "
    f"○ = covers it; colour = {CHANNEL}",
    fontsize=10.5, loc="left",  # cosmetic
)
ax.grid(lw=0.5, color=GRID)  # cosmetic
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()

_path = figure_path(ROOT, "4_recovery", "4e1_contraction_vs_bias", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)
print(f"Plot saved to {_path.relative_to(ROOT)}")
