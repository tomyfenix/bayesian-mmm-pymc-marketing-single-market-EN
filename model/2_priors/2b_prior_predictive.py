"""Prior predictive check: target range and variance decomposition implied by the priors of the model
that is fitted, before seeing a single data point.

Produces: model/2_priors/plots/2b1_target_range.png,
    plots/2b2_variance_by_source.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.mmm_dgp_config import adstock_mode  # noqa: E402
from config.diagnostics import (SAMPLING_SEED, N_PRIOR_SAMPLES, PRIOR_PREDICTIVE_INNER_BAND,  # noqa: E402
                                REFERENCE_VARIANCE_PROFILE, VARIANCE_DOMINANCE_THRESHOLD, TREND_GROWTH_THRESHOLDS)
from config.mmm_dgp_config import FOURIER_MU_PRIOR  # noqa: E402
from config.contracts import require_regime  # noqa: E402
require_regime("log", __file__)  # this script belongs to the log regime: with another dataset it stops here
from config.mmm_dgp_config import (L_MAX, CHANNEL_COLUMNS, adstock_alpha_prior_arrays, saturation_lam_prior,
                                   intercept_prior, check_prior_scale, YEARLY_SEASONALITY)
from config.contracts import contract
from viz.io import save_figure  # noqa: E402 (the provenance stamp)
from config.log_regime import fit_gamma_control, fit_priors  # noqa: E402
from viz.intervals import hdi, pointwise_hdi  # noqa: E402
from viz.labels import HDI  # noqa: E402

# defensive mkdir: plots/ is not versioned.
(ROOT / "model/2_priors/plots").mkdir(parents=True, exist_ok=True)

SEED = SAMPLING_SEED

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
GOLD = "#eda100"
PURPLE = "#8558c9"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRID,
    "font.family": "sans-serif",
    "font.size": 10,  # cosmetic
})

# ---------------------------------------------------------------------------
# 1. Data
df = pd.read_csv(ROOT / "synthetic_data" / "dataset.csv", parse_dates=["date"])
df_model = (
    df.query("geo == 'Geo_Centro' and trade_channel == 'Retail'")
    .sort_values("date")
    .reset_index(drop=True)
)

df_model["log_price"] = np.log(df_model["price"])
df_model["log_acv"] = np.log(df_model["acv"])
df_model["log_price_c"] = df_model["log_price"] - df_model["log_price"].mean()
df_model["log_acv_c"] = df_model["log_acv"] - df_model["log_acv"].mean()
N = len(df_model)
t_norm = np.arange(N) / N
df_model["t_norm_c"] = t_norm - t_norm.mean()
df_model["y_log_sales"] = np.log(df_model["sales"])

channel_columns = CHANNEL_COLUMNS  # single source in config/mmm_dgp_config.py
dates = df_model["date"]
y = df_model["y_log_sales"]

TS = float(np.maximum(np.abs(y).max(), 1.0))
print(f"target_scale_ = {TS:.4f}")
pri = fit_priors(TS, channel_columns)   # the priors of 3a_, from config/


# ---------------------------------------------------------------------------
# 2. LINEAR model: identical to 1a_
def build_mmm_linear():
    control_columns = ["log_price_c", "promo_flag", "log_acv_c", "t_norm_c"]
    X = df_model[["date"] + channel_columns + control_columns]

    # alpha by channel group, from config/
    _alphas, _betas = adstock_alpha_prior_arrays(channel_columns)
    _alpha_prior_alpha = xr.DataArray(_alphas, dims="channel", coords={"channel": channel_columns})
    _alpha_prior_beta = xr.DataArray(_betas, dims="channel", coords={"channel": channel_columns})
    adstock = GeometricAdstock(l_max=L_MAX, mode=adstock_mode(), priors={"alpha": Prior("Beta", alpha=_alpha_prior_alpha, beta=_alpha_prior_beta, dims="channel")})
    saturation = LogisticSaturation(
        priors={
            "lam": saturation_lam_prior(),  # the prior lives in config/; see the Priors section of the README
            "beta": Prior(
                "HalfNormal",  # array from config/; see the Priors section of the README
                sigma=xr.DataArray(pri["beta_sigma"], dims="channel", coords={"channel": channel_columns}),
                dims="channel",
            ),
        }
    )
    _mu, _sg, _lo, _hi = fit_gamma_control(TS, control_columns)   # the same priors as 3a_, from config/
    gamma_mu = xr.DataArray(_mu, dims="control", coords={"control": control_columns})
    gamma_sigma = xr.DataArray(_sg, dims="control", coords={"control": control_columns})
    gamma_lower = xr.DataArray(_lo, dims="control", coords={"control": control_columns})
    gamma_upper = xr.DataArray(_hi, dims="control", coords={"control": control_columns})
    gamma_control_prior = Prior(
        "TruncatedNormal", mu=gamma_mu, sigma=gamma_sigma, lower=gamma_lower, upper=gamma_upper, dims="control"
    )
    likelihood_prior = Prior("Normal", sigma=Prior("HalfNormal", sigma=pri["likelihood_sigma"], dims=()), dims=("date",))
    gamma_fourier_prior = Prior("Laplace", mu=FOURIER_MU_PRIOR, b=pri["fourier_b"], dims="fourier_mode")

    mmm = MMM(
        date_column="date",
        channel_columns=channel_columns,
        control_columns=control_columns,
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=YEARLY_SEASONALITY,
        time_varying_intercept=False,
        model_config={
            "intercept": intercept_prior(),
            "gamma_control": gamma_control_prior,
            "gamma_fourier": gamma_fourier_prior,
            "likelihood": likelihood_prior,
        },
    )
    mmm.build_model(X, y)
    check_prior_scale(mmm, contract("log"), {"target_scale": TS})   # the same check as 3a_
    return mmm, X, control_columns


print("\nSampling the prior predictive -- LINEAR model...")
mmm_lin, X_lin, control_columns_lin = build_mmm_linear()
mmm_lin.sample_prior_predictive(X_lin, y, samples=N_PRIOR_SAMPLES, random_seed=SEED)
idata_lin = mmm_lin.idata


# ===========================================================================
# STEP 1: range of the simulated target vs the real one
print("\n" + "=" * 80)  # cosmetic
# the real range is DERIVED from the target: a hand-written range from an earlier dataset would keep
# lying without anything flagging it.
print(f"STEP 1: range of the simulated target (prior) vs real target [{float(y.min()):.4f}, {float(y.max()):.4f}]")
print("=" * 80)  # cosmetic

y_prior_lin = idata_lin.prior["y_original_scale"].stack(sample=("chain", "draw"))

_lo, _hi = hdi(y_prior_lin.values)
print(f"  prior {HDI}, all weeks pooled: [{_lo:.2f}, {_hi:.2f}]")
print(f"  prior min-max:[{float(y_prior_lin.min()):.2f}, {float(y_prior_lin.max()):.2f}]")

_by_week = y_prior_lin.transpose("sample", ...).values      # (draws, weeks)
ext_lo, ext_hi = pointwise_hdi(_by_week)
int_lo, int_hi = pointwise_hdi(_by_week, PRIOR_PREDICTIVE_INNER_BAND)
_inner = f"HDI{round(100 * PRIOR_PREDICTIVE_INNER_BAND)}"  # trivial: unit
fig, ax = plt.subplots(figsize=(10, 5))  # cosmetic
ax.fill_between(dates, ext_lo, ext_hi, color=BLUE, alpha=0.15, label=f"prior {HDI}")  # cosmetic
ax.fill_between(dates, int_lo, int_hi, color=BLUE, alpha=0.30, label=f"prior {_inner}")  # cosmetic
ax.plot(dates, y, color=INK, lw=1.6, label="observed target")  # cosmetic
ax.set_ylabel("log(sales)")
ax.set_title(f"Prior predictive of the linear model: {HDI} and {_inner} by week, against the observed target",
             loc="left", fontsize=11)  # cosmetic
ax.legend(frameon=False, fontsize=8, loc="upper left")  # cosmetic
ax.grid(axis="y", lw=0.6)  # cosmetic
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
save_figure(fig, ROOT / "model/2_priors/plots/2b1_target_range.png",
            script_file=__file__, idata_name=None)   # stamp; this figure reads no .nc

# ===========================================================================
# STEP 2: WITHIN-DRAW variance decomposition, 7 sources (LINEAR model)
print("\n" + "=" * 80)  # cosmetic
print("STEP 2: WITHIN-DRAW variance decomposition by source -- 7 blocks (LINEAR model)")
print("=" * 80)  # cosmetic

control_contrib_lin = idata_lin.prior["control_contribution"].stack(sample=("chain", "draw"))
price_c = control_contrib_lin.sel(control="log_price_c")
promo_c = control_contrib_lin.sel(control="promo_flag")
acv_c = control_contrib_lin.sel(control="log_acv_c")
trend_c = control_contrib_lin.sel(control="t_norm_c")
channel_sum_lin = idata_lin.prior["channel_contribution"].stack(sample=("chain", "draw")).sum(dim="channel")
fourier_lin = idata_lin.prior["yearly_seasonality_contribution"].stack(sample=("chain", "draw"))
mu_lin = idata_lin.prior["mu"].stack(sample=("chain", "draw"))
y_lin_scaled = idata_lin.prior_predictive["y"].stack(sample=("chain", "draw"))
noise_lin = y_lin_scaled - mu_lin

blocks_lin = {
    "price": price_c,
    "ACV": acv_c,
    "promo": promo_c,
    "trend": trend_c,
    "seasonal": fourier_lin,
    "noise": noise_lin,
    "media": channel_sum_lin,
}

intra_draw_vars_lin = {}
for name, arr in blocks_lin.items():
    intra_draw_vars_lin[name] = float(arr.var(dim="date").mean(dim="sample"))

total_intra_lin = sum(intra_draw_vars_lin.values())

# Hand-written reference profile, from outside this repo: see config/diagnostics.py
real_profile = REFERENCE_VARIANCE_PROFILE

print(f"\n  {'Source':12s} {'Within-draw var':>15s} {'% model':>10s} {'% real (OLS)':>13s}")
sorted_blocks = sorted(intra_draw_vars_lin.items(), key=lambda kv: -kv[1])
for name, v in sorted_blocks:
    pct_model = 100 * v / total_intra_lin  # trivial: unit
    pct_real = real_profile.get(name, float("nan"))
    print(f"  {name:12s} {v:15.6f} {pct_model:9.1f}% {pct_real:12.1f}%")

max_src, min_src = max(intra_draw_vars_lin, key=intra_draw_vars_lin.get), min(intra_draw_vars_lin, key=intra_draw_vars_lin.get)
ratio = intra_draw_vars_lin[max_src] / intra_draw_vars_lin[min_src]
print(f"\n  Dominant source: {max_src} ({100*intra_draw_vars_lin[max_src]/total_intra_lin:.1f}%)")  # trivial: unit
print(f"  Smallest source: {min_src} ({100*intra_draw_vars_lin[min_src]/total_intra_lin:.1f}%)")  # trivial: unit
print(f"  Max/min ratio: {ratio:.2f}x "
      f"({'none dominates by more than 1 order of magnitude -- OK' if ratio < VARIANCE_DOMINANCE_THRESHOLD else 'ALERT: one source dominates by more than 10x'})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))  # cosmetic
ax = axes[0]
names_sorted = [n for n, _ in sorted_blocks]
vals_model = [100 * intra_draw_vars_lin[n] / total_intra_lin for n in names_sorted]  # trivial: unit
vals_real = [real_profile[n] for n in names_sorted]
y_pos = np.arange(len(names_sorted))
width = 0.38  # cosmetic
ax.barh(y_pos - width/2, vals_model, height=width, color=BLUE, label="Prior (linear model)")  # cosmetic
ax.barh(y_pos + width/2, vals_real, height=width, color=ORANGE, label="Real (OLS reconstruction, external reference)")  # cosmetic
ax.set_yticks(y_pos)
ax.set_yticklabels(names_sorted, fontsize=9)  # cosmetic
ax.set_xlabel("% of within-draw variance")
ax.set_title("Variance decomposition -- LINEAR model vs real profile", loc="left", fontsize=10)  # cosmetic
ax.legend(frameon=False, fontsize=8)  # cosmetic
ax.grid(axis="x", lw=0.6)  # cosmetic
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax = axes[1]
colors_map = {"price": BLUE, "ACV": ORANGE, "promo": AQUA, "trend": GOLD,
              "seasonal": PURPLE, "noise": RED, "media": INK_MUTED}
ax.pie(
    [intra_draw_vars_lin[n] for n in names_sorted],
    labels=names_sorted,
    colors=[colors_map[n] for n in names_sorted],
    autopct="%1.1f%%",
    textprops={"fontsize": 9},  # cosmetic
)
ax.set_title("Composition of the prior (LINEAR model)", loc="left", fontsize=10)  # cosmetic
fig.tight_layout()
save_figure(fig, ROOT / "model/2_priors/plots/2b2_variance_by_source.png",
            script_file=__file__, idata_name=None)   # stamp; this figure reads no .nc

# ===========================================================================
# STEP 3: percentiles of gamma_control[t_norm] in % of cumulative growth
print("\n" + "=" * 80)  # cosmetic
print("STEP 3: trend prior in % of cumulative growth over 156 weeks")
print("=" * 80)  # cosmetic
trend_raw = np.asarray(idata_lin.prior["gamma_control"].sel(control="t_norm_c")).ravel()
trend_growth_pct = (np.exp(trend_raw * TS) - 1) * 100  # trivial: unit
p5, p50, p95 = np.percentile(trend_growth_pct, [5, 50, 95])  # cosmetic
print(f"  % cumulative growth: p5={p5:.1f}%  p50={p50:.1f}%  p95={p95:.1f}%  "
      f"(declared goal: cover -20% to +40% with room to spare)")
pct_below_m20 = (trend_growth_pct < TREND_GROWTH_THRESHOLDS[0]).mean() * 100  # trivial: unit
pct_above_40 = (trend_growth_pct > TREND_GROWTH_THRESHOLDS[1]).mean() * 100  # trivial: unit
print(f"  % of draws < -20%: {pct_below_m20:.1f}%  |  % of draws > +40%: {pct_above_40:.1f}%")

print("\nDone. Plots saved in model/2_priors/plots/2b1, 2b2.")
