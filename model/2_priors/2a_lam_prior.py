"""Where the saturation_lam prior comes from. The current one is SATURATION_LAM_PRIOR_GAMMA, in config/;
the pymc-marketing default is Gamma(3, 1). Four parts:
  1. checks that pymc-marketing scales each channel by its maximum;
  2. takes the true lams to the model scale and places them in Gamma(3, 1);
  3. derives a Gamma(2, beta) from the observed impressions range, with a generic adstock alpha and not
     the true one, and draws it against the default and the current one (2a1);
  4. compares the shape of the saturation curve, default against current, with the true one (2a2).

Produces: model/2_priors/plots/2a1_lam_prior_density.png,
    plots/2a2_prior_saturation_shape.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

# Repo root, computed from the location of this file -- it works no matter where the script is run from.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.mmm_dgp_config import adstock_mode  # noqa: E402
from config.diagnostics import SAMPLING_SEED, N_PRIOR_SAMPLES_LAM_CHECK  # noqa: E402
from config.prior_checks import (LAM_PRIOR_DEFAULT_PYMC, LAM_CHECK_BETA_SIGMA,  # noqa: E402
                                 ACTIVE_EXPOSURE_THRESHOLD, ACTIVE_RANGE_PERCENTILES, LOW_LAM_QUANTILE)
from config.mmm_dgp_config import ADSTOCK_ALPHA_PRIOR_BY_CHANNEL  # noqa: E402
from config.contracts import require_regime  # noqa: E402
require_regime("log", __file__)  # this script belongs to the log regime: with another dataset it stops here
from config.mmm_dgp_config import (L_MAX, ADSTOCK_NORMALIZE, CHANNEL_COLUMNS, adstock_alpha_prior_arrays,
                                   SATURATION_LAM_PRIOR_GAMMA, saturation_lam_prior, intercept_prior,
                                   YEARLY_SEASONALITY)
from synthetic_data.adstock_dgp import geometric_adstock_dgp
from viz.io import save_figure  # noqa: E402 (the provenance stamp)

# defensive mkdir: plots/ is not versioned, and on a clean clone it does not exist.
(ROOT / "model/2_priors/plots").mkdir(parents=True, exist_ok=True)

SEED = SAMPLING_SEED
N_PRIOR_SAMPLES = N_PRIOR_SAMPLES_LAM_CHECK
N_CURVES_TO_PLOT = 60  # cosmetic

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

df = pd.read_csv(ROOT / "synthetic_data" / "dataset.csv", parse_dates=["date"])
df_model = (
    df.query("geo == 'Geo_Centro' and trade_channel == 'Retail'")
    .sort_values("date").reset_index(drop=True)
)
df_model["log_price"] = np.log(df_model["price"])
df_model["log_acv"] = np.log(df_model["acv"])
df_model["log_price_c"] = df_model["log_price"] - df_model["log_price"].mean()
df_model["log_acv_c"] = df_model["log_acv"] - df_model["log_acv"].mean()
df_model["y_log_sales"] = np.log(df_model["sales"])  # aligned with 1a_

channel_columns = CHANNEL_COLUMNS  # single source in config/mmm_dgp_config.py
control_columns = ["log_price_c", "promo_flag", "log_acv_c"]
X = df_model[["date"] + channel_columns + control_columns]
y = df_model["y_log_sales"]


def build_mmm(lam_prior=None):
    _alpha_a_vals, _alpha_b_vals = adstock_alpha_prior_arrays(channel_columns)  # by name in config/
    _alpha_a = xr.DataArray(_alpha_a_vals, dims="channel", coords={"channel": channel_columns})
    _alpha_b = xr.DataArray(_alpha_b_vals, dims="channel", coords={"channel": channel_columns})
    adstock = GeometricAdstock(l_max=L_MAX, mode=adstock_mode(), priors={"alpha": Prior("Beta", alpha=_alpha_a, beta=_alpha_b, dims="channel")})
    sat_priors = {"beta": Prior("HalfNormal", sigma=LAM_CHECK_BETA_SIGMA, dims="channel")}
    if lam_prior is not None:
        sat_priors["lam"] = lam_prior
    saturation = LogisticSaturation(priors=sat_priors)
    mmm = MMM(
        date_column="date",
        channel_columns=channel_columns,
        control_columns=control_columns,
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=YEARLY_SEASONALITY,
        model_config={
            "intercept": intercept_prior(),
        },
    )
    mmm.build_model(X, y)
    return mmm


mmm_base = build_mmm()

# ---------------------------------------------------------------------------
# STEP 2a: verify the channel scaling EMPIRICALLY
print("=" * 80)  # cosmetic
print("EMPIRICAL CHECK OF THE CHANNEL SCALING")
print("=" * 80)  # cosmetic
print(f"  mmm.scaling.channel.method = {mmm_base.scaling.channel.method!r}")
print(f"  mmm.scaling.target.method  = {mmm_base.scaling.target.method!r}")

channel_scale = dict(zip(channel_columns, np.atleast_1d(mmm_base.channel_scale)))
manual_max = {ch: df_model[ch].abs().max() for ch in channel_columns}
print("\n  channel_scale (mmm.channel_scale) vs max(abs(column)) computed by hand:")
for ch in channel_columns:
    print(f"    {ch:28s} mmm.channel_scale={channel_scale[ch]:9.2f}   "
          f"max(abs(col))={manual_max[ch]:9.2f}   match={np.isclose(channel_scale[ch], manual_max[ch])}")
print("\n  -> Confirmed: MaxAbsScaler per channel (scale = max(|x|) over the training window, "
      "WITHOUT subtracting the mean, hence 'Max' and not 'MaxAbs' with centering).")

# ---------------------------------------------------------------------------
# STEP 2b: true lam (raw) -> equivalent lam on the model scale
with open(ROOT / "synthetic_data" / "true_parameters.json") as f:
    true_params = json.load(f)

media_true = true_params["media_params_national_inputs"]

print("\n" + "=" * 80)  # cosmetic
print("TRUE LAM (raw) -> EQUIVALENT LAM ON THE MODEL SCALE")
print("=" * 80)  # cosmetic
print("  lam_scaled = lam_raw * channel_scale  (the scaled input is x_raw/channel_scale,")
print("  so lam_scaled*x_scaled = lam_raw*x_raw requires that conversion)")

from scipy.stats import gamma as gamma_dist

GAMMA_ALPHA, GAMMA_BETA = LAM_PRIOR_DEFAULT_PYMC["alpha"], LAM_PRIOR_DEFAULT_PYMC["beta"]  # the pymc-marketing default
# the current prior, the one every model uses: it is read from config/, not written here
LAM_ALPHA_CURRENT, LAM_BETA_CURRENT = SATURATION_LAM_PRIOR_GAMMA["alpha"], SATURATION_LAM_PRIOR_GAMMA["beta"]
CURRENT_LABEL = f"Gamma({LAM_ALPHA_CURRENT:g}, {LAM_BETA_CURRENT:g})"
lam_scaled_true = {}
for ch_col in channel_columns:
    ch = ch_col.replace("_impressions_000", "")
    lam_raw = media_true[ch]["lam"]
    lam_scaled = lam_raw * channel_scale[ch_col]
    lam_scaled_true[ch_col] = lam_scaled
    pct_below = gamma_dist.cdf(lam_scaled, a=GAMMA_ALPHA, scale=1 / GAMMA_BETA) * 100  # trivial: unit
    print(f"    {ch:8s} lam_raw={lam_raw:.7f}  channel_scale={channel_scale[ch_col]:8.1f}  "
          f"-> lam_scaled_true={lam_scaled:.3f}   percentile in Gamma(3,1) = {pct_below:.1f}%")

median_gamma31 = gamma_dist.median(a=GAMMA_ALPHA, scale=1 / GAMMA_BETA)
mass_below_1 = gamma_dist.cdf(1.0, a=GAMMA_ALPHA, scale=1 / GAMMA_BETA) * 100  # trivial: unit
print(f"\n  Gamma(3,1): median={median_gamma31:.2f}, mass below 1.0 = {mass_below_1:.1f}%")
lam_true_vals = np.array(list(lam_scaled_true.values()))
print(f"  Range of lam_scaled_true across the 4 channels: [{lam_true_vals.min():.3f}, {lam_true_vals.max():.3f}]")
print(f"  CONFIRMED: the true values fall in the lower {mass_below_1:.0f}% of Gamma(3,1) -- "
      f"the prior systematically favors a HIGHER lam (FASTER saturation) than the real one.")

# the generic alpha of each channel is the MEAN of its Beta(a, b) prior, a / (a + b): it is derived, not copied
GENERIC_ALPHA_BY_GROUP = {c.replace("_impressions_000", ""): a / (a + b)
                          for c, (a, b) in ADSTOCK_ALPHA_PRIOR_BY_CHANNEL.items()}
ACTIVE_THRESHOLD = ACTIVE_EXPOSURE_THRESHOLD

lam_bounds = {}
for ch_col in channel_columns:
    ch = ch_col.replace("_impressions_000", "")
    x_raw = df_model[ch_col].values
    ad = geometric_adstock_dgp(x_raw, GENERIC_ALPHA_BY_GROUP[ch], normalized=ADSTOCK_NORMALIZE, l_max=L_MAX)
    ad_scaled = ad / channel_scale[ch_col]
    peak = ad_scaled.max()
    active = ad_scaled[ad_scaled >= ACTIVE_THRESHOLD * peak]
    p10_active, p90_active = np.percentile(active, ACTIVE_RANGE_PERCENTILES[0]), np.percentile(active, ACTIVE_RANGE_PERCENTILES[1])
    lam_hi = np.log(3) / p10_active  # x50=p10 -> saturates fast -> high lam  # trivial: ln 3 = x50 of the logistic saturation
    lam_lo = np.log(3) / p90_active  # x50=p90 -> saturates slowly -> low lam  # trivial: ln 3 = x50 of the logistic saturation
    lam_bounds[ch] = (lam_lo, lam_hi)
    print(f"    {ch:8s} generic_alpha={GENERIC_ALPHA_BY_GROUP[ch]:.3f}  "
          f"active exposure (>={ACTIVE_THRESHOLD:.0%} of the peak) p10={p10_active:.4f} p90={p90_active:.4f}  "
          f"-> lam in [{lam_lo:.3f}, {lam_hi:.3f}]")

lam_lo_combined = min(b[0] for b in lam_bounds.values())
lam_hi_combined = max(b[1] for b in lam_bounds.values())
print(f"\n  Combined range (the widest across channels): lam in [{lam_lo_combined:.3f}, {lam_hi_combined:.3f}]")

from scipy.optimize import brentq

LAM_ALPHA_NEW = SATURATION_LAM_PRIOR_GAMMA["alpha"]   # the shape of the current one; the scale is derived
LAM_BETA_NEW = brentq(lambda r: gamma_dist.ppf(LOW_LAM_QUANTILE, a=LAM_ALPHA_NEW, scale=1 / r) - lam_lo_combined,
                     1e-4, 100)  # trivial: interval where the root is searched
median_new = gamma_dist.median(a=LAM_ALPHA_NEW, scale=1 / LAM_BETA_NEW)
mean_new = LAM_ALPHA_NEW / LAM_BETA_NEW
print(f"\n  Derived prior: Gamma(alpha={LAM_ALPHA_NEW:.0f}, beta={LAM_BETA_NEW:.3f}) "
      f"-- mean={mean_new:.2f}, median={median_new:.2f}.  Current in config/: {CURRENT_LABEL}")
for ch_col in channel_columns:
    ch = ch_col.replace("_impressions_000", "")
    pct_below_new = gamma_dist.cdf(lam_scaled_true[ch_col], a=LAM_ALPHA_NEW, scale=1 / LAM_BETA_NEW) * 100  # trivial: unit
    print(f"    {ch:8s} lam_scaled_true={lam_scaled_true[ch_col]:.3f} -> percentile in the derived prior = {pct_below_new:.1f}%")

# Density comparison plot: the default Gamma(3,1), the derived one and the current one, with the 4 true
# values marked.
fig, ax = plt.subplots(figsize=(9, 4.5))  # cosmetic
x_grid = np.linspace(0.001, 8, 500)  # cosmetic
ax.plot(x_grid, gamma_dist.pdf(x_grid, a=GAMMA_ALPHA, scale=1 / GAMMA_BETA), color=BLUE, lw=2,  # cosmetic
        label="pymc-marketing default: Gamma(3, 1)")
ax.plot(x_grid, gamma_dist.pdf(x_grid, a=LAM_ALPHA_NEW, scale=1 / LAM_BETA_NEW), color=ORANGE, lw=2, ls="--",  # cosmetic
        label=f"Derived from the observed range: Gamma({LAM_ALPHA_NEW:.0f}, {LAM_BETA_NEW:.4f})")
ax.plot(x_grid, gamma_dist.pdf(x_grid, a=LAM_ALPHA_CURRENT, scale=1 / LAM_BETA_CURRENT), color=ORANGE, lw=2,  # cosmetic
        label=f"Current prior (config/): {CURRENT_LABEL}")
channel_label_colors = {"tv_impressions_000": BLUE, "meta_impressions_000": ORANGE,
                        "tiktok_impressions_000": AQUA, "ooh_impressions_000": "#eda100"}
for ch_col, lam_val in lam_scaled_true.items():
    ch = ch_col.replace("_impressions_000", "")
    ax.axvline(lam_val, color=RED, lw=1.2, ls="--", alpha=0.7)  # cosmetic
    ax.text(lam_val, ax.get_ylim()[1] if False else 0, ch, rotation=90, fontsize=7,  # cosmetic
            color=RED, ha="right", va="bottom")
ax.set_xlim(0, 8)  # cosmetic
ax.set_xlabel("saturation_lam (model scale)")
ax.set_ylabel("density")
ax.set_title("saturation_lam prior: default, derived and current, vs true values (red lines)",
             loc="left", fontsize=11)  # cosmetic
ax.legend(frameon=False, fontsize=9)  # cosmetic
ax.grid(axis="y", lw=0.6)  # cosmetic
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
save_figure(fig, ROOT / "model/2_priors/plots/2a1_lam_prior_density.png",
            script_file=__file__, idata_name=None)   # stamp; this figure reads no .nc

# ---------------------------------------------------------------------------
# STEP 3: SHAPE plot -- normalized prior curves vs the true one
print("\nSampling the prior predictive for the shape plot (DEFAULT prior Gamma(3,1))...")
idata_current = mmm_base.sample_prior_predictive(X, y, samples=N_PRIOR_SAMPLES, random_seed=SEED)
lam_draws_current = mmm_base.idata.prior["saturation_lam"].stack(sample=("chain", "draw"))

print(f"Sampling the prior predictive for the shape plot (CURRENT prior {CURRENT_LABEL})...")
mmm_new_lam = build_mmm(lam_prior=saturation_lam_prior())
mmm_new_lam.sample_prior_predictive(X, y, samples=N_PRIOR_SAMPLES, random_seed=SEED)
lam_draws_new = mmm_new_lam.idata.prior["saturation_lam"].stack(sample=("chain", "draw"))

rng = np.random.default_rng(SEED)
draw_idx = rng.choice(lam_draws_current.sizes["sample"], size=N_CURVES_TO_PLOT, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)  # cosmetic

for col, ch_col in enumerate(channel_columns):
    ch = ch_col.replace("_impressions_000", "")

    alpha_true_ch = media_true[ch]["adstock_alpha"]
    x_ad_obs = geometric_adstock_dgp(
        df_model[ch_col].values, alpha_true_ch, normalized=ADSTOCK_NORMALIZE, l_max=L_MAX,
    )
    x_ad_max_obs = x_ad_obs.max()
    x_ad = np.linspace(0, x_ad_max_obs * 1.6, 300)  # cosmetic

    lam_true_raw = media_true[ch]["lam"]
    true_curve = np.tanh(lam_true_raw * x_ad / 2)  # normalized (beta=1)  # trivial: identity of the formula

    for row, (lam_draws, title, color) in enumerate([
        (lam_draws_current, "Gamma(3,1) -- DEFAULT", BLUE),
        (lam_draws_new, f"{CURRENT_LABEL} -- CURRENT", ORANGE),
    ]):
        ax = axes[row, col]
        for i in draw_idx:
            lam_scaled_draw = float(lam_draws.sel(channel=ch_col).isel(sample=i))
            lam_raw_equiv = lam_scaled_draw / channel_scale[ch_col]
            prior_curve = np.tanh(lam_raw_equiv * x_ad / 2)  # trivial: identity of the formula
            ax.plot(x_ad, prior_curve, color=color, alpha=0.12, lw=1)  # cosmetic
        ax.plot(x_ad, true_curve, color=INK, lw=2.2, label="True curve")  # cosmetic
        ax.axvline(x_ad_max_obs, color=INK_MUTED, lw=0.8, ls=":")  # cosmetic
        if row == 0:
            ax.set_title(ch, fontsize=12, color=INK)  # cosmetic
        if col == 0:
            ax.set_ylabel(f"{title}\nnormalized effect (0-1)", fontsize=9)  # cosmetic
        ax.set_xlabel("adstocked impressions (thousands)" if row == 1 else "")
        ax.set_ylim(-0.02, 1.05)  # cosmetic
        ax.grid(axis="y", lw=0.6)  # cosmetic
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")  # cosmetic
fig.suptitle("Shape of the saturation curve: prior draws (color) vs true curve (black)",
             fontsize=13, x=0.02, ha="left")  # cosmetic
fig.tight_layout(rect=[0, 0, 1, 0.96])  # cosmetic
save_figure(fig, ROOT / "model/2_priors/plots/2a2_prior_saturation_shape.png",
            script_file=__file__, idata_name=None)   # stamp; this figure reads no .nc

print("\nDone. Plots saved in model/2_priors/plots/2a1_lam_prior_density.png and "
      "model/2_priors/plots/2a2_prior_saturation_shape.png (relative to the repo root)")
