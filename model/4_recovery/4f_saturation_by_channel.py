"""The estimated saturation curve of each channel against the true one, with the observed spend range
marked: outside it, the curve is extrapolation.

Produces: model/4_recovery/plots/4f1_saturation_by_channel__<dataset_label>.png
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.diagnostics import FIGURE_SUBSAMPLE_SEED, N_DRAWS_BANDS  # noqa: E402
from config.mmm_dgp_config import L_MAX, CHANNEL_COLUMNS, ADSTOCK_NORMALIZE
from config.idata_paths import require_idata
from synthetic_data.adstock_dgp import geometric_adstock_aligned
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.canonical import validate_recovery
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, POSTERIOR_COLOR, apply_style, channel_color, channel_short_name
from viz.labels import POSTERIOR, HDI, CHANNEL, ADSTOCKED_IMPRESSIONS, OBSERVED_RANGE, EXTRAPOLATION_NOTE
from viz.io import figure_path, save_figure
from viz.intervals import pointwise_hdi

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
post = idata.posterior
channel_scale = {
    str(c): float(v) for c, v in zip(idata.constant_data["channel_scale"].coords["channel"].values,
                                      idata.constant_data["channel_scale"].values)
}

table = validate_recovery(idata, geo="Geo_Centro", trade="Retail").set_index("parameter")

df = pd.read_csv(ROOT / "synthetic_data" / "dataset.csv", parse_dates=["date"])
df_model = df.query("geo == 'Geo_Centro' and trade_channel == 'Retail'").sort_values("date")

N_CURVE_POINTS = 200  # cosmetic
N_DRAW_SUBSAMPLE = N_DRAWS_BANDS


def saturation(x_scaled, lam, beta):
    return beta * (1 - np.exp(-lam * x_scaled)) / (1 + np.exp(-lam * x_scaled))


fig, axes = plt.subplots(2, 2, figsize=(13, 11), gridspec_kw={"hspace": 0.65, "wspace": 0.3})  # cosmetic
rng = np.random.default_rng(FIGURE_SUBSAMPLE_SEED)

for ax, ch_col in zip(axes.ravel(), CHANNEL_COLUMNS):
    ch = channel_short_name(ch_col)
    color = channel_color(ch_col)
    cscale = channel_scale[ch_col]

    alpha_mean = float(table.loc[f"adstock_alpha[{ch}]", "posterior_mean"])
    raw_series = df_model[ch_col].values
    post_adstock = geometric_adstock_aligned(raw_series, alpha_mean, l_max=L_MAX, normalize=ADSTOCK_NORMALIZE)
    p10, p90, pmax = np.percentile(post_adstock, 10), np.percentile(post_adstock, 90), post_adstock.max()  # cosmetic

    x_max_raw = max(pmax * 1.15, p90 * 1.3)  # cosmetic
    x_grid_raw = np.linspace(0, x_max_raw, N_CURVE_POINTS)
    x_grid_scaled = x_grid_raw / cscale

    lam_draws = post["saturation_lam"].sel(channel=ch_col).values.ravel()
    beta_draws = post["saturation_beta"].sel(channel=ch_col).values.ravel()
    idx = rng.choice(len(lam_draws), size=min(N_DRAW_SUBSAMPLE, len(lam_draws)), replace=False)
    curves = np.array([saturation(x_grid_scaled, lam_draws[i], beta_draws[i]) for i in idx])
    curve_median = np.median(curves, axis=0)
    curve_lo, curve_hi = pointwise_hdi(curves)   # the HDI of each grid point, over the curves

    true_lam = float(table.loc[f"saturation_lam[{ch}]", "true_value"])
    true_beta = float(table.loc[f"saturation_beta[{ch}] (L)", "true_value"])
    true_curve = saturation(x_grid_scaled, true_lam, true_beta)

    ax.fill_between(x_grid_raw, curve_lo, curve_hi, color=color, alpha=0.25, label=f"{POSTERIOR} {HDI}")  # cosmetic
    ax.plot(x_grid_raw, curve_median, color=color, lw=2.0, label=f"{POSTERIOR}, median")  # cosmetic
    ax.plot(x_grid_raw, true_curve, color=INK, lw=1.6, ls="--", label=f"true curve")  # cosmetic

    ax.axvspan(pmax, x_max_raw, color=INK_MUTED, alpha=0.10, hatch="//", zorder=0)  # cosmetic
    ax.text(
        pmax + 0.02 * (x_max_raw - pmax), 0.93,  # cosmetic
        EXTRAPOLATION_NOTE,
        transform=ax.get_xaxis_transform(), fontsize=7, color=INK_SECONDARY,  # cosmetic
        ha="left", va="top", style="italic",
    )

    # rug of the observed post-adstock impressions, at the bottom
    rug_y = -0.04 * (true_curve.max())  # cosmetic
    ax.plot(post_adstock, np.full_like(post_adstock, rug_y), "|", color=INK_MUTED, markersize=8, alpha=0.5, clip_on=False)  # cosmetic

    # Vertical lines on the data, values written in a compact text box (not tied to the X position of
    # each line) -- it avoids p90 and max, which tend to fall close to each other on the X axis,
    # producing overlapping or cut text.
    for val in (p10, p90, pmax):
        ax.axvline(val, color=INK_MUTED, lw=0.9, ls=":")  # cosmetic
    ax.text(
        0.02, 0.98, f"p10={p10:,.0f}\np90={p90:,.0f}\nmax={pmax:,.0f}",  # cosmetic
        transform=ax.transAxes, fontsize=8, color=INK_SECONDARY, va="top", ha="left",  # cosmetic
        bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.3", alpha=0.85),  # cosmetic
    )

    ax.set_title(f"{ch}  (alpha used for adstocking: {alpha_mean:.3f})", fontsize=10.5, color=INK)  # cosmetic
    ax.set_xlabel(ADSTOCKED_IMPRESSIONS)
    ax.set_ylabel("contribution (validate_recovery scale)")
    ax.grid(lw=0.5, color=GRID)  # cosmetic
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    if ch_col == CHANNEL_COLUMNS[0]:
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")  # cosmetic

fig.suptitle(
    f"4f1 — Saturation curves per {CHANNEL}: posterior vs. true, with p10/p90/max of the\n"
    f"{OBSERVED_RANGE} in adstocked impressions (dataset design criteria: p10≤25%, p90≥85%, max≥93% saturation)",
    fontsize=12, x=0.02, ha="left",  # cosmetic
)
fig.text(0.02, 0.035,  # cosmetic
         f"Each of {N_DRAWS_BANDS} posterior draws (lam, beta) gives one curve. The line is the pointwise "
         f"median of those curves, not the curve at the mean lam; the band is their pointwise {HDI}.",
         fontsize=8.5, color=INK_SECONDARY, ha="left")  # cosmetic
fig.tight_layout(rect=[0, 0.06, 1, 0.90])  # cosmetic

_path = figure_path(ROOT, "4_recovery", "4f1_saturation_by_channel", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)
print(f"Plot saved to {_path.relative_to(ROOT)}")
