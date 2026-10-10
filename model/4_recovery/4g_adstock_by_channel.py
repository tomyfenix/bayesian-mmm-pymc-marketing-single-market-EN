"""The estimated adstock decay of each channel against the true one, week by week.

Produces: model/4_recovery/plots/4g1_adstock_by_channel__<dataset_label>.png
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.diagnostics import FIGURE_SUBSAMPLE_SEED, N_DRAWS_BANDS  # noqa: E402
from config.mmm_dgp_config import L_MAX, CHANNEL_COLUMNS
from config.idata_paths import require_idata
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.canonical import validate_recovery
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, apply_style, channel_color, channel_short_name
from viz.labels import POSTERIOR, HDI, TRUE_VALUE, CHANNEL, LAG_WEEKS
from viz.io import figure_path, save_figure
from viz.intervals import pointwise_hdi

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
post = idata.posterior
table = validate_recovery(idata, geo="Geo_Centro", trade="Retail").set_index("parameter")

lags = np.arange(L_MAX)


def weights(alpha):
    w = alpha ** lags
    return w / w.sum()


fig, axes = plt.subplots(2, 2, figsize=(12, 9))  # cosmetic
rng = np.random.default_rng(FIGURE_SUBSAMPLE_SEED)
N_SUB = N_DRAWS_BANDS

for ax, ch_col in zip(axes.ravel(), CHANNEL_COLUMNS):
    ch = channel_short_name(ch_col)
    color = channel_color(ch_col)
    alpha_draws = post["adstock_alpha"].sel(channel=ch_col).values.ravel()
    idx = rng.choice(len(alpha_draws), size=min(N_SUB, len(alpha_draws)), replace=False)
    w_draws = np.array([weights(a) for a in alpha_draws[idx]])
    w_median = np.median(w_draws, axis=0)
    w_lo, w_hi = pointwise_hdi(w_draws)   # the HDI of the weight of each lag, over the draws

    alpha_true = float(table.loc[f"adstock_alpha[{ch}]", "true_value"])
    alpha_post_mean = float(table.loc[f"adstock_alpha[{ch}]", "posterior_mean"])
    w_true = weights(alpha_true)

    ax.fill_between(lags, w_lo, w_hi, color=color, alpha=0.25, label=f"{POSTERIOR} {HDI}")  # cosmetic
    ax.plot(lags, w_median, color=color, lw=0, marker="o", markersize=5, label=f"{POSTERIOR}, median")  # cosmetic
    ax.plot(lags, w_true, color=INK, lw=1.6, ls="--", marker="x", markersize=6, label=TRUE_VALUE)  # cosmetic

    ax.set_title(f"{ch}   (alpha: posterior={alpha_post_mean:.3f}, true={alpha_true:.3f})", fontsize=10.5, color=INK)  # cosmetic
    ax.set_xlabel(LAG_WEEKS)
    ax.set_ylabel("normalised weight")
    ax.set_xticks(lags)
    ax.grid(lw=0.5, color=GRID)  # cosmetic
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    if ch_col == CHANNEL_COLUMNS[0]:
        ax.legend(fontsize=8, frameon=False)  # cosmetic

fig.suptitle(
    f"4g1 — Adstock decay per {CHANNEL}: normalised weight by lag (0 to L_MAX-1=11), posterior vs. true",
    fontsize=12, x=0.02, ha="left",  # cosmetic
)
fig.text(0.02, 0.035,  # cosmetic
         f"Each of {N_DRAWS_BANDS} posterior draws of alpha gives one set of weights. The dot is the "
         f"median weight at each lag, not the weights of the mean alpha; the band is the {HDI} at each lag.",
         fontsize=8.5, color=INK_SECONDARY, ha="left")  # cosmetic
fig.tight_layout(rect=[0, 0.06, 1, 0.95])  # cosmetic

_path = figure_path(ROOT, "4_recovery", "4g1_adstock_by_channel", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)
print(f"Plot saved to {_path.relative_to(ROOT)}")
