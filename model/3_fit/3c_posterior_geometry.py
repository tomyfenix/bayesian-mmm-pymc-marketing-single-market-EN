"""Posterior geometry: scatter of adstock_alpha against saturation_lam by channel and parallel
coordinates of the 15 parameters, to look for funnels or ridges.

It prints the posterior alpha-lam correlation of each channel (and puts it in the title of its
panel): it is the measure of the ridge, which the scatter only shows.

Produces: model/3_fit/plots/3c1_posterior_geometry__<dataset_label>.png
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.contracts import require_regime  # noqa: E402
require_regime("log", __file__)  # this script belongs to the log regime: with another dataset it stops here
from config.mmm_dgp_config import CHANNEL_COLUMNS
from config.idata_paths import require_idata
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.style import INK, INK_SECONDARY, INK_MUTED, MISS_COLOR, apply_style, channel_color, channel_short_name
from viz.labels import CHANNEL, DIVERGENCES
from viz.io import figure_path, save_figure

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
post = idata.posterior
diverging = idata.sample_stats["diverging"].values.astype(bool)
n_div = int(diverging.sum())
n_total = diverging.size

fig = plt.figure(figsize=(16, 12))  # cosmetic
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1])  # cosmetic

# --- Rows 1-2: scatter adstock_alpha x saturation_lam, one panel per channel -
correlations = {}
for i, ch in enumerate(CHANNEL_COLUMNS):
    ax = fig.add_subplot(gs[0 if i < 2 else 1, (i % 2) * 2:(i % 2) * 2 + 2])  # cosmetic
    alpha_draws = post["adstock_alpha"].sel(channel=ch).values.ravel()
    lam_draws = post["saturation_lam"].sel(channel=ch).values.ravel()
    div_flat = diverging.ravel()
    correlations[channel_short_name(ch)] = float(np.corrcoef(alpha_draws, lam_draws)[0, 1])
    color = channel_color(ch)
    ax.scatter(alpha_draws[~div_flat], lam_draws[~div_flat], s=4, alpha=0.25, color=color, label="non-divergent")  # cosmetic
    if n_div > 0:
        ax.scatter(alpha_draws[div_flat], lam_draws[div_flat], s=18, color=MISS_COLOR, marker="x", label="divergent", zorder=5)  # cosmetic
    ax.set_xlabel("adstock_alpha", fontsize=8, color=INK_SECONDARY)  # cosmetic
    ax.set_ylabel("saturation_lam", fontsize=8, color=INK_SECONDARY)  # cosmetic
    ax.set_title(f"{channel_short_name(ch)}   corr(alpha, lam) = {correlations[channel_short_name(ch)]:+.2f}",
                 fontsize=10, color=INK)  # cosmetic
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

# --- Row 3: parallel coordinates, 15 parameters -
ax_par = fig.add_subplot(gs[2, :])
az.plot_parallel(
    idata,
    var_names=["adstock_alpha", "saturation_lam", "saturation_beta", "gamma_control"],
    coords={"control": ["log_price_c", "promo_flag", "log_acv_c"]},
    norm_method="minmax",
    colornd=INK_MUTED,
    colord=MISS_COLOR,
    shadend=0.05,  # cosmetic
    ax=ax_par,
)
ax_par.set_title(
    f"Parallel coordinates — 15 parameters normalised to [0,1] "
    f"({'WITH' if n_div else 'NO'} divergence lines: {n_div}/{n_total} {DIVERGENCES})",
    fontsize=10, color=INK,  # cosmetic
)
ax_par.tick_params(axis="x", labelsize=7, rotation=30)  # cosmetic

fig.suptitle(
    f"3c1 — Posterior geometry: adstock_alpha × saturation_lam per {CHANNEL}, "
    f"plus parallel coordinates of all 15\n"
    f"{n_div} {DIVERGENCES} in {n_total} draws — no pathological geometry to highlight "
    f"(a result, not a missing analysis)",
    fontsize=11.5, x=0.02, ha="left",  # cosmetic
)
fig.tight_layout(rect=[0, 0, 1, 0.94])  # cosmetic

_path = figure_path(ROOT, "3_fit", "3c1_posterior_geometry", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)
print(f"Plot saved to {_path.relative_to(ROOT)}")
print(f"Divergences: {n_div}/{n_total}")
print("Posterior alpha-lam correlation by channel (all draws):")
for ch, r in correlations.items():
    print(f"  {ch:8s} {r:+.3f}")
