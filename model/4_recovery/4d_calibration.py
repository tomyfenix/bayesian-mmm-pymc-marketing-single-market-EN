"""Calibration: true value against posterior mean, with the HDI94, to see whether the intervals cover
what they claim to cover.

Produces: model/4_recovery/plots/4d1_calibration__<dataset_label>.png
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
from viz.canonical import validate_recovery
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, MISS_COLOR, apply_style, channel_color, channel_short_name
from viz.labels import TRUE_VALUE, HDI, MISSES_HDI
from viz.io import figure_path, save_figure

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)

table = validate_recovery(idata, geo="Geo_Centro", trade="Retail")
table = table.set_index("parameter")
hdi_lo_col = [c for c in table.columns if c.endswith("_lo")][0]
hdi_hi_col = [c for c in table.columns if c.endswith("_hi")][0]

PANELS = [
    ("adstock_alpha", [f"adstock_alpha[{c.replace('_impressions_000','')}]" for c in CHANNEL_COLUMNS], False),
    ("saturation_lam", [f"saturation_lam[{c.replace('_impressions_000','')}]" for c in CHANNEL_COLUMNS], False),
    ("saturation_beta (L)", [f"saturation_beta[{c.replace('_impressions_000','')}] (L)" for c in CHANNEL_COLUMNS], True),
    # gamma_control mixes signs (eta_true < 0, promo/ACV > 0) -- a log scale breaks with negatives, so it
    # stays linear on purpose.
    ("gamma_control", ["gamma_control[log_price_c]", "gamma_control[promo_flag]", "gamma_control[log_acv_c]"], False),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))  # cosmetic

for ax, (family, names, log_scale) in zip(axes, PANELS):
    xs, ys, lo, hi, colors, miss = [], [], [], [], [], []
    for name in names:
        r = table.loc[name]
        xs.append(float(r["true_value"]))
        ys.append(float(r["posterior_mean"]))
        lo.append(float(r["posterior_mean"] - r[hdi_lo_col]))
        hi.append(float(r[hdi_hi_col] - r["posterior_mean"]))
        miss.append(not bool(r["covers_hdi"]))
        # color by channel if the name carries a known channel; if not (gamma_control), neutral grey
        ch = next((c for c in CHANNEL_COLUMNS if channel_short_name(c) in name), None)
        colors.append(channel_color(ch) if ch else INK_SECONDARY)
    xs, ys = np.array(xs), np.array(ys)
    err = np.abs(np.array([lo, hi]))
    for i in range(len(names)):
        ax.errorbar(
            xs[i], ys[i], yerr=[[err[0, i]], [err[1, i]]], fmt="o",
            color=MISS_COLOR if miss[i] else colors[i],
            markeredgecolor="white", markeredgewidth=0.6, markersize=7,  # cosmetic
            elinewidth=1.6, capsize=3, zorder=3,  # cosmetic
        )
    lims = [min(xs.min(), ys.min()) * 0.9, max(xs.max(), ys.max()) * 1.1]  # cosmetic
    ax.plot(lims, lims, color=INK_MUTED, lw=1.0, ls="--", zorder=1, label="perfect calibration")
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(family, fontsize=10.5, color=INK)  # cosmetic
    ax.set_xlabel(TRUE_VALUE)
    if ax is axes[0]:
        ax.set_ylabel("posterior mean")
    ax.grid(lw=0.5, color=GRID)  # cosmetic
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.suptitle(
    f"4d1 — Calibration: {TRUE_VALUE} vs. posterior mean (bars = {HDI}; line = perfect calibration)\n"
    f"red = {MISSES_HDI}; log scale on saturation_beta and gamma_control because of their relative range",
    fontsize=11.5, x=0.02, ha="left",  # cosmetic
)
fig.tight_layout(rect=[0, 0, 1, 0.88])  # cosmetic

_path = figure_path(ROOT, "4_recovery", "4d1_calibration", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)
print(f"Plot saved to {_path.relative_to(ROOT)}")
