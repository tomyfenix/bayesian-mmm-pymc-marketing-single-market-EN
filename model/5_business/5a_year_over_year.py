"""Year-over-year duet: what explains the change in sales from one year to the next. In log-points,
where the model is additive and the decomposition closes with no residual.

Produces: model/5_business/plots/5a1_year_over_year__<dataset_label>.png
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
from config.idata_paths import require_idata
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.business import canonical_slice, log_change_decomposition, true_log_change_decomposition, short_channel
from viz.style import (INK, INK_SECONDARY, INK_MUTED, GRID, MISS_COLOR, POSTERIOR_COLOR,
                       TRUTH_COLOR, apply_style, channel_color)
from viz.labels import LOG_POINTS, TRUTH, HDI, CROSS_EFFECT
from viz.io import figure_path, save_figure

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
dfm = canonical_slice()

N = len(dfm)
week = np.arange(N)
YEARS = [week < 52, (week >= 52) & (week < 104), week >= 104]  # trivial: 52 weeks per year

# Presentation order: media, then economic drivers, then structure. Fixed, so that the two panels
# read against each other without reordering anything.
ORDER = ["tv", "meta", "tiktok", "ooh", "log_price_c", "promo_flag", "log_acv_c",
         "trend + seasonality", "intercept"]
DISPLAY = {
    "log_price_c": "price", "promo_flag": "promo", "log_acv_c": "ACV (distribution)",
    "trend + seasonality": "trend + seasonality", "intercept": "intercept",
}

fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.6), sharey=True)  # cosmetic

for ax, (label, i0, i1) in zip(axes, [("Year 2 vs. Year 1", 0, 1), ("Year 3 vs. Year 2", 1, 2)]):  # trivial: year index
    d = log_change_decomposition(idata, YEARS[i0], YEARS[i1]).set_index("component")
    v = true_log_change_decomposition(dfm, YEARS[i0], YEARS[i1]).set_index("component")
    order = [c for c in ORDER if c in d.index]
    y = np.arange(len(order))[::-1]

    for k, comp in enumerate(order):
        est = float(d.loc[comp, "delta_log"])
        lo, hi = float(d.loc[comp, "hdi_lo"]), float(d.loc[comp, "hdi_hi"])
        color = channel_color(comp) if comp in ("tv", "meta", "tiktok", "ooh") else INK_SECONDARY
        ax.barh(y[k], est, height=0.52, color=color, alpha=0.85, zorder=2)  # cosmetic
        ax.plot([lo, hi], [y[k], y[k]], color=INK, lw=1.4, zorder=3, solid_capstyle="butt")  # cosmetic
        if comp in v.index:
            ax.scatter([float(v.loc[comp, "delta_log"])], [y[k]], marker="D", s=34,  # cosmetic
                       facecolor="white", edgecolor=TRUTH_COLOR, lw=1.5, zorder=4)  # cosmetic

    ax.axvline(0, color=INK, lw=1.2, zorder=1)  # cosmetic
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY.get(c, c) for c in order], fontsize=9.5)  # cosmetic
    ax.set_xlabel(f"change in log(sales), {LOG_POINTS}")
    ax.grid(axis="x", lw=0.5, color=GRID)  # cosmetic
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    tot, res = d.attrs["total_log"], d.attrs["residual_log"]
    ax.set_title(
        f"{label}\ntotal {tot:+.4f} {LOG_POINTS} = {100*d.attrs['total_pct']:+.2f}%   ·   "  # trivial: unit
        f"residual {res:+.1e} (exact)",
        fontsize=10.5, loc="left", color=INK)  # cosmetic

    # Text column with the % version, and the cross-effect made EXPLICIT.
    lines = [f"{'component':<20s}{'%':>9s}"]
    for comp in order:
        lines.append(f"{DISPLAY.get(comp, comp):<20s}{100*float(d.loc[comp,'delta_pct']):>8.2f}%")  # trivial: unit
    ce = d.attrs["cross_effect_pct"]
    lines.append("-" * 29)  # cosmetic
    lines.append(f"{CROSS_EFFECT:<20s}{100*ce:>8.2f}%")  # trivial: unit
    lines.append(f"{'TOTAL':<20s}{100*d.attrs['total_pct']:>8.2f}%")  # trivial: unit
    ax.text(1.005, 0.99, "\n".join(lines), transform=ax.transAxes, fontsize=8,  # cosmetic
            family="monospace", va="top", ha="left", color=INK_SECONDARY,
            bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.5"))

axes[0].scatter([], [], marker="D", s=34, facecolor="white", edgecolor=TRUTH_COLOR, lw=1.5,  # cosmetic
                label=f"DGP {TRUTH}")
axes[0].plot([], [], color=INK, lw=1.4, label=HDI)  # cosmetic
axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)  # cosmetic

fig.suptitle(
    "5a1 — What moved sales, year over year: an EXACT change decomposition\n"
    f"Bars are the estimated change in {LOG_POINTS}; the model is additive in logs, so the "
    f"components add up with NO residual (see the title of each panel).\n"
    "The % column on the right is the same decomposition read multiplicatively — there the "
    f"components do NOT add, and the gap is shown as an explicit {CROSS_EFFECT} row.",
    fontsize=11, x=0.015, y=0.995, ha="left", va="top")  # cosmetic
fig.subplots_adjust(left=0.11, right=0.845, top=0.855, bottom=0.075, wspace=0.42)  # cosmetic

_path = figure_path(ROOT, "5_business", "5a1_year_over_year", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)   # with the provenance stamp
print(f"Plot saved to {_path.relative_to(ROOT)}")
for lbl, i0, i1 in [("year2 vs year1", 0, 1), ("year3 vs year2", 1, 2)]:  # trivial: year index
    d = log_change_decomposition(idata, YEARS[i0], YEARS[i1])
    print(f"  {lbl}: total={d.attrs['total_log']:+.6f} log-points ({100*d.attrs['total_pct']:+.2f}%), "  # trivial: unit
          f"residual={d.attrs['residual_log']:+.2e}, cross-effect={100*d.attrs['cross_effect_pct']:+.4f}pp "  # trivial: unit
          f"({100*abs(d.attrs['cross_effect_pct']/d.attrs['total_pct']):.1f}% of the total)")  # trivial: unit
