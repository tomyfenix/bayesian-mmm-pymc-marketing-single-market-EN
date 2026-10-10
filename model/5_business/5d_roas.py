"""Average and marginal ROAS by channel, both in the same figure and computed from the same sweep.

Produces: model/5_business/plots/5d1_roas__<dataset_label>.png
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.mmm_dgp_config import CHANNEL_COLUMNS, HDI_PROB
from config.budget import N_DRAWS, SWEEP_DRAWS_SEED
from config.idata_paths import require_idata
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.business import canonical_slice, short_channel, average_and_marginal_roas, true_roas
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, TRUTH_COLOR, apply_style, channel_color
from viz.labels import TRUTH, HDI, ROAS_AVG, ROAS_MARGINAL, SPEND
from viz.io import figure_path, save_figure

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
dfm = canonical_slice()

TOTAL_REVENUE = float((dfm["sales"] * dfm["price"]).sum())

rows = []
for col in CHANNEL_COLUMNS:
    ch = short_channel(col)
    avg, marg, spend = average_and_marginal_roas(idata, dfm, col, n_draws=N_DRAWS, seed=SWEEP_DRAWS_SEED)
    t_avg, t_marg, _ = true_roas(dfm, col)
    rows.append(dict(ch=ch, avg=avg, marg=marg, spend=spend, t_avg=t_avg, t_marg=t_marg))

fig = plt.figure(figsize=(16.5, 7.8))  # cosmetic
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.24,  # cosmetic
                      left=0.065, right=0.975, top=0.795, bottom=0.105)  # cosmetic

# ------------------------------------------------------- left panel: the two ROAS together
ax = fig.add_subplot(gs[0, 0])
step, height = 1.0, 0.32  # cosmetic
y_center = np.arange(len(rows))[::-1] * 1.0

for k, f in enumerate(rows):
    yc = y_center[k]
    ch_color = channel_color(f["ch"])
    for j, (key, tkey, offset, alpha, lbl) in enumerate(
            [("avg", "t_avg", +0.20, 0.88, ROAS_AVG), ("marg", "t_marg", -0.20, 0.42, ROAS_MARGINAL)]):  # cosmetic
        m = float(f[key].mean())
        lo, hi = az.hdi(f[key], hdi_prob=HDI_PROB)
        ax.barh(yc + offset, m, height=height, color=ch_color, alpha=alpha, zorder=2)  # cosmetic
        ax.plot([lo, hi], [yc + offset, yc + offset], color=INK, lw=1.5, zorder=3, solid_capstyle="butt")  # cosmetic
        ax.scatter([f[tkey]], [yc + offset], marker="D", s=36, facecolor="white",  # cosmetic
                   edgecolor=TRUTH_COLOR, lw=1.5, zorder=4)  # cosmetic
        ax.text(max(m, f[tkey]) + 0.55, yc + offset, f"{m:.2f}", va="center", ha="left",  # cosmetic
                fontsize=9, color=INK)  # cosmetic
        ax.text(0.22, yc + offset, lbl, va="center", ha="left", fontsize=8,  # cosmetic
                color="white" if alpha > 0.7 else INK, zorder=5)  # cosmetic

    mp, mm = float(f["avg"].mean()), float(f["marg"].mean())
    ax.add_patch(plt.Rectangle((mm, yc - 0.38), mp - mm, 0.76, facecolor="none",  # cosmetic
                               edgecolor="#c0392b", lw=1.0, ls=(0, (3, 2)), hatch="///",  # cosmetic
                               alpha=0.55, zorder=6))  # cosmetic
    ax.text((mp + mm) / 2, yc, f"−{100*(1 - mm/mp):.0f}%", ha="center", va="center",  # trivial: identity of the formula, unit
            fontsize=10, color="#c0392b", zorder=7, weight="bold",  # cosmetic
            bbox=dict(facecolor="white", edgecolor="none", pad=2.0))  # cosmetic

ax.set_yticks(y_center)
ax.set_yticklabels([f["ch"] for f in rows], fontsize=11.5)  # cosmetic
ax.set_xlim(0, max(max(f["avg"].mean(), f["t_avg"]) for f in rows) * 1.14)  # cosmetic
ax.set_xlabel("revenue returned per unit of spend  (sales × price / spend)")
ax.grid(axis="x", lw=0.5, color=GRID)  # cosmetic
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.set_title(f"{ROAS_AVG} vs. {ROAS_MARGINAL}   ·   the hatched band is what the next unit of "
             f"spend loses to saturation", fontsize=11, loc="left", color=INK)  # cosmetic
ax.scatter([], [], marker="D", s=36, facecolor="white", edgecolor=TRUTH_COLOR, lw=1.5,  # cosmetic
           label=f"DGP {TRUTH}")
ax.plot([], [], color=INK, lw=1.5, label=HDI)  # cosmetic
ax.legend(loc="lower right", fontsize=9, frameon=False)  # cosmetic

# ------------------------------------------------------- right panel: the spend, for weighting
ax2 = fig.add_subplot(gs[0, 1])
spends = np.array([f["spend"] for f in rows])
for k, f in enumerate(rows):
    ax2.barh(y_center[k], f["spend"] / 1e6, height=0.42, color=channel_color(f["ch"]),  # cosmetic
             alpha=0.70, zorder=2)  # cosmetic
    ax2.text(f["spend"] / 1e6 + 0.08, y_center[k],  # cosmetic
             f"{f['spend']/1e6:.2f}M  ({100*f['spend']/spends.sum():.0f}%)",  # trivial: unit
             va="center", ha="left", fontsize=9, color=INK)  # cosmetic
ax2.set_yticks(y_center)
ax2.set_yticklabels([])
ax2.set_xlim(0, spends.max() / 1e6 * 1.55)  # cosmetic
ax2.set_xlabel(f"{SPEND} over the period, millions")
ax2.grid(axis="x", lw=0.5, color=GRID)  # cosmetic
for s in ["top", "right", "left"]:
    ax2.spines[s].set_visible(False)
ax2.set_title("how much was actually spent", fontsize=11, loc="left", color=INK)  # cosmetic
ax2.text(0.97, 0.03,  # cosmetic
         f"total media spend {spends.sum()/1e6:.2f}M\nagainst period revenue "  # trivial: unit
         f"{TOTAL_REVENUE/1e6:.0f}M\n(sales are in UNITS;\nrevenue = sales × price)",  # trivial: unit
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY,  # cosmetic
         bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.45"))

fig.suptitle(
    "5d1 — What media returned, and what the next unit of spend would return\n"
    f"{ROAS_AVG} answers \"was this channel worth it\"; {ROAS_MARGINAL} answers \"should it get "
    "more\". Under saturation the second is always the smaller one, and here it is 62–77% "
    "smaller.\n"
    "Both come from the same counterfactual sweep as 5e1, so the two figures cannot disagree.",
    fontsize=11, x=0.015, y=0.995, ha="left", va="top")  # cosmetic

_path = figure_path(ROOT, "5_business", "5d1_roas", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)   # with the provenance stamp
print(f"Plot saved to {_path.relative_to(ROOT)}")
for f in rows:
    mp, mm = float(f["avg"].mean()), float(f["marg"].mean())
    lp, hp = az.hdi(f["avg"], hdi_prob=HDI_PROB)
    lm, hm = az.hdi(f["marg"], hdi_prob=HDI_PROB)
    print(f"  {f['ch']:8s} average={mp:6.2f} {HDI}[{lp:.2f},{hp:.2f}] truth={f['t_avg']:6.2f} "
          f"({100*abs(mp-f['t_avg'])/f['t_avg']:5.2f}% diff)   "  # trivial: unit
          f"marginal={mm:5.2f} {HDI}[{lm:.2f},{hm:.2f}] truth={f['t_marg']:5.2f} "
          f"({100*abs(mm-f['t_marg'])/f['t_marg']:5.2f}% diff)   gap={100*(1-mm/mp):.1f}%")  # trivial: unit
