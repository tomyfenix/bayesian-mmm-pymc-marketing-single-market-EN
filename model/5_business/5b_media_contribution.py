"""Media contribution in levels: how much of the period's sales is lost by switching off each
channel. The four bars do not add up to the joint effect, and the excess goes as a bar of its own.

# ! NOTE -- "not split" is not "cannot be split": 6b_ splits it, and as a range.

Produces: model/5_business/plots/5b1_media_contribution__<dataset_label>.png
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.mmm_dgp_config import CHANNEL_COLUMNS, HDI_PROB
from config.idata_paths import require_idata
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.business import (canonical_slice, short_channel, loss_by_channel, joint_loss,
                         true_loss_by_channel)
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, TRUTH_COLOR, apply_style, channel_color
from viz.labels import TRUTH, HDI, CROSS_EFFECT, DO_NOT_ADD_UP
from viz.io import figure_path, save_figure

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
dfm = canonical_slice()

CHANNELS = [short_channel(c) for c in CHANNEL_COLUMNS]
loss = loss_by_channel(idata)          # (n_draws, 4)
joint = joint_loss(idata)           # (n_draws)
total = loss.sum(axis=1)                  # (n_draws) the naive sum, draw by draw
excess = total - joint
truth = true_loss_by_channel(dfm)

m_loss, m_joint, m_total, m_exc = loss.mean(0), joint.mean(), total.mean(), excess.mean()
overstatement = m_exc / m_joint

# The HDI94 of these contributions measures hundredths of a percentage point: drawn, it is HIDDEN
# by the truth marker. A legend that names an invisible line is of the same kind as a header that
# lies about the cost, so the width is REPORTED as a number.
_widths = [float(np.diff(az.hdi(loss[:, k], hdi_prob=HDI_PROB))[0]) for k in range(len(CHANNELS))]
MAX_WIDTH = 100 * max(_widths)  # trivial: unit

fig = plt.figure(figsize=(16.5, 6.9))  # cosmetic
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.30,  # cosmetic
                      left=0.075, right=0.975, top=0.775, bottom=0.135)  # cosmetic

# ---------------------------------------------------------------- left panel: by channel
ax = fig.add_subplot(gs[0, 0])
y = np.arange(len(CHANNELS))[::-1]
for k, ch in enumerate(CHANNELS):
    lo, hi = az.hdi(loss[:, k], hdi_prob=HDI_PROB)
    ax.barh(y[k], 100 * m_loss[k], height=0.42, color=channel_color(ch), alpha=0.88, zorder=2)  # cosmetic
    ax.plot([100 * lo, 100 * hi], [y[k], y[k]], color=INK, lw=1.6, zorder=3, solid_capstyle="butt")  # cosmetic
    ax.scatter([100 * truth[ch]], [y[k]], marker="D", s=40, facecolor="white",  # cosmetic
               edgecolor=TRUTH_COLOR, lw=1.6, zorder=4)  # cosmetic
    ax.text(100 * max(m_loss[k], truth[ch]) + 0.55, y[k], f"{100*m_loss[k]:.2f}%",  # cosmetic
            va="center", ha="left", fontsize=9.5, color=INK)  # cosmetic

ax.set_yticks(y)
ax.set_yticklabels(CHANNELS, fontsize=11)  # cosmetic
ax.set_xlim(0, 100 * max(m_loss.max(), max(truth[c] for c in CHANNELS)) * 1.30)  # cosmetic
ax.set_xlabel("% of period sales lost if the channel is switched off")
ax.grid(axis="x", lw=0.5, color=GRID)  # cosmetic
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.set_title("Incremental contribution, one channel at a time", fontsize=11.5, loc="left", color=INK)  # cosmetic
ax.scatter([], [], marker="D", s=40, facecolor="white", edgecolor=TRUTH_COLOR, lw=1.6,  # cosmetic
           label=f"DGP {TRUTH}")
ax.plot([], [], color=INK, lw=1.6,  # cosmetic
        label=f"{HDI} (drawn, but at most {MAX_WIDTH:.2f} pp wide — narrower than the markers)")
ax.legend(loc="lower right", fontsize=9, frameon=False)  # cosmetic

# ---------------------------------------------------------------- right panel: the reconciliation
ax2 = fig.add_subplot(gs[0, 1])
labels = ["sum of the four\n(what a levels waterfall\nwould report)",
             f"{CROSS_EFFECT}\n(double-counted)",
             "all four switched off\nAT ONCE\n(the real joint effect)"]
values = [100 * m_total, -100 * m_exc, 100 * m_joint]  # trivial: unit
colors = [INK_MUTED, "#c0392b", INK_SECONDARY]
truths = [100 * sum(truth[c] for c in CHANNELS),  # trivial: unit
            -100 * (sum(truth[c] for c in CHANNELS) - truth["_joint"]),  # trivial: unit
            100 * truth["_joint"]]  # trivial: unit
y2 = np.arange(3)[::-1]  # trivial: the three bars of the panel
for k in range(3):  # trivial: the three bars of the panel
    ax2.barh(y2[k], values[k], height=0.40, color=colors[k], alpha=0.88, zorder=2)  # cosmetic
    ax2.scatter([truths[k]], [y2[k]], marker="D", s=40, facecolor="white",  # cosmetic
                edgecolor=TRUTH_COLOR, lw=1.6, zorder=4)  # cosmetic
    edge = max(values[k], truths[k]) if values[k] >= 0 else min(values[k], truths[k])
    ax2.text(edge + (0.75 if values[k] >= 0 else -0.75), y2[k],  # cosmetic
             f"{values[k]:+.2f}%" if k == 1 else f"{values[k]:.2f}%",
             va="center", ha="left" if values[k] >= 0 else "right", fontsize=10, color=INK)  # cosmetic

for k in (0, 2):  # trivial: bar indices
    lo, hi = az.hdi(total if k == 0 else joint, hdi_prob=HDI_PROB)
    ax2.plot([100 * lo, 100 * hi], [y2[k], y2[k]], color=INK, lw=1.6, zorder=3, solid_capstyle="butt")  # cosmetic

ax2.axvline(0, color=INK, lw=1.0, zorder=1)
ax2.set_yticks(y2)
ax2.set_yticklabels(labels, fontsize=9)  # cosmetic
ax2.set_xlim(-8, 34)  # cosmetic
ax2.set_xlabel("% of period sales")
ax2.grid(axis="x", lw=0.5, color=GRID)  # cosmetic
for s in ["top", "right", "left"]:
    ax2.spines[s].set_visible(False)
ax2.set_title("Why the four bars do not add up", fontsize=11.5, loc="left", color=INK)  # cosmetic
ax2.text(0.50, 0.50,  # cosmetic
         f"the sum overstates the joint effect\nby {100*m_exc:.2f} pp = {100*overstatement:.2f}%",  # trivial: unit
         transform=ax2.transAxes, ha="center", va="center", fontsize=11, color="#c0392b",  # cosmetic
         bbox=dict(facecolor="white", edgecolor="#c0392b", lw=0.9, boxstyle="round,pad=0.55"))  # cosmetic

fig.suptitle(
    "5b1 — What media is worth, in levels: switch a channel off and see what sales are lost\n"
    f"Valid in levels because saturation has f(0)=0, so \"no channel\" is a well-defined "
    f"counterfactual. But {DO_NOT_ADD_UP}.\n"
    "The right panel shows that gap as its own bar instead of quietly spreading it across the "
    "four channels. For a decomposition that DOES close, see 5a1.",
    fontsize=11, x=0.015, y=0.995, ha="left", va="top")  # cosmetic

_path = figure_path(ROOT, "5_business", "5b1_media_contribution", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)   # with the provenance stamp
print(f"Plot saved to {_path.relative_to(ROOT)}")
for k, ch in enumerate(CHANNELS):
    lo, hi = az.hdi(loss[:, k], hdi_prob=HDI_PROB)
    print(f"  {ch:8s} {100*m_loss[k]:6.3f}%  {HDI} [{100*lo:.3f}, {100*hi:.3f}]  "  # trivial: unit
          f"truth {100*truth[ch]:6.3f}%")  # trivial: unit
print(f"  SUM      {100*m_total:6.3f}%   JOINT    {100*m_joint:6.3f}%   "  # trivial: unit
      f"excess {100*m_exc:.3f}pp = {100*overstatement:.2f}% overstatement")  # trivial: unit
