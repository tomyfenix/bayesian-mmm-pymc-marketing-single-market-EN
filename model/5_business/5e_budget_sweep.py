"""Budget sweep from 0 to 1.5x by channel: what would happen to revenue, with the extrapolated
zone marked.

Produces: model/5_business/plots/5e1_budget_sweep__<dataset_label>.png
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.mmm_dgp_config import CHANNEL_COLUMNS, HDI_PROB
from config.budget import N_DRAWS, SWEEP_DRAWS_SEED, SWEEP_GRID, SWEEP_POINTS
from config.idata_paths import require_idata
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.business import canonical_slice, short_channel, channel_sweep, true_sweep
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, TRUTH_COLOR, apply_style, channel_color
from viz.labels import TRUTH, HDI, BUDGET_MULT
from viz.io import figure_path, save_figure

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
dfm = canonical_slice()

MULT = np.linspace(*SWEEP_GRID)
POINTS = list(SWEEP_POINTS)                  # where the deviations are reported

fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.4))  # cosmetic
deviations = []

for ax, col in zip(axes.ravel(), CHANNEL_COLUMNS):
    ch = short_channel(col)
    _, rev = channel_sweep(idata, dfm, col, MULT, n_draws=N_DRAWS, seed=SWEEP_DRAWS_SEED)   # (n_draws, n_mult), total revenue
    inc = (rev - rev[:, [0]]) / 1e6                        # incremental over "channel switched off"  # trivial: unit
    med = inc.mean(axis=0)
    lo, hi = az.hdi(inc[None, ...], hdi_prob=HDI_PROB).T
    tru = (true_sweep(dfm, col, MULT) - true_sweep(dfm, col, [0.0])[0]) / 1e6  # trivial: unit

    color = channel_color(ch)
    # The stretch above 1.0x is EXTRAPOLATION: it is shaded and labelled.
    ax.axvspan(1.0, MULT[-1], color=INK_MUTED, alpha=0.11, lw=0, zorder=0)  # cosmetic
    ax.fill_between(MULT, lo, hi, color=color, alpha=0.26, lw=0, zorder=1, label=HDI)  # cosmetic
    ax.plot(MULT, med, color=color, lw=2.0, zorder=3, label="posterior mean")  # cosmetic
    ax.plot(MULT, tru, color=TRUTH_COLOR, lw=1.3, ls="--", zorder=4, label=f"DGP {TRUTH}")  # cosmetic
    ax.axvline(1.0, color=INK, lw=1.3, ls=":", zorder=2)  # cosmetic

    for m in POINTS:
        j = int(np.argmin(np.abs(MULT - m)))
        d = 100 * (med[j] - tru[j]) / tru[j] if tru[j] != 0 else float("nan")  # trivial: unit
        inside = bool(lo[j] <= tru[j] <= hi[j])
        deviations.append((ch, m, med[j], tru[j], d, lo[j], hi[j], inside))
        ax.scatter([MULT[j]], [med[j]], s=26, color=INK, zorder=5)  # cosmetic
        if not inside:
            ax.scatter([MULT[j]], [tru[j]], s=95, facecolor="none", edgecolor="#c0392b",  # cosmetic
                       lw=1.5, zorder=6)  # cosmetic

    ax.text(0.995, 0.30, "observed spend (1.0×)  ", rotation=90, va="center", ha="right",  # cosmetic
            transform=ax.get_xaxis_transform(), fontsize=8, color=INK_SECONDARY)  # cosmetic
    ax.text((1.0 + MULT[-1]) / 2, 0.985, "EXTRAPOLATION\nbeyond any spend level the model saw",  # cosmetic
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8, color=INK_SECONDARY, linespacing=1.35)  # cosmetic

    table = [f"{'':>6s}{'model':>9s}{'truth':>9s}{'dev':>8s}"]
    for c2, m, mv, vv, d, _l, _h, inside in deviations[-len(POINTS):]:
        table.append(f"{m:>5.1f}×{mv:>9.1f}{vv:>9.1f}{d:>7.2f}%" + ("" if inside else " *"))
    if any(not f[7] for f in deviations[-len(POINTS):]):
        table.append(f"* truth outside {HDI}")
    ax.text(0.025, 0.965, "\n".join(table), transform=ax.transAxes, fontsize=8,  # cosmetic
            family="monospace", va="top", ha="left", color=INK_SECONDARY,
            bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.42"))

    ax.set_title(ch, fontsize=12, loc="left", color=INK)  # cosmetic
    ax.set_xlabel(f"{BUDGET_MULT} (1.0× = the spend actually observed)")
    ax.set_ylabel("incremental revenue over the period, millions")
    ax.set_xlim(-0.015, MULT[-1] + 0.025)  # cosmetic
    # Headroom: without it the EXTRAPOLATION label falls on top of the plateau of the curve, which is
    # exactly the stretch the label is warning must be read with care.
    ax.set_ylim(min(0.0, float(lo.min())) - 0.02 * float(hi.max()), float(max(hi.max(), tru.max())) * 1.20)  # cosmetic
    ax.grid(lw=0.5, color=GRID)  # cosmetic
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

axes[0, 0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.06), fontsize=8.5,  # cosmetic
                  frameon=True, framealpha=0.92, edgecolor=GRID)  # cosmetic

fig.suptitle(
    "5e1 — What each channel would have returned at a different budget\n"
    "Below 1.0× the curve interpolates between two anchored points (saturation has f(0)=0, so "
    "\"nothing spent\" is known exactly).\n"
    "Above 1.0× it EXTRAPOLATES past every spend level the model ever saw — that half is "
    "shaded. Same sweep as the ROAS in 5d1, so the two cannot disagree.\n"
    f"Where the DGP truth falls OUTSIDE {HDI} (red circle, * in the table) the band is "
    f"understating the real error: {sum(not f[7] for f in deviations)} of the {len(deviations)} reported points, "
    f"worst deviation {max(abs(f[4]) for f in deviations):.2f}%.",
    fontsize=11, x=0.012, y=0.995, ha="left", va="top")  # cosmetic
fig.subplots_adjust(left=0.062, right=0.985, top=0.845, bottom=0.065, hspace=0.34, wspace=0.20)  # cosmetic

_path = figure_path(ROOT, "5_business", "5e1_budget_sweep", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)   # with the provenance stamp
print(f"Plot saved to {_path.relative_to(ROOT)}")
print(f"  {'channel':8s}{'mult':>6s}{'model(M)':>11s}{'truth(M)':>11s}{'dev':>9s}   {HDI}")
for ch, m, mv, vv, d, l, h, inside in deviations:
    print(f"  {ch:8s}{m:>6.1f}{mv:>11.2f}{vv:>11.2f}{d:>8.2f}%   [{l:.2f}, {h:.2f}]"
          + ("" if inside else f"  <- truth OUTSIDE the {HDI}"))
_outside = sum(1 for f in deviations if not f[7])
_worst = max(abs(f[4]) for f in deviations)
print(f"  coverage at the reported points: {len(deviations)-_outside}/{len(deviations)} inside the "
      f"{HDI}; largest deviation {_worst:.2f}%")
