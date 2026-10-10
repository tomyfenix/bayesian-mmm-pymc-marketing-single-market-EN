"""Price and distribution (ACV) against their period average: how much they added to or took away
from sales, and how that compares with the true elasticity.

The title of each panel gives the estimated elasticity (gamma_control times target_scale, on the
elasticity scale), its HDI and the generator's true one.

Produces: model/5_business/plots/5c1_price_and_acv__<dataset_label>.png
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
from config.mmm_dgp_config import CANONICAL_TRADE, HDI_PROB
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.business import canonical_slice, centering_reference
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, TRUTH_COLOR, POSTERIOR_COLOR, apply_style
from viz.labels import TRUTH, HDI
from viz.io import figure_path, save_figure
from viz.intervals import hdi

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
dfm = canonical_slice()
dates = dfm["date"].to_numpy()

import json
TP = json.loads((ROOT / "synthetic_data" / "true_parameters.json").read_text(encoding="utf-8"))
tp = TP["trade_params"][CANONICAL_TRADE]

ctrl = idata.posterior["control_contribution_original_scale"]
names = [str(x) for x in ctrl.coords["control"].values]
# the estimated elasticity: gamma_control lives on the scale of the target divided by target_scale
target_scale = float(idata.constant_data["target_scale"])
gamma = idata.posterior["gamma_control"]

# (model variable, CSV column, true elasticity, title)
PANELS = [
    ("log_price_c", "price", tp["price_elasticity"], "Price"),
    ("log_acv_c", "acv", tp["distribution_elasticity"], "Distribution (ACV)"),
]

fig = plt.figure(figsize=(16.5, 8.4))  # cosmetic
gs = fig.add_gridspec(2, 2, width_ratios=[3.05, 1.0], wspace=0.20, hspace=0.42,  # cosmetic
                      left=0.065, right=0.975, top=0.815, bottom=0.075)  # cosmetic

summary = []
for row, (var, col, elast, title) in enumerate(PANELS):
    ref = centering_reference(dfm, col)                     # GEOMETRIC mean
    x = dfm[col].to_numpy()
    # Effect on sales in %: exp(contribution in log-points) - 1.
    c = ctrl.sel(control=var).stack(m=("chain", "draw")).transpose("m", "date").values
    pct = 100 * (np.exp(c) - 1)                             # (n_draws, n_weeks)  # trivial: unit
    med = pct.mean(axis=0)
    # pct[None] adds the chain axis: with 2d input arviz warns that the (draw, shape) interpretation
    # will change in a future version. Passing (chain, draw, shape) leaves no ambiguity.
    lo, hi = az.hdi(pct[None, ...], hdi_prob=HDI_PROB).T
    truth = 100 * (np.exp(elast * np.log(x / ref)) - 1)    # same reference as the model  # trivial: unit
    e_draws = gamma.sel(control=var).values.ravel() * target_scale
    e_med, (e_lo, e_hi) = float(e_draws.mean()), hdi(e_draws)
    inside = e_lo <= elast <= e_hi

    ax = fig.add_subplot(gs[row, 0])
    ax.axhline(0, color=INK, lw=1.1, zorder=3)  # cosmetic
    # Same criterion as 5b1: the band is drawn, but its maximum width is REPORTED, because at this
    # scale it is thinner than the line and a legend that names something invisible tells nothing.
    width = float(np.max(hi - lo))
    ax.fill_between(dates, lo, hi, color=POSTERIOR_COLOR, alpha=0.30, lw=0, zorder=1,  # cosmetic
                    label=f"{HDI} (at most {width:.2f} pp wide — thinner than the line)")
    ax.plot(dates, med, color=POSTERIOR_COLOR, lw=1.7, zorder=2, label="posterior mean")  # cosmetic
    ax.plot(dates, truth, color=TRUTH_COLOR, lw=1.2, ls="--", zorder=4, label=f"DGP {TRUTH}")  # cosmetic
    ax.set_ylabel("effect on sales, %")
    ax.grid(axis="y", lw=0.5, color=GRID)  # cosmetic
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title(
        f"{title}: elasticity {e_med:+.3f}  {HDI} [{e_lo:+.3f}, {e_hi:+.3f}]  ·  true {elast:+.3f}, "
        f"{'inside' if inside else 'OUTSIDE'} the {HDI}\n"
        f"Measured against its period average   ·   REFERENCE: {ref:.4f} "
        f"(geometric mean of {col}; the arithmetic mean, {x.mean():.4f}, is NOT the reference)",
        fontsize=10.5, loc="left", color=INK, pad=26)  # cosmetic
    if row == 0:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), fontsize=8.5,  # cosmetic
                  frameon=False, ncols=3)  # cosmetic

    # ---- summary on the right: period average, best and worst week
    ax2 = fig.add_subplot(gs[row, 1])
    avg_draws = pct.mean(axis=1)
    rows = [("period average", avg_draws.mean(), truth.mean(), az.hdi(avg_draws, hdi_prob=HDI_PROB)),
             ("most favourable week", med.max(), truth.max(), None),
             ("least favourable week", med.min(), truth.min(), None)]
    yy = np.arange(3)[::-1]  # trivial: the three rows of the panel
    for k, (lbl, val, tru, h) in enumerate(rows):
        ax2.barh(yy[k], val, height=0.42, color=INK_MUTED if k == 0 else INK_SECONDARY,  # cosmetic
                 alpha=0.85, zorder=2)  # cosmetic
        ax2.scatter([tru], [yy[k]], marker="D", s=34, facecolor="white", edgecolor=TRUTH_COLOR,  # cosmetic
                    lw=1.5, zorder=4)  # cosmetic
        if h is not None:
            ax2.plot(list(h), [yy[k], yy[k]], color=INK, lw=1.5, zorder=3, solid_capstyle="butt")  # cosmetic
        ax2.text(val + (0.55 if val >= 0 else -0.55), yy[k], f"{val:+.2f}%", va="center",  # cosmetic
                 ha="left" if val >= 0 else "right", fontsize=9, color=INK)  # cosmetic
    ax2.axvline(0, color=INK, lw=1.0, zorder=1)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([f[0] for f in rows], fontsize=8.5)  # cosmetic
    lim = max(abs(med.max()), abs(med.min())) * 1.22  # cosmetic
    ax2.set_xlim(-lim, lim)
    ax2.set_xlabel("effect on sales, %", fontsize=9)  # cosmetic
    ax2.grid(axis="x", lw=0.5, color=GRID)  # cosmetic
    for s in ["top", "right", "left"]:
        ax2.spines[s].set_visible(False)
    ax2.set_title("summary", fontsize=10, loc="left", color=INK_SECONDARY)  # cosmetic
    summary.append((title, ref, x.mean(), avg_draws.mean(), truth.mean(), med.min(), med.max(),
                    e_med, e_lo, e_hi, elast, inside))

fig.suptitle(
    "5c1 — Price and distribution, measured against the period average\n"
    "Zero is not \"no price\": it is the week where the variable sits exactly at its period "
    "reference, declared in each panel title. Above zero the variable pushed sales up, below "
    "zero it held them back.\n"
    "The reference is the GEOMETRIC mean, because that is what the model centres on — the "
    "arithmetic mean would move the zero line and flip which weeks read as favourable.",
    fontsize=11, x=0.015, y=0.995, ha="left", va="top")  # cosmetic

_path = figure_path(ROOT, "5_business", "5c1_price_and_acv", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)   # with the provenance stamp
print(f"Plot saved to {_path.relative_to(ROOT)}")
for title, ref, arith, avg, avg_t, mn, mx, e_med, e_lo, e_hi, elast, inside in summary:
    print(f"  {title:20s} geometric ref={ref:9.4f} (arithmetic {arith:9.4f})  "
          f"average={avg:+.3f}% (truth {avg_t:+.3f}%)  range [{mn:+.2f}%, {mx:+.2f}%]")
    print(f"  {'':20s} elasticity {e_med:+.4f}  {HDI} [{e_lo:+.4f}, {e_hi:+.4f}]  true {elast:+.4f}  "
          f"({'inside' if inside else 'OUTSIDE'} the {HDI})")
