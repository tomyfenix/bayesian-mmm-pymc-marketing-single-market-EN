"""Relative-error forest: the 15 parameters with their bias and the HDI94 of each one.

Produces: model/4_recovery/plots/4c1_relative_error_forest__<dataset_label>.png
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
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, MISS_COLOR, apply_style
from viz.labels import (AXIS_REL_ERROR, BIAS, TRUTH, TRUE_VALUE, HDI, CRITERION_A, CRITERION_B,
                        CRITERION_A_RULE, CRITERION_B_CAVEAT, CONTROL_DISPLAY)
from viz.io import figure_path, save_figure

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)

table = validate_recovery(idata, geo="Geo_Centro", trade="Retail", idata_path_for_staleness_check=IDATA_PATH)
table = table.set_index("parameter")
hdi_lo_col = [c for c in table.columns if c.endswith("_lo")][0]
hdi_hi_col = [c for c in table.columns if c.endswith("_hi")][0]

GRID_ORDER = (
    [f"adstock_alpha[{c.replace('_impressions_000', '')}]" for c in CHANNEL_COLUMNS]
    + [f"saturation_lam[{c.replace('_impressions_000', '')}]" for c in CHANNEL_COLUMNS]
    + [f"saturation_beta[{c.replace('_impressions_000', '')}] (L)" for c in CHANNEL_COLUMNS]
    + ["gamma_control[log_price_c]", "gamma_control[promo_flag]", "gamma_control[log_acv_c]"]
)
DISPLAY_NAME = {
    **CONTROL_DISPLAY,
}
FAMILY_BOUNDARIES = [4, 8, 12]  # after these rows (0-indexed) there is a family break  # cosmetic

n = len(GRID_ORDER)
y_pos = np.arange(n)[::-1]  # row 0 (adstock_alpha[tv]) on top

fig, ax = plt.subplots(figsize=(11, 9))  # cosmetic

max_observed_bias = float(table["relative_bias_pct"].abs().max())
n_not_covered = int((~table["covers_hdi"].astype(bool)).sum())

ax.axvspan(-max_observed_bias, max_observed_bias, color=INK_MUTED, alpha=0.12, zorder=0,  # cosmetic
           label=(f"{CRITERION_B} — band = ±{max_observed_bias:.2f}% = the largest {BIAS} "
                  f"OBSERVED in this very fit.\n15/15 fall inside — {CRITERION_B_CAVEAT}."))
ax.axvline(0, color=INK, lw=1.6, ls="-", zorder=4,  # cosmetic
           label=(f"{CRITERION_A} — {CRITERION_A_RULE}"
                  f"\n{n_not_covered}/15 fail it."))

for i, name in enumerate(GRID_ORDER):
    r = table.loc[name]
    bias = float(r["relative_bias_pct"])
    true_val = float(r["true_value"])
    hdi_lo_rel = 100 * (r[hdi_lo_col] - true_val) / true_val  # trivial: unit
    hdi_hi_rel = 100 * (r[hdi_hi_col] - true_val) / true_val  # trivial: unit
    covers = bool(r["covers_hdi"])
    color = MISS_COLOR if not covers else INK_SECONDARY
    y = y_pos[i]
    ax.plot([hdi_lo_rel, hdi_hi_rel], [y, y], color=color, lw=2.2, solid_capstyle="round", zorder=2)  # cosmetic
    ax.scatter([bias], [y], color=color, s=32, zorder=3, edgecolor="white", linewidth=0.6)  # cosmetic
    if not covers:
        # Dotted connector from the closest end of the bar to 0. It makes VISIBLE the gap that
        # separates the HDI94 from the truth; without it you have to follow the bar with your eyes
        # to the 0 line to realize it does not reach it.
        edge = hdi_hi_rel if hdi_hi_rel < 0 else hdi_lo_rel
        ax.plot([edge, 0], [y, y], color=MISS_COLOR, lw=1.0, ls=(0, (2, 2)), zorder=2)  # cosmetic

# Legend entry of the connector: it is added only once, by hand (drawing it inside the loop would add
# 6 identical entries).
ax.plot([], [], color=MISS_COLOR, lw=1.0, ls=(0, (2, 2)),  # cosmetic
        label=f"gap between the {HDI} and the {TRUTH} (those failing criterion A)")
# The legend goes BELOW the axes, not inside: inside (lower right) it covered the gamma row (ACV
# elasticity), whose point falls almost on 0. A legend that hides a data point is worse than a
# legend far away.
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.085), fontsize=7.5, frameon=False,  # cosmetic
          labelspacing=0.75, borderpad=0.0, handlelength=2.4)  # cosmetic

# family separators
for b in FAMILY_BOUNDARIES:
    y_sep = y_pos[b] + 0.5  # cosmetic
    ax.axhline(y_sep, color=GRID, lw=1.0, zorder=0)

ax.set_yticks(y_pos)
ax.set_yticklabels([DISPLAY_NAME.get(n, n) for n in GRID_ORDER], fontsize=9)  # cosmetic
ax.set_xlabel(AXIS_REL_ERROR)
ax.set_title(
    f"4c1 — BOTH recovery criteria on a single axis (0% = {TRUE_VALUE})\n"
    f"CRITERION A · coverage: {n_not_covered}/15 {HDI} intervals miss the {TRUTH} (in red)\n"
    f"CRITERION B · magnitude: max |{BIAS}| = {max_observed_bias:.2f}% — the grey band IS that "
    "maximum, not a threshold",
    fontsize=10.5, loc="left",  # cosmetic
)
ax.grid(axis="x", lw=0.5, color=GRID)  # cosmetic
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.set_ylim(-0.6, n - 0.4)  # cosmetic


fig.tight_layout()
_path = figure_path(ROOT, "4_recovery", "4c1_relative_error_forest", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)
print(f"Plot saved to {_path.relative_to(ROOT)}")
print(f"Band = +-{max_observed_bias:.4f}% (largest bias observed in this fit, not an independent threshold)")
