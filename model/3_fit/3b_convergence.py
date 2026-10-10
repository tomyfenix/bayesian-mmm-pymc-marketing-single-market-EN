"""Convergence diagnostic: rank plots by parameter and the sampler's energy histogram.

Produces: model/3_fit/plots/3b1_convergence__<dataset_label>.png
"""

import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.contracts import require_regime  # noqa: E402
require_regime("log", __file__)  # this script belongs to the log regime: with another dataset it stops here
from config.mmm_dgp_config import CHANNEL_COLUMNS
from config.idata_paths import require_idata
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.style import INK, INK_SECONDARY, apply_style
from viz.labels import CONVERGENCE, ENERGY, RANK_PLOTS, CONTROL_DISPLAY_SHORT
from viz.io import figure_path, save_figure

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)

# Same order as 4b1_prior_posterior, so that the two figures read side by side.
GRID_ORDER = (
    [("adstock_alpha", c) for c in CHANNEL_COLUMNS]
    + [("saturation_lam", c) for c in CHANNEL_COLUMNS]
    + [("saturation_beta", c) for c in CHANNEL_COLUMNS]
    + [("gamma_control", c) for c in ["log_price_c", "promo_flag", "log_acv_c"]]
)
DISPLAY_NAME = CONTROL_DISPLAY_SHORT


def _short(dim, coord):
    return coord.replace("_impressions_000", "") if dim == "channel" else DISPLAY_NAME.get(coord, coord)


fig, axes = plt.subplots(4, 4, figsize=(16, 13.6))  # cosmetic

for idx, (param, coord) in enumerate(GRID_ORDER):
    row, col = divmod(idx, 4)  # cosmetic
    ax = axes[row, col]
    dim = "control" if param == "gamma_control" else "channel"
    az.plot_rank(idata, var_names=[param], coords={dim: [coord]}, kind="bars", ax=ax)
    label = _short(dim, coord)
    ax.set_title(f"{param}[{label}]" if dim == "channel" else label, fontsize=9, color=INK)  # cosmetic
    ax.set_xlabel("")
    ax.set_ylabel("")

# Panel 16: energy + BFMI
ax_energy = axes[3, 3]
az.plot_energy(idata, ax=ax_energy)
bfmi = az.bfmi(idata)
ax_energy.set_title(
    f"{ENERGY.capitalize()} (BFMI per chain: {', '.join(f'{b:.2f}' for b in bfmi)})",
    fontsize=8.5, color=INK,  # cosmetic
)
ax_energy.legend(fontsize=7, frameon=False)  # cosmetic

fig.suptitle(
    f"3b1 — Sampling {CONVERGENCE}: {RANK_PLOTS} (15 parameters) + {ENERGY}/BFMI\n"
    "(uniform ranks across chains = good mixing; see Gabry et al. 2019, section 4)",
    fontsize=12, x=0.02, ha="left",  # cosmetic
)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # cosmetic

_path = figure_path(ROOT, "3_fit", "3b1_convergence", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)
print(f"Plot saved to {_path.relative_to(ROOT)}")
print(f"BFMI per chain: {bfmi}")
n_div = int(idata.sample_stats["diverging"].values.sum())
print(f"Divergences: {n_div}/{idata.sample_stats['diverging'].size}")
