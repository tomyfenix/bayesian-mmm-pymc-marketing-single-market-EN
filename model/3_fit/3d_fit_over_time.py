"""Fit over time: observed against fitted sales week by week, with their interval.

Produces: model/3_fit/plots/3d1_fit_over_time__<dataset_label>.png
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.contracts import require_regime  # noqa: E402
require_regime("log", __file__)  # this script belongs to the log regime: with another dataset it stops here
from config.idata_paths import require_idata
from config.mmm_dgp_config import HDI_PROB
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale
from viz.style import INK, INK_SECONDARY, INK_MUTED, GRID, POSTERIOR_COLOR, apply_style
from viz.labels import WEEK, WEEKS, HDI
from viz.io import figure_path, save_figure

apply_style()

DATASET_LABEL = get_dataset_label()
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)

dates = pd.to_datetime(idata.posterior.coords["date"].values)
target_scale = float(idata.constant_data["target_scale"].values)

observed_sales = np.exp(idata.observed_data["y"].values * target_scale)
# NOTE: y_original_scale ALREADY has target_scale applied (it is pm.Deterministic("y_original_scale",
# mu*target_scale) on the pymc-marketing side) -- it is NOT multiplied again (the range of this
# variable was checked before using it, which avoids the double-scaling error).
fitted_draws = np.exp(idata.posterior["y_original_scale"].values)  # (chain, draw, date)
fitted_flat = fitted_draws.reshape(-1, fitted_draws.shape[-1])  # (sample, date)
fitted_median = np.median(fitted_flat, axis=0)
hdi = az.hdi(idata.posterior["y_original_scale"], hdi_prob=HDI_PROB)
hdi_lo = np.exp(hdi["y_original_scale"].sel(hdi="lower").values)
hdi_hi = np.exp(hdi["y_original_scale"].sel(hdi="higher").values)

fig, ax = plt.subplots(figsize=(14, 6.8))  # cosmetic
ax.fill_between(dates, hdi_lo, hdi_hi, color=POSTERIOR_COLOR, alpha=0.30, label=f"fitted mean, {HDI}")  # cosmetic
ax.plot(dates, fitted_median, color=POSTERIOR_COLOR, lw=1.6, label="fitted mean, median")  # cosmetic
ax.plot(dates, observed_sales, color=INK, lw=1.0, ls="-", marker="o", markersize=2.5, label="observed")  # cosmetic

max_abs_pct_err = float(np.max(np.abs(observed_sales - fitted_median) / observed_sales) * 100)  # trivial: unit
ax.set_title(
    f"3d1 — Observed vs. FITTED MEAN, 156 {WEEKS} (largest pointwise error: {max_abs_pct_err:.2f}%)\n"
    f"This is the fitted mean with PARAMETER uncertainty (y_original_scale), NOT a posterior\n"
    f"predictive replicate with observation noise — it is not a PPC. At sigma_eps=0.001 the two are\n"
    f"indistinguishable at this scale, and the fit is NEAR-TAUTOLOGICAL: it is not evidence of anything.",
    fontsize=9.5, color=INK, loc="left",  # cosmetic
)
ax.set_xlabel(WEEK)
ax.set_ylabel("sales")
ax.legend(frameon=False, fontsize=9, loc="upper left")  # cosmetic
ax.grid(axis="y", lw=0.5, color=GRID)  # cosmetic
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()

_path = figure_path(ROOT, "3_fit", "3d1_fit_over_time", DATASET_LABEL)
save_figure(fig, _path, script_file=__file__)
print(f"Plot saved to {_path.relative_to(ROOT)}")
print(f"Largest pointwise absolute % error: {max_abs_pct_err:.4f}%")
