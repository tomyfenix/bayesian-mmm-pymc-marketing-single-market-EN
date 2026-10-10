"""The specification of the linear model with trend. It builds the model and prints it: it fits nothing
and writes nothing to disk.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.mmm_dgp_config import adstock_mode  # noqa: E402
from config.diagnostics import COLLINEARITY_THRESHOLD  # noqa: E402
from config.mmm_dgp_config import FOURIER_MU_PRIOR  # noqa: E402
from config.contracts import contract, require_regime  # noqa: E402
require_regime("log", __file__)  # this script belongs to the log regime: with another dataset it stops here
from config.mmm_dgp_config import (L_MAX, CHANNEL_COLUMNS, adstock_alpha_prior_arrays, saturation_beta_sigma_array,
                                   saturation_lam_prior,
                                   intercept_prior, YEARLY_SEASONALITY)

# ---------------------------------------------------------------------------
# ! The prior of each parameter and where it comes from: see the Priors section of the README.

df = pd.read_csv(ROOT / "synthetic_data" / "dataset.csv", parse_dates=["date"])

df_model = (
    df
    .query("geo == 'Geo_Centro' and trade_channel == 'Retail'")
    .sort_values("date")
    .reset_index(drop=True)
)

print(f"Total rows: {len(df)} -> rows after filter: {len(df_model)}")

df_model["log_price"] = np.log(df_model["price"])
df_model["log_acv"] = np.log(df_model["acv"])
df_model["y_log_sales"] = np.log(df_model["sales"])

TARGET_SCALE_EMPIRICAL = float(np.maximum(np.abs(df_model["y_log_sales"]).max(), 1.0))
print(f"\nempirical target_scale_ (log(sales)): {TARGET_SCALE_EMPIRICAL:.5f}")

df_model["log_price_c"] = df_model["log_price"] - df_model["log_price"].mean()
df_model["log_acv_c"] = df_model["log_acv"] - df_model["log_acv"].mean()

N = len(df_model)
t_norm = np.arange(N) / N
df_model["t_norm_c"] = t_norm - t_norm.mean()
print(f"t_norm_c: range [{df_model['t_norm_c'].min():.4f}, {df_model['t_norm_c'].max():.4f}]")

corr_acv_trend = np.corrcoef(df_model["log_acv_c"], df_model["t_norm_c"])[0, 1]
# The label comes from the measured value against the threshold.
_level = "HIGH" if abs(corr_acv_trend) >= COLLINEARITY_THRESHOLD else "low"
print(f"Correlation log_acv_c vs t_norm_c: {corr_acv_trend:+.4f}  "
      f"({_level}; threshold {COLLINEARITY_THRESHOLD}). Centering does not change it.")

channel_columns = CHANNEL_COLUMNS  # single source in config/mmm_dgp_config.py
control_columns = ["log_price_c", "promo_flag", "log_acv_c", "t_norm_c"]
target_column = "y_log_sales"

_alphas, _betas = adstock_alpha_prior_arrays(channel_columns)  # by name in config, not positional
_alpha_prior_alpha = xr.DataArray(_alphas, dims="channel", coords={"channel": channel_columns})
_alpha_prior_beta = xr.DataArray(_betas, dims="channel", coords={"channel": channel_columns})
adstock = GeometricAdstock(
    l_max=L_MAX,  # imported from config
    mode=adstock_mode(),
    priors={"alpha": Prior("Beta", alpha=_alpha_prior_alpha, beta=_alpha_prior_beta, dims="channel")},
)

saturation = LogisticSaturation(
    priors={
        "lam": saturation_lam_prior(),   # one single place in config/
        "beta": Prior(
            "HalfNormal",
            sigma=xr.DataArray(
                saturation_beta_sigma_array(channel_columns), dims="channel", coords={"channel": channel_columns}
            ),
            dims="channel",
        ),
    }
)

# ---------------------------------------------------------------------------
# The priors come from config/, not from copied literals: a belief in two places gets out of sync. It
# is the same function 3a_ uses, so the two models cannot differ.
_pri = contract("log").fit_priors(TARGET_SCALE_EMPIRICAL, channel_columns)
assert _pri["controls"] == control_columns, (_pri["controls"], control_columns)
_ctrl = {"control": control_columns}
gamma_control_prior = Prior(
    "TruncatedNormal",
    mu=xr.DataArray(_pri["gamma_mu"], dims="control", coords=_ctrl),
    sigma=xr.DataArray(_pri["gamma_sigma"], dims="control", coords=_ctrl),
    lower=xr.DataArray(_pri["gamma_lower"], dims="control", coords=_ctrl),
    upper=xr.DataArray(_pri["gamma_upper"], dims="control", coords=_ctrl),
    dims="control",
)

likelihood_prior = Prior(
    "Normal",
    sigma=Prior("HalfNormal", sigma=_pri["likelihood_sigma"], dims=()),
    dims=("date",),
)
gamma_fourier_prior = Prior("Laplace", mu=FOURIER_MU_PRIOR, b=_pri["fourier_b"], dims="fourier_mode")

mmm = MMM(
    date_column="date",
    channel_columns=channel_columns,
    control_columns=control_columns,
    adstock=adstock,
    saturation=saturation,
    yearly_seasonality=YEARLY_SEASONALITY,
    time_varying_intercept=False,
    model_config={
        "intercept": intercept_prior(),
        "gamma_control": gamma_control_prior,
        "gamma_fourier": gamma_fourier_prior,
        "likelihood": likelihood_prior,
    },
)

X = df_model[["date"] + channel_columns + control_columns]
y = df_model[target_column]

mmm.build_model(X, y)

assert abs(mmm.target_scale - TARGET_SCALE_EMPIRICAL) < 1e-6, (  # trivial: floating-point tolerance
    f"the library's target_scale_ ({mmm.target_scale}) does not match the one "
    f"computed by hand ({TARGET_SCALE_EMPIRICAL}) -- check before going on."
)
print(f"\ntarget_scale_ (mmm.target_scale) = {mmm.target_scale:.4f} -- OK, it matches")

print("\n" + "=" * 80)  # cosmetic
print("MODEL SPECIFICATION (review before fitting)")
print("=" * 80)  # cosmetic
print(mmm.model.str_repr())

print("\nModel coordinates:")
for coord_name, coord_vals in mmm.model.coords.items():
    vals = list(coord_vals)
    if coord_name == "date":
        print(f"  - {coord_name}: {len(vals)} weeks, from {vals[0].date()} to {vals[-1].date()}")
    elif len(vals) > 10:  # cosmetic
        print(f"  - {coord_name}: {len(vals)} values (0..{len(vals) - 1})")
    else:
        print(f"  - {coord_name}: {vals}")

print("\nShapes of the free variables:")
for rv in mmm.model.free_RVs:
    print(f"  - {rv.name}: {rv.eval().shape}")

# ---------------------------------------------------------------------------
# ! log_acv_c and t_norm_c correlate -0.05 in this dataset: their effects DO separate.
