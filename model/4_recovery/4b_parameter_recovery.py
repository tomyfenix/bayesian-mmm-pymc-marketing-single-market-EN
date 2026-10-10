"""Parameter recovery: prior, posterior and true value of the 15 parameters in a single figure, plus
the table of relative bias and coverage.

Produces: results/4b_parameter_recovery__<dataset_label>.csv,
    results/4b_contraction_displacement__<dataset_label>.csv,
    plots/4b1_prior_posterior__<dataset_label>.png
"""

import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.mmm_dgp_config import adstock_mode  # noqa: E402
from config.diagnostics import (SAMPLING_SEED, N_PRIOR_SAMPLES,  # noqa: E402
                                INFORMED_CONTRACTION_THRESHOLD, INFORMED_DISPLACEMENT_THRESHOLD)
from config.mmm_dgp_config import FOURIER_MU_PRIOR  # noqa: E402
from config.contracts import require_regime  # noqa: E402
require_regime("log", __file__)  # this script belongs to the log regime: with another dataset it stops here
from config.mmm_dgp_config import (L_MAX, CHANNEL_COLUMNS, adstock_alpha_prior_arrays, contraction_sd,
                                   saturation_lam_prior, intercept_prior, check_prior_scale,
                                   YEARLY_SEASONALITY, HDI_PROB, CATEGORY_TREND_PER_WEEK)
from config.contracts import contract
from config.idata_paths import require_idata
from config.log_regime import fit_gamma_control, fit_priors
from synthetic_data.true_params_reader import get_true_L, get_channel_params, get_demand_params, get_dataset_label, warn_if_idata_stale

from viz.canonical import validate_recovery
from viz.style import (
    INK, INK_SECONDARY, INK_MUTED, GRID, BASELINE, SURFACE,
    PRIOR_COLOR as BLUE, POSTERIOR_COLOR as ORANGE, MISS_COLOR as MISS_RED,
    apply_style,
)
from viz.overlay import prior_posterior_two_scales
from viz.labels import (BIAS, TRUE_VALUE, HDI, MISSES_HDI, CONTRACTION, CONTRACTION_SHORT,
                        FULL_PRIOR, ZOOM_TO_WINDOW, POSTERIOR_WINDOW, DATASET_LABEL_NOTE,
                        CONTROL_DISPLAY)
from viz.io import figure_path, save_figure
from viz.intervals import hdi

AQUA = "#1baf7a"  # accent color for the annotation -- it is not part of the semantic palette of
                   # viz.style (it does not stand for prior/posterior/truth/miss), so it stays local
                   # to this script.

apply_style()

SEED = SAMPLING_SEED
# the interval of the table is the repo-wide HDI (HDI_PROB, viz/intervals.py), the same as 4a_'s
_LO, _HI = f"hdi{round(HDI_PROB * 100)}_lo", f"hdi{round(HDI_PROB * 100)}_hi"  # trivial: unit

CONTROL_COLUMNS = ["log_price_c", "promo_flag", "log_acv_c", "t_norm_c"]


def build_mmm():
    df = pd.read_csv(ROOT / "synthetic_data" / "dataset.csv", parse_dates=["date"])
    df_model = (
        df.query("geo == 'Geo_Centro' and trade_channel == 'Retail'")
        .sort_values("date").reset_index(drop=True)
    )
    df_model["log_price"] = np.log(df_model["price"])
    df_model["log_acv"] = np.log(df_model["acv"])
    df_model["log_price_c"] = df_model["log_price"] - df_model["log_price"].mean()
    df_model["log_acv_c"] = df_model["log_acv"] - df_model["log_acv"].mean()
    N = len(df_model)
    t_norm = np.arange(N) / N
    df_model["t_norm_c"] = t_norm - t_norm.mean()
    df_model["y_log_sales"] = np.log(df_model["sales"])

    X = df_model[["date"] + CHANNEL_COLUMNS + CONTROL_COLUMNS]
    y = df_model["y_log_sales"]
    TS = float(np.maximum(np.abs(y).max(), 1.0))
    pri = fit_priors(TS, CHANNEL_COLUMNS)   # the priors of 3a_, from config/

    _alphas, _betas = adstock_alpha_prior_arrays(CHANNEL_COLUMNS)
    _alpha_prior_alpha = xr.DataArray(_alphas, dims="channel", coords={"channel": CHANNEL_COLUMNS})
    _alpha_prior_beta = xr.DataArray(_betas, dims="channel", coords={"channel": CHANNEL_COLUMNS})
    adstock = GeometricAdstock(l_max=L_MAX, mode=adstock_mode(), priors={"alpha": Prior("Beta", alpha=_alpha_prior_alpha, beta=_alpha_prior_beta, dims="channel")})
    _beta_sigma = xr.DataArray(pri["beta_sigma"], dims="channel", coords={"channel": CHANNEL_COLUMNS})
    saturation = LogisticSaturation(priors={
        "lam": saturation_lam_prior(),   # one single place in config/
        "beta": Prior("HalfNormal", sigma=_beta_sigma, dims="channel"),
    })
    _mu, _sg, _lo, _hi = fit_gamma_control(TS, CONTROL_COLUMNS)   # the same priors as 3a_, from config/
    gamma_mu = xr.DataArray(_mu, dims="control", coords={"control": CONTROL_COLUMNS})
    gamma_sigma = xr.DataArray(_sg, dims="control", coords={"control": CONTROL_COLUMNS})
    gamma_lower = xr.DataArray(_lo, dims="control", coords={"control": CONTROL_COLUMNS})
    gamma_upper = xr.DataArray(_hi, dims="control", coords={"control": CONTROL_COLUMNS})
    gamma_control_prior = Prior(
        "TruncatedNormal", mu=gamma_mu, sigma=gamma_sigma, lower=gamma_lower, upper=gamma_upper, dims="control"
    )
    likelihood_prior = Prior("Normal", sigma=Prior("HalfNormal", sigma=pri["likelihood_sigma"], dims=()), dims=("date",))
    gamma_fourier_prior = Prior("Laplace", mu=FOURIER_MU_PRIOR, b=pri["fourier_b"], dims="fourier_mode")

    mmm = MMM(
        date_column="date", channel_columns=CHANNEL_COLUMNS, control_columns=CONTROL_COLUMNS,
        adstock=adstock, saturation=saturation, yearly_seasonality=YEARLY_SEASONALITY, time_varying_intercept=False,
        model_config={
            "intercept": intercept_prior(),
            "gamma_control": gamma_control_prior,
            "gamma_fourier": gamma_fourier_prior,
            "likelihood": likelihood_prior,
        },
    )
    mmm.build_model(X, y)
    check_prior_scale(mmm, contract("log"), {"target_scale": TS})   # the same check as 3a_
    return mmm, X, y, df_model


mmm, X, y, df_model = build_mmm()

_DATASET_LABEL = get_dataset_label()
_IDATA_PATH = require_idata("linear_trend", _DATASET_LABEL)
warn_if_idata_stale(_IDATA_PATH)
idata = az.from_netcdf(_IDATA_PATH)
post = idata.posterior

channel_scale = dict(zip(CHANNEL_COLUMNS, np.atleast_1d(mmm.channel_scale)))
target_scale = float(mmm.target_scale)
print("=" * 80)  # cosmetic
print("SCALE FACTORS (verified empirically)")
print("=" * 80)  # cosmetic
for ch in CHANNEL_COLUMNS:
    print(f"  channel_scale[{ch}] = {channel_scale[ch]:.2f}")
print(f"  target_scale_ = {target_scale:.4f}")

with open(ROOT / "synthetic_data" / "true_parameters.json") as f:
    true_params = json.load(f)

# L, lam and the demand parameters are read ALREADY COMPOSED from effective_params via
# true_params_reader -- they are not rebuilt by hand (a rebuild by hand goes stale without warning).
demand_true = get_demand_params("Geo_Centro", "Retail")

N_WEEKS = len(df_model)
# The truth, in the SAME unit as the posterior: growth in log from the first to the last week. The
# generator puts the trend inside the log (log(1 + seasonality + trend)), so its trend part at t = N-1
# is log(1 + CATEGORY_TREND_PER_WEEK * (N-1)); the seasonality, which is also inside, makes it an
# approximation and not an identity.
TREND_TRUE_TOTAL = float(np.log1p(CATEGORY_TREND_PER_WEEK * (N_WEEKS - 1)))
# the posterior of gamma_control[t_norm_c] is per unit of t_norm_c: times the range of t_norm_c in
# the slice (first to last week) it gives the growth in log over the period
_T_NORM_RANGE = float(df_model["t_norm_c"].iloc[-1] - df_model["t_norm_c"].iloc[0])
print(f"\n  True cumulative trend log(1 + {CATEGORY_TREND_PER_WEEK:g} * {N_WEEKS-1}) = {TREND_TRUE_TOTAL:.4f} "
      f"({(np.exp(TREND_TRUE_TOTAL)-1)*100:.1f}% growth)")  # trivial: unit

# ===========================================================================
# STEP 0: prior -> posterior CONTRACTION AND DISPLACEMENT (ALL parameters)
print("\n" + "=" * 80)  # cosmetic
print("STEP 0: prior -> posterior CONTRACTION AND DISPLACEMENT")
print("=" * 80)  # cosmetic
print(f"Sampling {N_PRIOR_SAMPLES} prior draws (final model, not fitted)...")

mmm_prior, _, _, _ = build_mmm()
mmm_prior.sample_prior_predictive(X, y, samples=N_PRIOR_SAMPLES, random_seed=SEED, extend_idata=True)
prior = mmm_prior.idata.prior

PARAM_GROUPS = {
    "adstock_alpha": ("channel", CHANNEL_COLUMNS),
    "saturation_lam": ("channel", CHANNEL_COLUMNS),
    "saturation_beta": ("channel", CHANNEL_COLUMNS),
    "gamma_control": ("control", CONTROL_COLUMNS),
    "gamma_fourier": ("fourier_mode", ["sin_1", "sin_2", "cos_1", "cos_2"]),
}
SCALAR_PARAMS = ["intercept", "y_sigma"]

rows = []
for param, (dim_name, coords) in PARAM_GROUPS.items():
    for coord in coords:
        prior_vals = prior[param].sel({dim_name: coord}).values.ravel()
        post_vals = post[param].sel({dim_name: coord}).values.ravel()
        sd_prior = prior_vals.std()
        # contraction in SD, not in variance -- see the docstring of contraction_sd in
        # config/mmm_dgp_config.py.
        contraction = contraction_sd(prior_vals, post_vals)
        displacement = abs(post_vals.mean() - prior_vals.mean()) / sd_prior
        rows.append({
            "param": param, "coord": coord,
            "prior_mean": prior_vals.mean(), "prior_sd": sd_prior,
            "post_mean": post_vals.mean(), "post_sd": post_vals.std(),
            "contraction": contraction, "displacement": displacement,
        })
for param in SCALAR_PARAMS:
    prior_vals = prior[param].values.ravel()
    post_vals = post[param].values.ravel()
    sd_prior = prior_vals.std()
    contraction = contraction_sd(prior_vals, post_vals)
    displacement = abs(post_vals.mean() - prior_vals.mean()) / sd_prior
    rows.append({
        "param": param, "coord": "-",
        "prior_mean": prior_vals.mean(), "prior_sd": sd_prior,
        "post_mean": post_vals.mean(), "post_sd": post_vals.std(),
        "contraction": contraction, "displacement": displacement,
    })

contraction_df = pd.DataFrame(rows)




def informed_label(row):
    if row["contraction"] > INFORMED_CONTRACTION_THRESHOLD or row["displacement"] > INFORMED_DISPLACEMENT_THRESHOLD:
        return "informed by the data"
    return "DOMINATED BY THE PRIOR"


contraction_df["diagnostic"] = contraction_df.apply(informed_label, axis=1)
_n_informed = int((contraction_df["diagnostic"] == "informed by the data").sum())
if _n_informed == len(contraction_df):
    print(f"\n  NOTE on the `diagnostic` column: {_n_informed}/{len(contraction_df)} "
          f"parameters fall on the same side of the criterion")
    print(f"  (contraction > {INFORMED_CONTRACTION_THRESHOLD} or displacement > "
          f"{INFORMED_DISPLACEMENT_THRESHOLD}). The MINIMUM observed contraction is "
          f"{contraction_df['contraction'].min():.4f},")
    print(f"  that is {contraction_df['contraction'].min()/INFORMED_CONTRACTION_THRESHOLD:.0f}x the threshold: "
          f"the criterion NO longer discriminates in this regime and the column")
    print(f"  is kept as a historical record, not as a test.")
pd.set_option("display.width", 180)  # cosmetic
pd.set_option("display.max_rows", 60)  # cosmetic
print("\n" + contraction_df.round(4).to_string(index=False))  # cosmetic
contraction_df.to_csv(ROOT / "results" / f"4b_contraction_displacement__{_DATASET_LABEL}.csv", index=False)


# ===========================================================================
# STEP 1: full RECOVERY
print("\n" + "=" * 80)  # cosmetic
print("STEP 1: RECOVERY -- adstock_alpha")
print("=" * 80)  # cosmetic
summary_rows = []
for ch in CHANNEL_COLUMNS:
    ch_name = ch.replace("_impressions_000", "")
    post_vals = post["adstock_alpha"].sel(channel=ch).values.ravel()
    true_val = get_channel_params(ch_name)["adstock_alpha"]
    lo, hi = hdi(post_vals)
    mean_val = post_vals.mean()
    contains = lo <= true_val <= hi
    contraction = float(contraction_df[(contraction_df.param == "adstock_alpha") & (contraction_df.coord == ch)]["contraction"].iloc[0])
    displacement = float(contraction_df[(contraction_df.param == "adstock_alpha") & (contraction_df.coord == ch)]["displacement"].iloc[0])
    summary_rows.append({
        "parameter": f"adstock_alpha[{ch_name}]", "true_value": true_val,
        "posterior_mean": mean_val, _LO: lo, _HI: hi,
        "contains_truth": contains, "contraction": contraction, "displacement": displacement,
    })
    print(f"  {ch_name:8s} true={true_val:.3f}  post_mean={mean_val:.3f}  "
          f"{HDI}=[{lo:.3f},{hi:.3f}]  contains={contains}  contr={contraction:.3f}  disp={displacement:.3f}")

print("\n" + "=" * 80)  # cosmetic
print("STEP 1b: RECOVERY -- saturation (lam, L)")
print("=" * 80)  # cosmetic
print("\n-- lam (channel scale: lam_scaled = lam_raw * channel_scale) --")
for ch in CHANNEL_COLUMNS:
    ch_name = ch.replace("_impressions_000", "")
    post_vals = post["saturation_lam"].sel(channel=ch).values.ravel()
    lam_raw_true = get_channel_params(ch_name)["lam"]
    lam_scaled_true = lam_raw_true * channel_scale[ch]
    lo, hi = hdi(post_vals)
    mean_val = post_vals.mean()
    contains = lo <= lam_scaled_true <= hi
    contraction = float(contraction_df[(contraction_df.param == "saturation_lam") & (contraction_df.coord == ch)]["contraction"].iloc[0])
    displacement = float(contraction_df[(contraction_df.param == "saturation_lam") & (contraction_df.coord == ch)]["displacement"].iloc[0])
    summary_rows.append({
        "parameter": f"saturation_lam[{ch_name}]", "true_value": lam_scaled_true,
        "posterior_mean": mean_val, _LO: lo, _HI: hi,
        "contains_truth": contains, "contraction": contraction, "displacement": displacement,
    })
    print(f"  {ch_name:8s} true={lam_scaled_true:.3f}  post_mean={mean_val:.3f}  "
          f"{HDI}=[{lo:.3f},{hi:.3f}]  contains={contains}  contr={contraction:.3f}  disp={displacement:.3f}")

print("\n-- L / beta (target scale: beta_scaled = L_geo_true / target_scale_) --")
for ch in CHANNEL_COLUMNS:
    ch_name = ch.replace("_impressions_000", "")
    post_vals = post["saturation_beta"].sel(channel=ch).values.ravel()
    L_geo_true = get_true_L(ch_name, "Geo_Centro", "Retail")
    beta_scaled_true = L_geo_true / target_scale
    lo, hi = hdi(post_vals)
    mean_val = post_vals.mean()
    contains = lo <= beta_scaled_true <= hi
    contraction = float(contraction_df[(contraction_df.param == "saturation_beta") & (contraction_df.coord == ch)]["contraction"].iloc[0])
    displacement = float(contraction_df[(contraction_df.param == "saturation_beta") & (contraction_df.coord == ch)]["displacement"].iloc[0])
    summary_rows.append({
        "parameter": f"saturation_beta[{ch_name}]", "true_value": beta_scaled_true,
        "posterior_mean": mean_val, _LO: lo, _HI: hi,
        "contains_truth": contains, "contraction": contraction, "displacement": displacement,
    })
    print(f"  {ch_name:8s} true={beta_scaled_true:.4f}  post_mean={mean_val:.4f}  "
          f"{HDI}=[{lo:.4f},{hi:.4f}]  contains={contains}  contr={contraction:.3f}  disp={displacement:.3f}")

print("\n" + "=" * 80)  # cosmetic
print("STEP 1c: RECOVERY -- elasticities and trend")
print("=" * 80)  # cosmetic

price_elasticity_true = demand_true["eta_true"]
distribution_elasticity_true = demand_true["gamma_true"]
promo_lift_true = demand_true["promo_lift"]

post_vals = post["gamma_control"].sel(control="log_price_c").values.ravel() * target_scale
lo, hi = hdi(post_vals)
mean_val = post_vals.mean()
contains = lo <= price_elasticity_true <= hi
contraction = float(contraction_df[(contraction_df.param == "gamma_control") & (contraction_df.coord == "log_price_c")]["contraction"].iloc[0])
displacement = float(contraction_df[(contraction_df.param == "gamma_control") & (contraction_df.coord == "log_price_c")]["displacement"].iloc[0])
summary_rows.append({
    "parameter": "price elasticity", "true_value": price_elasticity_true,
    "posterior_mean": mean_val, _LO: lo, _HI: hi,
    "contains_truth": contains, "contraction": contraction, "displacement": displacement,
})
print(f"  price:       true={price_elasticity_true:.3f}  post_mean={mean_val:.3f}  "
      f"{HDI}=[{lo:.3f},{hi:.3f}]  contains={contains}  contr={contraction:.3f}  disp={displacement:.3f}")

post_vals = post["gamma_control"].sel(control="log_acv_c").values.ravel() * target_scale
lo, hi = hdi(post_vals)
mean_val = post_vals.mean()
contains = lo <= distribution_elasticity_true <= hi
contraction = float(contraction_df[(contraction_df.param == "gamma_control") & (contraction_df.coord == "log_acv_c")]["contraction"].iloc[0])
displacement = float(contraction_df[(contraction_df.param == "gamma_control") & (contraction_df.coord == "log_acv_c")]["displacement"].iloc[0])
summary_rows.append({
    "parameter": "ACV elasticity", "true_value": distribution_elasticity_true,
    "posterior_mean": mean_val, _LO: lo, _HI: hi,
    "contains_truth": contains, "contraction": contraction, "displacement": displacement,
})
print(f"  ACV:         true={distribution_elasticity_true:.3f}  post_mean={mean_val:.3f}  "
      f"{HDI}=[{lo:.3f},{hi:.3f}]  contains={contains}  contr={contraction:.3f}  disp={displacement:.3f}")

gamma_promo_raw = post["gamma_control"].sel(control="promo_flag").values.ravel()
promo_lift_draws = np.exp(gamma_promo_raw * target_scale) - 1
# the HDI is computed on the parameter scale (gamma, log-points), as in 4a_, and its edges are taken to
# lift with exp(.) - 1: the HDI is not invariant to transformations, and this way the promo coverage is
# the same as 4a_'s.
_g_lo, _g_hi = hdi(gamma_promo_raw * target_scale)
lo, hi = float(np.exp(_g_lo) - 1), float(np.exp(_g_hi) - 1)
mean_val = promo_lift_draws.mean()
contains = _g_lo <= np.log1p(promo_lift_true) <= _g_hi
contraction = float(contraction_df[(contraction_df.param == "gamma_control") & (contraction_df.coord == "promo_flag")]["contraction"].iloc[0])
displacement = float(contraction_df[(contraction_df.param == "gamma_control") & (contraction_df.coord == "promo_flag")]["displacement"].iloc[0])
summary_rows.append({
    "parameter": "promo_lift", "true_value": promo_lift_true,
    "posterior_mean": mean_val, _LO: lo, _HI: hi,
    "contains_truth": contains, "contraction": contraction, "displacement": displacement,
})
print(f"  promo:       true={promo_lift_true:.3f}  post_mean={mean_val:.3f}  "
      f"{HDI}=[{lo:.3f},{hi:.3f}]  contains={contains}  contr={contraction:.3f}  disp={displacement:.3f}")

post_vals_trend = post["gamma_control"].sel(control="t_norm_c").values.ravel() * target_scale * _T_NORM_RANGE
lo, hi = hdi(post_vals_trend)
mean_val = post_vals_trend.mean()
contains = lo <= TREND_TRUE_TOTAL <= hi
contraction = float(contraction_df[(contraction_df.param == "gamma_control") & (contraction_df.coord == "t_norm_c")]["contraction"].iloc[0])
displacement = float(contraction_df[(contraction_df.param == "gamma_control") & (contraction_df.coord == "t_norm_c")]["displacement"].iloc[0])
summary_rows.append({
    "parameter": "trend (cumulative log growth)", "true_value": TREND_TRUE_TOTAL,
    "posterior_mean": mean_val, _LO: lo, _HI: hi,
    "contains_truth": contains, "contraction": contraction, "displacement": displacement,
})
print(f"  trend:       true={TREND_TRUE_TOTAL:.4f}  post_mean={mean_val:.4f}  "
      f"{HDI}=[{lo:.4f},{hi:.4f}]  contains={contains}  contr={contraction:.3f}  disp={displacement:.3f}")

summary_df = pd.DataFrame(summary_rows)
# 16 rows: the 15 parameters of 4a_ (4 adstock_alpha, 4 saturation_lam, 4 saturation_beta, price, ACV
# and promo) plus the trend, which 4a_ does not evaluate because t_norm_c has no closed true coefficient
# in the generator (the trend enters inside the log); here it is compared against the true cumulative
# growth. The reported coverage is that of the 15, the same count as 4a_.
summary_df["in_the_15"] = ~summary_df["parameter"].str.startswith("trend")
_n15 = int(summary_df["in_the_15"].sum())
assert _n15 == 15, _n15  # trivial: the 15 parameters 4a_ evaluates
print(f"\n  The true value falls inside the {HDI} in "
      f"{int(summary_df.loc[summary_df['in_the_15'], 'contains_truth'].sum())} of {_n15} parameters "
      f"(the trend, separately: {'inside' if bool(summary_df.loc[~summary_df['in_the_15'], 'contains_truth'].iloc[0]) else 'outside'}).")
summary_df.to_csv(ROOT / "results" / f"4b_parameter_recovery__{_DATASET_LABEL}.csv", index=False)

# ===========================================================================
# STEP 2: divergences -- LOCATION, not just a count
print("\n" + "=" * 80)  # cosmetic
print("STEP 2: LOCATION of the divergences")
print("=" * 80)  # cosmetic
diverging = idata.sample_stats["diverging"].values  # (chain, draw)
n_div = int(diverging.sum())
n_total = diverging.size
print(f"  Divergences: {n_div} of {n_total} draws ({100*n_div/n_total:.2f}%)")  # trivial: unit

if n_div > 0:
    div_flat = diverging.ravel()
    for param in ["adstock_alpha", "saturation_lam", "gamma_control"]:
        dim_name = "channel" if param != "gamma_control" else "control"
        coords = post[param].coords[dim_name].values
        print(f"\n  {param} -- mean in DIVERGENT vs NON-divergent draws:")
        for coord in coords:
            vals = post[param].sel({dim_name: coord}).values.ravel()
            mean_div = vals[div_flat].mean() if div_flat.sum() > 0 else float("nan")
            mean_ok = vals[~div_flat].mean()
            print(f"    {coord:24s} divergent={mean_div:.4f}  non_divergent={mean_ok:.4f}")

# ===========================================================================
# STEP 3: CONTRAST against the pre-registered media prediction
print("\n" + "=" * 80)  # cosmetic
print("STEP 3: CONTRAST AGAINST THE PRE-REGISTERED MEDIA PREDICTION")
print("=" * 80)  # cosmetic

acv_contraction = float(contraction_df[(contraction_df.param == "gamma_control") & (contraction_df.coord == "log_acv_c")]["contraction"].iloc[0])
acv_displacement = float(contraction_df[(contraction_df.param == "gamma_control") & (contraction_df.coord == "log_acv_c")]["displacement"].iloc[0])
print(f"  ACV contraction = {acv_contraction:.3f}  |  ACV displacement = {acv_displacement:.3f}")

print(f"\nPREDICTION B (media): contraction of adstock_alpha and saturation_lam, the 4 channels:")
media_contractions = contraction_df[contraction_df.param.isin(["adstock_alpha", "saturation_lam"])]
print(media_contractions[["param", "coord", "contraction", "displacement"]].round(4).to_string(index=False))  # cosmetic
mean_media_contraction = media_contractions["contraction"].mean()
print(f"  AVERAGE media contraction (adstock+lam, 8 parameters): {mean_media_contraction:.4f}")
print(f"  The pre-registered prediction expected contraction NEAR 0 (threshold 0.15): it was")
print(f"  widely exceeded and the matter is CLOSED. The redesign of the DGP made")
print(f"  the media parameters identifiable: {mean_media_contraction:.1%} average")
print(f"  contraction, against the ~0% from before.")
print(f"  This contrast is kept as a historical record, not as a test: its threshold no longer")
print(f"  discriminates anything in the current regime.")

print("\n" + "=" * 80)  # cosmetic
print("STEP 0b: plot of prior vs posterior vs truth (4b1)")
print("=" * 80)  # cosmetic

recovery_table = validate_recovery(
    idata, geo="Geo_Centro", trade="Retail", idata_path_for_staleness_check=_IDATA_PATH,
)
print("\n" + recovery_table.round(4).to_string(index=False))  # cosmetic
n_miss = int((~recovery_table["covers_hdi"]).sum())
print(f"\n{n_miss}/{len(recovery_table)} parameters with the true value OUTSIDE the {HDI}.")


def _parse_param_name(name: str) -> tuple[str, str]:
    """'adstock_alpha[tv]' -> ('adstock_alpha', 'tv_impressions_000'); 'saturation_beta[tv] (L)' ->
    ('saturation_beta', 'tv_impressions_000'); 'gamma_control[log_price_c]' -> ('gamma_control',
    'log_price_c').
    """
    base = name.split("[")[0]
    inner = name.split("[", 1)[1].split("]")[0]
    if base in ("adstock_alpha", "saturation_lam", "saturation_beta"):
        coord = f"{inner}_impressions_000"
    else:
        coord = inner
    return base, coord


# Fixed order (grouped by parameter type and channel), not the table's "descending bias" one -- this
# way the plot reads as a stable grid, not as a ranking that reorders the panels from run to run.
GRID_ORDER = (
    [f"adstock_alpha[{c.replace('_impressions_000', '')}]" for c in CHANNEL_COLUMNS]
    + [f"saturation_lam[{c.replace('_impressions_000', '')}]" for c in CHANNEL_COLUMNS]
    + [f"saturation_beta[{c.replace('_impressions_000', '')}] (L)" for c in CHANNEL_COLUMNS]
    + ["gamma_control[log_price_c]", "gamma_control[promo_flag]", "gamma_control[log_acv_c]"]
)
DISPLAY_NAME = dict(CONTROL_DISPLAY)

N_COLS = 4  # cosmetic
N_ROWS = 4  # 4 groups x 4 columns = 16 cells, 15 used + 1 note  # cosmetic

fig = plt.figure(figsize=(16, 4.5 * N_ROWS))  # cosmetic
_outer = fig.add_gridspec(N_ROWS, N_COLS, hspace=0.42, wspace=0.16)  # cosmetic
axes = np.empty((N_ROWS, N_COLS), dtype=object)      # "main" axis = the top one (wide)
axes_zoom = np.empty((N_ROWS, N_COLS), dtype=object)
for _r in range(N_ROWS):
    for _c in range(N_COLS):
        _sub = _outer[_r, _c].subgridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.30)  # cosmetic
        axes[_r, _c] = fig.add_subplot(_sub[0])
        axes_zoom[_r, _c] = fig.add_subplot(_sub[1])

table_by_name = recovery_table.set_index("parameter")
_prior_mass_zoom = []

for idx, name in enumerate(GRID_ORDER):
    row_idx, col_idx = divmod(idx, N_COLS)
    ax = axes[row_idx, col_idx]
    ax_zoom = axes_zoom[row_idx, col_idx]
    base, coord = _parse_param_name(name)
    dim = "control" if base == "gamma_control" else "channel"
    prior_vals = prior[base].sel({dim: coord}).values.ravel()
    post_vals = post[base].sel({dim: coord}).values.ravel()
    r = table_by_name.loc[name]
    true_val = float(r["true_value"])
    covers = bool(r["covers_hdi"])
    bias = float(r["relative_bias_pct"])
    contraction = contraction_sd(prior_vals, post_vals)  # SD, not variance -- see config/

    # all the logic of scales, densities, shaded HDI94 and truth line lives in viz.overlay -- see that
    # module for the full reasoning.
    _info = prior_posterior_two_scales(ax, ax_zoom, post_vals, true_val, prior_vals, hdi_prob=HDI_PROB)
    _prior_mass_zoom.append((name, _info["prior_mass_in_zoom"]))

    label = DISPLAY_NAME.get(name, name)
    title_color = MISS_RED if not covers else INK
    marker = f"  ✗ {HDI} misses"  if not covers else ""
    ax.set_title(f"{label}\n{BIAS}={bias:+.2f}%{marker}  |  contr.={contraction*100:.1f}%",  # trivial: unit
                 fontsize=8.5, color=title_color, fontweight="bold" if not covers else "normal")  # cosmetic
    if not covers:  # visual mark -- red border, redundant with the title text
        for _a in (ax, ax_zoom):
            for spine in _a.spines.values():
                spine.set_visible(True)
                spine.set_color(MISS_RED)
                spine.set_linewidth(2.2)  # cosmetic

    if name == "saturation_lam[tv]":  # the central result of the project
        # It goes in the WIDE panel at the bottom left: at the top center it overlapped with the
        # "full prior" label in the corner.
        ax.text(0.02, 0.10, f"{CONTRACTION} = {contraction*100:.1f}%",  # cosmetic
                transform=ax.transAxes, fontsize=8, color=AQUA, fontweight="bold",  # cosmetic
                va="bottom", ha="left")

    if covers:  # normal panel: only the bottom axis. A panel with a miss already has the 4 in red.
        for _a in (ax, ax_zoom):
            for side in ["top", "right", "left"]:
                _a.spines[side].set_visible(False)
    ax.grid(axis="x", lw=0.5)  # cosmetic
    ax_zoom.grid(axis="x", lw=0.5)  # cosmetic
    ax.tick_params(labelsize=7)  # cosmetic
    ax_zoom.tick_params(labelsize=7)  # cosmetic
    # label of what each half is, small and once per panel
    ax.text(0.985, 0.90, FULL_PRIOR, transform=ax.transAxes, fontsize=6.5,  # cosmetic
            color=INK_SECONDARY, ha="right", va="top")
    ax_zoom.text(0.985, 0.88, ZOOM_TO_WINDOW, transform=ax_zoom.transAxes,  # cosmetic
                 fontsize=6.5, color=INK_SECONDARY, ha="right", va="top")  # cosmetic

axes_zoom[3, 3].axis("off")
ax_note = axes[3, 3]
ax_note.axis("off")
ax_note.text(
    0, 1,
    f"{n_miss}/15 parameters whose\n{TRUE_VALUE} falls outside the\n{HDI} (red border above).\n\n"
    f"Precision floor of the\nnoise-free pipeline: up to\n~5% relative {BIAS}.",
    transform=ax_note.transAxes, fontsize=8.5, color=INK_SECONDARY, va="top",  # cosmetic
)

axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")  # cosmetic
fig.suptitle(
    f"Prior vs. posterior vs. {TRUE_VALUE} — the 15 parameters of validate_recovery()\n"
    f"Each parameter at TWO scales: ABOVE, the {FULL_PRIOR} — {CONTRACTION_SHORT} reads off the "
    "WIDTH (both curves normalised to their own max; the orange needle marks the posterior mean);\n"
    f"BELOW, a zoom to the {POSTERIOR_WINDOW} — that is where {BIAS} reads off, with the {HDI} shaded. "
    f"Blue=prior, orange=posterior, dashed={TRUE_VALUE}, red border={MISSES_HDI}. "
    f"{DATASET_LABEL_NOTE}={_DATASET_LABEL}",
    fontsize=11, x=0.02, y=0.992, ha="left", va="top",  # cosmetic
)
# NO tight_layout: the nested grid already fixes the spacing (hspace/wspace of add_gridspec and of each
# subgridspec). tight_layout over 32 axes rearranges them and breaks the visual pairing.
fig.subplots_adjust(left=0.035, right=0.985, top=0.905, bottom=0.03)  # cosmetic
_plot_path = figure_path(ROOT, "4_recovery", "4b1_prior_posterior", _DATASET_LABEL)
save_figure(fig, _plot_path, script_file=__file__)
print(f"\nPlot saved to {_plot_path.relative_to(ROOT)}")

# How much prior mass falls inside the ZOOM window. It is the number that explains why the prior was
# invisible when that window was the only scale of the panel; it is printed so it can be checked
# without re-deriving it.
_ms = sorted(_prior_mass_zoom, key=lambda r: r[1])
print(f"Prior mass inside the zoom window -- median "
      f"{100*np.median([m for _, m in _prior_mass_zoom]):.2f}%, "  # trivial: unit
      f"range [{100*_ms[0][1]:.2f}%, {100*_ms[-1][1]:.2f}%]")  # trivial: unit
for _n, _m in _ms[:3]:
    print(f"    worst: {_n:30s} {100*_m:7.3f}%")  # trivial: unit
for _n, _m in _ms[-2:]:
    print(f"    best:  {_n:30s} {100*_m:7.3f}%")  # trivial: unit

# raw data of the parameters outside the HDI, to judge in the report whether the posterior is visually
# indistinguishable from the true value or not -- that judgment comes from reading the plot, not
# something this script should assign itself.
misses = recovery_table[~recovery_table["covers_hdi"]].copy()
if len(misses):
    print("\n" + "=" * 80)  # cosmetic
    print(f"detail of the {len(misses)} parameters outside the {HDI} (for the report)")
    print("=" * 80)  # cosmetic
    hdi_col_lo = [c for c in misses.columns if c.endswith("_lo")][0]
    hdi_col_hi = [c for c in misses.columns if c.endswith("_hi")][0]
    for _, r in misses.iterrows():
        hdi_width = r[hdi_col_hi] - r[hdi_col_lo]
        gap_true_to_hdi = min(abs(r["true_value"] - r[hdi_col_lo]), abs(r["true_value"] - r[hdi_col_hi]))
        print(f"  {r['parameter']:28s} true={r['true_value']:.4f}  post_mean={r['posterior_mean']:.4f}  "
              f"{HDI}=[{r[hdi_col_lo]:.4f},{r[hdi_col_hi]:.4f}]  hdi_width={hdi_width:.4f}  "
              f"truth_to_hdi_distance={gap_true_to_hdi:.5f} ({100*gap_true_to_hdi/hdi_width:.1f}% of the HDI width)")  # trivial: unit

print("\nDone.")
