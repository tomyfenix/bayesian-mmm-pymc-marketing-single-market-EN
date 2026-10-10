"""The fit: NUTS with 4 chains on the linear model with trend, and its convergence diagnostics. It is the
only script that runs real MCMC. Its .nc is read by 4a_, test_decomposition and every figure script
except 1a_, 2a_ and 2b_, which do not look at the posterior.

Produces: model/3_fit/3a_posterior__<dataset_label>.nc (+ .sha256 sidecar)
"""

import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.mmm_dgp_config import adstock_mode  # noqa: E402
from config.diagnostics import (SAMPLING_SEED, SAMPLER, SAMPLER_TRIAL, MAX_FIT_TIME_MIN,  # noqa: E402
                                R_HAT_THRESHOLD, ESS_BULK_THRESHOLD)
from config.mmm_dgp_config import FOURIER_MU_PRIOR  # noqa: E402
from config.mmm_dgp_config import (L_MAX, CHANNEL_COLUMNS, adstock_alpha_prior_arrays, check_prior_scale,
                                   saturation_lam_prior, intercept_prior, YEARLY_SEASONALITY)
from config.contracts import contract, current_contract
from config.idata_paths import idata_path
from config.compilation import NUMBA_KWARGS, all_in_numba  # one single place
from config.saving import save_idata_atomic
from synthetic_data.true_params_reader import get_dataset_label
SEED = SAMPLING_SEED


def build_mmm(regime: str | None = None):
    """Identical to 1a_specification.py in the log regime. `regime` chooses NOTHING -- the regime is
    still declared by the dataset.
    """
    get_dataset_label()
    # the regime is READ from the dataset (config.contracts), not from an editable constant.
    current = current_contract().NAME
    if regime is not None and regime != current:
        raise ValueError(f"build_mmm(regime={regime!r}), but the live dataset is of the {current!r} regime. "
                         f"Regenerate the dataset with step 1 of the README before rebuilding that .nc.")
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

    C = contract("log")   # the log controls and priors live in config/log_regime.py
    channel_columns = CHANNEL_COLUMNS  # single source in config/mmm_dgp_config.py
    control_columns = C.CONTROLS + ["t_norm_c"]

    X = df_model[["date"] + channel_columns + control_columns]
    y = df_model["y_log_sales"]
    TS = float(np.maximum(np.abs(y).max(), 1.0))
    pri = C.fit_priors(TS, channel_columns)

    _alphas, _betas = adstock_alpha_prior_arrays(channel_columns)  # by name in config, not positional
    _alpha_prior_alpha = xr.DataArray(_alphas, dims="channel", coords={"channel": channel_columns})
    _alpha_prior_beta = xr.DataArray(_betas, dims="channel", coords={"channel": channel_columns})
    adstock = GeometricAdstock(l_max=L_MAX, mode=adstock_mode(), priors={"alpha": Prior("Beta", alpha=_alpha_prior_alpha, beta=_alpha_prior_beta, dims="channel")})
    saturation = LogisticSaturation(
        priors={
            "lam": saturation_lam_prior(),   # one single place in config/
            "beta": Prior(
                # array from config/; see the Priors section of the README
                "HalfNormal",
                sigma=xr.DataArray(
                    pri["beta_sigma"], dims="channel", coords={"channel": channel_columns}
                ),
                dims="channel",
            ),
        }
    )
    gamma_mu = xr.DataArray(
        pri["gamma_mu"], dims="control", coords={"control": control_columns}
    )
    gamma_sigma = xr.DataArray(
        pri["gamma_sigma"], dims="control", coords={"control": control_columns}
    )
    gamma_lower = xr.DataArray(pri["gamma_lower"], dims="control", coords={"control": control_columns})
    gamma_upper = xr.DataArray(pri["gamma_upper"], dims="control", coords={"control": control_columns})
    gamma_control_prior = Prior(
        "TruncatedNormal", mu=gamma_mu, sigma=gamma_sigma, lower=gamma_lower, upper=gamma_upper, dims="control"
    )
    likelihood_prior = Prior("Normal", sigma=Prior("HalfNormal", sigma=pri["likelihood_sigma"], dims=()), dims=("date",))
    gamma_fourier_prior = Prior("Laplace", mu=FOURIER_MU_PRIOR, b=pri["fourier_b"], dims="fourier_mode")

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
    mmm.build_model(X, y)
    check_prior_scale(mmm, C, {"target_scale": TS})
    return mmm, X, y
if __name__ == "__main__":
    # the regime this run REALLY uses (the live dataset's). It is passed to the build, which checks it,
    # and it is what the .nc stores in build_mmm_kwargs.
    SPEC = {"regime": current_contract().NAME}
    mmm, X, y = build_mmm(**SPEC)

    print("=" * 80)  # cosmetic
    print("STEP 1: SPEED TEST (draws=200, tune=200, chains=1)")
    print("=" * 80)  # cosmetic

    TEST_DRAWS, TEST_TUNE, TEST_CHAINS = SAMPLER_TRIAL["draws"], SAMPLER_TRIAL["tune"], SAMPLER_TRIAL["chains"]
    FULL_DRAWS, FULL_TUNE, FULL_CHAINS = SAMPLER["draws"], SAMPLER["tune"], SAMPLER["chains"]
    TARGET_ACCEPT = SAMPLER["target_accept"]

    t0 = time.perf_counter()
    mmm_test, X_test, y_test = build_mmm(**SPEC)
    # all_in_numba also takes the deterministics to NUMBA
    with all_in_numba():
        mmm_test.fit(
            X_test, y_test,
            draws=TEST_DRAWS, tune=TEST_TUNE, chains=TEST_CHAINS, cores=TEST_CHAINS,
            target_accept=TARGET_ACCEPT, random_seed=SEED, progressbar=True,
            **NUMBA_KWARGS,
        )
    elapsed_test = time.perf_counter() - t0
    steps_per_sec = (TEST_DRAWS + TEST_TUNE) / elapsed_test
    print(f"\n  Time (1 chain, {TEST_DRAWS+TEST_TUNE} steps): {elapsed_test:.1f} sec "
          f"({steps_per_sec:.2f} steps/sec)")

    est_seconds_parallel = (FULL_DRAWS + FULL_TUNE) / steps_per_sec
    print(f"  Extrapolation to the full fit ({FULL_CHAINS} chains in parallel): "
          f"~{est_seconds_parallel/60:.1f} minutes (conservative, includes the fixed JIT compile)")  # trivial: unit

    if est_seconds_parallel > MAX_FIT_TIME_MIN * 60:  # trivial: seconds per minute
        raise SystemExit("Estimated time exceeds 30 minutes. Stopping before the full fit.")

    print("\n" + "=" * 80)  # cosmetic
    print(f"STEP 2: FULL FIT (chains={FULL_CHAINS}, tune={FULL_TUNE}, draws={FULL_DRAWS}, "
          f"target_accept={TARGET_ACCEPT})")
    print("=" * 80)  # cosmetic

    t0 = time.perf_counter()
    with all_in_numba():
        idata = mmm.fit(
            X, y,
            draws=FULL_DRAWS, tune=FULL_TUNE, chains=FULL_CHAINS, cores=FULL_CHAINS,
            target_accept=TARGET_ACCEPT, random_seed=SEED, progressbar=True,
            **NUMBA_KWARGS,
        )
    elapsed_full = time.perf_counter() - t0
    print(f"\n  Real time of the full fit: {elapsed_full/60:.1f} minutes")  # trivial: unit

    # The name includes the current dataset_label (config/idata_paths.py): fitting on another dataset
    # variant does not overwrite this file.
    OUT_PATH = idata_path("linear_trend", get_dataset_label())
    save_idata_atomic(idata, OUT_PATH, script="3a_fit.py", build_mmm_kwargs=SPEC)
    print(f"\n  InferenceData saved to {OUT_PATH} (+ .sha256 sidecar)")

    print("\n" + "=" * 80)  # cosmetic
    print("STEP 4: DIAGNOSTICS (arviz.summary)")
    print("=" * 80)  # cosmetic

    free_rv_names = [rv.name for rv in mmm.model.free_RVs]
    summary = az.summary(idata, var_names=free_rv_names, round_to=4)  # cosmetic
    pd.set_option("display.max_rows", 200)  # cosmetic
    pd.set_option("display.width", 160)  # cosmetic
    print(summary)

    n_divergences = int(idata.sample_stats["diverging"].sum())
    print(f"\n  Total divergences: {n_divergences} of {FULL_CHAINS * FULL_DRAWS} draws "
          f"({100*n_divergences/(FULL_CHAINS*FULL_DRAWS):.2f}%)")  # trivial: unit

    bad_rhat = summary[summary["r_hat"] > R_HAT_THRESHOLD]
    print(f"\n  Variables with r_hat > 1.01: {len(bad_rhat)}")
    if len(bad_rhat) > 0:
        print(bad_rhat[["r_hat", "ess_bulk", "ess_tail"]])

    bad_ess = summary[summary["ess_bulk"] < ESS_BULK_THRESHOLD]
    print(f"\n  Variables with ess_bulk < 400: {len(bad_ess)}")
    if len(bad_ess) > 0:
        print(bad_ess[["r_hat", "ess_bulk", "ess_tail"]])

    print(f"\n  max r_hat: {summary['r_hat'].max():.4f} ({summary['r_hat'].idxmax()})")
    print(f"  min ess_bulk: {summary['ess_bulk'].min():.0f} ({summary['ess_bulk'].idxmin()})")

    print("\nDone.")
