"""Constants SHARED between the synthetic data generator (synthetic_data/generate_dataset.py) and every
model script (model/<stage>/*.py) that builds a GeometricAdstock.
"""

from pathlib import Path

import numpy as np

# Repo root, anchored to the location of THIS file (config/ lives one level below the root): it
# does not depend on where it is imported from.
ROOT = Path(__file__).resolve().parents[1]

CANONICAL_GEO: str = "Geo_Centro"
CANONICAL_TRADE: str = "Retail"


REGIMES: tuple[str, ...] = ("log",)


def split_label(label: str) -> tuple[str, str]:
    """A label WITHOUT a regime prefix ('no_noise') belongs to the log regime: before, the regime
    did not exist as a variable and the DGP was always multiplicative.
    """
    head, sep, rest = label.partition("__")
    if sep and head in REGIMES:
        return head, rest
    return "log", label


def normalize_label(label: str) -> str:
    """Canonical form <regime>__<noise>. 'no_noise' -> 'log__no_noise'; idempotent."""
    regime, rest = split_label(label)
    return f"{regime}__{rest}"


L_MAX: int = 12

ADSTOCK_NORMALIZE: bool = True

ADSTOCK_MODE: str = "After"   # adstock convolution: spend weighs from its own week on


def adstock_mode():
    """ADSTOCK_MODE as the ConvMode that the generator and GeometricAdstock expect. Lazy import."""
    from pymc_marketing.mmm.transformers import ConvMode
    return ConvMode(ADSTOCK_MODE)

CHANNEL_COLUMNS: list[str] = [
    "tv_impressions_000",
    "meta_impressions_000",
    "tiktok_impressions_000",
    "ooh_impressions_000",
]

ADSTOCK_ALPHA_PRIOR_BY_CHANNEL: dict[str, tuple[float, float]] = {
    "tv_impressions_000": (3, 2),
    "meta_impressions_000": (2, 4),
    "tiktok_impressions_000": (2, 4),
    "ooh_impressions_000": (3, 2),
}


def adstock_alpha_prior_arrays(channel_columns: list[str] | None = None) -> tuple[list[float], list[float]]:
    """Turns ADSTOCK_ALPHA_PRIOR_BY_CHANNEL (by name) into the two positional lists (alpha, beta)
    that pymc-marketing's `Prior("Beta", alpha=..., beta=..., dims="channel")` expects, in the order
    of `channel_columns` (default: CHANNEL_COLUMNS).
    """
    cols = channel_columns if channel_columns is not None else CHANNEL_COLUMNS
    alphas = [ADSTOCK_ALPHA_PRIOR_BY_CHANNEL[c][0] for c in cols]
    betas = [ADSTOCK_ALPHA_PRIOR_BY_CHANNEL[c][1] for c in cols]
    return alphas, betas


# ---------------------------------------------------------------------------
# ! Several scripts read this constant: changing it invalidates the existing fits.
SATURATION_BETA_SIGMA_BY_CHANNEL: dict[str, float] = {
    "tv_impressions_000": 0.045,
    "meta_impressions_000": 0.02,
    "tiktok_impressions_000": 0.02,
    "ooh_impressions_000": 0.02,
}


SATURATION_LAM_PRIOR_GAMMA: dict[str, float] = {"alpha": 2, "beta": 0.471}


def saturation_lam_prior(dims="channel"):
    """The saturation_lam prior that EVERY model uses. Lazy import: the generator imports this
    module too, and does not need pymc-marketing for it.
    """
    from pymc_marketing.prior import Prior
    return Prior("Gamma", **SATURATION_LAM_PRIOR_GAMMA, dims=dims)


def saturation_beta_sigma_array(channel_columns: list[str] | None = None) -> list[float]:
    """Turns SATURATION_BETA_SIGMA_BY_CHANNEL (by name) into the positional list that
    `Prior("HalfNormal", sigma=xr.DataArray([...]), dims="channel")` expects, in the order of
    `channel_columns` (default: CHANNEL_COLUMNS).
    """
    cols = channel_columns if channel_columns is not None else CHANNEL_COLUMNS
    return [SATURATION_BETA_SIGMA_BY_CHANNEL[c] for c in cols]


# Model specification shared by every script that builds it (1a_, 2a_, 3a_, 4b_, 2b_, 4e_, 4h_):
# Fourier harmonics and the intercept prior, on the scale divided by target_scale.
YEARLY_SEASONALITY: int = 2
FOURIER_MU_PRIOR: float = 0      # center of the Laplace prior of the harmonics
INTERCEPT_HALFNORMAL_SIGMA: float = 1


def intercept_prior():
    """The intercept prior that EVERY model uses. Lazy import, like saturation_lam_prior()."""
    from pymc_marketing.prior import Prior
    return Prior("HalfNormal", sigma=INTERCEPT_HALFNORMAL_SIGMA, dims=())


# The level of the credible intervals the whole repo reports (HDI94): tables, bands and figures
# compute it with viz/intervals.py.
HDI_PROB: float = 0.94


def contraction_sd(prior_draws, post_draws) -> float:
    """Prior->posterior contraction in standard-deviation units: 1 - sd(post)/sd(prior). 0 = the
    posterior did not move from the prior; 1 = degenerate posterior (sd=0).
    """
    import numpy as _np
    prior_sd = _np.asarray(prior_draws).std()
    post_sd = _np.asarray(post_draws).std()
    return 1 - (post_sd / prior_sd)


# max|log(sales)| of the canonical slice: the target scale the priors were written against.
TS_LOG_REFERENCE: float = 12.150931
# Three priors written on the scale DIVIDED by target_scale (the one of 3a_). Here they are on that
# scale; in log-points they are the value times TS_LOG_REFERENCE, and fit_priors() rescales them by r.
GAMMA_PROMO_SIGMA_SCALED: float = 0.0250
FOURIER_B_SCALED: float = 0.0095
LIKELIHOOD_SIGMA_SCALED: float = 0.010

GAMMA_CONTROL_LOGPOINTS: dict[str, dict] = {
    "log_price_c": {"mu": -2.62, "sigma": 1.3, "lower": None, "upper": 0.0},
    "promo_flag":  {"mu": 0.0,   "sigma": GAMMA_PROMO_SIGMA_SCALED * TS_LOG_REFERENCE, "lower": None, "upper": None},
    "log_acv_c":   {"mu": 1.0,   "sigma": 0.5, "lower": 0.0,  "upper": None},
    "t_norm_c":    {"mu": 0.118, "sigma": 0.15, "lower": None, "upper": None},
}


FOURIER_B_LOGPOINTS: float = FOURIER_B_SCALED * TS_LOG_REFERENCE
LIKELIHOOD_SIGMA_LOGPOINTS: float = LIKELIHOOD_SIGMA_SCALED * TS_LOG_REFERENCE


class PriorWrongScaleError(ValueError):
    """A prior the model received is not on the scale of the target it fits."""


def check_prior_scale(mmm, contract, ctx: dict, rtol: float = 1e-6) -> None:
    """Fails if any prior THE MODEL RECEIVED is not on the scale of its target."""
    import numpy as _np

    def as_list(x):
        return _np.atleast_1d(_np.asarray(getattr(x, "values", x), dtype=float))

    problems = []

    def compare(name, implied, canonical):
        if not _np.isclose(implied, canonical, rtol=rtol, atol=1e-12):
            factor = f"x{implied / canonical:.4g}" if canonical else "canonical = 0"
            problems.append(f"  {name:32} implies {implied:+.6g} log-points, "
                            f"the derived value is {canonical:+.6g}  ({factor})")

    def to_lp(value, control=None):
        return contract.to_logpoints(value, control, ctx)

    beta_sigma = mmm.saturation.function_priors["beta"].parameters["sigma"]
    for ch, s in zip(mmm.channel_columns, as_list(beta_sigma)):
        compare(f"saturation_beta.sigma[{ch.split('_')[0]}]", to_lp(s),
                SATURATION_BETA_SIGMA_BY_CHANNEL[ch] * TS_LOG_REFERENCE)
    gp = mmm.model_config["gamma_control"].parameters
    controls = [str(c) for c in gp["mu"].coords["control"].values]
    for field in ("mu", "sigma"):
        for c, v in zip(controls, as_list(gp[field])):
            compare(f"gamma_control.{field}[{c}]", to_lp(v, c),
                    GAMMA_CONTROL_LOGPOINTS[contract.CANONICAL_KEY[c]][field])
    compare("gamma_fourier.b", to_lp(float(mmm.model_config["gamma_fourier"].parameters["b"])),
            FOURIER_B_LOGPOINTS)
    sig_lik = mmm.model_config["likelihood"].parameters["sigma"].parameters["sigma"]
    compare("likelihood.sigma", to_lp(float(sig_lik)), LIKELIHOOD_SIGMA_LOGPOINTS)
    if problems:
        raise PriorWrongScaleError(
            f"{len(problems)} model prior(s) are not on the scale of the '{contract.NAME}' regime "
            f"(target_scale={ctx['target_scale']:.6g}):\n" + "\n".join(problems))


# Seed of the dataset DESIGN: flighting, price, ACV, CPM. Fixed: moving it is another experiment. The
# NOISE seed is something else and lives in config/dgp_axes.py (CANONICAL_NOISE_SEED).
DESIGN_SEED = 42


# --- true demand of the DGP: read by the generator and viz/business.py (and 4b_ for the trend)
DEMAND_SEASONALITY_AMPLITUDE: float = 0.08   # amplitude of the yearly seasonality, as a proportion
DEMAND_SEASONALITY_PHASE_WEEKS: int = 6      # phase shift, in weeks, of the yearly sine
CATEGORY_TREND_PER_WEEK: float = 0.0008      # category growth per week


def true_demand(week_of_year, t):
    """(seasonality, trend) of the DGP demand, week by week. They enter sales together as
    log(1 + seasonality + trend).
    """
    seasonality = DEMAND_SEASONALITY_AMPLITUDE * np.sin(
        2 * np.pi * (week_of_year + DEMAND_SEASONALITY_PHASE_WEEKS) / 52)
    return seasonality, CATEGORY_TREND_PER_WEEK * t


# --- (moved from the generator: grid) -
N_WEEKS = 156
START_DATE = "2022-01-03"   # string; the generator builds the dates with pd.date_range

# The geo and trade names are proper names of the synthetic markets: they enter the generator's
# random seeds, so renaming them would change the dataset.
GEOS = ["Geo_Norte", "Geo_Centro", "Geo_Sur"]
TRADE_CHANNELS = ["Retail", "Conveniencia"]
MEDIA_CHANNELS = ["tv", "meta", "tiktok", "ooh"]


SIGMA_EPS_FLOOR = 0.001


def sigma_for_snr(sales_deterministic, transform, snr: float) -> float:
    """The sales noise sigma that gives that SNR in ONE geo x trade cell. SNR = sd(signal)/sd(noise)."""
    return float(np.std(transform(np.asarray(sales_deterministic, dtype=float)))) / snr

ROUND_OUTPUTS: bool = True
CSV_DECIMALS = 2                   # decimals of the CSV columns
MEDIA_SHARE_DECIMALS = 5           # decimals of the media shares


NOISE_CONFIG = {
    "impressions_noise_pct": 0.12,
    "cpm_noise_pct": 0.05,
    "acv_noise_sd": 1.2,
    "price_noise_pct": 0.16,
    "sales_noise_pct": 0.05,
    "geo_regional_variation_impressions": (0.9, 1.1),
    "geo_regional_variation_cpm": (0.95, 1.05),
    "promo_discount_range": (0.15, 0.30),
}
PROMO_DISCOUNT_MID = 0.225
PROMO_FREQUENCY = 0.18             # weekly probability of a promotion
Q4_WEEKS = (45, 53)                # weeks of the year with the year-end CPM (np.arange: 45..52)
CPM_RELATIVE_FLOOR = 0.5           # the CPM does not go below this fraction of its base


# ---------------------------------------------------------------------------
# ! THESE TWO ARE NOT EDITED. They are the ALIGNED values, derived from the constants above.
ADSTOCK_NORMALIZED_DGP: bool = ADSTOCK_NORMALIZE  # True = aligned (default)
ADSTOCK_L_MAX_DGP: int | None = L_MAX             # None = infinite memory (old, stress test)


media_true_params = {
    "tv":     {"adstock_alpha": 0.70, "lam": 0.0003662},
    "meta":   {"adstock_alpha": 0.35, "lam": 0.0004994},
    "tiktok": {"adstock_alpha": 0.20, "lam": 0.0009987},
    "ooh":    {"adstock_alpha": 0.50, "lam": 0.0015694},
}


# --- (moved from the generator: geo hyperparameters) -
geo_hyperparams = {
    "tv":     {"sigma_geo": 0.15},
    "meta":   {"sigma_geo": 0.20},
    "tiktok": {"sigma_geo": 0.25},
    "ooh":    {"sigma_geo": 0.10},
}


geo_population = {"Geo_Norte": 1_000_000, "Geo_Centro": 1_300_000, "Geo_Sur": 600_000}
geo_size = {geo: pop / 1_000_000 for geo, pop in geo_population.items()}
geo_share = {geo: size / sum(geo_size.values()) for geo, size in geo_size.items()}


trade_params = {
    "Retail":       {"base_sales": 45000, "price_elasticity": -1.8, "promo_lift": 0.22,
                      "distribution_elasticity": 1.05, "n_skus": 6, "media_attenuation": 0.6},
    "Conveniencia": {"base_sales": 18000, "price_elasticity": -0.9, "promo_lift": 0.35,
                      "distribution_elasticity": 0.85, "n_skus": 3, "media_attenuation": 0.4},
}

acv_true_params = {
    ("Geo_Centro", "Retail"):       {"acv_min": 70, "acv_max": 96,
                                      "phases": [(36, +1.058, 0.750), (74, -1.237, 0.839), (127, +1.333, 0.644), (138, -0.969, 0.462)]},
    ("Geo_Norte", "Retail"):        {"acv_min": 55, "acv_max": 92,
                                      "phases": [(49, +0.992, 0.647), (63, -1.270, 0.847), (115, +0.884, 0.672)]},
    ("Geo_Sur", "Retail"):          {"acv_min": 40, "acv_max": 78,
                                      "phases": [(37, +0.919, 0.585), (87, -0.972, 0.548), (102, +0.603, 0.781)]},
    ("Geo_Centro", "Conveniencia"): {"acv_min": 45, "acv_max": 75,
                                      "phases": [(10, -1.031, 0.804), (43, +1.231, 0.405), (93, -1.326, 0.803), (120, +0.710, 0.628)]},
    ("Geo_Norte", "Conveniencia"):  {"acv_min": 35, "acv_max": 65,
                                      "phases": [(49, +0.615, 0.528), (83, -1.385, 0.808), (139, +1.251, 0.621)]},
    ("Geo_Sur", "Conveniencia"):    {"acv_min": 25, "acv_max": 55,
                                      "phases": [(8, +1.126, 0.461), (61, -1.039, 0.698), (74, +1.132, 0.665), (128, -0.768, 0.701)]},
}


PRICE_TRUE_PARAMS = {
    "Retail":       {"step_weeks": [20, 58, 96, 134], "step_pct": [0.22, -0.19, 0.21, -0.18]},
    "Conveniencia": {"step_weeks": [33, 71, 109, 147], "step_pct": [0.20, -0.17, 0.19, -0.16]},
}
GEO_PRICE_INDEX = {"Geo_Norte": 0.97, "Geo_Centro": 1.00, "Geo_Sur": 1.03}
PRICE_BASE = 100

_forbidden_step_weeks = set(range(26, N_WEEKS + 1, 26)) | set(range(52, N_WEEKS + 1, 52))
for _trade, _cfg in PRICE_TRUE_PARAMS.items():
    assert not (set(_cfg["step_weeks"]) & _forbidden_step_weeks), (
        f"step_weeks of {_trade} falls on a multiple of 26 or 52 -- it collides with the seasonal harmonics"
    )
    assert len(_cfg["step_weeks"]) == len(_cfg["step_pct"])


# --- (moved from the generator: cpm) -
cpm_true_params = {
    "tv":     {"cpm_base": 8.0, "cpm_trend": 0.004, "q4_multiplier": 1.25},
    "meta":   {"cpm_base": 5.0, "cpm_trend": 0.015, "q4_multiplier": 1.40},  # rises with auction competition
    "tiktok": {"cpm_base": 4.0, "cpm_trend": 0.020, "q4_multiplier": 1.35},  # growing platform, more competition
    "ooh":    {"cpm_base": 6.0, "cpm_trend": 0.002, "q4_multiplier": 1.10},
}


# --- (moved from the generator: saturation) -
def logistic_saturation(x_adstocked, L, lam):
    """Logistic saturation, the SAME formula as pymc-marketing's LogisticSaturation: f(x) = L * (1 -
    exp(-lam*x)) / (1 + exp(-lam*x)).
    """
    import pytensor.xtensor as ptx
    from pymc_marketing.mmm.components.saturation import LogisticSaturation
    x = np.asarray(x_adstocked, dtype=float)
    flat = ptx.as_xtensor(x.reshape(-1), dims=("date",))
    y = LogisticSaturation().function(flat, lam=lam, beta=L).eval()
    return np.asarray(y, dtype=float).reshape(x.shape)


FLIGHT_DESIGN = {
    "tv":     {"on_weeks": 17, "off_weeks": 21, "phase": 30, "peak_multiple_of_x50": 3.05},
    "meta":   {"on_weeks": 8,  "off_weeks": 9,  "phase": 12, "peak_multiple_of_x50": 3.05},
    "tiktok": {"on_weeks": 12, "off_weeks": 13, "phase": 22, "peak_multiple_of_x50": 3.05},
    "ooh":    {"on_weeks": 18, "off_weeks": 21, "phase": 20, "peak_multiple_of_x50": 3.05},
}


GEO_PHASE_OFFSET = {
    "Geo_Centro": {"tv": 0, "meta": 0, "tiktok": 0, "ooh": 0},
    "Geo_Norte":  {"tv": 2, "meta": 1, "tiktok": 1, "ooh": 3},
    "Geo_Sur":    {"tv": 5, "meta": 2, "tiktok": 3, "ooh": 5},
}
GEO_MEDIA_SHARE_MULTIPLIER = {"Geo_Centro": 1.0, "Geo_Norte": 1.15, "Geo_Sur": 0.82}


# --- (moved from the generator: impressions) -
impressions_config = {
    "tv":     dict(seasonal_amplitude=0.10, seasonal_phase=0, trend_slope=0.15),
    "meta":   dict(seasonal_amplitude=0.10, seasonal_phase=0, trend_slope=0.20),
    "tiktok": dict(seasonal_amplitude=0.10, seasonal_phase=0, trend_slope=0.25),
    "ooh":    dict(seasonal_amplitude=0.08, seasonal_phase=0, trend_slope=0.05),
}
