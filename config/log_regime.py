"""LOG REGIME (multiplicative). Everything specific to this regime lives in this file, and nothing else:
it reads on its own, with no conditional branches.
"""

import numpy as np

from config.mmm_dgp_config import (
    CHANNEL_COLUMNS, FOURIER_B_SCALED, GAMMA_CONTROL_LOGPOINTS, GAMMA_PROMO_SIGMA_SCALED, CANONICAL_GEO,
    LIKELIHOOD_SIGMA_SCALED, MEDIA_CHANNELS, N_WEEKS, PRICE_BASE,
    SIGMA_EPS_FLOOR, CANONICAL_TRADE, TS_LOG_REFERENCE, logistic_saturation,
    saturation_beta_sigma_array, sigma_for_snr,
)

NAME = "log"
DESCRIPTION = "multiplicative: sales = base x effects; the model fits log(sales)"

L_NATIONAL: dict[str, float] = {"tv": 0.3636, "meta": 0.2273, "tiktok": 0.1364, "ooh": 0.1136}
L_NATIONAL_RESCALE_CONSTANT = 22000


SHARE_TITLE = "(1 - exp(-media_log_multiplier))"


# ===========================================================================
# GENERATOR SIDE
def calibrate(ing_ref):
    """The log regime calibrates nothing against the data: its ceilings are L_NATIONAL, fixed. It only
    records whether it runs with the 'gate' stress, for the metadata.
    """
    return {"media_gated": bool(ing_ref["media_gated"])}


def sales_equation(ing, state=None) -> dict:
    """The sales of ONE slice (geo, trade)."""
    tp = ing["tp"]
    e = tp["distribution_elasticity"]

    media_effect_by_channel = {}
    for ch in MEDIA_CHANNELS:
        L_geo = L_NATIONAL[ch] * ing["geo_mult"][ch]
        media_effect_by_channel[ch] = logistic_saturation(ing["adstocked"][ch], L=L_geo, lam=ing["lam"][ch])
    media_log_effect_total = sum(media_effect_by_channel.values())
    L_effective = {ch: L_NATIONAL[ch] * ing["geo_mult"][ch] * tp["media_attenuation"] for ch in MEDIA_CHANNELS}

    distribution_effect = (ing["acv"] / 100) ** e
    base = tp["base_sales"] * ing["geo_size"]
    price_effect = (ing["price"] / PRICE_BASE) ** tp["price_elasticity"]
    promo_effect = 1 + ing["promo_flag"] * tp["promo_lift"]
    seasonal_trend_effect = 1 + ing["seasonality"] + ing["trend"]
    base_combined = base * price_effect * promo_effect * seasonal_trend_effect * distribution_effect

    if ing["media_gated"]:   # 'gate' stress (formerly the constant MEDIA_GATED_BY_DISTRIBUTION)
        media_log_multiplier = media_log_effect_total * tp["media_attenuation"] * distribution_effect
        by_channel = {ch: media_effect_by_channel[ch] * tp["media_attenuation"] * distribution_effect
                      for ch in MEDIA_CHANNELS}
    else:
        media_log_multiplier = media_log_effect_total * tp["media_attenuation"]
        by_channel = {ch: media_effect_by_channel[ch] * tp["media_attenuation"] for ch in MEDIA_CHANNELS}

    # multiplicative form: sales = base_combined * exp(media_log_multiplier). Explicit MULTIPLICATIVE
    # noise: it matches EXACTLY the estimator's likelihood, log(S) = mu + eps.
    sales_deterministic = base_combined * np.exp(media_log_multiplier)
    # zero level = the usual floor (same code, bit for bit); with an SNR, the sigma that gives it in
    # THIS slice (config.mmm_dgp_config.sigma_for_snr, one single definition of signal).
    if ing["sales_snr"] is None:
        sigma_eps = SIGMA_EPS_FLOOR  # noise switched off -- finished model
    else:
        sigma_eps = sigma_for_snr(sales_deterministic, transform_target, ing["sales_snr"])
    eps = ing["noise_rng"].normal(0, sigma_eps, N_WEEKS)
    sales = sales_deterministic * np.exp(eps)

    # media_share: exact counterfactual of switching ALL media off; by channel, INDIVIDUAL ablation.
    # The sum of the 4 does NOT equal the total (cross terms of the multiplicative form).
    media_share = 1 - np.exp(-media_log_multiplier)
    media_share_by_channel = {ch: 1 - np.exp(-by_channel[ch]) for ch in MEDIA_CHANNELS}
    return {"sales_deterministic": sales_deterministic, "sales": sales, "media_share": media_share,
            "media_share_by_channel": media_share_by_channel, "L_effective": L_effective, "summary": {}}


def demand_effects(tp, geo_size_geo, acv_p) -> dict:
    """The truth of the controls that gets exported: elasticities and lift."""
    return {
        "gamma_true": tp["distribution_elasticity"],  # exact ONLY without the 'gate' stress
        "eta_true": tp["price_elasticity"],
        "promo_lift": tp["promo_lift"],
    }


def media_inputs(media_true_params) -> dict:
    """What gets exported as media_params_national_inputs: the shared values plus the log ceiling."""
    return {ch: {**v, "L_national": L_NATIONAL[ch]} for ch, v in media_true_params.items()}


FIXED_METADATA = {"l_national_rescale_constant": L_NATIONAL_RESCALE_CONSTANT}


def run_metadata(state, summary) -> dict:
    return {"media_gated_by_distribution": state["media_gated"]}


ABLATION_COMMENT = (
    "Per-channel ablation convention (media_share_by_channel / "
    "{ch}_media_share in the CSV): INDIVIDUAL, not sequential -- each "
    "channel is switched off separately against the SAME base with the other "
    "channels on. The sum of the 4 does NOT equal the total media_share "
    "because of the cross terms of the multiplicative form "
    "(exp(sum) != product of individual exps via the subtraction of 1)."
)

JSON_NOTE = (
    "Adstock and saturation act on impressions (*_impressions_000), not on spend. "
    "spend = impressions_000 * cpm. Saturation uses the SAME formula as pymc-marketing's "
    "LogisticSaturation: f(x) = L*(1-exp(-lam*x))/(1+exp(-lam*x)), with f(0)=0 exactly. "
    "Half-saturation point = ln(3)/lam (see effective_params.media.x50_half_saturation). "
    "FUNCTIONAL FORM: the media effect is MULTIPLICATIVE, not additive: "
    "sales_deterministic = base_combined * exp(media_log_multiplier), where media_log_multiplier "
    "is the sum of the saturation curves by channel (each with ceiling L_national * "
    "geo_multiplier, WITHOUT geo_size) attenuated by media_attenuation (formerly 'trade_factor', "
    "see trade_params). The L_national of each channel was rescaled (divided by ~22000, see "
    "metadata.l_national_rescale_constant) relative to an earlier additive version so that the "
    "media uplift would land in a reasonable order of magnitude. "
    "distribution_elasticity (in trade_params) IS the true elasticity of ACV on sales "
    "(gamma_true in effective_params.demand) -- before the media x distribution interaction was removed "
    "(MEDIA_GATED_BY_DISTRIBUTION=True) the effective gamma was not constant and exceeded "
    "distribution_elasticity. 'population' is fixed over time per geo (see geo_population), "
    "proportional to geo_size. See effective_params for the values ALREADY COMPOSED and ready to "
    "compare against a posterior -- do not rebuild L/gamma by hand from the input dictionaries "
    "(media_params_national_inputs, trade_params): use synthetic_data/true_params_reader.py."
)


def check_output(df) -> None:
    """Log-regime sales are positive by construction (product of positives times exp)."""
    return None


def report(state, summary, df) -> None:
    return None


# ===========================================================================
# ESTIMATOR SIDE
CONTROLS: list[str] = ["log_price_c", "promo_flag", "log_acv_c"]
CANONICAL_KEY: dict[str, str] = {c: c for c in CONTROLS + ["t_norm_c"]}

# The target scale gives away the regime of a .nc: max|log(sales)| is ~5-12.
TARGET_SCALE_RANGE: tuple[float, float] = (1.0, 100.0)


def transform_target(sales):
    return np.log(sales)


# fit_data columns that the CSV does not carry under that name, and how they are derived (config/dataset_sync).
FIT_DATA_DERIVED = {
    "y": ("sales", np.log, False),
    "log_price_c": ("price", np.log, True),
    "log_acv_c": ("acv", np.log, True),
}


def control_truth(d: dict) -> dict:
    """Truth of the controls in the units of the term the model estimates (before dividing by
    target_scale).
    """
    return {"log_price_c": d["eta_true"], "promo_flag": np.log(1 + d["promo_lift"]),
            "log_acv_c": d["gamma_true"]}


def fit_priors(target_scale: float, channel_columns: list[str] | None = None) -> dict:
    """The priors of 3a_ in the log regime. 3a_ builds them from this function, and the figure scripts
    that rebuild the model read the gamma_control ones with fit_gamma_control().
    """
    TS = target_scale
    cols = channel_columns if channel_columns is not None else CHANNEL_COLUMNS
    r = canonical_target_scale() / TS
    controls = CONTROLS + ["t_norm_c"]
    g = [GAMMA_CONTROL_LOGPOINTS[c] for c in controls]
    return {
        "controls": controls,
        "beta_sigma": [float(s) * r for s in saturation_beta_sigma_array(cols)],
        "gamma_mu": [p["mu"] / TS for p in g],
        # the promo one was written on the divided scale: it is rescaled by r, not divided by TS
        "gamma_sigma": [GAMMA_PROMO_SIGMA_SCALED * r if c == "promo_flag" else p["sigma"] / TS
                        for c, p in zip(controls, g)],
        "gamma_lower": [-np.inf if p["lower"] is None else p["lower"] for p in g],
        "gamma_upper": [np.inf if p["upper"] is None else p["upper"] for p in g],
        "fourier_b": FOURIER_B_SCALED * r,
        "likelihood_sigma": LIKELIHOOD_SIGMA_SCALED * r,
    }


def fit_gamma_control(target_scale: float, control_columns: list[str]):
    """(mu, sigma, lower, upper) of gamma_control from fit_priors(), in the order of control_columns.
    It accepts any subset of the controls, in the order they are passed.
    """
    pri = fit_priors(target_scale)
    idx = [pri["controls"].index(c) for c in control_columns]
    return tuple([pri[k][i] for i in idx] for k in ("gamma_mu", "gamma_sigma", "gamma_lower", "gamma_upper"))


def to_logpoints(value: float, control: str | None, ctx: dict) -> float:
    """Inverse of the prior translation: in the log regime, log-points = scaled value * target_scale."""
    return value * ctx["target_scale"]


def canonical_target_scale(csv_path=None) -> float:
    """max|log(sales)| of the canonical slice -- the scale the literals of 3a_ are written against."""
    import pandas as _pd
    from pathlib import Path as _Path
    if csv_path is None:
        csv_path = (_Path(__file__).resolve().parents[1] / "synthetic_data" / "regimes" / "log__no_noise"
                    / "dataset.csv")
    _df = _pd.read_csv(csv_path)
    _c = _df[(_df["geo"] == CANONICAL_GEO) & (_df["trade_channel"] == CANONICAL_TRADE)]
    _y = transform_target(_c["sales"].to_numpy())
    return float(max(abs(_y.max()), abs(_y.min()), 1.0))


def saturation_beta_sigma_scaled(target_scale: float, channel_columns=None) -> list[float]:
    """The SAME saturation_beta prior, expressed on the internal scale of another target."""
    _ref = canonical_target_scale()
    return [s * _ref / target_scale for s in saturation_beta_sigma_array(channel_columns)]
def gamma_control_prior_arrays(target_scale: float, control_columns: list[str]):
    """(mu, sigma, lower, upper) of the gamma_control TruncatedNormal, on the internal scale of the
    target that is passed.
    """
    mu, sg, lo, hi = [], [], [], []
    for c in control_columns:
        p = GAMMA_CONTROL_LOGPOINTS[c]
        mu.append(p["mu"] / target_scale)
        sg.append(p["sigma"] / target_scale)
        lo.append(-np.inf if p["lower"] is None else p["lower"])
        hi.append(np.inf if p["upper"] is None else p["upper"])
    return mu, sg, lo, hi


def check_fit_equivalence(tol: float = 1e-6) -> None:
    """The translations above reproduce, at the canonical target_scale, the literals of fit_priors().
    If someone touches one side without the other, this fails.
    """
    from config.contracts import require_regime   # it only makes sense with the log dataset
    require_regime(NAME, "check_fit_equivalence")
    TS = canonical_target_scale()
    if abs(TS - TS_LOG_REFERENCE) > 1e-6 * TS_LOG_REFERENCE:
        raise AssertionError(f"canonical target_scale {TS!r} != TS_LOG_REFERENCE {TS_LOG_REFERENCE!r}: "
                             f"the log dataset changed, or the dataset on disk is not of the log regime")
    lit = fit_priors(TS)
    obtained_beta = saturation_beta_sigma_scaled(TS)
    if max(abs(a - b) for a, b in zip(lit["beta_sigma"], obtained_beta)) > tol:
        raise AssertionError(f"saturation_beta: {lit['beta_sigma']} != {obtained_beta}")
    mu, sg, lo, hi = gamma_control_prior_arrays(TS, lit["controls"])
    for name, a, b in (("mu", mu, lit["gamma_mu"]), ("sigma", sg, lit["gamma_sigma"]),
                       ("lower", lo, lit["gamma_lower"]), ("upper", hi, lit["gamma_upper"])):
        for x, y in zip(a, b):
            if not (x == y or abs(x - y) <= tol * max(1.0, abs(y))):
                raise AssertionError(f"gamma_control[{name}]: {a} != {b}")
