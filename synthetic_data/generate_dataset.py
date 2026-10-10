"""Synthetic dataset generator: 3 geos x 2 trade channels x 156 weeks, with geometric adstock, logistic
saturation, price, promo, ACV, trend and seasonality. It also writes the true parameters, which are the
reference the model is validated against.

Produces: synthetic_data/dataset.csv, true_parameters.json, media_share.csv, dataset_manifest.md (the
configuration, with the target and the measured SNR)
"""

import sys
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pymc_marketing
from scipy.special import expit, logit

# Repo root, computed from the location of this file -- it works no matter where the script is run from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.mmm_dgp_config import CANONICAL_GEO, CANONICAL_TRADE, ADSTOCK_MODE
from config.mmm_dgp_config import (
    DESIGN_SEED, N_WEEKS, START_DATE, GEOS, TRADE_CHANNELS, MEDIA_CHANNELS, ROUND_OUTPUTS,
    NOISE_CONFIG, PROMO_DISCOUNT_MID, media_true_params,
    geo_hyperparams, geo_population, geo_size, geo_share, trade_params, acv_true_params,
    PRICE_TRUE_PARAMS, GEO_PRICE_INDEX, PRICE_BASE, cpm_true_params, logistic_saturation,
    FLIGHT_DESIGN, GEO_PHASE_OFFSET, GEO_MEDIA_SHARE_MULTIPLIER, impressions_config, true_demand,
    CSV_DECIMALS, MEDIA_SHARE_DECIMALS, PROMO_FREQUENCY, Q4_WEEKS, CPM_RELATIVE_FLOOR,
)
from config.contracts import contract, content_sha256
from config.dgp_axes import REGRESSOR_WARNING, choose_configuration, manifest, measure_snr

CFG = choose_configuration(sys.argv[1:])
CHOSEN_REGIME = CFG.regime
if CFG.noise_in_regressors:
    print("\n" + "!" * 78 + "\n" + REGRESSOR_WARNING + "\n" + "!" * 78 + "\n")  # cosmetic
from synthetic_data.adstock_dgp import geometric_adstock_dgp

# ---------------------------------------------------------------------------
# 0. General configuration

DATASET_LABEL = CFG.label
_ADSTOCK_NORMALIZED, _ADSTOCK_L_MAX = CFG.adstock   # the one aligned with the estimator, unless stressed


def _rng_for(component: str, *key_parts) -> np.random.Generator:
    """Independent rng stream per (component, *keys). Deterministic by KEY, not by position or call
    order: reordering the geo/trade loop, or switching off any other noise source, does not alter the
    draws of this one.
    """
    key = "|".join([component, *map(str, key_parts)])
    digest = zlib.crc32(key.encode())
    return np.random.default_rng(np.random.SeedSequence(entropy=DESIGN_SEED, spawn_key=(digest,)))


def _noise_rng(component: str, *key_parts) -> np.random.Generator:
    """Like _rng_for, but with the chosen NOISE seed (--seed). Used ONLY by the noise sources: the
    sales noise and the measurement error of the regressors.
    """
    key = "|".join([component, *map(str, key_parts)])
    digest = zlib.crc32(key.encode())
    return np.random.default_rng(np.random.SeedSequence(entropy=CFG.seed, spawn_key=(digest,)))


def _with_error(x, variable: str, component: str, *key, multiplicative: bool = False):
    """The OBSERVED regressor: the true one plus measurement error, if the custom configuration put
    noise on that variable; if not, the true one untouched -- not a single rng call, so that the
    canonical CSV does not change a bit.
    """
    snr = CFG.snr(variable)
    if snr is None:
        return x
    rng = _noise_rng(component, *key)
    if multiplicative:
        s2 = float(np.log1p(np.var(x) / (snr ** 2 * np.mean(x ** 2))))  # trivial: identity of the formula
        return x * np.exp(rng.normal(-s2 / 2, np.sqrt(s2), len(x)))  # trivial: identity of the formula
    return x + rng.normal(0, float(np.std(x)) / snr, len(x))


def _r(x, n):
    return round(x, n) if ROUND_OUTPUTS else x


dates = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")
t = np.arange(N_WEEKS)
week_of_year = dates.isocalendar().week.values


geo_multipliers = {}
for ch in MEDIA_CHANNELS:
    sigma = geo_hyperparams[ch]["sigma_geo"]
    # Category (c) -- structural heterogeneity, NEVER switched off.
    mults = _rng_for("geo_multipliers", ch).lognormal(mean=0.0, sigma=sigma, size=len(GEOS))
    geo_multipliers[ch] = dict(zip(GEOS, mults))


def _acv_path(phases, n=N_WEEKS):
    """path(t) in [0,1]: sum of logistic level transitions (each phase is (week, relative magnitude,
    speed); positive magnitude = gains distribution, negative = loses it), renormalized.
    """
    raw = np.zeros(n)
    for week, magnitude, speed in phases:
        raw = raw + magnitude / (1.0 + np.exp(-speed * (np.arange(n) - week)))
    return (raw - raw.min()) / (raw.max() - raw.min())


def acv_series(acv_min, acv_max, phases, geo, trade):
    """Non-monotonic ACV in logit space: it goes through [acv_min, acv_max] following `_acv_path` (0 to
    1, but not monotonic in between -- it goes up and down), compressed near the ends by the
    logit/expit transformation.
    """
    L_min, L_max = logit(acv_min / 100), logit(acv_max / 100)  # trivial: unit
    path = _acv_path(phases)
    acv = 100 * expit(L_min + (L_max - L_min) * path)  # trivial: unit
    # Category (a) -- measurement noise (reporting variability of syndicated data). Off:
    noise = 0.0  # noise switched off -- finished model
    # noise = _rng_for("acv_noise", geo, trade).normal(0, NOISE_CONFIG["acv_noise_sd"], N_WEEKS)
    return np.clip(acv + noise, 0, 100)  # trivial: ACV in %, 100 = full distribution


def price_series(trade, geo, promo_flag):
    """Price with 3 sources of variation INDEPENDENT of promo (list-price steps of mixed sign, geo
    level) plus the promo effect itself and idiosyncratic noise.
    """
    cfg = PRICE_TRUE_PARAMS[trade]
    weeks = np.array(cfg["step_weeks"])
    pcts = np.array(cfg["step_pct"])
    n_steps_passed = np.searchsorted(weeks, t, side="right")
    cum_factor = np.concatenate([[1.0], np.cumprod(1 + pcts)])
    step_multiplier = cum_factor[n_steps_passed]

    list_price = PRICE_BASE * GEO_PRICE_INDEX[geo] * step_multiplier

    # Category (a) -- discount depth. Off, and NOT at 0 -- it stays at PROMO_DISCOUNT_MID:
    discount_pct = np.full(N_WEEKS, PROMO_DISCOUNT_MID)
    # discount_pct = _rng_for("promo_discount", geo, trade).uniform(*NOISE_CONFIG["promo_discount_range"], N_WEEKS)
    price = list_price * (1 - promo_flag * discount_pct)
    # Category (a) -- idiosyncratic price noise. Off:
    price = price * 1.0
    # price = price * (1 + _rng_for("price_noise", geo, trade).normal(0, NOISE_CONFIG["price_noise_pct"], N_WEEKS))
    return price, list_price


def cpm_series(cpm_base, cpm_trend, q4_multiplier, ch):
    is_q4 = np.isin(week_of_year, np.arange(*Q4_WEEKS))
    trend = cpm_trend * t
    seasonal_bump = np.where(is_q4, cpm_base * (q4_multiplier - 1), 0)
    noise = 0.0  # noise switched off -- finished model
    # noise = _rng_for("cpm_noise", ch).normal(0, cpm_base * NOISE_CONFIG["cpm_noise_pct"], N_WEEKS)
    series = cpm_base + trend + seasonal_bump + noise
    return np.clip(series, cpm_base * CPM_RELATIVE_FLOOR, None)

national_cpm = {ch: cpm_series(**cpm_true_params[ch], ch=ch) for ch in MEDIA_CHANNELS}


def _flight_pattern(on_weeks, off_weeks, phase, n=N_WEEKS):
    """DETERMINISTIC pattern (designed, not drawn): 1.0 in ON weeks, 0.0 in OFF weeks, a repeated
    (on_weeks+off_weeks) cycle, shifted by `phase`.
    """
    cycle = on_weeks + off_weeks
    return ((np.arange(n) - phase) % cycle < on_weeks).astype(float)


def _calibrate_on_level(ch):
    """National ON level (mean_impr) such that, in Geo_Centro, the adstocked peak reaches
    peak_multiple_of_x50 * x_50 -- x_50 = ln(3)/lam (it does NOT change with the aligned adstock).
    """
    p = media_true_params[ch]
    cfg = FLIGHT_DESIGN[ch]
    pattern_unit = _flight_pattern(cfg["on_weeks"], cfg["off_weeks"], cfg["phase"])
    adstocked_unit = geometric_adstock_dgp(
        pattern_unit, p["adstock_alpha"], normalized=_ADSTOCK_NORMALIZED, l_max=_ADSTOCK_L_MAX,
    )
    peak_unit = adstocked_unit.max()
    x50_ch = np.log(3) / p["lam"]  # trivial: ln 3 = x50 of the logistic saturation
    target_adstock_center = cfg["peak_multiple_of_x50"] * x50_ch
    X_on_center = target_adstock_center / peak_unit
    return X_on_center / geo_share["Geo_Centro"]


def base_impressions_series(mean_impr, flight_pattern, seasonal_amplitude, seasonal_phase, trend_slope, ch):
    """Impressions = ON level * fractional seasonality * flighting * noise. mean_impr is the ON level
    (plateau during the burst), not an average -- see _calibrate_on_level().
    """
    level = mean_impr * (1 + trend_slope * t / N_WEEKS)
    seasonal = 1 + seasonal_amplitude * np.sin(2 * np.pi * (week_of_year - seasonal_phase) / 52)  # trivial: 52 weeks per year, trigonometric identity
    # Category (a) -- measurement noise (media buying variability). Off:
    noise = 1.0  # noise switched off -- finished model
    # noise = 1 + _rng_for("impressions_noise", ch).normal(0, NOISE_CONFIG["impressions_noise_pct"], N_WEEKS)
    return level * seasonal * flight_pattern * noise


_mean_impr_by_ch = {ch: _calibrate_on_level(ch) for ch in impressions_config}

geo_national_impressions = {}
for geo in GEOS:
    for ch, cfg in impressions_config.items():
        flight_cfg = FLIGHT_DESIGN[ch]
        cycle = flight_cfg["on_weeks"] + flight_cfg["off_weeks"]
        phase_geo = (flight_cfg["phase"] + GEO_PHASE_OFFSET[geo][ch]) % cycle
        pattern_geo = _flight_pattern(flight_cfg["on_weeks"], flight_cfg["off_weeks"], phase_geo)
        geo_national_impressions[(geo, ch)] = base_impressions_series(
            mean_impr=_mean_impr_by_ch[ch], flight_pattern=pattern_geo, ch=ch, **cfg,
        )

DEMAND_SEASONALITY, CATEGORY_TREND = true_demand(week_of_year, t)   # the truth lives in config/


def _geo_impressions(geo):
    """Impressions of the geo (thousands), shifted by geo and with its own per-capita pressure."""
    return {ch: geo_national_impressions[(geo, ch)] * geo_share[geo] * GEO_MEDIA_SHARE_MULTIPLIER[geo] * 1.0
            for ch in MEDIA_CHANNELS}


def _regressors(geo, trade):
    """(promo_flag, price, list_price, acv) of the slice. promo_flag is Category (c) -- regressor design,
    NEVER switched off.
    """
    promo_flag = (_rng_for("promo_flag", geo, trade).uniform(0, 1, N_WEEKS) < PROMO_FREQUENCY).astype(int)
    price, list_price = price_series(trade, geo, promo_flag)
    acv = acv_series(**acv_true_params[(geo, trade)], geo=geo, trade=trade)
    return promo_flag, price, list_price, acv


def _geo_adstock(geo):
    """Adstock of the geo's impressions, by channel."""
    imp = _geo_impressions(geo)
    return {ch: geometric_adstock_dgp(imp[ch], media_true_params[ch]["adstock_alpha"],
                                      normalized=_ADSTOCK_NORMALIZED, l_max=_ADSTOCK_L_MAX)
            for ch in MEDIA_CHANNELS}


def _ingredients(geo, trade, adstocked_by_channel):
    """Everything a sales equation needs from ONE slice: regressors, adstock, seasonality, slice
    parameters and the noise rng.
    """
    promo_flag, price, _list_price, acv = _regressors(geo, trade)
    return {
        "geo": geo, "trade": trade, "tp": trade_params[trade], "geo_size": geo_size[geo],
        "acv_p": acv_true_params[(geo, trade)], "promo_flag": promo_flag, "price": price, "acv": acv,
        "seasonality": DEMAND_SEASONALITY, "trend": CATEGORY_TREND,
        "adstocked": adstocked_by_channel,
        "lam": {ch: media_true_params[ch]["lam"] for ch in MEDIA_CHANNELS},
        "geo_mult": {ch: geo_multipliers[ch][geo] for ch in MEDIA_CHANNELS},
        "noise_rng": _noise_rng("sales_eps", geo, trade),
        "sales_snr": CFG.snr("sales"),       # None = the usual technical floor
        "media_gated": CFG.media_gated,      # 'gate' stress (only the log regime reads it)
    }


# THE REGIME: its sales equation, its truth and what gets exported live in config/<regime>_regime.py.
# Here the contract is just requested and handed the ingredients.
REG = contract(CHOSEN_REGIME)
_STATE = REG.calibrate(_ingredients(CANONICAL_GEO, CANONICAL_TRADE, _geo_adstock(CANONICAL_GEO)))
_SUMMARY = {}   # (geo, trade) -> whatever the regime's equation wants to report

# ---------------------------------------------------------------------------
# 7. Panel assembly: geo x trade_channel x week
rows = []
media_share_records = []  # diagnostic only, it does not go to the final CSV
effective_L = {}  # (channel, geo, trade) -> effective L, for effective_params (section 8)

for geo in GEOS:
    geo_impressions = _geo_impressions(geo)
    # the OBSERVED impressions (with measurement error if the custom configuration asks for it; if
    # not, the true ones). Key (geo, channel): a fact of the geo, the same for its two trades. Sales and
    # spend are computed with the TRUE ones.
    geo_impressions_obs = {ch: _with_error(geo_impressions[ch], f"impressions_{ch}", "impressions_noise",
                                           geo, ch, multiplicative=True) for ch in MEDIA_CHANNELS}
    geo_cpm = {ch: national_cpm[ch] * 1.0 for ch in MEDIA_CHANNELS}
    # geo_cpm = {ch: national_cpm[ch]
    # _rng_for("geo_regional_cpm", geo, ch).uniform(*NOISE_CONFIG["geo_regional_variation_cpm"], N_WEEKS)
    # for ch in MEDIA_CHANNELS}

    adstocked_by_channel = _geo_adstock(geo)
    # derived spend: impressions (thousands) x CPM ($ per thousand)
    geo_spend = {ch: geo_impressions[ch] * geo_cpm[ch] for ch in MEDIA_CHANNELS}

    for trade in TRADE_CHANNELS:
        ing = _ingredients(geo, trade, adstocked_by_channel)
        promo_flag, price, acv = ing["promo_flag"], ing["price"], ing["acv"]
        # OBSERVED price and ACV; the sales equation below uses the TRUE ones.
        price_obs = _with_error(price, "price", "price_noise", geo, trade)
        acv_obs = _with_error(acv, "acv", "acv_noise", geo, trade)
        if acv_obs is not acv:
            acv_obs = np.clip(acv_obs, 0, 100)  # trivial: ACV in %, 100 = full distribution
        # THE SALES EQUATION of the regime (config/<regime>_regime.py). What follows records the CSV
        # rows and the share diagnostic.
        out = REG.sales_equation(ing, _STATE)
        sales, sales_deterministic = out["sales"], out["sales_deterministic"]
        media_share, media_share_by_channel = out["media_share"], out["media_share_by_channel"]
        for ch in MEDIA_CHANNELS:
            effective_L[(ch, geo, trade)] = out["L_effective"][ch]
        _SUMMARY[(geo, trade)] = out["summary"]

        for i in range(N_WEEKS):
            media_share_records.append({
                "geo": geo, "trade": trade, "media_share": media_share[i],
                **{f"{ch}_media_share": media_share_by_channel[ch][i] for ch in MEDIA_CHANNELS},
            })

        for i in range(N_WEEKS):
            sales_r = _r(sales[i], CSV_DECIMALS)
            sales_det_r = _r(sales_deterministic[i], CSV_DECIMALS)
            row = {
                "date": dates[i],
                "geo": geo,
                "trade_channel": trade,
                "sales": sales_r,
                "sales_deterministic": sales_det_r,
                "noise_realized": round(sales_r - sales_det_r, CSV_DECIMALS),
                "population": geo_population[geo],
                "price": _r(price_obs[i], CSV_DECIMALS),
                "promo_flag": int(promo_flag[i]),
                "acv": _r(acv_obs[i], CSV_DECIMALS),
                "media_share": _r(media_share[i], MEDIA_SHARE_DECIMALS),
            }
            for ch in MEDIA_CHANNELS:
                row[f"{ch}_media_share"] = _r(media_share_by_channel[ch][i], MEDIA_SHARE_DECIMALS)
            for ch in MEDIA_CHANNELS:
                row[f"{ch}_impressions_000"] = _r(geo_impressions_obs[ch][i], CSV_DECIMALS)
                row[f"{ch}_cpm"] = _r(geo_cpm[ch][i], CSV_DECIMALS)
                row[f"{ch}_spend"] = _r(geo_spend[ch][i], CSV_DECIMALS)
            # the value WITHOUT error of each noisy regressor, at the end of the row. They only exist
            # if the custom configuration put noise on that variable: the canonical CSV does not have them.
            if price_obs is not price:
                row["price_true"] = _r(price[i], CSV_DECIMALS)
            if acv_obs is not acv:
                row["acv_true"] = _r(acv[i], CSV_DECIMALS)
            for ch in MEDIA_CHANNELS:
                if geo_impressions_obs[ch] is not geo_impressions[ch]:
                    row[f"{ch}_impressions_000_true"] = _r(geo_impressions[ch][i], CSV_DECIMALS)
            rows.append(row)

df = pd.DataFrame(rows).sort_values(["trade_channel", "geo", "date"]).reset_index(drop=True)

REG.check_output(df)

# ---------------------------------------------------------------------------
# 8. Save dataset + true parameters
output_dir = ROOT / "synthetic_data"
output_dir.mkdir(parents=True, exist_ok=True)  # it used to fail on a clean clone
df.to_csv(output_dir / "dataset.csv", index=False)
# the CSV cannot carry its regime inside without changing its content. The truth JSON declares it,
# tied to THIS CSV by the sha256 of its content: config.contracts.dataset_regime fails if someone
# changes one without the other. The same content hash (line endings normalized to LF) goes to the
# manifest, the log and the figure footers: that way it is the same on Windows (to_csv writes CRLF) and
# on Linux. The one of the canonical dataset starts with 1cc979a4bf19fb5c.
_CSV_SHA256 = content_sha256(output_dir / "dataset.csv")
# the SNR MEASURED on the data exactly as it was written, and the readable manifest next to it.
_MEASURED_SNR = measure_snr(df, REG.transform_target)
(output_dir / "dataset_manifest.md").write_text(
    manifest(CFG, _MEASURED_SNR, _CSV_SHA256[:16]), encoding="utf-8")


def _json_default(obj):
    """Recursive sanitization for json.dump: any numpy value (np.float64, np.int64, np.bool_,
    np.ndarray) would otherwise raise TypeError.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object not serializable to JSON: {type(obj)!r}")


# Metadata keys that change on EVERY run without a single true value having changed: the time and the
# HEAD commit. They are listed here, and not in the comparer, so that it is visible that they are
# exactly two and why.
_VOLATILE_METADATA = ("generated_at_utc", "git_commit")


def _without_volatile(payload: dict) -> dict:
    """Copy of the payload without the metadata keys that change on their own."""
    clean = dict(payload)
    clean["metadata"] = {k: v for k, v in payload.get("metadata", {}).items()
                         if k not in _VOLATILE_METADATA}
    return clean


def _write_json_if_changed(path: Path, payload: dict) -> bool:
    """Writes `payload` to `path` ONLY if some substantive value changed. The generator is deterministic
    by seed: running it twice produces the SAME dataset, and rewriting the JSON would dirty the tree.
    """
    new = json.loads(json.dumps(payload, indent=2, default=_json_default))  # cosmetic
    if path.exists():
        try:
            with open(path) as f:
                old = json.load(f)
        except (OSError, json.JSONDecodeError):
            old = None
        if old is not None and _without_volatile(old) == _without_volatile(new):
            return False
    with open(path, "w") as f:
        json.dump(new, f, indent=2)  # cosmetic
    return True


def _git_commit_hash(root: Path) -> str | None:
    """Double try/except: git may not be installed, or the folder may not be versioned -- both cases
    return None without breaking the dataset generation.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5,  # trivial: infrastructure: git timeout
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, OSError):
        return None


# effective_params: values ALREADY COMPOSED, not input dictionaries -- whoever validates does not have
# to rebuild anything by reading the code.
effective_params_media = {}
for ch in MEDIA_CHANNELS:
    p = media_true_params[ch]
    for geo in GEOS:
        for trade in TRADE_CHANNELS:
            key = f"{ch}|{geo}|{trade}"
            effective_params_media[key] = {
                "L_effective": float(effective_L[(ch, geo, trade)]),
                "adstock_alpha": p["adstock_alpha"],
                "lam": p["lam"],
                "x50_half_saturation": float(np.log(3) / p["lam"]),  # trivial: ln 3 = x50 of the logistic saturation
            }

# The truth of the controls, in the format of its regime (config/<regime>_regime.py). The keys differ on
# purpose: a conversion of one regime applied to the truth of another gives KeyError, not a plausible number.
effective_params_demand = {
    f"{geo}|{trade}": REG.demand_effects(trade_params[trade], geo_size[geo], acv_true_params[(geo, trade)])
    for geo in GEOS for trade in TRADE_CHANNELS
}

true_params_export = {
    "metadata": {
        "dataset_label": DATASET_LABEL,
        "regime": CHOSEN_REGIME,
        "csv_sha256": _CSV_SHA256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_script": "synthetic_data/generate_dataset.py",
        "git_commit": _git_commit_hash(ROOT),
        "pymc_marketing_version": pymc_marketing.__version__,
        "seed": DESIGN_SEED,                        # the DESIGN seed; the noise one goes in configuration
        "n_weeks": N_WEEKS,
        "adstock": {"normalized": _ADSTOCK_NORMALIZED, "l_max": _ADSTOCK_L_MAX, "mode": ADSTOCK_MODE},
        **REG.FIXED_METADATA,
        "noise_config": NOISE_CONFIG,
        **CFG.metadata(),
        "measured_snr": _MEASURED_SNR,
        "geo_trade_being_modeled": {"geo": "Geo_Centro", "trade": "Retail"},
    },
    "media_params_national_inputs": REG.media_inputs(media_true_params),
    "cpm_true_params": cpm_true_params,
    "geo_hyperparams": geo_hyperparams,
    "geo_multipliers": {ch: {g: float(v) for g, v in d.items()} for ch, d in geo_multipliers.items()},
    "geo_population": geo_population,
    "geo_size": geo_size,
    "geo_share": geo_share,
    "trade_params": trade_params,
    "price_true_params": {**PRICE_TRUE_PARAMS, "geo_price_index": GEO_PRICE_INDEX},
    "acv_true_params": {f"{g}|{tr}": v for (g, tr), v in acv_true_params.items()},
    "effective_params": {
        "_ablation_comment": REG.ABLATION_COMMENT,
        "media": effective_params_media,
        "demand": effective_params_demand,
    },
    "note": REG.JSON_NOTE,
}
true_params_export["metadata"].update(REG.run_metadata(_STATE, _SUMMARY))

_json_path = output_dir / "true_parameters.json"
if _write_json_if_changed(_json_path, true_params_export):
    print(f"Written: {_json_path.name} (some true value changed)")
else:
    print(f"No changes: {_json_path.name} -- the true values are identical, "
          f"it is not rewritten (avoids dirtying the tree; see _write_json_if_changed)")

print("Dataset generated:", df.shape)
print(df.columns.tolist())
print(df.head(3))  # cosmetic

media_share_df = pd.DataFrame(media_share_records)
print("\n" + "=" * 60)  # cosmetic
print("MEDIA SHARE OF SALES " + REG.SHARE_TITLE)
print("=" * 60)  # cosmetic
print(media_share_df["media_share"].describe(percentiles=[0.1, 0.5, 0.9]))  # cosmetic

print("\nBy (geo, trade):")
geo_trade_summary = media_share_df.groupby(["geo", "trade"])["media_share"].describe(
    percentiles=[0.1, 0.5, 0.9]  # cosmetic
)[["mean", "10%", "50%", "90%", "min", "max"]]
print(geo_trade_summary)

print("\nBy channel (individual ablation, not additive -- see note):")
channel_cols = [f"{ch}_media_share" for ch in MEDIA_CHANNELS]
channel_summary = media_share_df[channel_cols].describe(percentiles=[0.1, 0.5, 0.9]).T[  # cosmetic
    ["mean", "10%", "50%", "90%", "min", "max"]
]
print(channel_summary)

media_share_summary = (
    media_share_df.groupby(["geo", "trade"])[["media_share"] + channel_cols]
    .describe(percentiles=[0.1, 0.5, 0.9])  # cosmetic
)
media_share_summary.to_csv(output_dir / "media_share.csv")
print(f"\nSaved: {output_dir / 'media_share.csv'}")


REG.report(_STATE, _SUMMARY, df)

print()
print("=" * 60)  # cosmetic
print(f"  GENERATED REGIME: {CHOSEN_REGIME.upper()}   (dataset_label = {DATASET_LABEL})")
print(f"  {REG.DESCRIPTION}")
print(f"  noise {CFG.level}  ·  seed {CFG.seed}  ·  stress {', '.join(CFG.stress) or 'none'}")
print(f"  dataset.csv  content sha256 {_CSV_SHA256[:16]}")
print("  manifest: synthetic_data/dataset_manifest.md")
print("=" * 60)  # cosmetic
