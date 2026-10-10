"""THE DGP CONFIGURATION: ALL ITS AXES, IN ONE SINGLE PLACE.
"""

import hashlib
from dataclasses import dataclass, field

from config.mmm_dgp_config import (
    ADSTOCK_L_MAX_DGP, ADSTOCK_NORMALIZED_DGP, MEDIA_CHANNELS, REGIMES, DESIGN_SEED,
)

# NOISE seed of the usual dataset: with it the label carries no -s<n>. It is not the design seed
# (DESIGN_SEED, in mmm_dgp_config): the seed you choose moves only the noise.
CANONICAL_NOISE_SEED = 42

# level -> (target SNR of sales or None, label token, description)
LEVELS = {
    "zero": (None, "no_noise", "technical floor, NOT absence of noise -- sigma 0.001 in log; "
                               "measured SNR ~300-420. It is the usual dataset"),
    "low": (8.0, "low", "SNR 8   (achievable R2 ~0.98)"),
    "medium": (4.0, "medium", "SNR 4   (achievable R2 ~0.94)"),
    "high": (2.0, "high", "SNR 2   (achievable R2 ~0.80)"),
    "very_high": (1.0, "very_high", "SNR 1   (achievable R2 ~0.50)"),
}

# custom variable -> (CSV column, what it is)
NOISY_VARIABLES = {
    "sales": ("sales", "equation noise, in the model space"),
    "price": ("price", "measurement error of the price"),
    "acv": ("acv", "measurement error of the ACV"),
    **{f"impressions_{ch}": (f"{ch}_impressions_000", f"measurement error of the {ch} impressions")
       for ch in MEDIA_CHANNELS},
}

# scenario -> (description, regimes where it makes sense)
STRESS = {
    "gate": ("media gated by distribution: the media effect is attenuated with the ACV",
             ("log",)),
    "nonorm": ("non-normalized adstock: the carryover weights do not add up to 1", ("log",)),
    "inf": ("infinite memory: the adstock is not truncated at L_MAX", ("log",)),
}

REGRESSOR_WARNING = (
    "WARNING -- NOISE IN THE REGRESSORS. With measurement error in price, ACV or impressions, the "
    "estimator is BIASED by construction: attenuation toward zero of the effect of the variable with "
    "error (regression dilution). That is NOT a failure of the model. validate_recovery() stops "
    "measuring 'does the model work?' and starts measuring 'how much can it take?'. With adstock and "
    "saturation in between, and with correlated regressors (price and promo, ~0.7), the DIRECTION of "
    "the bias on the variable itself is predictable and the MAGNITUDE is not; the bias also leaks into "
    "the parameters of the correlated variables, in an unpredictable direction. A bad recovery here is "
    "not a bug.")

FULL_COMMAND = ("python synthetic_data/generate_dataset.py --regime log "
                "--noise zero|low|medium|high|very_high|custom "
                "[--noise-custom sales=4,price=10,...  (only with --noise custom)] "
                "--seed N --stress none|gate,nonorm,inf")


# ------------------------------------------------------------------------------ validators
def _validate_regime(v, partial):
    if v not in REGIMES:
        raise ValueError(f"valid regimes: {list(REGIMES)}")
    return v


def _validate_level(v, partial):
    if v not in LEVELS and v != "custom":
        raise ValueError(f"valid levels: {list(LEVELS) + ['custom']}")
    return v


def _validate_custom(v, partial) -> tuple:
    pairs = {}
    for chunk in [t.strip() for t in v.split(",") if t.strip()]:
        name, sep, num = chunk.partition("=")
        name = name.strip()
        if not sep or name not in NOISY_VARIABLES:
            raise ValueError(f"'{chunk}': each entry is variable=SNR, with variable in "
                             f"{list(NOISY_VARIABLES)}")
        try:
            snr = float(num)
        except ValueError:
            raise ValueError(f"'{chunk}': the SNR has to be a number") from None
        if not snr > 0:
            raise ValueError(f"'{chunk}': the SNR has to be > 0 (0 would be infinite noise)")
        pairs[name] = snr
    if not pairs:
        raise ValueError("at least one variable has to be named")
    return tuple(sorted(pairs.items()))


def _validate_seed(v, partial) -> int:
    try:
        n = int(v)
    except ValueError:
        raise ValueError("the seed is an integer >= 0") from None
    if n < 0:
        raise ValueError("the seed is an integer >= 0")
    return n


def _validate_stress(v, partial) -> tuple:
    if v.strip() == "none":
        return ()
    chosen = sorted({t.strip() for t in v.split(",") if t.strip()})
    bad = [t for t in chosen if t not in STRESS]
    if bad or not chosen:
        raise ValueError(f"'none' or a comma-separated list of {list(STRESS)}")
    reg = partial.get("regime")
    not_applicable = [t for t in chosen if reg and reg not in STRESS[t][1]]
    if not_applicable:
        raise ValueError(f"{not_applicable} makes no sense in the {reg} regime: "
                         + "; ".join(f"{t} only in {list(STRESS[t][1])}" for t in not_applicable))
    return tuple(chosen)


# ------------------------------------------------------------------------------ the axes
@dataclass(frozen=True)
class Axis:
    key: str
    flag: str
    question: str
    validate: object                      # (text, partial) -> value; ValueError if it does not work
    options: dict | None = None           # closed menu; None = free text
    help_text: str = ""                   # for the free text
    needed: object = field(default=lambda partial: True)


AXES: list[Axis] = [
    Axis("regime", "--regime", "Which dataset regime do you want to generate?", _validate_regime,
         options=None),   # the options come from the contract of each regime (see _options)
    Axis("level", "--noise", "Which noise level? (SNR = sd(signal)/sd(noise); see config/dgp_axes.py)",
         _validate_level,
         options={**{k: d for k, (_s, _t, d) in LEVELS.items()},
                  "custom": "you choose the SNR per variable (sales, price, ACV, impressions). "
                            "With noise in the regressors the recovery is BIASED on purpose"}),
    Axis("noise_custom", "--noise-custom",
         "Which variables carry noise, and with what SNR each?", _validate_custom,
         help_text=("  variable=SNR separated by commas, e.g.  sales=4,price=10,impressions_tv=5\n"
                    f"  variables: {', '.join(NOISY_VARIABLES)}\n"
                    "  sales not listed = the technical floor (sigma=0 exactly degenerates the likelihood)\n"
                    f"  {REGRESSOR_WARNING}"),
         needed=lambda partial: partial.get("level") == "custom"),
    Axis("seed", "--seed", "Which NOISE seed? (it moves only the noise realization; "
         "the usual dataset is 42)", _validate_seed,
         help_text="  an integer >= 0. The design (flighting, price, ACV, CPM, parameters) does not change."),
    Axis("stress", "--stress", "Which stress scenarios? ('none' is an answer, not a default)",
         _validate_stress,
         help_text="  'none', or a comma-separated list:\n" + "\n".join(
             f"    {k:8} {d}  [{', '.join(r)}]" for k, (d, r) in STRESS.items())),
]


def _options(axis: Axis, partial: dict) -> dict | None:
    if axis.key == "regime":
        from config.contracts import CONTRACTS
        return {n: c.DESCRIPTION for n, c in CONTRACTS.items()}
    return axis.options


# ------------------------------------------------------------------------------ the configuration
@dataclass(frozen=True)
class DGPConfig:
    regime: str
    level: str                                  # key of LEVELS or "custom"
    noise: tuple                                # ((variable, snr)...) sorted; () = only the floor
    seed: int
    stress: tuple                               # scenarios, sorted

    def snr(self, variable: str) -> float | None:
        """Target SNR of that variable, or None if it carries no noise (sales: the floor)."""
        return dict(self.noise).get(variable)

    @property
    def noise_in_regressors(self) -> bool:
        return any(v != "sales" for v, _ in self.noise)

    @property
    def adstock(self) -> tuple[bool, int | None]:
        """(normalized, l_max) of the DGP: the one aligned with the estimator, unless stressed."""
        return (ADSTOCK_NORMALIZED_DGP and "nonorm" not in self.stress,
                None if "inf" in self.stress else ADSTOCK_L_MAX_DGP)

    @property
    def media_gated(self) -> bool:
        return "gate" in self.stress

    @property
    def canonical(self) -> bool:
        return self.level == "zero" and self.seed == CANONICAL_NOISE_SEED and not self.stress

    @property
    def noise_token(self) -> str:
        if self.level != "custom":
            return LEVELS[self.level][1]
        signature = ",".join(f"{v}={s:g}" for v, s in self.noise)
        return "c" + hashlib.sha256(signature.encode()).hexdigest()[:6]

    @property
    def label(self) -> str:
        """<regime>__<noise>[-s<seed>][-<stress>...]. The canonical configuration has no seed in its
        name (log__no_noise); in any other the seed ALWAYS goes in the name.
        """
        parts = [self.noise_token]
        if not self.canonical:
            parts.append(f"s{self.seed}")
        parts += list(self.stress)
        rest = "-".join(parts)
        assert "__" not in rest, rest
        return f"{self.regime}__{rest}"

    def metadata(self) -> dict:
        return {"configuration": {
            "regime": self.regime, "level": self.level,
            "noise_by_variable": {v: s for v, s in self.noise},
            "custom_signature": self.noise_token if self.level == "custom" else None,
            "noise_seed": self.seed, "design_seed": DESIGN_SEED,
            "stress": list(self.stress),
            "adstock_dgp": {"normalized": self.adstock[0], "l_max": self.adstock[1]},
            "media_gated_by_distribution": self.media_gated,
            "noise_in_regressors": self.noise_in_regressors,
            "snr_definition": "sd(signal)/sd(noise); signal = temporal sd of the deterministic target "
                              "per geo x trade cell, in the estimator space (config/dgp_axes.py)",
        }}


def _build(p: dict) -> DGPConfig:
    if p["level"] == "custom":
        noise = p["noise_custom"]
    else:
        snr = LEVELS[p["level"]][0]
        noise = () if snr is None else (("sales", snr),)
    return DGPConfig(regime=p["regime"], level=p["level"], noise=noise,
                     seed=p["seed"], stress=p["stress"])


def choose_configuration(argv: list[str]) -> DGPConfig:
    """Each axis: explicit by flag (and already validated), or asked, or missing -- never guessed."""
    from config.interaction import (MissingDecision, choose, missing_decision, someone_watching,
                                    ask_text, flag_value)
    partial = {}
    for axis in AXES:                    # 1. what came by argument, in order
        v = flag_value(axis.flag, argv)
        if v is None:
            continue
        if not axis.needed(partial):
            raise MissingDecision(f"\n{axis.flag} is not needed with this configuration "
                                  f"(it only goes with --noise custom).\n")
        try:
            partial[axis.key] = axis.validate(v, partial)
        except ValueError as exc:
            raise MissingDecision(f"\n{axis.flag} {v!r}: {exc}\n") from None
    missing = [a.flag for a in AXES if a.key not in partial and a.needed(partial)]
    if missing and not someone_watching():   # 2. with nobody: ONE message, with the full command
        missing_decision(f"the dataset configuration (missing: {' '.join(missing)})", FULL_COMMAND)
    for axis in AXES:                    # 3. with a person: what is missing is asked, in order
        if axis.key in partial or not axis.needed(partial):
            continue
        ops = _options(axis, partial)
        if ops:
            partial[axis.key] = axis.validate(
                choose(axis.question, ops, flag=axis.flag, argv=[], how_to_pass_it=FULL_COMMAND),
                partial)
        else:
            text = ask_text(axis.question, validate=lambda s, a=axis: a.validate(s, partial),
                            help_text=axis.help_text, flag=axis.flag, argv=[],
                            how_to_pass_it=FULL_COMMAND)
            partial[axis.key] = axis.validate(text, partial)
    return _build(partial)


# ------------------------------------------------------------------------------ measure and report
def measure_snr(df, transform) -> dict:
    """SNR MEASURED on the data, by variable and cell: sd(signal)/sd(realized noise). Sales: in the
    estimator space (transform = the contract's transform_target).
    """
    import numpy as np
    out = {}
    for (geo, trade), d in df.groupby(["geo", "trade_channel"]):
        cell = f"{geo}|{trade}"
        det = transform(d["sales_deterministic"].to_numpy())
        obs = transform(d["sales"].to_numpy())
        out.setdefault("sales", {})[cell] = float(np.std(det) / np.std(obs - det))
        for var, (col, _q) in NOISY_VARIABLES.items():
            if var != "sales" and f"{col}_true" in d:
                v, o = d[f"{col}_true"].to_numpy(), d[col].to_numpy()
                out.setdefault(var, {})[cell] = float(np.std(v) / np.std(o - v))
    return out


def manifest(cfg: DGPConfig, measured: dict, csv_sha256: str, n_negative: int = 0) -> str:
    """The readable manifest that sits next to the dataset. Deterministic: the same configuration gives
    the same file (no date), so that regenerating does not dirty the tree.
    """
    L = [f"# Manifest of the dataset `{cfg.label}`", "",
         "> GENERATED by synthetic_data/generate_dataset.py -- do not edit by hand.", "",
         f"- **Regime:** {cfg.regime}",
         f"- **Noise level:** {cfg.level}" + (f" (signature `{cfg.noise_token}`)" if cfg.level == "custom" else ""),
         f"- **Noise seed:** {cfg.seed}  ·  design seed: {DESIGN_SEED} (fixed)",
         "- **Stress:** " + ("none" if not cfg.stress else
                             "; ".join(f"{s} ({STRESS[s][0]})" for s in cfg.stress)),
         f"- **CSV:** `dataset.csv`, content sha256 (line endings in LF) `{csv_sha256}`", "",
         "## Noise: SNR = sd(signal)/sd(noise)", "",
         "Signal = temporal deviation of the deterministic target of each geo×trade cell, in the "
         "estimator space: log(sales).", "",
         "| variable | target SNR | measured SNR (min · median · max over cells) |",
         "|---|---|---|"]
    for var in ["sales"] + [v for v in NOISY_VARIABLES if v != "sales"]:
        if var not in measured:
            continue
        target = cfg.snr(var)
        vals = sorted(measured[var].values())
        med = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
        L.append(f"| {var} | {'technical floor' if target is None else f'{target:g}'} | "
                 f"{vals[0]:.3g} · {med:.3g} · {vals[-1]:.3g} |")
    without = [v for v in NOISY_VARIABLES if v != "sales" and v not in measured]
    L += ["", f"No noise (true value in the CSV): {', '.join(without) if without else 'none'}."]
    if cfg.noise_in_regressors:
        L += ["", f"**{REGRESSOR_WARNING}**", "",
              "The `<col>_true` columns keep the value without error; sales were generated with the "
              "true value."]
    if n_negative:
        L += ["", f"⚠️ {n_negative} negative sales."]
    return "\n".join(L) + "\n"
