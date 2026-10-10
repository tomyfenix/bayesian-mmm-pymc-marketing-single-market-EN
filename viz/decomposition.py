"""DECOMPOSITION OF A MULTIPLICATIVE MODEL INTO FOUR SPLITS.

# ! WATCH OUT -- the price and ACV reference is mandatory: it changes the split of ALL drivers.
# ! WATCH OUT -- LMDI is only defined for a multiplicative v; with an additive one it refuses.
"""

import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from config.mmm_dgp_config import PRICE_BASE
from viz.business import (log_components, true_log_components, canonical_slice,
                          centering_reference)

_ROOT = Path(__file__).resolve().parents[1]

# The nine drivers of the model side (linear 3a_), in the order they are reported.
DRIVERS_K9 = ["log_price_c", "promo_flag", "log_acv_c", "t_norm_c", "seasonality",
              "tv", "meta", "tiktok", "ooh"]
# The only grouping used: the DGP puts the trend INSIDE the seasonality, log(1 + 0.08*sin + 0.0008*t),
# so on the DGP side they are one single factor. To compare model against DGP they have to be grouped
# explicitly with group(comp, GROUPING_K8).
GROUPING_K8 = {"trend + seasonality": ("t_norm_c", "seasonality")}
# The order of (B) that is reported: base -> price -> promo -> ACV -> seasonality -> media.
BASE_ORDER_K9 = ["log_price_c", "promo_flag", "log_acv_c", "t_norm_c", "seasonality",
                 "tv", "meta", "tiktok", "ooh"]
BASE_ORDER_K8 = ["log_price_c", "promo_flag", "log_acv_c", "trend + seasonality",
                 "tv", "meta", "tiktok", "ooh"]
VALID_REFERENCES = ("model", "dgp")


class MissingReference(ValueError):
    """A decomposition was requested without saying against which level of price and ACV."""


class NotMultiplicative(ValueError):
    """LMDI was requested on a v that is not a product of factors."""


# ---------------------------------------------------------------------------
# references
def reference_levels(reference: str, dfm=None) -> dict:
    """{'price': r_p, 'acv': r_a} for the requested reference. No default: it fails if it is missing."""
    if reference not in VALID_REFERENCES:
        raise MissingReference(
            f"reference={reference!r}. One of {VALID_REFERENCES} has to be chosen: 'model' takes price and "
            f"ACV to their geometric mean of the period (the one the model is centered on); 'dgp' takes "
            f"them to 100 (PRICE_BASE and full distribution). It changes the base and the whole split.")
    if reference == "dgp":
        return {"price": float(PRICE_BASE), "acv": 100.0}  # trivial: ACV in %, 100 = full distribution
    dfm = canonical_slice() if dfm is None else dfm
    return {"price": centering_reference(dfm, "price"), "acv": centering_reference(dfm, "acv")}


# ---------------------------------------------------------------------------
# components: ln f_k per week, and the base
@dataclass
class Components:
    """ln f_k per (draw, week) and the base in log. v(N) = exp(logbase + SUM_k logf_k)."""
    names: list
    logf: np.ndarray          # (K, M, T)
    logbase: np.ndarray       # (M, T)
    side: str                 # 'model' | 'dgp'
    reference: str            # 'model' | 'dgp'
    grouped: dict = field(default_factory=dict)


def _stack(da, draws=None) -> np.ndarray:
    """(m, date) with m = chain*n_draw + draw, the same order as stack(m=("chain","draw")). With `draws`
    it does NOT stack the whole posterior only to throw 99% away: it translates the flat indices into the
    (chain, draw) pair and indexes point-wise.
    """
    if draws is None:
        return da.stack(m=("chain", "draw")).transpose("m", "date").values
    import xarray as xr
    ch, dr = np.divmod(np.asarray(draws), da.sizes["draw"])
    return da.isel(chain=xr.DataArray(ch, dims="m"),
                   draw=xr.DataArray(dr, dims="m")).transpose("m", "date").values


def _stack_param(da, draws=None) -> np.ndarray:
    """Same as _stack but for a parameter without a `date` dimension (returns (m,))."""
    if draws is None:
        return da.stack(m=("chain", "draw")).values
    import xarray as xr
    ch, dr = np.divmod(np.asarray(draws), da.sizes["draw"])
    return da.isel(chain=xr.DataArray(ch, dims="m"), draw=xr.DataArray(dr, dims="m")).values


def model_components(idata, reference: str, dfm=None, draws=None) -> Components:
    """The K=9 drivers of the model side, with price and ACV taken to `reference`. The price component is
    gamma * target_scale * (log p - log gm), with gm the geometric mean of the slice (verified to 1.1e-16).
    """
    dfm = canonical_slice() if dfm is None else dfm
    ref = reference_levels(reference, dfm)
    comp = log_components(idata)
    ts = float(idata.constant_data["target_scale"].values)
    g = idata.posterior["gamma_control"]
    logf = []
    for k in DRIVERS_K9:
        # the draw subset is taken INSIDE _stack, before materializing the whole posterior: stacking the
        # 4000 draws of the 9 drivers and only then taking the test's 40 gives the same numbers with 9
        # times the work.
        x = _stack(comp[k], draws)
        if k in ("log_price_c", "log_acv_c"):
            col = "price" if k == "log_price_c" else "acv"
            gm = centering_reference(dfm, col)
            gam = _stack_param(g.sel(control=k), draws) * ts
            x = x + (gam * (math.log(gm) - math.log(ref[col])))[:, None]
        logf.append(x)
    logf = np.stack(logf)
    logS = _stack(idata.posterior["y_original_scale"], draws)
    return Components(list(DRIVERS_K9), logf, logS - logf.sum(0), "model", reference)


def dgp_components(reference: str, dfm=None) -> Components:
    """The K=8 drivers of the DGP (trend and seasonality are one single factor), M=1.
    true_log_components() already gives them with the DGP reference (price and ACV over 100).
    """
    dfm = canonical_slice() if dfm is None else dfm
    ref = reference_levels(reference, dfm)
    V = true_log_components(dfm)
    TP = json.loads((_ROOT / "synthetic_data" / "true_parameters.json").read_text(encoding="utf-8"))
    from config.mmm_dgp_config import CANONICAL_TRADE
    tp = TP["trade_params"][CANONICAL_TRADE]
    elast = {"price": tp["price_elasticity"], "acv": tp["distribution_elasticity"]}
    names = ["log_price_c", "promo_flag", "log_acv_c", "trend + seasonality", "tv", "meta", "tiktok", "ooh"]
    source = {**V, "trend + seasonality": V["seasonality"]}
    logf = []
    for k in names:
        x = np.asarray(source[k], dtype=float).copy()
        if k in ("log_price_c", "log_acv_c"):
            col = "price" if k == "log_price_c" else "acv"
            x = x + elast[col] * (math.log(reference_levels("dgp")[col]) - math.log(ref[col]))
        logf.append(x[None, :])
    logf = np.stack(logf)
    logS = sum(np.asarray(v, dtype=float) for k, v in V.items() if k != "t_norm_c")[None, :]
    return Components(names, logf, logS - logf.sum(0), "dgp", reference)


def group(comp: Components, groups: dict) -> Components:
    """Adds up the ln f of each group into a new driver. EXPLICIT: it never groups behind your back."""
    used = {x for g in groups.values() for x in g}
    missing = used - set(comp.names)
    if missing:
        raise KeyError(f"group: {sorted(missing)} are not among the drivers {comp.names}")
    names, logf = [], []
    for i, k in enumerate(comp.names):
        if k in used:
            continue
        names.append(k)
        logf.append(comp.logf[i])
    for new, pieces in groups.items():
        names.append(new)
        logf.append(sum(comp.logf[comp.names.index(p)] for p in pieces))
    order = [n for n in (BASE_ORDER_K8 if set(names) == set(BASE_ORDER_K8) else names)]
    idx = [names.index(n) for n in order]
    return Components(order, np.stack([logf[i] for i in idx]), comp.logbase, comp.side, comp.reference,
                      {**comp.grouped, **groups})


# ---------------------------------------------------------------------------
# THE value function
class ValueFunction:
    """v(S): sales per (draw, week) with the drivers in S on and the rest at their reference. It is the
    ONLY door through which the methods read the model.
    """

    def __init__(self, names, weekly_fn, label=""):
        self.names = list(names)
        self.K = len(self.names)
        self.N = (1 << self.K) - 1
        self._weekly = weekly_fn
        self._totals = {}
        self.label = label
        self.calls = 0

    @classmethod
    def from_components(cls, comp: Components) -> "ValueFunction":
        def weekly(mask):
            s = comp.logbase.copy()
            for i in range(comp.logf.shape[0]):
                if mask >> i & 1:
                    s = s + comp.logf[i]
            return np.exp(s)
        return cls(comp.names, weekly, f"{comp.side}, reference={comp.reference}")

    def mask(self, switched_on) -> int:
        m = 0
        for k in switched_on:
            m |= 1 << self.names.index(k)
        return m

    def weekly(self, mask: int) -> np.ndarray:
        self.calls += 1
        return np.asarray(self._weekly(mask), dtype=float)

    def total(self, mask: int) -> np.ndarray:
        if mask not in self._totals:
            self._totals[mask] = self.weekly(mask).sum(axis=-1)
        return self._totals[mask]

    def incremental(self) -> np.ndarray:
        return self.total(self.N) - self.total(0)


def v_dgp(switched_off, reference: str, dfm=None) -> np.ndarray:
    """Weekly DGP sales with `switched_off` at their reference and everything else as observed. The
    truth of the counterfactual without running the generator: the log DGP (without the 'gate' stress)
    is an exact product of factors, so switching a subset off is subtracting its ln f.
    """
    comp = dgp_components(reference, dfm)
    fv = ValueFunction.from_components(comp)
    switched_on = [k for k in comp.names if k not in set(switched_off)]
    unknown = set(switched_off) - set(comp.names)
    if unknown:
        raise KeyError(f"v_dgp: {sorted(unknown)} are not DGP drivers ({comp.names})")
    return fv.weekly(fv.mask(switched_on))[0]


# ---------------------------------------------------------------------------
# the four methods -- they consume ONLY v
def removal(fv: ValueFunction) -> dict:
    """(A) A_k = v(N) - v(N without k), per draw. It does NOT add up to the incremental: it misses the cross term."""
    return {k: fv.total(fv.N) - fv.total(fv.N & ~(1 << i)) for i, k in enumerate(fv.names)}


def sequential(fv: ValueFunction, order) -> dict:
    """(B) The drivers enter in `order`; each one takes what it adds on top of the ones already in."""
    order = list(order)
    if sorted(order) != sorted(fv.names):
        raise ValueError(f"sequential: the order has to be a permutation of {fv.names}, it is {order}")
    out, S = {}, 0
    for k in order:
        b = 1 << fv.names.index(k)
        out[k] = fv.total(S | b) - fv.total(S)
        S |= b
    return out


def shapley_weight(s: int, K: int) -> float:
    """|S|! (K - |S| - 1)! / K!: the probability that, in a random order, exactly those in S enter before."""
    return math.factorial(s) * math.factorial(K - s - 1) / math.factorial(K)


def shapley(fv: ValueFunction, weights=None) -> dict:
    """(C) Exact Shapley from the 2^K totals of v. `weights` exists ONLY for the test's sabotage."""
    weights = weights or shapley_weight
    out = {}
    for i, k in enumerate(fv.names):
        b = 1 << i
        acc = 0.0
        for S in range(fv.N + 1):
            if S & b:
                continue
            acc = acc + weights(bin(S).count("1"), fv.K) * (fv.total(S | b) - fv.total(S))
        out[k] = acc
    return out


def shapley_by_orders(fv: ValueFunction, orders, divisor=None) -> dict:
    """Shapley as the average of (B) over a list of orders. With ALL the orders and divisor=None it
    matches shapley(); it exists for the test (and for the sabotage of dropping one order).
    """
    orders = list(orders)
    divisor = divisor or len(orders)
    out = {k: 0.0 for k in fv.names}
    for o in orders:
        for k, x in sequential(fv, o).items():
            out[k] = out[k] + x
    return {k: x / divisor for k, x in out.items()}


def log_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """L(a, b) = (a - b) / (ln a - ln b), with L(a, a) = a. Continuous at a = b."""
    d = np.log(a) - np.log(b)
    almost = np.abs(d) < 1e-12  # trivial: floating-point tolerance
    return np.where(almost, a, (a - b) / np.where(almost, 1.0, d))


def lmdi(fv: ValueFunction, tol: float = 1e-9) -> dict:  # trivial: floating-point tolerance
    """(D) Weekly LMDI with the logarithmic mean: D_k = SUM_t L(v_t(N), v_t(empty)) * ln f_k,t. ln f_k,t
    is read from v: ln v_t({k}) - ln v_t(empty).
    """
    Y, B = fv.weekly(fv.N), fv.weekly(0)
    lnf = {k: np.log(fv.weekly(1 << i)) - np.log(B) for i, k in enumerate(fv.names)}
    gap = np.abs(sum(lnf.values()) - (np.log(Y) - np.log(B))).max()
    if gap > tol:
        raise NotMultiplicative(
            f"LMDI is not defined for this v: SUM ln f_k differs from ln v(N) - ln v(empty) by {gap:.3e}. "
            f"LMDI splits in proportion to ln f_k and that only closes if v is a product of factors.")
    L = log_mean(Y, B)
    return {k: (L * lnf[k]).sum(axis=-1) for k in fv.names}


def sequential_range(fv: ValueFunction, normalize=None) -> dict:
    """min and max of (B)_k over ALL the orders, without enumerating them. (B)_k depends only on the set
    S that entered before k: v(S u k) - v(S).
    """
    f = normalize or (lambda x: float(np.mean(x)))
    out = {}
    for i, k in enumerate(fv.names):
        b = 1 << i
        vals = [f(fv.total(S | b) - fv.total(S)) for S in range(fv.N + 1) if not S & b]
        out[k] = (min(vals), max(vals))
    return out


def efficiency(fv: ValueFunction, split: dict) -> float:
    """max|SUM_k split_k - (v(N) - v(empty))| over the draws, relative to the incremental."""
    total = sum(split[k] for k in fv.names)
    inc = fv.incremental()
    return float(np.max(np.abs(total - inc) / np.maximum(np.abs(inc), 1e-300)))  # trivial: floating-point tolerance


def all_subsets(names):
    for r in range(len(names) + 1):
        yield from combinations(names, r)
