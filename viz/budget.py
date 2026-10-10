"""BUDGET OPTIMIZATION ON THE POSTERIOR.

# ! WATCH OUT -- it optimizes expected sales in LEVELS, not the sum of contributions in log-points.
# ! WATCH OUT -- the priors were chosen by whoever wrote the generator: optimistic result by construction.
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.budget import OPTIMIZATION_MARGINAL_EPS, CROSS_CHANNEL_OPTIM, REBALANCE_ITER
from config.mmm_dgp_config import ADSTOCK_NORMALIZE, CHANNEL_COLUMNS, L_MAX
from config.budget import LOWER_BOUND, UPPER_BOUND, N_DRAWS, OPTIMIZATION_DRAWS_SEED
# _saturate is the LogisticSaturation formula that channel_sweep already uses. It is imported instead of
# copied: two copies of the same saturation is exactly how two different ROAS come about.
from viz.business import _saturate, short_channel


class InfeasibleScenario(ValueError):
    """The sum of the bounds does not enclose the budget: there is no admissible allocation."""


class Surface:
    """Period sales per draw as a function of the four spend multipliers. `m` is a vector of multipliers
    in the order of `self.channels`, each relative to the REFERENCE spend of its channel (the historical
    one by default, but the saturated-channel check changes it).
    """

    def __init__(self, idata, dfm, reference=None, n_draws=N_DRAWS, seed=OPTIMIZATION_DRAWS_SEED):
        from synthetic_data.adstock_dgp import geometric_adstock_aligned
        post = idata.posterior
        self.channels = [short_channel(c) for c in CHANNEL_COLUMNS]
        self.dfm = dfm
        self.price = dfm["price"].values
        self.historical_spend = {ch: float(dfm[f"{ch}_spend"].sum()) for ch in self.channels}
        # The reference spend is the point where the curve is evaluated with m=1. The saturated-channel
        # check moves it; everything else in the scenario is defined RELATIVE to it.
        self.ref_spend = dict(self.historical_spend if reference is None else reference)

        tscale = float(idata.constant_data["target_scale"].values)
        rng = np.random.default_rng(seed)
        n_total = post.sizes["chain"] * post.sizes["draw"]
        self.idx = rng.choice(n_total, size=min(n_draws, n_total), replace=False)

        def st(da):
            return da.stack(m=("chain", "draw")).values[self.idx]

        def st_t(da):
            return da.stack(m=("chain", "draw")).transpose("m", "date").values[self.idx]

        logS = st_t(post["y_original_scale"])
        self._lam, self._bet, self._ad, self._c1, self._cscale = {}, {}, {}, {}, {}
        for col in CHANNEL_COLUMNS:
            ch = short_channel(col)
            self._lam[ch] = st(post["saturation_lam"].sel(channel=col))
            self._bet[ch] = st(post["saturation_beta"].sel(channel=col))
            self._cscale[ch] = float(idata.constant_data["channel_scale"].sel(channel=col).values)
            alpha = st(post["adstock_alpha"].sel(channel=col))
            # Adstock PER DRAW, like channel_sweep: with the mean alpha a residual would remain
            self._ad[ch] = np.stack([geometric_adstock_aligned(dfm[col].values, float(a), l_max=L_MAX,
                                                               normalize=ADSTOCK_NORMALIZE) for a in alpha])
            self._c1[ch] = st_t(post["channel_contribution_original_scale"].sel(channel=col))
        self._tscale = tscale
        # sales with ALL media off: the base the surface is rebuilt on
        self._S_without = np.exp(logS - sum(self._c1.values()))
        self._logS = logS

        # CHECK, not decorative: at the multipliers that leave each channel at its historical spend, the
        # surface has to give EXACTLY exp(y_original_scale). If not, the scale conversion is wrong and the
        # optimum below maximizes something else.
        m_hist = np.array([self.historical_spend[ch] / self.ref_spend[ch] for ch in self.channels])
        err = float(np.abs(self.weekly_sales(m_hist) - np.exp(logS)).max())
        rel = err / float(np.abs(np.exp(logS)).max())
        if rel > 1e-12:  # trivial: floating-point tolerance
            raise AssertionError(
                f"Surface: with the historical spend it does not reproduce exp(y_original_scale) "
                f"(relative max|diff| = {rel:.3e}). Check the scale conversion before trusting any optimum.")

    # ------------------------------------------------------------------ the surface
    def contribution(self, ch: str, m: float) -> np.ndarray:
        """c_k(m) in log-points, per (draw, week). m is relative to the reference spend."""
        factor = m * self.ref_spend[ch] / self.historical_spend[ch]
        x = (self._ad[ch] * factor) / self._cscale[ch]
        return _saturate(x, self._lam[ch][:, None], self._bet[ch][:, None]) * self._tscale

    def weekly_sales(self, m) -> np.ndarray:
        s = np.log(self._S_without)
        for i, ch in enumerate(self.channels):
            s = s + self.contribution(ch, float(m[i]))
        return np.exp(s)

    def revenue(self, m) -> np.ndarray:
        """Period revenue per draw, in monetary units."""
        return (self.weekly_sales(m) * self.price).sum(axis=1)

    def expected_revenue(self, m) -> float:
        return float(self.revenue(m).mean())

    def spend(self, m) -> dict:
        return {ch: float(m[i]) * self.ref_spend[ch] for i, ch in enumerate(self.channels)}

    def total_spend(self, m) -> float:
        return sum(self.spend(m).values())

    # ------------------------------------------------------------------ returns
    def marginal_return(self, m, eps=OPTIMIZATION_MARGINAL_EPS) -> dict:
        """d(expected revenue) / d(spend of channel k), by centered finite difference. The same computation
        average_and_marginal_roas() does for one channel, but evaluated at the JOINT point m, which is what
        is needed to verify KKT at the optimum.
        """
        out = {}
        for i, ch in enumerate(self.channels):
            plus, minus = np.array(m, float), np.array(m, float)
            plus[i] += eps
            minus[i] -= eps
            d_spend = 2 * eps * self.ref_spend[ch]  # trivial: identity of the formula
            out[ch] = (self.expected_revenue(plus) - self.expected_revenue(minus)) / d_spend
        return out

    def average_roas(self, m) -> dict:
        """Incremental revenue of having the channel on / its spend, at the point m."""
        out = {}
        base = self.expected_revenue(m)
        for i, ch in enumerate(self.channels):
            zero = np.array(m, float)
            zero[i] = 0.0
            out[ch] = (base - self.expected_revenue(zero)) / (float(m[i]) * self.ref_spend[ch])
        return out


# ---------------------------------------------------------------------------
# the optimum
def _bounds(sup: Surface, lo=LOWER_BOUND, hi=UPPER_BOUND):
    return [(lo, hi)] * len(sup.channels)


def optimize(sup: Surface, budget: float, lo=LOWER_BOUND, hi=UPPER_BOUND):
    """Maximizes the EXPECTED revenue of the period with the total budget fixed. Returns (m_optimal,
    scipy_result).
    """
    from scipy.optimize import minimize
    g = np.array([sup.ref_spend[ch] for ch in sup.channels])
    if not (lo * g).sum() <= budget <= (hi * g).sum():
        raise InfeasibleScenario(
            f"budget={budget:.1f} outside [{(lo * g).sum():.1f}, {(hi * g).sum():.1f}], "
            f"the sum of the bounds. No allocation satisfies both at once.")
    # start: the feasible point closest to the split proportional to the reference spend
    m0 = np.clip(np.full(len(g), budget / g.sum()), lo, hi)
    m0 = m0 * (budget / float((m0 * g).sum()))
    m0 = np.clip(m0, lo, hi)
    res = minimize(
        lambda m: -sup.expected_revenue(m) / 1e6,  # trivial: numerical scale of the objective, in millions
        m0, method="SLSQP", bounds=_bounds(sup, lo, hi),
        constraints=[{"type": "eq", "fun": lambda m: float((m * g).sum() - budget) / budget}],
        options=dict(CROSS_CHANNEL_OPTIM),
    )
    return np.clip(res.x, lo, hi), res


def average_roi_split(sup: Surface, budget: float, lo=LOWER_BOUND, hi=UPPER_BOUND):
    """CONTRAST, not a proposal: splits the budget in proportion to the AVERAGE ROI. It exists so that 7a_
    can show that the optimum is NOT this.
    """
    g = np.array([sup.ref_spend[ch] for ch in sup.channels])
    m_ref = np.ones(len(g))
    roas = np.array([sup.average_roas(m_ref)[ch] for ch in sup.channels])
    weight = roas / roas.sum()
    spend = weight * budget
    m = np.clip(spend / g, lo, hi)
    # rescale what was left free to spend the whole budget again
    for _ in range(REBALANCE_ITER):
        missing = budget - float((m * g).sum())
        if abs(missing) < 1e-6 * budget:  # trivial: floating-point tolerance
            break
        free = np.array([(m[i] < hi - 1e-12) if missing > 0 else (m[i] > lo + 1e-12)  # trivial: floating-point tolerance
                         for i in range(len(g))], dtype=float)
        if free.sum() == 0:
            break
        w = free * weight
        w = w / w.sum() if w.sum() > 0 else free / free.sum()
        m = np.clip(m + missing * w / g, lo, hi)
    return m


def saturated_scenario(sup_base: Surface, idata, dfm, channel: str, factor: float,
                       n_draws=N_DRAWS, seed=OPTIMIZATION_DRAWS_SEED) -> Surface:
    """The SAME surface with the reference spend of `channel` multiplied by `factor`. It is the check: the
    channel is evaluated in the flat zone of its saturation.
    """
    ref = dict(sup_base.ref_spend)
    ref[channel] = ref[channel] * factor
    return Surface(idata, dfm, reference=ref, n_draws=n_draws, seed=seed)
