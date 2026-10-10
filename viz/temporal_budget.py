"""BUDGET OPTIMIZATION OVER TIME.

# ! WATCH OUT -- the problem is not concave: that is why it starts from three points and compares them.
# ! WATCH OUT -- the saturation is tanh, concave from the origin: this model cannot ask for pulsing.
# ! WATCH OUT -- it adds L_MAX tail weeks: the level is not the period's; the gains do compare.
# ! WATCH OUT -- the priors were chosen by whoever wrote the generator: optimistic result by construction.
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.budget import (RELATIVE_BUDGET_TOL, FTOL_TEMPORAL, MAXITER_MULTIPLIERS,
                           FTOL_MULTIPLIERS, TEMPORAL_REBALANCE_ITER)
from config.mmm_dgp_config import (ADSTOCK_NORMALIZE, CHANNEL_COLUMNS, CANONICAL_GEO, L_MAX,
                                   CANONICAL_TRADE)
from config.budget import (CARRYOVER_TAIL, LOWER_BOUND, WEEKLY_CAP_MAX,
                           UPPER_BOUND)
from viz.business import short_channel, true_log_components


def adstock_weights(alpha) -> np.ndarray:
    """w[..., l] = alpha^l, normalized by the FULL sum of the L_MAX-lag kernel."""
    a = np.asarray(alpha, dtype=float)[..., None]
    w = a ** np.arange(L_MAX)
    return w / w.sum(axis=-1, keepdims=True) if ADSTOCK_NORMALIZE else w


def _stack_lags(x: np.ndarray, t_eval: int) -> np.ndarray:
    """(L_MAX, t_eval) with the series shifted l weeks and zeros before the start and after the end.
    `t_eval` can be larger than the length of `x`: those extra weeks are the TAIL, where there is no new
    spend but the carryover of the earlier spend does arrive.
    """
    T = x.shape[-1]
    X = np.zeros((L_MAX, t_eval))
    for l in range(L_MAX):
        end = min(T + l, t_eval)
        if end > l:
            X[l, l:end] = x[:end - l]
    return X


def _sat_and_derivative(z):
    """sat(z) = (1-e^-z)/(1+e^-z) = tanh(z/2) and its derivative. The SAME formula as the estimator's."""
    e = np.exp(-z)
    s = (1 - e) / (1 + e)
    return s, 2 * e / (1 + e) ** 2  # trivial: identity of the formula


class TemporalSurface:
    """Period revenue per draw as a function of the weekly spend of the four channels. The plan is a
    (K, T) array of weekly spend in money.
    """

    def __init__(self, idata, dfm, n_draws: int, seed: int, flat_base=False,
                 flat_cpm: bool = False, indices=None):
        post = idata.posterior
        self.channels = [short_channel(c) for c in CHANNEL_COLUMNS]
        self.K, self.T = len(self.channels), len(dfm)
        # T_eval = the weeks of the period PLUS the carryover tail. Spend is decided over T; sales are
        # collected over T_eval.
        self.tail = CARRYOVER_TAIL
        self.T_eval = self.T + self.tail
        self.dfm = dfm
        self.price = dfm["price"].values
        self.tscale = float(idata.constant_data["target_scale"].values)

        n_total = post.sizes["chain"] * post.sizes["draw"]
        if indices is not None:
            # `indices` allows evaluating on draws that were NOT used to optimize: if the plan is measured
            # with the same draws it was chosen with, the gain is overestimated.
            self.idx = np.sort(np.asarray(indices, dtype=int))
        else:
            rng = np.random.default_rng(seed)
            self.idx = np.sort(rng.choice(n_total, size=min(n_draws, n_total), replace=False))
        self.M = len(self.idx)

        def st(da):
            return da.stack(m=("chain", "draw")).values[self.idx]

        def st_t(da):
            return da.stack(m=("chain", "draw")).transpose("m", "date").values[self.idx]

        self.spend_obs = np.stack([dfm[f"{ch}_spend"].values for ch in self.channels])
        self.imp_obs = np.stack([dfm[c].values for c in CHANNEL_COLUMNS])
        cpm_csv = np.stack([dfm[f"{ch}_cpm"].values for ch in self.channels])
        with np.errstate(divide="ignore", invalid="ignore"):
            cpm_eff = np.where(self.imp_obs > 0, self.spend_obs / np.maximum(self.imp_obs, 1e-12),  # trivial: floating-point tolerance
                               cpm_csv)
        self.cpm = np.full_like(cpm_eff, cpm_eff.mean()) if flat_cpm else cpm_eff

        self.lam = np.stack([st(post["saturation_lam"].sel(channel=c)) for c in CHANNEL_COLUMNS])
        self.bet = np.stack([st(post["saturation_beta"].sel(channel=c)) for c in CHANNEL_COLUMNS])
        self.cscale = np.array([float(idata.constant_data["channel_scale"].sel(channel=c).values)
                                for c in CHANNEL_COLUMNS])
        alpha = np.stack([st(post["adstock_alpha"].sel(channel=c)) for c in CHANNEL_COLUMNS])
        self.w = adstock_weights(alpha)                    # (K, M, L_MAX)

        logS = st_t(post["y_original_scale"])
        c1 = np.stack([st_t(post["channel_contribution_original_scale"].sel(channel=c))
                       for c in CHANNEL_COLUMNS])
        base_log = logS - c1.sum(axis=0)
        if flat_base in (True, "trend_seas"):
            trend = st_t(post["control_contribution_original_scale"].sel(control="t_norm_c"))
            seas = st_t(post["yearly_seasonality_contribution_original_scale"])
            base_log = base_log - (trend + seas) + (trend + seas).mean(axis=1, keepdims=True)
        elif flat_base == "total":
            base_log = np.repeat(base_log.mean(axis=1, keepdims=True), self.T, axis=1)
            self.price = np.full(self.T, float(self.price.mean()))
        elif flat_base is not False:
            raise ValueError(f"flat_base={flat_base!r}: use False, 'trend_seas' or 'total'")
        self.period_base = np.exp(base_log)                # (M, T) -- only the real weeks
        # the tail: average weeks. There is no data for them, and any other choice (repeating the last
        # one, extrapolating the trend) brings in a stronger assumption without gaining anything.
        if self.tail:
            tail_base = np.repeat(self.period_base.mean(axis=1, keepdims=True), self.tail, axis=1)
            self.base = np.concatenate([self.period_base, tail_base], axis=1)
            self.price = np.concatenate([self.price,
                                         np.full(self.tail, float(self.price.mean()))])
        else:
            self.base = self.period_base
        self._logS = logS

        # bounds
        self.hist_spend_by_channel = self.spend_obs.sum(axis=1)
        self.budget = float(self.hist_spend_by_channel.sum())
        self.imp_cap = WEEKLY_CAP_MAX * self.imp_obs.max(axis=1)        # (K)
        self.spend_cap = self.imp_cap[:, None] * self.cpm                # (K, T)

        # CHECK, not decorative: with the observed spend the surface has to reproduce exp(y_original_scale).
        # If not, the spend -> impressions -> adstock conversion is wrong and every optimum below
        # maximizes something else.
        if flat_base is False and not flat_cpm:
            err = np.abs(self.sales(self.spend_obs)[:, :self.T] - np.exp(logS)).max()
            rel = err / float(np.abs(np.exp(logS)).max())
            if rel > 1e-10:  # trivial: floating-point tolerance
                raise AssertionError(
                    f"TemporalSurface: with the observed spend it does not reproduce exp(y_original_scale) "
                    f"(relative max|diff| {rel:.3e}). Check the conversion before optimizing.")

    # ------------------------------------------------------------------ the surface
    def impressions(self, spend: np.ndarray) -> np.ndarray:
        return np.asarray(spend, dtype=float) / self.cpm

    def adstock(self, imp: np.ndarray) -> np.ndarray:
        """(K, M, T_eval). w[k] @ stack_lags(imp[k]) is exactly the aligned adstock."""
        return np.stack([self.w[k] @ _stack_lags(imp[k], self.T_eval) for k in range(self.K)])

    def contributions(self, spend: np.ndarray):
        ad = self.adstock(self.impressions(spend))
        z = self.lam[:, :, None] * ad / self.cscale[:, None, None]
        s, ds = _sat_and_derivative(z)
        c = self.tscale * self.bet[:, :, None] * s
        return c, ds, z

    def sales(self, spend: np.ndarray) -> np.ndarray:
        c, _, _ = self.contributions(spend)
        return self.base * np.exp(c.sum(axis=0))

    def revenue_per_draw(self, spend: np.ndarray) -> np.ndarray:
        return (self.sales(spend) * self.price).sum(axis=1)

    def objective(self, spend: np.ndarray) -> float:
        return float(self.revenue_per_draw(spend).mean())

    def gradient(self, spend: np.ndarray, no_stock: bool = False) -> np.ndarray:
        """d(expected revenue) / d(spend_{k,t}), analytic. (K, T). d revenue / d imp_{k,s} =
        SUM_l w_{k,l} * [revenue_{s+l} * tscale * beta_k * sat'(z_{k,s+l}) / channel_scale_k],
        and d/d spend = that / cpm_{k,s}.

        `no_stock=True` evaluates sat' at z = 0, as if the week did not inherit stock from the earlier
        ones: it is the "first unit" marginal that ignores the adstock. It exists only as a contrast.
        """
        c, ds, _ = self.contributions(spend)
        if no_stock:
            ds = np.full_like(ds, _sat_and_derivative(np.zeros(1))[1][0])
        revenue_t = self.base * np.exp(c.sum(axis=0)) * self.price        # (M, T)
        g = np.zeros((self.K, self.T))
        for k in range(self.K):
            B = revenue_t * self.tscale * self.bet[k][:, None] * ds[k] * \
                (self.lam[k][:, None] / self.cscale[k])                    # (M, T_eval)
            acc = np.zeros((self.M, self.T))
            for l in range(L_MAX):
                # week s contributes to week s+l; with the tail, s+l can fall after the period, and THAT
                # is what the tail is there to collect
                end = min(self.T, self.T_eval - l)
                acc[:, :end] += self.w[k][:, l:l + 1] * B[:, l:l + end]
            g[k] = acc.mean(axis=0) / self.cpm[k]
        return g

    # ------------------------------------------------------------------ plans
    def historical_plan(self) -> np.ndarray:
        return self.spend_obs.copy()

    def even_plan(self) -> np.ndarray:
        """The historical total of each channel split evenly over the T weeks."""
        return np.repeat((self.hist_spend_by_channel / self.T)[:, None], self.T, axis=1)

    def plan_from_multipliers(self, m) -> np.ndarray:
        """The 7a_ plan: it scales the whole weekly series of each channel by its multiplier."""
        return self.spend_obs * np.asarray(m, dtype=float)[:, None]

    def violations(self, spend: np.ndarray, tol=1e-6) -> list:  # trivial: floating-point tolerance
        """List of violations. Empty = admissible plan. The weekly cap is measured in IMPRESSIONS."""
        spend = np.asarray(spend, dtype=float)
        imp = self.impressions(spend)
        bad = []
        if spend.min() < -tol:
            bad.append(f"negative spend (min {spend.min():.2f})")
        for k, ch in enumerate(self.channels):
            if imp[k].max() > self.imp_cap[k] * (1 + 1e-9) + tol:  # trivial: floating-point tolerance
                bad.append(f"{ch}: a week of {imp[k].max():.1f} impressions above the cap "
                           f"{self.imp_cap[k]:.1f}")
            tot = spend[k].sum()
            lo, hi = LOWER_BOUND * self.hist_spend_by_channel[k], UPPER_BOUND * self.hist_spend_by_channel[k]
            if not (lo - tol <= tot <= hi + tol):
                bad.append(f"{ch}: total {tot:.0f} outside [{lo:.0f}, {hi:.0f}]")
        if abs(spend.sum() - self.budget) > RELATIVE_BUDGET_TOL * self.budget:
            bad.append(f"total budget {spend.sum():.0f} != {self.budget:.0f}")
        return bad


# ---------------------------------------------------------------------------
# the optimizer
def optimize_temporal(sup: TemporalSurface, x0: np.ndarray, maxiter: int, fixed_totals: bool = False):
    """SLSQP over the 624 variables, with analytic gradient. Constraints: the total budget as an
    EQUALITY, the total of each channel between its two bounds, and a box per week.

    `fixed_totals=True` fixes the total of each channel at its historical value (EQUALITY per channel):
    only spend across weeks moves. Against the same optimum with free totals, it is the fair test of how
    much reallocating across channels adds.
    """
    from scipy.optimize import minimize
    K, T, B = sup.K, sup.T, sup.budget
    sc = B / (K * T)                        # scale unit: the average spend per cell

    def unpack(v):
        return v.reshape(K, T) * sc

    def neg(v):
        return -sup.objective(unpack(v)) / B

    def neg_grad(v):
        return -(sup.gradient(unpack(v)) * sc / B).ravel()

    # with fixed totals, the total budget is the sum of the K per-channel equalities: repeating it leaves
    # the constraints linearly dependent and SLSQP stops at the first iteration ("Singular matrix C in
    # LSQ subproblem") returning the start.
    cons = [] if fixed_totals else [{"type": "eq",
                                     "fun": lambda v: v.sum() * sc / B - 1.0,
                                     "jac": lambda v: np.full(v.size, sc / B)}]
    for k in range(K):
        gk = sup.hist_spend_by_channel[k]
        sel = np.zeros((K, T))
        sel[k] = 1.0
        sel = sel.ravel()
        if fixed_totals:
            cons.append({"type": "eq",
                         "fun": (lambda v, k=k, gk=gk: (v.reshape(K, T)[k].sum() * sc - gk) / B),
                         "jac": (lambda v, sel=sel: sel * sc / B)})
            continue
        cons.append({"type": "ineq",
                     "fun": (lambda v, k=k, gk=gk: (UPPER_BOUND * gk - v.reshape(K, T)[k].sum() * sc) / B),
                     "jac": (lambda v, sel=sel: -sel * sc / B)})
        cons.append({"type": "ineq",
                     "fun": (lambda v, k=k, gk=gk: (v.reshape(K, T)[k].sum() * sc - LOWER_BOUND * gk) / B),
                     "jac": (lambda v, sel=sel: sel * sc / B)})
    bounds = [(0.0, float(sup.spend_cap[k, t]) / sc) for k in range(K) for t in range(T)]
    res = minimize(neg, (np.asarray(x0, float) / sc).ravel(), jac=neg_grad, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": maxiter, "ftol": FTOL_TEMPORAL})
    plan = np.clip(unpack(res.x), 0.0, sup.spend_cap)
    return plan, res


def optimize_multipliers(sup: TemporalSurface, maxiter: int = MAXITER_MULTIPLIERS):
    """The 7a_ problem solved with THIS engine: one multiplier per channel, 4 variables. It exists so that
    S0, S1, S2 and S3 are measured with the same objective function and the comparison is between PLANS
    and not between implementations.
    """
    from scipy.optimize import minimize
    g = sup.hist_spend_by_channel

    def neg(m):
        return -sup.objective(sup.plan_from_multipliers(m)) / sup.budget

    def neg_grad(m):
        return -(sup.gradient(sup.plan_from_multipliers(m)) * sup.spend_obs).sum(axis=1)             / sup.budget

    ceiling = np.minimum(UPPER_BOUND, sup.imp_cap / sup.imp_obs.max(axis=1))
    res = minimize(neg, np.minimum(np.ones(sup.K), ceiling), jac=neg_grad, method="SLSQP",
                   bounds=list(zip([LOWER_BOUND] * sup.K, ceiling)),
                   constraints=[{"type": "eq", "fun": lambda m: (m @ g) / sup.budget - 1.0,
                                 "jac": lambda m: g / sup.budget}],
                   options={"maxiter": maxiter, "ftol": FTOL_MULTIPLIERS})
    return np.clip(res.x, LOWER_BOUND, ceiling), res


def repair_budget(sup: TemporalSurface, plan: np.ndarray, fixed_totals: bool = False) -> np.ndarray:
    """Rescales the plan so that it spends EXACTLY the budget, respecting the weekly box. It is needed
    because the clip after SLSQP can leave the total a few units short, and comparing objectives between
    plans that spend differently compares nothing. With `fixed_totals`, it repairs each channel toward its
    historical total separately.
    """
    plan = np.array(plan, dtype=float)
    if fixed_totals:
        for k in range(sup.K):
            for _ in range(TEMPORAL_REBALANCE_ITER):
                missing = sup.hist_spend_by_channel[k] - plan[k].sum()
                if abs(missing) < 1e-7 * sup.hist_spend_by_channel[k]:  # trivial: floating-point tolerance
                    break
                margin = (sup.spend_cap[k] - plan[k]) if missing > 0 else plan[k]
                if margin.sum() <= 0:
                    break
                plan[k] = np.clip(plan[k] + missing * margin / margin.sum(), 0.0, sup.spend_cap[k])
        return plan
    for _ in range(TEMPORAL_REBALANCE_ITER):
        missing = sup.budget - plan.sum()
        if abs(missing) < 1e-7 * sup.budget:  # trivial: floating-point tolerance
            break
        margin = (sup.spend_cap - plan) if missing > 0 else plan
        if margin.sum() <= 0:
            break
        plan = np.clip(plan + missing * margin / margin.sum(), 0.0, sup.spend_cap)
    return plan


def random_start(sup: TemporalSurface, seed: int) -> np.ndarray:
    """A random feasible plan: uniform in the weekly box, rescaled to the total of each channel."""
    rng = np.random.default_rng(seed)
    plan = rng.uniform(0.0, 1.0, size=(sup.K, sup.T)) * sup.spend_cap
    plan = plan * (sup.hist_spend_by_channel / plan.sum(axis=1))[:, None]
    return np.clip(plan, 0.0, sup.spend_cap)


# ---------------------------------------------------------------------------
# the same plan, evaluated in the GENERATOR
def dgp_revenue(dfm, spend: np.ndarray, cpm: np.ndarray) -> float:
    """Period revenue the DGP would produce with that plan. No model in between. It uses the generator's
    exact equation: ceiling L per channel, true lam on the adstock WITHOUT channel_scale, and the true
    base with media switched off.
    """
    from synthetic_data.adstock_dgp import geometric_adstock_aligned
    from synthetic_data.true_params_reader import get_channel_params, get_true_L
    comp = true_log_components(dfm)
    logS = sum(v for k, v in comp.items() if k != "t_norm_c")
    channels = [short_channel(c) for c in CHANNEL_COLUMNS]
    base = np.exp(logS - sum(np.asarray(comp[ch], float) for ch in channels))
    price = dfm["price"].values
    # the SAME tail as the model, with the same average-week convention: otherwise the model-generator
    # comparison would be measuring two different horizons
    T = len(dfm)
    t_eval = T + CARRYOVER_TAIL
    base = np.concatenate([base, np.full(CARRYOVER_TAIL, base.mean())])
    price = np.concatenate([price, np.full(CARRYOVER_TAIL, price.mean())])
    imp = np.asarray(spend, float) / cpm
    total = np.zeros(t_eval)
    for k, ch in enumerate(channels):
        p = get_channel_params(ch)
        L = get_true_L(ch, CANONICAL_GEO, CANONICAL_TRADE)
        x = np.concatenate([imp[k], np.zeros(CARRYOVER_TAIL)])
        ad = geometric_adstock_aligned(x, p["adstock_alpha"], l_max=L_MAX,
                                       normalize=ADSTOCK_NORMALIZE)
        z = p["lam"] * ad
        total = total + L * (1 - np.exp(-z)) / (1 + np.exp(-z))
    return float(((base * np.exp(total)) * price).sum())
