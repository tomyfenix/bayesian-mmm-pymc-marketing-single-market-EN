"""FOUR MEDIA PLANS ON THE SAME BUDGET, AND THE FAIR TEST OF THE CROSS-CHANNEL SPLIT.

S1 (one multiplier per channel) coincides with S0 by construction under the weekly cap: it is an
identity, not a finding. What measures the cross-channel split is comparing the temporal optimum with
free channel totals (S3) against the same optimum with the totals fixed at the historical ones.

# ! NOTE -- the flighting is deliberate: flattening measures the design of the dataset, not a real plan.
# ! NOTE -- the priors were chosen by whoever wrote the generator: optimistic result by construction.

Produces: model/7_optimization/plots/7b1_weekly_plans__<dataset_label>.png,
    model/7_optimization/plots/7b2_paired_gains__<dataset_label>.png,
    results/7b_temporal_optimization__<dataset_label>.md
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.budget import SCENARIOS  # noqa: E402
from config.budget import RANDOM_START_SEED, OFF_THRESHOLD, EDGE_VERDICT_THRESHOLDS  # noqa: E402
from config.idata_paths import require_idata  # noqa: E402
from config.mmm_dgp_config import L_MAX, HDI_PROB, Q4_WEEKS  # noqa: E402
from config.budget import (STARTS, CARRYOVER_TAIL, CONTROL_CV_MAX, LOWER_BOUND,
                                WEEKLY_CAP_MAX,
                                UPPER_BOUND, MAXITER_TEMPORAL, N_DRAWS_EVALUATION,
                                N_DRAWS, N_DRAWS_OPTIMIZATION, OPTIMIZATION_DRAWS_SEED,
                                STARTS_TOLERANCE)  # noqa: E402
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale  # noqa: E402
import viz.temporal_budget as PT  # noqa: E402
from viz.io import figure_path, provenance_fields, save_figure  # noqa: E402
from viz.labels import HDI  # noqa: E402
from viz.business import canonical_slice  # noqa: E402
from viz.style import BASELINE, GRID, INK, INK_MUTED, INK_SECONDARY, apply_style, channel_color  # noqa: E402

apply_style()

DATASET_LABEL = get_dataset_label()
if DATASET_LABEL != "log__no_noise":
    sys.exit(f"7b_: these scenarios are ONLY for the no-noise regime. The live dataset is "
             f"{DATASET_LABEL!r}; regenerate the dataset with step 1 of the README.")
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
dfm = canonical_slice()
OUTPUT = ROOT / "results" / f"7b_temporal_optimization__{DATASET_LABEL}.md"


def table(headers, rows):
    widths = [max(len(str(h)), *(len(str(f[i])) for f in rows)) for i, h in enumerate(headers)]
    out = ["| " + " | ".join(str(h).ljust(a) for h, a in zip(headers, widths)) + " |",
           "|" + "|".join("-" * (a + 2) for a in widths) + "|"]  # cosmetic
    out += ["| " + " | ".join(str(c).ljust(a) for c, a in zip(f, widths)) + " |" for f in rows]
    return out


def paired(ia, ib):
    """PAIRED gain in % : the same draw in both plans. Median and HDI94."""
    d = 100 * (ia - ib) / ib  # trivial: unit
    lo, hi = az.hdi(d, hdi_prob=HDI_PROB)
    return float(np.median(d)), float(lo), float(hi), float((d > 0).mean())


def interior_cv(sup, plan):
    sl = slice(L_MAX, sup.T - L_MAX)
    return {sup.channels[k]: float(plan[k, sl].std() / plan[k, sl].mean()) for k in range(sup.K)}


def longest_run(weeks):
    """(first, last) of the longest run of consecutive weeks."""
    best, start = (weeks[0], weeks[0]), weeks[0]
    for a, b in zip(weeks, weeks[1:]):
        if b != a + 1:
            start = b
        if b - start > best[1] - best[0]:
            best = (start, b)
    return int(best[0]), int(best[1])


def main():
    t_total = time.perf_counter()
    sup_o = PT.TemporalSurface(idata, dfm, n_draws=N_DRAWS_OPTIMIZATION, seed=OPTIMIZATION_DRAWS_SEED)
    # The evaluation runs on the draws that did NOT enter the optimization. Measuring a plan with the
    # same draws it was chosen with is evaluating on the training set: the gain comes out inflated by
    # the part of the optimum that is noise of those particular draws.
    _n_total = idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"]
    _rest = np.setdiff1d(np.arange(_n_total), sup_o.idx)
    sup_e = PT.TemporalSurface(idata, dfm, n_draws=len(_rest), seed=OPTIMIZATION_DRAWS_SEED, indices=_rest)
    assert len(_rest) == _n_total - N_DRAWS_OPTIMIZATION, (len(_rest), _n_total)
    assert len(_rest) == N_DRAWS_EVALUATION, (f"config says {N_DRAWS_EVALUATION} evaluation draws and the "
                                            f"complement has {len(_rest)}: update config/budget.py")
    assert not set(sup_e.idx) & set(sup_o.idx), "the evaluation draws overlap with the optimization ones"
    channels, T = sup_o.channels, sup_o.T

    # ---------------------------------------------------------------- the four plans
    S0 = sup_o.historical_plan()
    # S1 is optimized with the SAME 400 draws 7a_ uses, not with the 200 of the temporal
    # optimization: the point of S1 is to be the same plan as 7a_, and with another subset of draws
    # the multipliers move in the fourth decimal and the cross-check between the two engines is lost.
    sup_s1 = PT.TemporalSurface(idata, dfm, n_draws=N_DRAWS, seed=OPTIMIZATION_DRAWS_SEED)
    mult, res_m = PT.optimize_multipliers(sup_s1)
    S1 = sup_o.plan_from_multipliers(mult)
    S2 = sup_o.even_plan()
    viol_S2 = sup_o.violations(S2)
    # With the weekly cap in IMPRESSIONS, S1 collapses to S0 and it is not a bug: the peak week of each
    # channel is already at its observed maximum, so no multiplier can go above 1, and with the budget
    # fixed as an equality the only feasible point is m = (1,1,1,1).
    s1_collapses = bool(np.allclose(mult, 1.0, atol=1e-6))  # trivial: floating-point tolerance

    starts = {"S0": S0, "S2": S2, "random": PT.random_start(sup_o, RANDOM_START_SEED)}
    assert tuple(starts) == STARTS, (tuple(starts), STARTS)
    runs = {}
    for name, x0 in starts.items():
        t = time.perf_counter()
        plan, res = PT.optimize_temporal(sup_o, x0, maxiter=MAXITER_TEMPORAL)
        plan = PT.repair_budget(sup_o, plan)
        runs[name] = {"plan": plan, "obj": sup_o.objective(plan), "nit": res.nit,
                      "ok": bool(res.success), "msg": str(res.message),
                      "s": time.perf_counter() - t, "viol": sup_o.violations(plan)}
    # an SLSQP that stops returns its start: without this, an "optimum" could be S0 or S2
    for name, c in runs.items():
        assert c["ok"] and c["nit"] > 1, f"7b_: start {name} did not converge ({c['nit']} it.: {c['msg']})"
    objs = np.array([c["obj"] for c in runs.values()])
    starts_spread = float((objs.max() - objs.min()) / objs.mean())
    if starts_spread > STARTS_TOLERANCE:
        sys.exit(f"7b_: the three starts do NOT reach the same objective (relative spread "
                 f"{starts_spread:.2e} > {STARTS_TOLERANCE:.0e}). There are local optima: that is "
                 f"the result and no 'optimum' is published without saying so.")
    S3 = runs[max(runs, key=lambda n: runs[n]["obj"])]["plan"]

    plans = dict(zip(SCENARIOS, (S0, S1, S2, S3)))   # the plan labels come from config/
    viol = {n: sup_o.violations(p) for n, p in plans.items()}
    for name in ("S0_historical", "S2_even_pacing", "S3_temporal_optimum"):
        assert not viol[name], (name, viol[name])

    # ---------------------------------------------------------------- the fair test (D5)
    # The same temporal optimum, with each channel's total fixed at its historical one: only the spend
    # across weeks moves. S3 minus this plan is what reallocating ACROSS channels adds, measured with
    # the same optimizer and the same weekly cap (S1 - S0 does not measure it: it is zero by construction).
    t = time.perf_counter()
    S3_fixed, res_fixed = PT.optimize_temporal(sup_o, S2, maxiter=MAXITER_TEMPORAL, fixed_totals=True)
    S3_fixed = PT.repair_budget(sup_o, S3_fixed, fixed_totals=True)
    sec_fixed = time.perf_counter() - t
    # an SLSQP that stops returns the start: without this assert, "S3 fixed" would silently be S2
    assert res_fixed.success and res_fixed.nit > 1, (res_fixed.nit, res_fixed.message)
    assert not sup_o.violations(S3_fixed), sup_o.violations(S3_fixed)
    assert np.allclose(S3_fixed.sum(axis=1), sup_o.hist_spend_by_channel, rtol=1e-6), "S3 fixed moved a total"  # trivial: floating-point tolerance
    # S3 has more freedom than S3_fixed: if its objective is lower, the optimizer did not get there
    assert sup_o.objective(S3) >= sup_o.objective(S3_fixed) - 1e-6 * abs(sup_o.objective(S3)), (  # trivial: floating-point tolerance
        "7b_: the optimum with free totals yields less than with fixed totals")

    # ---------------------------------------------------------------- feasibility
    rev = {n: sup_e.revenue_per_draw(p) for n, p in plans.items()}
    rev_fixed = sup_e.revenue_per_draw(S3_fixed)
    means = {n: float(v.mean()) for n, v in rev.items()}
    for n in ("S0_historical", "S1_cross_channel", "S2_even_pacing"):
        assert means["S3_temporal_optimum"] >= means[n] - 1e-6, (  # trivial: floating-point tolerance
            f"7b_: the objective of S3 ({means['S3_temporal_optimum']:.1f}) is lower than that of {n} "
            f"({means[n]:.1f}), which is feasible. The optimizer did not find the optimum.")

    # ---------------------------------------------------------------- KKT
    g = sup_o.gradient(S3)
    at_zero = S3 < 1e-6 * sup_o.spend_cap  # trivial: floating-point tolerance
    at_cap = S3 > sup_o.spend_cap * (1 - 1e-9)  # trivial: floating-point tolerance
    free = ~(at_zero | at_cap)
    gl = g[free]
    kkt_spread = float((gl.max() - gl.min()) / gl.mean()) if gl.size else float("nan")
    kkt_by_channel = []
    for k, ch in enumerate(channels):
        lk = free[k]
        gk = g[k][lk]
        kkt_by_channel.append([ch, int(at_zero[k].sum()), int(at_cap[k].sum()), int(lk.sum()),
                               f"{gk.mean():.4f}" if lk.any() else "—",
                               f"{(gk.max() - gk.min()) / gk.mean():.2e}" if lk.sum() > 1 else "—"])

    CONTROL = "constant base and price + constant CPM"
    controls = {}
    for label, kw in (("context: trend and seasonality at their mean",
                       dict(flat_base="trend_seas")),
                      ("context: constant base and price", dict(flat_base="total")),
                      (CONTROL, dict(flat_base="total", flat_cpm=True))):
        sc = PT.TemporalSurface(idata, dfm, n_draws=N_DRAWS_OPTIMIZATION, seed=OPTIMIZATION_DRAWS_SEED, **kw)
        pc, rc = PT.optimize_temporal(sc, sc.even_plan(), maxiter=MAXITER_TEMPORAL)
        pc = PT.repair_budget(sc, pc)
        b = sc.base.mean(axis=0) * sc.price
        controls[label] = {"cv": interior_cv(sc, pc), "cv_base": float(b.std() / b.mean()),
                           "nit": rc.nit, "ok": bool(rc.success), "plan": pc, "sup": sc}
    cv_literal = max(controls["context: trend and seasonality at their mean"]["cv"].values())
    cv_isolated = max(controls[CONTROL]["cv"].values())
    assert cv_isolated < CONTROL_CV_MAX, (
        f"7b_ CONTROL FAILED: with constant base, price and CPM, the optimum is NOT even pacing in "
        f"the interior (maximum CV {cv_isolated:.4f} >= {CONTROL_CV_MAX}). There is nothing left there "
        f"for the optimum to follow except the adstock at the edges: if it still comes out with a shape, "
        f"the optimizer has a bug and no number in this script holds.")

    # ---------------------------------------------------------------- shape of the optimum
    # `period_base` and not `base`: the correlation is against the 156 real weeks, not against the
    # 168 of the evaluation -- the 12 of the tail are an invented constant and have no spend.
    base_t = sup_e.period_base.mean(axis=0)
    weekly_spend = S3.sum(axis=0)
    corr = float(np.corrcoef(weekly_spend, base_t)[0, 1])
    corr_by_channel = {ch: float(np.corrcoef(S3[k], base_t)[0, 1]) for k, ch in enumerate(channels)}
    flights = {}
    for k, ch in enumerate(channels):
        on = S3[k] > OFF_THRESHOLD * sup_o.spend_cap[k]
        gaps = [t for t in range(1, T - 1) if not on[t] and on[:t].any() and on[t + 1:].any()]
        flights[ch] = (int((~on).sum()), len(gaps))
    free_marg = float(gl.mean())
    marg_at_zero = g[at_zero]
    base_rel_zero = (float(np.mean([base_t[t] for k in range(sup_o.K)
                                    for t in np.where(at_zero[k])[0]]) / base_t.mean())
                     if at_zero.any() else float("nan"))
    assert marg_at_zero.size == 0 or marg_at_zero.max() <= free_marg * (1 + 1e-6), (  # trivial: floating-point tolerance
        f"7b_: there are weeks at zero whose marginal ({marg_at_zero.max():.4f}) exceeds that of the "
        f"free weeks ({free_marg:.4f}). That violates KKT: the optimum should have put spend on them. "
        f"It is not a flight, it is a bug.")

    # Why a cell ends at zero. The marginal of the first unit is NOT Y_t*beta*f'(0)/cpm_t: the week
    # inherits stock from the previous ones (the adstock), so f' is evaluated where that stock already
    # left it, not at zero. The explanation is checked in both directions: with the stock, every cell at
    # zero falls below the common marginal (it explains it); ignoring it (f'(0)), some fall above, that
    # is, without the stock the optimum would have put spend on them.
    def explains(marg):
        return bool(marg[at_zero].size == 0 or marg[at_zero].max() <= free_marg * (1 + 1e-6))  # trivial: floating-point tolerance

    g_no_stock = sup_o.gradient(S3, no_stock=True)
    assert explains(g), "7b_: with the stock, the cells at zero do not fall below the common marginal"
    assert not explains(g_no_stock), (
        "7b_: ignoring the stock the cells at zero ALSO fall below the common marginal: the stock is "
        "not what switches them off and the text of the .md would say something false")
    n_stock_only = int((g_no_stock[at_zero] > free_marg).sum())
    is_q4 = np.isin(dfm["date"].dt.isocalendar().week.values, np.arange(*Q4_WEEKS))
    promo = dfm["promo_flag"].values > 0
    tv = channels.index("tv") if "tv" in channels else None
    off = {}
    if tv is not None and at_zero[tv].any():
        wk = np.where(at_zero[tv])[0]
        cpm_rel = sup_o.cpm[tv] / sup_o.cpm[tv].mean()
        off = {"n": len(wk), "q4": int(is_q4[wk].sum()), "q4_total": int(is_q4.sum()),
               "post_promo": int(sum(promo[s - 1] for s in wk if s > 0)),
               "promo": int(promo[wk].sum()),
               "cpm_q4": float(cpm_rel[is_q4].mean()), "cpm_rest": float(cpm_rel[~is_q4].mean()),
               "run": longest_run(wk)}

    tail_sl, interior_sl = slice(T - L_MAX, T), slice(L_MAX, T - L_MAX)
    december = {ch: (float(S3[k, tail_sl].mean()), float(S3[k, interior_sl].mean()))
                for k, ch in enumerate(channels)}
    dec_total = float(S3[:, tail_sl].sum(axis=0).mean() / S3[:, interior_sl].sum(axis=0).mean())
    pc_ctrl = controls[CONTROL]["plan"]
    dec_control = float(pc_ctrl[:, tail_sl].sum(axis=0).mean() /
                        pc_ctrl[:, interior_sl].sum(axis=0).mean())

    # ---------------------------------------------------------------- the generator
    dgp = {n: PT.dgp_revenue(dfm, p, sup_o.cpm) for n, p in plans.items()}

    # ---------------------------------------------------------------- figures
    # 7b1: the base in its own panel (index 100 = its mean), and per channel the spend of S0, S2 and S3
    # plus the PRESSURE of S3 -- impressions with adstock, valued at the channel's mean CPM so that it
    # stays in the same money as the spend. The adstock uses the alpha of each evaluation draw.
    weeks = np.arange(T)
    pressure = sup_e.adstock(sup_e.impressions(S3)).mean(axis=1)[:, :T] * sup_o.cpm.mean(axis=1)[:, None]
    fig, axes = plt.subplots(5, 1, figsize=(13.5, 12.5), sharex=True,  # cosmetic
                             gridspec_kw={"height_ratios": [0.8, 1, 1, 1, 1]})  # cosmetic
    ax = axes[0]
    ax.plot(weeks, 100 * base_t / base_t.mean(), color=INK, linewidth=1.1)  # trivial: index 100  # cosmetic
    ax.axhline(100, color=INK_MUTED, linewidth=0.8, linestyle=":")  # cosmetic
    for s in np.where(is_q4)[0]:
        ax.axvspan(s - 0.5, s + 0.5, color=BASELINE, alpha=0.35, linewidth=0)  # cosmetic
    ax.set_ylabel("base $Y_t$ without media\n(index, mean = 100)", fontsize=9)  # cosmetic
    ax.set_title("shaded: year-end weeks, with high CPM", fontsize=8.5, loc="left",  # cosmetic
                 color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID)
    for k, ch in enumerate(channels):
        ax = axes[k + 1]
        ax.fill_between(weeks, S0[k] / 1e3, color=BASELINE, step="mid", alpha=0.85,  # cosmetic
                        label="S0 historical")
        ax.plot(weeks, S2[k] / 1e3, color=INK_SECONDARY, linewidth=1.0, linestyle="--",  # cosmetic
                label="S2 flattened")
        ax.step(weeks, S3[k] / 1e3, where="mid", color=channel_color(ch), linewidth=1.5,  # cosmetic
                label="S3 temporal optimum")
        ax.plot(weeks, pressure[k] / 1e3, color=INK, linewidth=1.0, alpha=0.8,  # cosmetic
                label="pressure of S3 (with adstock)")
        ax.plot(weeks, sup_o.spend_cap[k] / 1e3, color=INK_MUTED, linestyle=":", linewidth=0.9)  # cosmetic
        ax.set_ylabel(f"{ch}\n(thousand $/wk)", fontsize=9)  # cosmetic
        ax.grid(axis="y", color=GRID)
        if k == 0:
            ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center")  # cosmetic
    axes[-1].set_xlabel("week")
    fig.suptitle("Weekly spend by channel: historical, flattened and temporal optimum",
                 fontsize=13, x=0.007, ha="left", y=0.995)  # cosmetic
    tv_text = (f" tv is switched off in {off['n']} weeks, {off['q4']} of them at year end "
               f"(CPM {off['cpm_q4']:.2f}× its mean): the optimum does not follow the base alone." if off else "")
    fig.text(0.007, 0.976,  # cosmetic
             f"Same total budget ({sup_o.budget / 1e6:.2f} M) and same bounds per channel. The "  # trivial: unit
             f"dotted line is the weekly cap, in impressions converted to money with the CPM of "
             f"each week.\nThe total optimal spend correlates {corr:+.2f} with the base.{tv_text}"
             f"\nSales are evaluated with a {CARRYOVER_TAIL}-week tail: the right edge receives "
             f"{100 * dec_total:.0f}% of an interior week.",  # trivial: unit
             fontsize=9, color=INK_SECONDARY, va="top")  # cosmetic
    # above the provenance stamp, which viz.io.stamp_figure writes at the bottom
    fig.text(0.007, 0.024,  # cosmetic
             "The dataset's flighting is deliberate, put there to identify the saturation curve: "
             "the gain from flattening measures that design, not a real plan.",
             fontsize=8.5, color=INK_MUTED)  # cosmetic
    fig.tight_layout(rect=(0, 0.04, 1, 0.935))  # cosmetic
    p1 = figure_path(ROOT, "7_optimization", "7b1_weekly_plans", DATASET_LABEL)
    save_figure(fig, p1, script_file=__file__)

    pairs = [("S1 − S0\ncross-channel split", rev["S1_cross_channel"], rev["S0_historical"]),
             ("S2 − S0\nflatten", rev["S2_even_pacing"], rev["S0_historical"]),
             ("S3 − S2\nfollow base and adstock", rev["S3_temporal_optimum"], rev["S2_even_pacing"]),
             ("S3 − S0\ntotal", rev["S3_temporal_optimum"], rev["S0_historical"]),
             ("S3 − S3 fixed\ncross-channel split,\nfair test", rev["S3_temporal_optimum"], rev_fixed)]
    fig, ax = plt.subplots(figsize=(11.0, 5.6))  # cosmetic
    for i, (name, a, b) in enumerate(pairs):
        med, lo, hi, _ = paired(a, b)
        ax.plot([i, i], [lo, hi], color=channel_color(channels[i % len(channels)]), linewidth=11,  # cosmetic
                solid_capstyle="butt", alpha=0.9, zorder=3)  # cosmetic
        ax.plot([i], [med], marker="D", ms=6, color=INK, zorder=4)  # cosmetic
        # the HDI94 is THINNER than the marker at this scale: if it is not written, it cannot be read
        ax.annotate(f"{med:+.3f}%\n{HDI} [{lo:+.3f}, {hi:+.3f}]", (i, med),
                    textcoords="offset points", xytext=(14, -6), fontsize=9.5,  # cosmetic
                    color=INK_SECONDARY)
    ax.axhline(0, color=INK, linewidth=1.0)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p[0] for p in pairs], fontsize=9)  # cosmetic
    ax.set_xlim(-0.6, len(pairs) + 0.85)     # room on the right for the label of the last point  # cosmetic
    ax.set_ylabel("paired gain per draw (% of revenue)")
    ax.set_title(f"The gains, paired by draw — median and {HDI}\n"
                 "S1 − S0 is zero by construction: with the weekly cap, the only feasible multiplier "
                 "plan is the historical one.\nThe cross-channel split is measured in the "
                 "last column: the temporal optimum with free totals against the same one with fixed totals.",
                 fontsize=10.5, loc="left")  # cosmetic
    ax.grid(axis="y", color=GRID, zorder=0)
    # above the provenance stamp, which viz.io.stamp_figure writes at the bottom
    fig.text(0.005, 0.048,  # cosmetic
             "Each draw evaluates both plans and subtracts: the parameter uncertainty cancels out. "
             "The priors were chosen by whoever wrote the generator.",
             fontsize=8.5, color=INK_MUTED)  # cosmetic
    fig.tight_layout(rect=(0, 0.082, 1, 1))  # cosmetic
    p2 = figure_path(ROOT, "7_optimization", "7b2_paired_gains", DATASET_LABEL)
    save_figure(fig, p2, script_file=__file__)

    # ---------------------------------------------------------------- the .md
    prov = provenance_fields("linear_trend")
    L = [
        "# Budget optimization over TIME — four plans, the same money",
        "",
        "GENERATED by `model/7_optimization/7b_temporal_optimization.py` — "
        "**do not edit by hand**.",
        "",
        f"> **Provenance:** dataset=`{DATASET_LABEL}` | idata=`{prov['idata_name']}`@`{prov['idata_hash']}` "
        f"| csv=`{prov['csv_name']}`@`{prov['csv_hash']}` | commit=`{prov['commit']}` "
        f"| draws: {N_DRAWS_OPTIMIZATION} to optimize, {len(rev['S0_historical'])} to evaluate "
        f"| generated {datetime.now().isoformat(timespec='seconds')}",
        "",
        "> **READ BEFORE CITING.** The priors were chosen by whoever wrote the generator, so the response",
        "> curves are **optimistic by construction**. And the dataset's flighting is **deliberate**: it is",
        "> there so that the saturation curve is identifiable. The gain from flattening measures that",
        "> design, not a real media plan. This demonstrates the method.",
        "",
        "## 1. The four plans",
        "",
        f"All of them spend **{sup_o.budget:,.0f}** over the period, with each channel between "
        f"{LOWER_BOUND:g}× and {UPPER_BOUND:g}× of its historical total and each week between 0 and "
        f"{WEEKLY_CAP_MAX:g}× its observed weekly maximum of IMPRESSIONS — the saturation curve "
        f"is estimated on impressions, and with a variable CPM the same money buys different "
        f"quantities, so a cap in money would let weeks above the maximum seen go through. "
        f"Sales are evaluated over the {T} weeks plus **{CARRYOVER_TAIL} tail weeks with zero spend**, "
        f"so that the carryover generated at the end is collected; the tail uses the period's mean "
        f"base and price and is applied equally to the four scenarios and to the generator.",
        "",
    ]
    L += table(["Plan", "What it decides", "Variables", "Total spend"],
               [["S0 historical", "nothing: it is the observed plan", "0", f"{S0.sum():,.0f}"],
                ["S1 cross-channel", "one multiplier per channel (the 7a_ one)", f"{sup_o.K}",
                 f"{S1.sum():,.0f}"],
                ["S2 even pacing", "nothing: each channel's total / 156", "0", f"{S2.sum():,.0f}"],
                ["S3 temporal optimum", "spend by channel and week", f"{sup_o.K * T}",
                 f"{S3.sum():,.0f}"],
                ["S3 with fixed totals", "spend by week; each channel's total, the historical one",
                 f"{sup_o.K * T}", f"{S3_fixed.sum():,.0f}"]])
    L += [
        "",
        f"**S1 is S0 by construction.** S1 scales the weekly series of each channel by a "
        f"multiplier, like `7a_`, but here with the weekly cap in impressions, which `7a_` does not have. "
        f"The optimal multipliers come out "
        f"{', '.join(f'{c}={m:.4f}' for c, m in zip(channels, mult))}"
        + (": all 1. " if s1_collapses else ": **not all 1**, against what the identity predicts. ")
        + f"It is an identity of the problem, not a finding: the peak week of each channel is already at its "
        f"observed impressions maximum, so no multiplier can go above 1, and since the "
        f"budget enters as an equality and the four totals add up to the historical one, the only "
        f"feasible point is `m = (1,1,1,1)`. The `7a_` multipliers are different because `7a_` lets "
        f"two channels go above their observed weekly maximum (see its table). What measures the "
        f"cross-channel split with the cap in place is the test of section 2: S3 against S3 with fixed totals.",
        "",
        f"**S2 meets the weekly cap in all four channels**{': ' + '; '.join(viol_S2) if viol_S2 else ''}. "
        f"Its maximum weekly impressions go from "
        f"{', '.join(f'{c} {v:,.0f}' for c, v in zip(channels, sup_o.impressions(S2).max(axis=1)))} "
        f"against caps of "
        f"{', '.join(f'{c} {v:,.0f}' for c, v in zip(channels, sup_o.imp_cap))}.",
        "",
        "## 2. The gains, paired by draw",
        "",
        f"Each draw evaluates both plans and subtracts; the median and the {HDI} are of that difference.",
        "",
    ]
    L += table(["Comparison", "What it measures", "median", HDI, "draws with a gain"],
               [[n.replace("\n", " — "), q,
                 f"{paired(a, b)[0]:+.4f}%",
                 f"[{paired(a, b)[1]:+.4f}%, {paired(a, b)[2]:+.4f}%]",
                 f"{100 * paired(a, b)[3]:.1f}%"]  # cosmetic
                for (n, a, b), q in zip(pairs, ["cross-channel split", "flattening the flighting",
                                                "following the base and the adstock", "both things",
                                                "reallocating across channels (fair test)"])])
    med10, lo10, hi10, fr10 = paired(rev["S1_cross_channel"], rev["S0_historical"])
    medj, loj, hij, frj = paired(rev["S3_temporal_optimum"], rev_fixed)
    medf, lof, hif, _ = paired(rev_fixed, rev["S0_historical"])
    medl, lol, hil, _ = paired(rev["S3_temporal_optimum"], rev["S0_historical"])
    L += [
        "",
        f"**S1 − S0** gives {med10:+.4f}% with {HDI} [{lo10:+.4f}%, {hi10:+.4f}%] and "
        f"{100 * fr10:.1f}% of draws with a gain: it is the identity of section 1, not a measurement.",  # trivial: unit
        "",
        f"**The fair test of the cross-channel split.** The temporal optimum with each channel's total "
        f"fixed at its historical one gains {medf:+.4f}% over S0 ({HDI} [{lof:+.4f}%, {hif:+.4f}%]); with the "
        f"totals free between {LOWER_BOUND:g}× and {UPPER_BOUND:g}×, {medl:+.4f}% ({HDI} "
        f"[{lol:+.4f}%, {hil:+.4f}%]). The paired difference, what reallocating across channels adds "
        f"once the spend already moves over time, is **{medj:+.4f}%** with {HDI} "
        f"[{loj:+.4f}%, {hij:+.4f}%], with a gain in {100 * frj:.1f}% of the draws. The channel totals "  # trivial: unit
        f"of the free optimum end at "
        f"{', '.join(f'{c} {S3[k].sum() / sup_o.hist_spend_by_channel[k]:.3f}×' for k, c in enumerate(channels))} "
        f"of the historical ones. (Optimized in {sec_fixed:.0f} s, {res_fixed.nit} iterations.)",
        "",
        "## 3. The model against the generator",
        "",
        "The same plan, evaluated with the exact DGP equation (true ceilings, `lam` and adstock), "
        "with no model in between. It says whether the recommendation survives outside.",
        "",
    ]
    L += table(["Plan", "Revenue according to the model", "Δ% vs S0", "Revenue in the generator", "Δ% vs S0"],
               [[n.replace("_", " "), f"{means[n]:,.0f}",
                 f"{100 * (means[n] / means['S0_historical'] - 1):+.3f}",  # trivial: unit
                 f"{dgp[n]:,.0f}", f"{100 * (dgp[n] / dgp['S0_historical'] - 1):+.3f}"]  # trivial: unit
                for n in plans])
    L += [
        "",
        "## 4. The controls",
        "",
        f"**Feasibility.** The objective of S3 ({means['S3_temporal_optimum']:,.0f}) is greater than or "
        f"equal to that of S0, S1 and S2, which are the three feasible ones. `assert` in the script.",
        "",
        f"**Multiple starts.** The three reach the same objective with a relative spread "
        f"of **{starts_spread:.1e}**, against a tolerance of {STARTS_TOLERANCE:.0e}. There is no "
        f"evidence of local optima even though the problem is not concave.",
        "",
    ]
    L += table(["Start", "iterations", "converged", "objective", "seconds"],
               [[n, c["nit"], "yes" if c["ok"] else "NO", f"{c['obj']:,.1f}", f"{c['s']:.0f}"]
                for n, c in runs.items()])
    L += [
        "",
        "**KKT.** At the optimum, the weeks that are not stuck at a bound have to have the "
        "same marginal return per unit of money, within and across channels.",
        "",
    ]
    L += table(["Channel", "weeks at 0", "weeks at the cap", "free weeks",
                "mean marginal", "relative spread"], kkt_by_channel)
    L += [
        "",
        f"Over the **{int(free.sum())}** free cells of the four channels together, the relative spread "
        f"of the marginal is **{kkt_spread:.2e}**.",
        "",
        "**The control that can fail — flat base.** With a shapeless base, the optimum has to "
        f"be ~even pacing except for the first and last {L_MAX} weeks, where the adstock falls outside "
        f"the period. Criterion: coefficient of variation of the optimal weekly spend in the interior "
        f"< {CONTROL_CV_MAX:.0%}.",
        "",
    ]
    L += table(["Version", "CV of the base", "interior CV by channel", "maximum", "passes?"],
               [[("**" + e + "**" if e == CONTROL else e), f"{c['cv_base']:.4f}",
                 " ".join(f"{k}={v:.4f}" for k, v in c["cv"].items()),
                 f"{max(c['cv'].values()):.4f}",
                 "**yes**" if max(c["cv"].values()) < CONTROL_CV_MAX else "no"]
                for e, c in controls.items()])
    L += [
        "",
        f"**The control PASSES: maximum CV {cv_isolated:.4f}**, well below {CONTROL_CV_MAX:.0%}. "
        f"With constant base, price and CPM there is nothing left for the optimum to follow except "
        f"the adstock at the edges, and it does come out as even pacing in the interior. The script "
        f"checks it with an `assert`.",
        "",
        f"The first two rows are context, not the control, and they explain why it had to be corrected. "
        f"Fixing only trend and seasonality leaves the base with a CV of "
        f"{controls['context: trend and seasonality at their mean']['cv_base']:.4f} against "
        f"{np.std(base_t * sup_e.price[:T]) / np.mean(base_t * sup_e.price[:T]):.4f} without flattening "
        f"anything: **the weekly shape of the base comes from price, promo and ACV**, not from seasonality, "
        f"so in that version the premise of the control does not hold and its CV of {cv_literal:.4f} "
        f"says nothing about the optimizer. Leaving base and price constant but the CPM "
        f"variable, the CV stays at "
        f"{max(controls['context: constant base and price']['cv'].values()):.4f}: that is no longer a "
        f"defect but the optimum buying where impressions are cheaper.",
        "",
        "## 5. Does it still empty December?",
        "",
        f"With the {CARRYOVER_TAIL}-week tail, the carryover of the final spend is collected. Comparing the "
        f"mean spend of the last {L_MAX} weeks against that of the interior:",
        "",
    ]
    L += table(["Channel", "mean spend, last 12", "mean spend, interior", "ratio"],
               [[ch, f"{a:,.0f}", f"{b:,.0f}", f"{a / b:.3f}"]
                for ch, (a, b) in december.items()])
    L += [
        "",
        f"In total, the right edge receives **{100 * dec_total:.1f}%** of what an interior week "  # trivial: unit
        f"receives. " +
        ("**It no longer empties it.**" if dec_total > EDGE_VERDICT_THRESHOLDS[1] else
         "**It still empties it.**" if dec_total < EDGE_VERDICT_THRESHOLDS[0] else "**It eases it but does not even it out.**") +
        f" In the flat-base control, where the only thing that can move the edge is the adstock, the "
        f"ratio is {dec_control:.3f}: that is the part of the effect that is purely about the horizon.",
        "",
        "## 6. What the optimum looks like",
        "",
        f"**Correlation with the base.** The total weekly spend of S3 against the base `Y_t` without media: "
        f"**{corr:+.3f}**. By channel: "
        f"{', '.join(f'{c} {v:+.3f}' for c, v in corr_by_channel.items())}. It is positive, as it has to "
        f"be in a multiplicative model: `dY_t/dx_t = Y_t · β · f'(x_t)`, so the same impression "
        f"yields more where the base is already high.",
        "",
        f"**Are there flights?** Yes, and they are not pulsing.",
        "",
    ]
    L += table(["Channel", "weeks switched off in S3", "gaps between switched-on weeks"],
               [[ch, f"{a}", f"{b}"] for ch, (a, b) in flights.items()])
    L += [
        "",
        f"`tv` stays switched off in {flights['tv'][0]} weeks, and {flights['tv'][1]} of them are between "
        f"switched-on weeks. **It is not pulsing, and the functional form proves it:** `LogisticSaturation` is "
        f"`tanh`, concave from the origin, so splitting a spend between two weeks always yields at "
        f"least as much as concentrating it. This model **cannot** recommend pulsing because of saturation.",
        "",
        f"What it can do, and what happens, is a **corner solution**: KKT says a cell goes "
        f"to zero when the marginal of its first unit of money falls below the common marginal of the free ones "
        f"({free_marg:.4f}). That marginal is **not** `Y_t · β · f'(0) / cpm_t`: the week inherits stock "
        f"from the previous ones through the adstock, so `f'` is evaluated where that stock left it, higher up "
        f"the curve and with less slope. With the stock, the highest marginal among the "
        f"{int(at_zero.sum())} cells at zero is **{float(marg_at_zero.max()):.4f}**, below the "
        f"common one. Ignoring the stock (`f'(0)`), {n_stock_only} of those {int(at_zero.sum())} cells "
        f"would be **above** it, and the optimum would have put spend on them: the explanation by the base "
        f"alone is not enough. The script checks both things with an `assert`.",
        "",
    ]
    if off:
        L += [
            f"**The switched-off stretch of `tv`.** Of its {off['n']} weeks at zero, {off['q4']} fall "
            f"in the {off['q4_total']} year-end weeks, where the CPM of `tv` is at "
            f"{off['cpm_q4']:.2f}× its mean (against {off['cpm_rest']:.2f}× the rest of the year): "
            f"the same money buys fewer impressions. The longest consecutive stretch goes from week "
            f"{off['run'][0]} to week {off['run'][1]}. {off['promo']} weeks at zero have "
            f"promo, with a high base: there the base alone would ask for spend, and what switches `tv` off is the CPM. "
            f"{off['post_promo']} follow a week with promo, whose spend leaves stock. Those weeks "
            f"have on average a base of {100 * base_rel_zero:.1f}% of the period mean: the "  # trivial: unit
            f"optimum does not follow the base alone.",
            "",
        ]
    L += [
        f"Full run: {time.perf_counter() - t_total:.0f} s.",
    ]
    OUTPUT.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"  {p1.relative_to(ROOT).as_posix()}")
    print(f"  {p2.relative_to(ROOT).as_posix()}")
    print(f"  {OUTPUT.relative_to(ROOT).as_posix()}")
    print("  S1 multipliers: " + " ".join(f"{c}={m:.4f}" for c, m in zip(channels, mult)))
    for n, a, b in pairs:
        med, lo, hi, fr = paired(a, b)
        # the label carries a typographic minus, which the figure needs and the Windows console
        # (cp1252) cannot write: for stdout it is swapped for a plain hyphen.
        print(("  %-26s median %+.4f%%  " + HDI + " [%+.4f%%, %+.4f%%]  positive in %.1f%%")
              % (n.replace("\n", " ").replace("−", "-"), med, lo, hi, 100 * fr))  # trivial: unit
    print(f"  starts: spread {starts_spread:.2e} (tol {STARTS_TOLERANCE:.0e}); "
          f"KKT spread {kkt_spread:.2e} over {int(free.sum())} free cells")
    print(f"  flat-base control (constant base+price+CPM): max CV {cv_isolated:.4f} -- PASSES "
          f"(context: literal {cv_literal:.4f})")
    print(f"  S1 collapses to S0: {s1_collapses}  (all multipliers 1.0)")
    print(f"  December: the edge receives {100 * dec_total:.1f}% of the interior "  # trivial: unit
          f"(flat-base control: {100 * dec_control:.1f}%)")  # trivial: unit
    print(f"  correlation optimal spend vs base: {corr:+.3f}")
    print("  generator: " + " ".join(f"{n.split('_')[0]}={100 * (dgp[n] / dgp['S0_historical'] - 1):+.2f}%"  # trivial: unit
                                     for n in plans))
    print(f"  flights in S3: {flights}")
    print(f"  fair test: S3 fixed vs S0 {medf:+.4f}%  S3 free vs S0 {medl:+.4f}%  "
          f"free - fixed {medj:+.4f}% {HDI} [{loj:+.4f}%, {hij:+.4f}%]")
    print(f"  switched off: cells at zero {int(at_zero.sum())}; without stock above the common marginal "
          f"{n_stock_only}; tv {off}")
    print(f"  total {time.perf_counter() - t_total:.0f} s")


if __name__ == "__main__":
    main()
