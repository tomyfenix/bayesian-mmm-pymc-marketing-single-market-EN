"""BUDGET OPTIMIZATION: reallocate the historical spend across the four channels.

# ! NOTE -- the gain is tiny and that is the result, not a defect.
# ! NOTE -- the priors were chosen by whoever wrote the generator: optimistic result by construction.
# ! NOTE -- this optimum does not respect the weekly impressions cap; 7b_ does, and there it gains nothing.

Produces:
    model/7_optimization/plots/7a1_cross_channel_optimization__<dataset_label>.png,
    results/7a_cross_channel_optimization__<dataset_label>.md
"""

import sys
from datetime import datetime
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.budget import KKT_THRESHOLD  # noqa: E402
from config.idata_paths import require_idata  # noqa: E402
from config.mmm_dgp_config import HDI_PROB  # noqa: E402
from config.budget import (SATURATED_CHANNEL_CONTROL, CONTROL_FACTOR, LOWER_BOUND, UPPER_BOUND,
                                N_DRAWS, OBJECTIVE, BUDGET, OPTIMIZATION_DRAWS_SEED)  # noqa: E402
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale  # noqa: E402
import viz.budget as PR  # noqa: E402
from viz.io import figure_path, provenance_fields, save_figure  # noqa: E402
from viz.labels import HDI  # noqa: E402
from viz.business import canonical_slice  # noqa: E402
from viz.style import BASELINE, GRID, INK, INK_MUTED, INK_SECONDARY, apply_style, channel_color  # noqa: E402

apply_style()

DATASET_LABEL = get_dataset_label()
if DATASET_LABEL != "log__no_noise":
    sys.exit(f"7a_: this scenario is ONLY for the no-noise regime. The live dataset is "
             f"{DATASET_LABEL!r}; regenerate the dataset with step 1 of the README.")
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
dfm = canonical_slice()
OUTPUT = ROOT / "results" / f"7a_cross_channel_optimization__{DATASET_LABEL}.md"


def table(headers, rows):
    widths = [max(len(str(h)), *(len(str(f[i])) for f in rows)) for i, h in enumerate(headers)]
    out = ["| " + " | ".join(str(h).ljust(a) for h, a in zip(headers, widths)) + " |",
           "|" + "|".join("-" * (a + 2) for a in widths) + "|"]  # cosmetic
    out += ["| " + " | ".join(str(c).ljust(a) for c, a in zip(f, widths)) + " |" for f in rows]
    return out


def main():
    sup = PR.Surface(idata, dfm, n_draws=N_DRAWS, seed=OPTIMIZATION_DRAWS_SEED)
    channels = sup.channels
    assert BUDGET == "historical_spend", BUDGET
    B = sum(sup.historical_spend.values())
    one = np.ones(len(channels))

    m_opt, res = PR.optimize(sup, B, LOWER_BOUND, UPPER_BOUND)
    if not res.success:
        sys.exit(f"7a_: the optimizer does not converge ({res.message}). No optimum is published.")
    m_roi = PR.average_roi_split(sup, B, LOWER_BOUND, UPPER_BOUND)

    rev_h, rev_o, rev_r = sup.revenue(one), sup.revenue(m_opt), sup.revenue(m_roi)
    mg_h, mg_o = sup.marginal_return(one), sup.marginal_return(m_opt)
    mg_r = sup.marginal_return(m_roi)
    roas_h = sup.average_roas(one)
    g_h, g_o, g_r = sup.spend(one), sup.spend(m_opt), sup.spend(m_roi)

    # --- KKT: the channels that are not at a bound have the SAME marginal at the optimum
    free = [ch for i, ch in enumerate(channels)
            if LOWER_BOUND + 1e-6 < m_opt[i] < UPPER_BOUND - 1e-6]  # trivial: floating-point tolerance
    vals = [mg_o[ch] for ch in free]
    kkt_spread = (max(vals) - min(vals)) / np.mean(vals) if len(vals) > 1 else 0.0
    assert kkt_spread < KKT_THRESHOLD, (
        f"7a_: at the optimum the {len(free)} interior channels do NOT equalize their marginal return "
        f"(relative spread {kkt_spread:.2e}). That violates the optimality condition: the reported "
        f"optimum is not an optimum.")

    # --- the control that can fail: an artificially saturated channel
    sup_c = PR.saturated_scenario(sup, idata, dfm, SATURATED_CHANNEL_CONTROL, CONTROL_FACTOR,
                                  n_draws=N_DRAWS, seed=OPTIMIZATION_DRAWS_SEED)
    m_c, res_c = PR.optimize(sup_c, B, LOWER_BOUND, UPPER_BOUND)
    if not res_c.success:
        sys.exit(f"7a_: the control scenario does not converge ({res_c.message}).")
    i_sat = channels.index(SATURATED_CHANNEL_CONTROL)
    sat_spend_ref = sup_c.ref_spend[SATURATED_CHANNEL_CONTROL]
    sat_spend_opt = sup_c.spend(m_c)[SATURATED_CHANNEL_CONTROL]
    assert sat_spend_opt < sat_spend_ref - 1e-6, (  # trivial: floating-point tolerance
        f"7a_ CONTROL FAILED: with the spend of {SATURATED_CHANNEL_CONTROL} doubled, the optimum does NOT "
        f"take budget away from it ({sat_spend_opt:.1f} >= {sat_spend_ref:.1f}). An optimizer that does not "
        f"cut a channel placed on the flat zone of its saturation is not looking at the marginal return.")
    m_c_roi = PR.average_roi_split(sup_c, B, LOWER_BOUND, UPPER_BOUND)

    lo_h, hi_h = az.hdi(rev_h, hdi_prob=HDI_PROB)
    lo_o, hi_o = az.hdi(rev_o, hdi_prob=HDI_PROB)
    lo_r, hi_r = az.hdi(rev_r, hdi_prob=HDI_PROB)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4),  # cosmetic
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})  # cosmetic
    ax = axes[0]
    x = np.arange(len(channels))
    ax.bar(x - 0.21, [g_h[c] / 1e6 for c in channels], width=0.42, color=BASELINE,  # cosmetic
           edgecolor=INK, linewidth=0.6, label="historical", zorder=3)  # cosmetic
    for i, c in enumerate(channels):
        ax.bar(x[i] + 0.21, g_o[c] / 1e6, width=0.42, color=channel_color(c),  # cosmetic
               edgecolor=INK, linewidth=0.6, zorder=3,  # cosmetic
               label="optimum (channel colour)" if i == 0 else None)
    for i, c in enumerate(channels):
        d = 100 * (g_o[c] / g_h[c] - 1)  # trivial: unit
        ax.annotate(f"{d:+.1f}%", (x[i] + 0.21, g_o[c] / 1e6), textcoords="offset points",  # cosmetic
                    xytext=(0, 4), ha="center", fontsize=9, color=INK_SECONDARY)  # cosmetic
        ax.annotate(f"mg {mg_h[c]:.2f}\n→ {mg_o[c]:.2f}", (x[i] - 0.21, g_h[c] / 1e6),  # cosmetic
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8,  # cosmetic
                    color=INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("spend over the period (millions)")
    ax.set_title(f"Historical spend against optimum\nbudget fixed at {B / 1e6:.2f} M, "  # trivial: unit
                 f"bounds {LOWER_BOUND:g}x–{UPPER_BOUND:g}x", fontsize=11, loc="left")  # cosmetic
    ax.legend(frameon=False, fontsize=9, loc="upper right")  # cosmetic
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_ylim(0, max(g_h.values()) / 1e6 * 1.32)  # cosmetic

    ax = axes[1]
    labels = ["historical", "optimum", "average ROI\n(contrast)"]
    for i, (rev, col) in enumerate(((rev_h, BASELINE), (rev_o, channel_color("tv")),
                                    (rev_r, INK_MUTED))):
        lo, hi = az.hdi(rev, hdi_prob=HDI_PROB)
        ax.plot([i, i], [lo / 1e6, hi / 1e6], color=col, linewidth=7, solid_capstyle="butt",  # cosmetic
                alpha=0.85, zorder=3)  # cosmetic
        ax.plot([i], [rev.mean() / 1e6], marker="D", ms=6, color=INK, zorder=4)  # cosmetic
        ax.annotate(f"{rev.mean() / 1e6:,.1f}", (i, rev.mean() / 1e6), textcoords="offset points",  # trivial: unit
                    xytext=(11, -3), fontsize=9, color=INK_SECONDARY)  # cosmetic
    ax.set_xticks(range(3))  # trivial: the three scenarios
    ax.set_xticklabels(labels, fontsize=9)  # cosmetic
    ax.set_xlim(-0.55, 2.75)  # cosmetic
    ax.set_ylabel("expected revenue over the period (millions)")
    ax.set_title(f"Expected sales with {HDI}\nthe HDI measures "
                 f"{(hi_h - lo_h) / (rev_o.mean() - rev_h.mean()):.1f}x what separates the optimum from the "
                 f"historical plan", fontsize=11, loc="left")  # cosmetic
    ax.grid(axis="y", color=GRID, zorder=0)
    # At this scale the historical/optimum pair is ONE line: the inset zooms in so that the two HDIs can
    # be seen to overlap. It is the point of the figure, not a presentation detail.
    ins = ax.inset_axes((0.37, 0.30, 0.59, 0.40))  # cosmetic
    for i, (rev, col) in enumerate(((rev_h, BASELINE), (rev_o, channel_color("tv")))):
        lo, hi = az.hdi(rev, hdi_prob=HDI_PROB)
        ins.plot([i, i], [lo / 1e6, hi / 1e6], color=col, linewidth=13, solid_capstyle="butt",  # cosmetic
                 alpha=0.85, zorder=3)  # cosmetic
        ins.plot([i], [rev.mean() / 1e6], marker="D", ms=5, color=INK, zorder=4)  # cosmetic
    ins.set_xlim(-0.6, 1.6)  # cosmetic
    ins.set_xticks([0, 1])
    ins.set_xticklabels(["hist.", "opt."], fontsize=8)  # cosmetic
    ins.tick_params(labelsize=8)  # cosmetic
    ins.set_title("zoomed in", fontsize=8.5, color=INK_MUTED, loc="left")  # cosmetic
    ins.grid(axis="y", color=GRID, zorder=0)

    ax = axes[2]
    diff = 100 * (rev_o - rev_h) / rev_h  # trivial: unit
    frac = float((diff > 0).mean())
    ax.hist(diff, bins=30, color=channel_color("meta"), edgecolor=INK, linewidth=0.5, zorder=3)  # cosmetic
    ax.axvline(0, color=INK, linewidth=1.0, zorder=4)  # cosmetic
    ax.axvline(diff.mean(), color=INK, linestyle="--", linewidth=1.0, zorder=4)  # cosmetic
    ax.set_xlabel("optimum − historical, per draw (% of revenue)")
    ax.set_ylabel("draws")
    ax.set_title(f"PAIRED comparison per draw\nmean {diff.mean():+.4f}%, positive in "
                 f"{100 * frac:.0f}% of the draws", fontsize=11, loc="left")  # cosmetic
    ax.grid(axis="y", color=GRID, zorder=0)
    # Above the provenance stamp, which viz.io.stamp_figure writes at the bottom.
    fig.text(0.005, 0.050,  # cosmetic
             "The priors were chosen by whoever wrote the generator: this demonstrates the method, not "
             "the precision of the model.",
             fontsize=8.5, color=INK_MUTED)  # cosmetic
    fig.tight_layout(rect=(0, 0.078, 1, 1))  # cosmetic
    _path = figure_path(ROOT, "7_optimization", "7a1_cross_channel_optimization", DATASET_LABEL)
    save_figure(fig, _path, script_file=__file__)

    # ---------------------------------------------------------------- tables
    prov = provenance_fields("linear_trend")
    L = [
        "# Budget optimization — reallocating the historical spend across channels",
        "",
        "GENERATED by `model/7_optimization/7a_cross_channel_optimization.py` — "
        "**do not edit by hand**.",
        "",
        f"> **Provenance:** dataset=`{DATASET_LABEL}` | idata=`{prov['idata_name']}`@`{prov['idata_hash']}` "
        f"| csv=`{prov['csv_name']}`@`{prov['csv_hash']}` | commit=`{prov['commit']}` "
        f"| draws={len(rev_h)} of {idata.posterior.sizes['chain'] * idata.posterior.sizes['draw']} "
        f"| generated {datetime.now().isoformat(timespec='seconds')}",
        "",
        "> **READ BEFORE CITING.** The priors were chosen by whoever wrote the synthetic generator, so the",
        "> posterior response curves are **optimistic by construction**. What this scenario demonstrates",
        "> is the **method** —that the optimum comes from the MARGINAL return of the posterior and not",
        "> from the average ROI— and **not** how much should be spent on each channel.",
        "",
        "## 1. The scenario",
        "",
    ]
    L += table(["Parameter", "Value", "Where it comes from"], [
        ["total budget", f"{B:,.0f}", f"`config/budget.py:BUDGET = {BUDGET!r}` "
                                     f"(sum of the spend of the 4 channels over the period)"],
        ["bounds per channel", f"{LOWER_BOUND:g}x to {UPPER_BOUND:g}x", "`LOWER_BOUND` / `UPPER_BOUND`, "
                                                                         "on the historical spend of each channel"],
        ["objective", OBJECTIVE, "`OBJECTIVE` — mean over the draws of the period's sales in LEVELS"],
        ["draws", f"{len(rev_h)}", "`N_DRAWS`, seed `OPTIMIZATION_DRAWS_SEED`"],
        ["what does NOT move", "the weekly flighting", "the scenario reallocates across CHANNELS, not across weeks"],
    ])
    L += [
        "",
        "## 2. Historical spend against optimum",
        "",
        "`mg` is the marginal return: expected revenue per additional unit of spend, at that point.",
        "",
    ]
    L += table(["Channel", "historical spend", "optimal spend", "Δ%", "historical mg", "optimal mg",
                "hist. average ROAS"],
               [[c, f"{g_h[c]:,.0f}", f"{g_o[c]:,.0f}", f"{100 * (g_o[c] / g_h[c] - 1):+.2f}",  # trivial: unit
                 f"{mg_h[c]:.4f}", f"{mg_o[c]:.4f}", f"{roas_h[c]:.2f}"] for c in channels] +
               [["**total**", f"{B:,.0f}", f"{sup.total_spend(m_opt):,.0f}", "+0.00", "—", "—", "—"]])
    L += [
        "",
        f"**The optimality condition holds.** The {len(free)} channels that did not end at a bound "
        f"({', '.join(free)}) have the SAME marginal return at the optimum, "
        f"{np.mean(vals):.4f}, with a relative spread of {kkt_spread:.1e}. In the historical plan they go from "
        f"{min(mg_h.values()):.4f} to {max(mg_h.values()):.4f}: that is why the historical plan is not optimal, and "
        f"why the optimum takes spend away from the low-marginal channels and gives it to the high-marginal ones.",
        "",
        "## 3. What is gained — and why it is less than it sounds",
        "",
    ]
    L += table(["Scenario", "expected revenue", HDI, "Δ% against historical"], [
        ["historical", f"{rev_h.mean():,.0f}", f"{lo_h:,.0f} to {hi_h:,.0f}", "—"],
        ["optimum (marginal)", f"{rev_o.mean():,.0f}", f"{lo_o:,.0f} to {hi_o:,.0f}",
         f"{100 * (rev_o.mean() / rev_h.mean() - 1):+.4f}"],  # trivial: unit
        ["split by average ROI", f"{rev_r.mean():,.0f}", f"{lo_r:,.0f} to {hi_r:,.0f}",
         f"{100 * (rev_r.mean() / rev_h.mean() - 1):+.4f}"],  # trivial: unit
    ])
    L += [
        "",
        f"The {HDI} of revenue measures {(hi_h - lo_h) / 1e6:,.2f} million, that is "  # trivial: unit
        f"{100 * (hi_h - lo_h) / rev_h.mean():.4f}% of revenue. The gain of the optimum is "  # trivial: unit
        f"{100 * (rev_o.mean() / rev_h.mean() - 1):+.4f}%, that is "  # trivial: unit
        f"**{(hi_h - lo_h) / (rev_o.mean() - rev_h.mean()):.1f} times smaller than the width of the "
        f"uncertainty**: the HDIs of the historical plan and of the optimum overlap almost completely and "
        f"comparing them says nothing. What can be claimed is the PAIRED comparison, draw by draw "
        f"—the same draw in both scenarios, so the parameter uncertainty cancels out—: the "
        f"optimum beats the historical plan in **{100 * frac:.0f}%** of the {len(diff)} draws, with a mean "  # trivial: unit
        f"of {diff.mean():+.4f}% and a range from {diff.min():+.4f}% to {diff.max():+.4f}%.",
        "",
        "The split by **average ROI** is here only as a contrast: it is "
        f"{100 * (rev_r.mean() / rev_h.mean() - 1):+.4f}% against the historical plan, that is **worse than doing "  # trivial: unit
        f"nothing**. It splits in proportion to the average ROAS, which ignores that the return falls with the "
        f"spend level; in its allocation the marginal returns go from {min(mg_r.values()):.4f} to "
        f"{max(mg_r.values()):.4f}, far from equal.",
        "",
        "## 4. The control that can fail — an artificially saturated channel",
        "",
        f"The reference spend of **{SATURATED_CHANNEL_CONTROL}** is multiplied by **{CONTROL_FACTOR:g}**, "
        f"leaving it on the flat zone of its saturation curve, and it is re-optimized with the SAME "
        f"budget and the SAME relative bounds. Nothing else changes.",
        "",
    ]
    L += table(["Channel", "reference spend", "optimal spend", "multiple of its reference"],
               [[c, f"{sup_c.ref_spend[c]:,.0f}", f"{sup_c.spend(m_c)[c]:,.0f}", f"{m_c[i]:.4f}"]
                for i, c in enumerate(channels)])
    L += [
        "",
        f"**It passes.** The optimum takes "
        f"{sat_spend_ref - sat_spend_opt:,.0f} away from {SATURATED_CHANNEL_CONTROL} —it takes it to "
        f"{m_c[i_sat]:.4f}x of its reference, that is to its lower bound— and splits that among the other "
        f"three. The script checks it with an `assert`: if the saturated channel did not go down, the run "
        f"fails and publishes nothing.",
        "",
        f"**And it is necessary but NOT sufficient.** The split by average ROI ALSO cuts its spend in this "
        f"scenario (it takes it to {m_c_roi[i_sat]:.4f}x), because the average of a channel whose spend was "
        f"doubled also falls. What separates the marginal from the average is not this control but the "
        f"optimality condition of section 2: at the optimum the marginals of the free channels are equal, "
        f"and in the split by average ROI they are not.",
    ]
    OUTPUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  {_path.relative_to(ROOT).as_posix()}")
    print(f"  {OUTPUT.relative_to(ROOT).as_posix()}")
    print(f"  budget {B:,.0f} | optimum " +
          " ".join(f"{c}={m_opt[i]:.4f}" for i, c in enumerate(channels)))
    print(f"  historical revenue {rev_h.mean():,.0f} -> optimum {rev_o.mean():,.0f} "
          f"({100 * (rev_o.mean() / rev_h.mean() - 1):+.4f}%), paired positive in {100 * frac:.0f}% of the draws")  # trivial: unit
    print(f"  KKT: marginal of the free channels {np.mean(vals):.4f} (spread {kkt_spread:.1e}); "
          f"historical {min(mg_h.values()):.4f}-{max(mg_h.values()):.4f}")
    print(f"  average ROI (contrast): {100 * (rev_r.mean() / rev_h.mean() - 1):+.4f}%")  # trivial: unit
    print(f"  CONTROL {SATURATED_CHANNEL_CONTROL} x{CONTROL_FACTOR:g}: optimum {m_c[i_sat]:.4f}x of its "
          f"reference ({sat_spend_opt:,.0f} against {sat_spend_ref:,.0f}) -- PASSES")


if __name__ == "__main__":
    main()
