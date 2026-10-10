"""THE DECOMPOSITION BENCH FIGURE: the same model, four split rules.

# ! NOTE -- "split range", never "error": the difference between methods is not imprecision.
# ! NOTE -- the reference changes the split of drivers that do not even use it.

Produces: model/6_cross_term/plots/6c1_split_range__<dataset_label>.png
"""

import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config.diagnostics import SPLIT_TABLE_DECIMALS  # noqa: E402
from config.idata_paths import require_idata  # noqa: E402
from synthetic_data.true_params_reader import get_dataset_label, warn_if_idata_stale  # noqa: E402
import viz.decomposition as D  # noqa: E402
from viz.io import figure_path, save_figure  # noqa: E402
from viz.business import canonical_slice  # noqa: E402
from viz.style import (BASELINE, GRID, INK, INK_MUTED, INK_SECONDARY, POSTERIOR_COLOR,
                       PRIOR_COLOR, TRUTH_COLOR, apply_style)  # noqa: E402

apply_style()

DATASET_LABEL = get_dataset_label()
if DATASET_LABEL != "log__no_noise":
    sys.exit(f"6c_: this bench is ONLY for the no-noise regime. The live dataset is "
             f"{DATASET_LABEL!r}; regenerate the dataset with step 1 of the README.")
IDATA_PATH = require_idata("linear_trend", DATASET_LABEL)
warn_if_idata_stale(IDATA_PATH)
idata = az.from_netcdf(IDATA_PATH)
dfm = canonical_slice()
TABLE = ROOT / "results" / f"6b_cross_term_split__{DATASET_LABEL}.md"
LABEL = {"log_price_c": "price", "promo_flag": "promo", "log_acv_c": "ACV", "t_norm_c": "trend",
         "seasonality": "seasonality", "tv": "tv", "meta": "meta", "tiktok": "tiktok", "ooh": "ooh"}


def split(reference: str) -> dict:
    """The same four splits as 6b_, through the same calls."""
    fv = D.ValueFunction.from_components(D.model_components(idata, reference, dfm))
    V = fv.total(fv.N)
    pct = lambda x: float(np.mean(100 * x / V))  # noqa: E731  # trivial: unit
    A, B, C = D.removal(fv), D.sequential(fv, D.BASE_ORDER_K9), D.shapley(fv)
    Dm, R = D.lmdi(fv), D.sequential_range(fv, normalize=pct)
    return {k: {"A": pct(A[k]), "B": pct(B[k]), "C": pct(C[k]), "D": pct(Dm[k]),
                "Bmin": R[k][0], "Bmax": R[k][1]} for k in fv.names}


def read_table(md: str) -> dict:
    """The 6b_ table, parsed. It is the cross-check: the figure and the .md have to say the same."""
    out, section = {}, None
    for line in md.splitlines():
        if line.startswith("## 1."):
            section = "model"
        elif line.startswith("## 2."):
            section = "dgp"
        elif line.startswith("## 3."):
            section = None
        if section is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 7 or cells[0] not in LABEL.values():  # trivial: columns of the 6b_ table
            continue
        try:
            v = [float(c) for c in cells[1:]]
        except ValueError:
            continue
        out[(section, cells[0])] = dict(zip(("A", "B", "Bmin", "Bmax", "C", "D"), v))
    return out


METHODS = ("A", "B", "C", "D")   # the four that are drawn: the grey band goes from one to another


def panel(ax, data, title, order):
    y = np.arange(len(order))[::-1]
    for i, k in enumerate(order):
        r = data[k]
        # the band is the range of the DRAWN methods: each edge is a marker
        lo, hi = min(r[m] for m in METHODS), max(r[m] for m in METHODS)
        assert any(r[m] == lo for m in METHODS) and any(r[m] == hi for m in METHODS), (k, lo, hi)
        ax.plot([lo, hi], [y[i], y[i]], color=BASELINE, linewidth=9,  # cosmetic
                solid_capstyle="butt", zorder=2)  # cosmetic
    markers = (("B", "o", PRIOR_COLOR, "(B) sequential, base order"),
               ("C", "s", POSTERIOR_COLOR, "(C) Shapley"),
               ("D", "^", "#1baf7a", "(D) LMDI"),
               ("A", "x", TRUTH_COLOR, "(A) removal — does NOT add up to the joint effect"))
    for m, marker, color, label in markers:
        ax.scatter([data[k][m] for k in order], y, marker=marker, s=46 if m != "A" else 54,  # cosmetic
                   color=color, zorder=4, label=label, linewidths=1.5 if m == "A" else 0.6,  # cosmetic
                   edgecolors=INK if m != "A" else color)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8, zorder=1)  # cosmetic
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[k] for k in order])
    ax.set_xlabel("% of the period's sales")
    ax.set_title(title, fontsize=11, loc="left")  # cosmetic
    ax.grid(axis="x", color=GRID, zorder=0)


def main():
    mod, dgp = split("model"), split("dgp")
    # CROSS-CHECK: the figure cannot say anything other than the table. If 6b_ did not run after the
    # last change, this check finds it out here and not when someone compares the two artifacts.
    if not TABLE.exists():
        sys.exit(f"6c_: {TABLE.relative_to(ROOT).as_posix()} is missing. Run "
                 f"6b_cross_term_split.py first: this figure is cross-checked against that table.")
    ref_md = read_table(TABLE.read_text(encoding="utf-8"))
    worst, where = 0.0, ""
    for side, data in (("model", mod), ("dgp", dgp)):
        for k, r in data.items():
            expected = ref_md.get((side, LABEL[k]))
            if expected is None:
                sys.exit(f"6c_: the 6b_ table has no row ({side}, {LABEL[k]}).")
            for m in ("A", "B", "Bmin", "Bmax", "C", "D"):
                # The value ROUNDED to the 3 decimals 6b_ writes in the table is compared: that way the
                # cross-check is exact and there is no need to pick a tolerance that hides a real difference.
                d = abs(round(r[m], SPLIT_TABLE_DECIMALS) - expected[m])
                if d > worst:
                    worst, where = d, f"{side}/{LABEL[k]}/{m}"
    if worst > 1e-9:  # trivial: floating-point tolerance
        sys.exit(f"6c_: the figure and {TABLE.name} do NOT say the same (worst difference {worst:.2e} at "
                 f"{where}). Regenerate 6b_ before publishing the figure.")

    order = list(D.BASE_ORDER_K9)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), sharey=True)  # cosmetic
    panel(axes[0], mod, "1. MODEL reference — price and ACV at their geometric mean", order)
    panel(axes[1], dgp, "2. DGP reference — price and ACV at 100", order)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")  # cosmetic
    fig.suptitle("Split range: the SAME model, four rules — and two references",
                 fontsize=13, x=0.007, ha="left", y=0.985)  # cosmetic
    fig.text(0.007, 0.918,  # cosmetic
             "The grey band goes from the smallest to the largest of the four drawn methods. The difference "
             "between them is SPLIT RANGE, not error:\nthe split of the cross term is not a "
             "parameter of the generator and no experiment tells it apart.",
             fontsize=9, color=INK_SECONDARY, va="top")  # cosmetic
    d_tv = dgp["tv"]["C"] - mod["tv"]["C"]   # how much tv moves in Shapley when the reference changes
    fig.text(0.007, 0.043,  # cosmetic
             f"Changing the reference moves drivers that do not use it (tv {d_tv:+.2f} pp in Shapley without "
             f"anything about tv changing): what changed is the base they are measured against.",
             fontsize=8.5, color=INK_MUTED)  # cosmetic
    fig.tight_layout(rect=(0, 0.072, 1, 0.895))  # cosmetic
    _path = figure_path(ROOT, "6_cross_term", "6c1_split_range", DATASET_LABEL)
    save_figure(fig, _path, script_file=__file__)
    print(f"  {_path.relative_to(ROOT).as_posix()}")
    print(f"  cross-check against {TABLE.name}: worst difference {worst:.2e} ({where or 'no differences'})")
    widths = {LABEL[k]: max(mod[k][m] for m in METHODS) - min(mod[k][m] for m in METHODS) for k in order}
    max_width = max(widths, key=widths.get)
    print(f"  model reference: the widest band (across the 4 methods) is {max_width} "
          f"({widths[max_width]:.3f} pp)")
    # the range of (B) over the 9! orders stays in the 6b_ table (and in the cross-check above)
    widths_b = {LABEL[k]: mod[k]["Bmax"] - mod[k]["Bmin"] for k in order}
    width_b = max(widths_b, key=widths_b.get)
    print(f"  (B) over the 9! orders: the widest range is {width_b} ({widths_b[width_b]:.3f} pp)")


if __name__ == "__main__":
    main()
