"""TEST OF THE FOUR SPLITS. Runs standalone: `python tests/test_decomposition.py`.

# ! NOTE -- the sabotages have to FAIL efficiency; if one day they pass, the test is worth nothing.
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import viz.decomposition as D  # noqa: E402
from viz.business import log_components, true_log_components, canonical_slice, centering_reference  # noqa: E402

TOL = 1e-12
_n = {"ok": 0}


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    _n["ok"] += 1
    print(f"  OK  {msg}")


def fv_toy(names, weekly):
    return D.ValueFunction(names, weekly)


def multiplicative(b, f):
    """v_t(S) = b_t * PROD_{k in S} f_k,t, with b (T,) and f (K, T). M=1."""
    b, f = np.asarray(b, float), np.asarray(f, float)
    return lambda m: (b * np.prod([f[i] if m >> i & 1 else np.ones_like(b) for i in range(len(f))], axis=0))[None, :]


def methods(fv, order):
    return {"A": D.removal(fv), "B": D.sequential(fv, order), "C": D.shapley(fv), "D": D.lmdi(fv)}


print("2.1 -- the four read the same v, and v matches recomputing from the components")
dfm = canonical_slice()
import arviz as az  # noqa: E402
# The path is resolved by config/idata_paths.py, not by a literal: the name of the .nc carries the
# dataset_label and also has legacy name resolution. Written by hand, the test only ran in a repo
# where the file with the old name existed.
from config.idata_paths import require_idata  # noqa: E402
from synthetic_data.true_params_reader import get_dataset_label  # noqa: E402
idata = az.from_netcdf(require_idata("linear_trend", get_dataset_label()))
draws = np.random.default_rng(0).choice(4000, 40, replace=False)

for ref in D.VALID_REFERENCES:
    lev = D.reference_levels(ref, dfm)
    # --- model: recompute DIRECTLY from log_components, without going through Components
    comp_m = D.model_components(idata, ref, dfm, draws=draws)
    fv = D.ValueFunction.from_components(comp_m)
    raw = log_components(idata)
    st = lambda da: da.stack(m=("chain", "draw")).transpose("m", "date").values[draws]  # noqa: E731
    st_raw = {k: st(raw[k]) for k in D.DRIVERS_K9}
    logS = st(idata.posterior["y_original_scale"])
    ts = float(idata.constant_data["target_scale"].values)
    g = idata.posterior["gamma_control"].stack(m=("chain", "draw"))
    corr = {}
    for k, col in (("log_price_c", "price"), ("log_acv_c", "acv")):
        gam = g.sel(control=k).values[draws] * ts
        corr[k] = (gam * (math.log(centering_reference(dfm, col)) - math.log(lev[col])))[:, None]
    worst = 0.0
    for m in range(fv.N + 1):
        off = [k for i, k in enumerate(fv.names) if not m >> i & 1]
        direct = np.exp(logS - sum(st_raw[k] + corr.get(k, 0.0) for k in off)) if off else np.exp(logS)
        worst = max(worst, float(np.max(np.abs(fv.weekly(m) - direct) / direct)))
    ok(worst < TOL, f"model, reference={ref}: v(S) = direct recomputation in the {fv.N + 1} worlds (max rel {worst:.1e})")
    ok(np.allclose(fv.weekly(fv.N), np.exp(logS), rtol=TOL, atol=0),
       f"model, reference={ref}: v(N) = exp(y_original_scale), it does not depend on the reference")

    # --- DGP: recompute from true_log_components
    comp_d = D.dgp_components(ref, dfm)
    fvd = D.ValueFunction.from_components(comp_d)
    V = true_log_components(dfm)
    source = {**V, "trend + seasonality": V["seasonality"]}
    import json  # noqa: E402
    tp = json.loads((ROOT / "synthetic_data/true_parameters.json").read_text(encoding="utf-8"))["trade_params"]["Retail"]
    elast = {"log_price_c": tp["price_elasticity"] * (math.log(100) - math.log(lev["price"])),
          "log_acv_c": tp["distribution_elasticity"] * (math.log(100) - math.log(lev["acv"]))}
    logSd = sum(np.asarray(v, float) for k, v in V.items() if k != "t_norm_c")
    worst = 0.0
    for m in range(fvd.N + 1):
        off = [k for i, k in enumerate(fvd.names) if not m >> i & 1]
        direct = np.exp(logSd - sum(np.asarray(source[k], float) + elast.get(k, 0.0) for k in off))
        worst = max(worst, float(np.max(np.abs(fvd.weekly(m)[0] - direct) / direct)))
    ok(worst < TOL, f"DGP, reference={ref}: v(S) = direct recomputation in the {fvd.N + 1} worlds (max rel {worst:.1e})")

res = np.abs(np.log(D.v_dgp([], "dgp", dfm)) - np.log(dfm["sales_deterministic"].values)).max()
ok(res < 2e-4, f"v_dgp(nothing switched off) rebuilds sales_deterministic to {res:.2e} in log (CSV rounding)")

# the four methods go ONLY through v: an instrumented v that records every mask requested
comp_m = D.model_components(idata, "model", dfm, draws=draws)
requested = set()
base = D.ValueFunction.from_components(comp_m)
spy = D.ValueFunction(base.names, lambda m: (requested.add(m), base.weekly(m))[1])
methods(spy, D.BASE_ORDER_K9)
ok(requested <= set(range(spy.N + 1)) and spy.calls > 0,
   f"the four methods read the model ONLY through v ({len(requested)} distinct masks requested)")

print("2.2 -- (A)_k = v(N) - v(N without k), exact")
A = D.removal(base)
for i, k in enumerate(base.names):
    direct = base.weekly(base.N).sum(-1) - base.weekly(base.N & ~(1 << i)).sum(-1)
    assert np.array_equal(A[k], direct), k
ok(True, f"(A) matches the definition bit for bit in the {base.K} drivers")

print("2.3 -- (B), (C) and (D) add up to v(N) - v(empty)")
for ref in D.VALID_REFERENCES:
    fvr = D.ValueFunction.from_components(D.model_components(idata, ref, dfm, draws=draws))
    for name, rep in (("B", D.sequential(fvr, D.BASE_ORDER_K9)), ("C", D.shapley(fvr)), ("D", D.lmdi(fvr))):
        e = D.efficiency(fvr, rep)
        ok(e < TOL, f"model K=9, reference={ref}: ({name}) adds up to the incremental (rel dev {e:.1e})")
    fvg = D.ValueFunction.from_components(D.group(D.model_components(idata, ref, dfm, draws=draws), D.GROUPING_K8))
    fvd = D.ValueFunction.from_components(D.dgp_components(ref, dfm))
    for side, f in (("model K=8", fvg), ("DGP K=8", fvd)):
        for name, rep in (("B", D.sequential(f, D.BASE_ORDER_K8)), ("C", D.shapley(f)), ("D", D.lmdi(f))):
            e = D.efficiency(f, rep)
            ok(e < TOL, f"{side}, reference={ref}: ({name}) adds up to the incremental (rel dev {e:.1e})")

print("2.4 -- Shapley: symmetry and dummy")
b = np.array([100.0, 120.0, 90.0])
f = np.array([[1.3, 1.1, 1.2], [1.3, 1.1, 1.2], [1.0, 1.0, 1.0], [0.9, 1.05, 0.8]])  # 0 and 1 symmetric, 2 dummy
fv = fv_toy(["a", "b", "dummy", "c"], multiplicative(b, f))
C = D.shapley(fv)
ok(np.allclose(C["a"], C["b"], rtol=0, atol=1e-12), "symmetry: two interchangeable drivers receive the same")
ok(np.allclose(C["dummy"], 0, atol=1e-12), "dummy: a driver with f=1 in every week receives 0")
for name, rep in (("A", D.removal(fv)), ("B", D.sequential(fv, ["c", "dummy", "a", "b"])), ("D", D.lmdi(fv))):
    ok(np.allclose(rep["dummy"], 0, atol=1e-12), f"dummy: ({name}) also gives it 0")

print("2.5 -- K=2: the closed formulas")
bb, f1, f2 = 1000.0, 1.25, 1.4
fv = fv_toy(["1", "2"], multiplicative([bb], [[f1], [f2]]))
Y = bb * f1 * f2
L = (Y - bb) / math.log(Y / bb)
expected = {
    "A": {"1": bb * f2 * (f1 - 1), "2": bb * f1 * (f2 - 1)},
    "B": {"1": bb * (f1 - 1), "2": bb * f1 * (f2 - 1)},
    "C": {"1": bb * (f1 - 1) * (1 + f2) / 2, "2": bb * (f2 - 1) * (1 + f1) / 2},
    "D": {"1": L * math.log(f1), "2": L * math.log(f2)},
}
got = methods(fv, ["1", "2"])
for mtd in "ABCD":
    for k in ("1", "2"):
        assert abs(float(got[mtd][k][0]) - expected[mtd][k]) < 1e-9, (mtd, k, got[mtd][k], expected[mtd][k])
ok(True, "K=2: (A), (B), (C) and (D) give the closed formulas")

print("2.6 -- positive control: without interaction, the four coincide")
# Multiplicative WITHOUT interaction: each driver differs from 1 only in weeks where the others equal 1.
# In those weeks the product of factors has no cross terms, and the split is not ambiguous.
b = np.array([100.0, 200.0, 150.0, 80.0])
f = np.array([[1.5, 1, 1, 1.2], [1, 0.7, 1, 1], [1, 1, 1.3, 1]])
fv = fv_toy(["x", "y", "z"], multiplicative(b, f))
got = methods(fv, ["z", "x", "y"])
for k in fv.names:
    vals = [float(got[m][k][0]) for m in "ABCD"]
    assert max(vals) - min(vals) < 1e-9, (k, vals)
ok(True, "multiplicative without interaction: (A) = (B) = (C) = (D) in the three drivers")
# Additive: (A), (B) and (C) coincide; LMDI is NOT defined and has to refuse
fa = fv_toy(["x", "y"], lambda m: (100.0 + 30.0 * (m & 1) + 50.0 * (m >> 1 & 1)) * np.ones((1, 3)))
ga = {"A": D.removal(fa), "B": D.sequential(fa, ["y", "x"]), "C": D.shapley(fa)}
for k in fa.names:
    vals = [float(ga[m][k][0]) for m in "ABC"]
    assert max(vals) - min(vals) < 1e-9, (k, vals)
ok(True, "additive: (A) = (B) = (C)")
try:
    D.lmdi(fa)
    raise AssertionError("LMDI accepted an additive v")
except D.NotMultiplicative:
    ok(True, "additive: LMDI refuses (NotMultiplicative) instead of splitting something that does not close")

print("2.7 -- sabotage: efficiency has to FAIL")
fv = D.ValueFunction.from_components(D.dgp_components("dgp", dfm))
K = fv.K
w = [D.shapley_weight(s, K) for s in range(K)]
# ! Reversing the weights does not sabotage (they are symmetric in s and K-1-s); rotating them does.
reversed_w = w[::-1]
e_inv = D.efficiency(fv, D.shapley(fv, weights=lambda s, K: reversed_w[s]))
ok(e_inv < TOL, f"(sabotage control) REVERSED weights: they are symmetric, efficiency passes ({e_inv:.1e})")
rotated = w[1:] + w[:1]
e = D.efficiency(fv, D.shapley(fv, weights=lambda s, K: rotated[s]))
ok(e > 1e-6, f"Shapley weights ROTATED by hand -> efficiency fails (rel dev {e:.2e})")
fv3 = fv_toy(["a", "b", "c"], multiplicative([100.0], [[1.4], [1.2], [0.9]]))
every = list(itertools.permutations(fv3.names))
e_ok = D.efficiency(fv3, D.shapley_by_orders(fv3, every))
ok(e_ok < TOL and np.allclose(D.shapley_by_orders(fv3, every)["a"], D.shapley(fv3)["a"]),
   "sabotage control: with the 3! orders, the average of (B) = Shapley and it is efficient")
e = D.efficiency(fv3, D.shapley_by_orders(fv3, every[:-1], divisor=len(every)))
ok(e > 1e-6, f"one order removed (still dividing by K!) -> efficiency fails (rel dev {e:.2e})")

print("2.8 -- negative control: what HAS to come out different, comes out different")
fv = D.ValueFunction.from_components(D.dgp_components("dgp", dfm))
o1 = D.BASE_ORDER_K8
o2 = list(reversed(o1))
b1, b2 = D.sequential(fv, o1), D.sequential(fv, o2)
diff = max(abs(float(b1[k][0] - b2[k][0])) for k in fv.names)
ok(diff > 1e-3 * float(fv.total(fv.N)[0]) / 100,
   f"(B) with two different orders splits differently: max|diff| = {diff:.1f} sales units")
fm = D.ValueFunction.from_components(D.group(D.model_components(idata, "dgp", dfm, draws=draws), D.GROUPING_K8))
diffv = max(float(np.max(np.abs(fm.total(m).mean() - fv.total(m)[0]) / fv.total(m)[0])) for m in range(fv.N + 1))
ok(diffv > 1e-6, f"v_model != v_DGP: max relative difference over the 256 worlds = {diffv:.2e}")
try:
    D.reference_levels(None)
    raise AssertionError("an empty reference was accepted")
except D.MissingReference:
    ok(True, "without a price and ACV reference, it fails with MissingReference (no silent default)")

print(f"\nPASSED: {_n['ok']} checks.")
