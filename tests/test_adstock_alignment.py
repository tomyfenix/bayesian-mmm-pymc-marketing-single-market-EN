"""Regression test: the generator's adstock and the estimator's are the same transformation.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytensor.xtensor as ptx
from pymc_marketing.mmm.transformers import geometric_adstock as _pymc_geometric_adstock_fresh

from config.mmm_dgp_config import L_MAX, ADSTOCK_NORMALIZE
from synthetic_data.adstock_dgp import geometric_adstock_aligned

# The 4 true alphas of the DGP, copied ON PURPOSE from media_true_params in
# config/mmm_dgp_config.py: the test is a control independent of that source.
DGP_ALPHAS = {"tv": 0.70, "meta": 0.35, "tiktok": 0.20, "ooh": 0.50}

RTOL, ATOL = 1e-10, 1e-12


def _fresh_pymc_marketing_call(x: np.ndarray, alpha: float, l_max: int, normalize: bool) -> np.ndarray:
    """Independent call to the library, without going through the generator's wrapper -- if someone
    breaks geometric_adstock_aligned() without breaking this, the test catches it anyway.
    """
    x_xt = ptx.as_xtensor(np.asarray(x, dtype=float), dims=("date",))
    y = _pymc_geometric_adstock_fresh(x_xt, alpha=alpha, l_max=l_max, normalize=normalize, dim="date").eval()
    return np.asarray(y, dtype=float)


def _geometric_adstock_closed_form(x: np.ndarray, alpha: float, l_max: int) -> np.ndarray:
    """Closed formula, independent of pytensor and of pymc-marketing. It documents EXPLICITLY the
    convention checked against the source code of pymc_marketing/mmm/transformers.py
    (geometric_adstock + batched_convolution with mode="After"): - kernel weights: w[k] =
    alpha**k, for k = 0 ..
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = alpha ** np.arange(l_max)
    w = w / w.sum()
    out = np.zeros(n)
    for t in range(n):
        k_max = min(t, l_max - 1)
        out[t] = np.dot(w[: k_max + 1], x[t - k_max : t + 1][::-1])
    return out


def test_adstock_alignment_all_dgp_alphas():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1000, 156)
    x[::11] = 0.0  # includes interspersed zeros, like a series with real flighting

    for channel, alpha in DGP_ALPHAS.items():
        y_wrapper = geometric_adstock_aligned(x, alpha, l_max=L_MAX, normalize=ADSTOCK_NORMALIZE)
        y_fresh = _fresh_pymc_marketing_call(x, alpha, l_max=L_MAX, normalize=ADSTOCK_NORMALIZE)
        y_closed = _geometric_adstock_closed_form(x, alpha, l_max=L_MAX)

        assert np.allclose(y_wrapper, y_fresh, rtol=RTOL, atol=ATOL), (
            f"[{channel}] geometric_adstock_aligned() (generator) != fresh pymc-marketing call"
        )
        assert np.allclose(y_wrapper, y_closed, rtol=RTOL, atol=ATOL), (
            f"[{channel}] geometric_adstock_aligned() (generator) != closed formula in numpy"
        )
        assert np.allclose(y_fresh, y_closed, rtol=RTOL, atol=ATOL), (
            f"[{channel}] pymc-marketing (fresh) != closed formula in numpy"
        )
        max_abs_diff = max(
            np.max(np.abs(y_wrapper - y_fresh)),
            np.max(np.abs(y_wrapper - y_closed)),
        )
        print(f"  OK  channel={channel:8s} alpha={alpha:.2f}  max|diff| across the 3 implementations = {max_abs_diff:.3e}")

    print(
        f"\nPASSED: the 3 implementations (generator wrapper, fresh pymc-marketing, "
        f"closed formula) agree with rtol={RTOL} for the {len(DGP_ALPHAS)} DGP alphas, "
        f"l_max={L_MAX}, normalize={ADSTOCK_NORMALIZE}."
    )


if __name__ == "__main__":
    test_adstock_alignment_all_dgp_alphas()
