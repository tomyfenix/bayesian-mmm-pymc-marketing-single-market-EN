"""Regression test: the generator's saturation and the estimator's are the same function.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytensor.xtensor as ptx
from pymc_marketing.mmm.components.saturation import LogisticSaturation

from config.mmm_dgp_config import MEDIA_CHANNELS, logistic_saturation, media_true_params
from config.log_regime import L_NATIONAL

RTOL, ATOL = 1e-10, 1e-12


def _estimator(x: np.ndarray, lam: float, L: float) -> np.ndarray:
    """The saturation as the model evaluates it: LogisticSaturation.function, called fresh. In
    0.19.2 the function works on xtensor: a bare ndarray gives a TypeError.
    """
    x_xt = ptx.as_xtensor(np.asarray(x, dtype=float), dims=("date",))
    return np.asarray(LogisticSaturation().function(x_xt, lam=lam, beta=L).eval(), dtype=float)


def _closed_form(x: np.ndarray, lam: float, L: float) -> np.ndarray:
    """(1 - e^-y) / (1 + e^-y) = tanh(y/2): algebraic identity, independent of the other two."""
    return L * np.tanh(lam * np.asarray(x, dtype=float) / 2.0)


def test_saturation_alignment_all_dgp_channels():
    for channel in MEDIA_CHANNELS:
        lam = media_true_params[channel]["lam"]
        x50 = np.log(3.0) / lam
        x = np.concatenate([[0.0], np.linspace(x50 / 1000, 10 * x50, 400)])
        for L in (L_NATIONAL[channel], 1.0):
            y_dgp = np.asarray(logistic_saturation(x, L=L, lam=lam), dtype=float)
            y_est = _estimator(x, lam, L)
            y_cf = _closed_form(x, lam, L)

            assert np.allclose(y_dgp, y_est, rtol=RTOL, atol=ATOL), (
                f"[{channel}, L={L}] DGP logistic_saturation() != estimator LogisticSaturation"
            )
            assert np.allclose(y_dgp, y_cf, rtol=RTOL, atol=ATOL), (
                f"[{channel}, L={L}] DGP logistic_saturation() != L*tanh(lam*x/2)"
            )
            assert np.allclose(y_est, y_cf, rtol=RTOL, atol=ATOL), (
                f"[{channel}, L={L}] estimator LogisticSaturation != L*tanh(lam*x/2)"
            )
            assert y_dgp[0] == 0.0 and y_est[0] == 0.0, f"[{channel}, L={L}] f(0) != 0"
            mid = float(np.asarray(logistic_saturation(np.array([x50]), L=L, lam=lam))[0])
            assert np.isclose(mid, L / 2, rtol=RTOL), f"[{channel}, L={L}] f(ln3/lam) = {mid} != L/2"

            diff = max(np.max(np.abs(y_dgp - y_est)), np.max(np.abs(y_dgp - y_cf)))
            print(f"  OK  channel={channel:7s} lam={lam:.7f} L={L:.4f}  max|diff| across the 3 = {diff:.3e}")

    print(f"\nPASSED: the DGP saturation, the estimator's and the closed form agree with rtol={RTOL} "
          f"in the {len(MEDIA_CHANNELS)} channels, with the log ceilings and L=1.0.")


if __name__ == "__main__":
    test_saturation_alignment_all_dgp_channels()
