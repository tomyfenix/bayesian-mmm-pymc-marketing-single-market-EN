"""Adstock transformations for the synthetic data generator (DGP).
"""

import numpy as np
import pytensor.xtensor as ptx
from config.mmm_dgp_config import adstock_mode
from pymc_marketing.mmm.transformers import geometric_adstock as _pymc_geometric_adstock


def geometric_adstock_old(x: np.ndarray, alpha: float) -> np.ndarray:
    """ORIGINAL formula of the generator (early versions): cumulative geometric memory, NOT normalized
    and NOT truncated (equivalent to l_max=len(x), normalize=False).
    """
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + alpha * out[i - 1]
    return out


def geometric_adstock_aligned(
    x: np.ndarray, alpha: float, l_max: int, normalize: bool = True
) -> np.ndarray:
    """Geometric adstock using pymc-marketing's REAL function (direct import, not reimplemented) -- it
    guarantees by construction that the DGP applies exactly the same transformation as the estimator.
    """
    x_xt = ptx.as_xtensor(np.asarray(x, dtype=float), dims=("date",))
    y = _pymc_geometric_adstock(
        x_xt, alpha=alpha, l_max=l_max, normalize=normalize, dim="date", mode=adstock_mode()
    ).eval()
    return np.asarray(y, dtype=float)


def geometric_adstock_dgp(
    x: np.ndarray,
    alpha: float,
    *,
    normalized: bool,
    l_max: int | None,
) -> np.ndarray:
    """Configurable dispatcher of the DGP adstock -- explicit stress scenario. The confirmed design of
    the project is that the DGP and the estimator use the SAME transformation (normalized=True,
    l_max=config.L_MAX -- see generate_dataset.py).
    """
    if not normalized and l_max is None:
        return geometric_adstock_old(x, alpha)
    effective_l_max = l_max if l_max is not None else len(x)
    return geometric_adstock_aligned(x, alpha, l_max=effective_l_max, normalize=normalized)
