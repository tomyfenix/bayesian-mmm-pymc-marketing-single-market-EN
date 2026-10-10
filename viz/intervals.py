"""The repo-wide interval: the HDI of HDI_PROB (config/), computed with arviz. The bands of the figures
and the intervals of the tables come from here, so that no figure uses a different one.
"""

import numpy as np

from config.mmm_dgp_config import HDI_PROB


def hdi(draws, prob: float = HDI_PROB) -> tuple[float, float]:
    """(lower, upper) of the HDI of a 1-D sample of draws."""
    import arviz as az
    lo, hi = az.hdi(np.asarray(draws, dtype=float).ravel(), hdi_prob=prob)
    return float(lo), float(hi)


def pointwise_hdi(draws, prob: float = HDI_PROB) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise band: `draws` has shape (n_draws, n_points) and each column is summarized by its own
    HDI. Returns (lower, upper), each of length n_points.
    """
    import arviz as az
    x = np.asarray(draws, dtype=float)
    band = az.hdi(x[np.newaxis, ...], hdi_prob=prob)     # arviz expects (chain, draw, *shape)
    return band[..., 0], band[..., 1]
