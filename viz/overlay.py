"""The ONE density + HDI94 + truth overlay function.
"""

import arviz as az
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from config.mmm_dgp_config import HDI_PROB
from viz.style import PRIOR_COLOR, POSTERIOR_COLOR, TRUTH_COLOR
from viz.labels import PRIOR, POSTERIOR, TRUE_VALUE


def density_with_truth(
    ax,
    post_draws,
    true_value: float,
    *,
    prior_draws=None,
    hdi_prob: float = HDI_PROB,
    post_color: str = POSTERIOR_COLOR,
    prior_color: str = PRIOR_COLOR,
    truth_color: str = TRUTH_COLOR,
    shade_hdi: bool = True,
    n_grid: int = 400,  # cosmetic
    xlim: tuple[float, float] | None = None,
    label_prior: str = PRIOR,
    label_post: str = POSTERIOR,
    label_truth: str = TRUE_VALUE,
) -> dict:
    """Draws, on `ax`: the posterior density (+ optionally the prior's, same grid_x), the HDI94 band
    shaded over the posterior density, and a dashed vertical line at `true_value`.
    """
    post_draws = np.asarray(post_draws).ravel()
    post_sd = post_draws.std()

    if xlim is not None:
        xlim_lo, xlim_hi = xlim
    else:
        center_lo, center_hi = min(post_draws.min(), true_value), max(post_draws.max(), true_value)
        pad = max(1.5 * post_sd, 0.1 * (center_hi - center_lo), 1e-9)  # cosmetic
        xlim_lo, xlim_hi = center_lo - pad, center_hi + pad

    grid_x = np.linspace(xlim_lo, xlim_hi, n_grid)

    if prior_draws is not None:
        prior_draws = np.asarray(prior_draws).ravel()
        ax.fill_between(grid_x, gaussian_kde(prior_draws)(grid_x), color=prior_color, alpha=0.30, label=label_prior)  # cosmetic

    post_dens = gaussian_kde(post_draws)(grid_x)
    ax.fill_between(grid_x, post_dens, color=post_color, alpha=0.50, label=label_post)  # cosmetic

    hdi_lo, hdi_hi = az.hdi(post_draws, hdi_prob=hdi_prob)
    hdi_lo, hdi_hi = float(hdi_lo), float(hdi_hi)
    if shade_hdi:
        mask = (grid_x >= hdi_lo) & (grid_x <= hdi_hi)
        if mask.any():
            ax.fill_between(grid_x[mask], post_dens[mask], color=post_color, alpha=0.28, lw=0)  # cosmetic

    ax.axvline(true_value, color=truth_color, lw=1.6, ls="--", label=label_truth)  # cosmetic
    ax.set_xlim(xlim_lo, xlim_hi)

    return {
        "xlim_lo": xlim_lo, "xlim_hi": xlim_hi,
        "hdi_lo": hdi_lo, "hdi_hi": hdi_hi,
        "contains_truth": hdi_lo <= true_value <= hdi_hi,
    }


def prior_posterior_two_scales(
    ax_wide,
    ax_zoom,
    post_draws,
    true_value: float,
    prior_draws,
    *,
    hdi_prob: float = HDI_PROB,
    post_color: str = POSTERIOR_COLOR,
    prior_color: str = PRIOR_COLOR,
    truth_color: str = TRUTH_COLOR,
    n_grid: int = 600,  # cosmetic
    tail: float = 0.001,  # cosmetic
) -> dict:
    """Prior and posterior on TWO x-axis scales, one above the other. It exists because a single scale
    cannot show the two things this figure has to say: CONTRACTION (wide prior vs narrow posterior)
    needs to see the whole prior; BIAS (posterior vs truth) needs the posterior window.
    """
    post_draws = np.asarray(post_draws).ravel()
    prior_draws = np.asarray(prior_draws).ravel()

    # --- wide panel: the whole prior ---------------------------------------------
    lo_p, hi_p = np.quantile(prior_draws, [tail, 1 - tail])
    lo = min(lo_p, post_draws.min(), true_value)
    hi = max(hi_p, post_draws.max(), true_value)
    pad = 0.03 * (hi - lo)  # cosmetic
    x_wide = np.linspace(lo - pad, hi + pad, n_grid)

    d_prior = gaussian_kde(prior_draws)(x_wide)
    d_post = gaussian_kde(post_draws)(x_wide)
    ax_wide.fill_between(x_wide, d_prior / d_prior.max(), color=prior_color, alpha=0.35,  # cosmetic
                          lw=0, label=PRIOR)
    if d_post.max() > 0:
        ax_wide.fill_between(x_wide, d_post / d_post.max(), color=post_color, alpha=0.65,  # cosmetic
                              lw=0, label=POSTERIOR)
    ax_wide.axvline(post_draws.mean(), color=post_color, lw=3.0, zorder=4, solid_capstyle="butt")  # cosmetic
    ax_wide.axvline(true_value, color=truth_color, lw=1.1, ls=(0, (3, 2)), zorder=6,  # cosmetic
                     label=TRUE_VALUE)
    ax_wide.set_xlim(lo - pad, hi + pad)
    ax_wide.set_ylim(0, 1.12)  # cosmetic
    ax_wide.set_yticks([])
    ax_wide.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))  # cosmetic

    # --- zoom panel: the posterior window ----------------------------------------
    post_sd = post_draws.std()
    c_lo, c_hi = min(post_draws.min(), true_value), max(post_draws.max(), true_value)
    z_pad = max(1.5 * post_sd, 0.1 * (c_hi - c_lo), 1e-9)  # cosmetic
    z_lo, z_hi = c_lo - z_pad, c_hi + z_pad
    x_zoom = np.linspace(z_lo, z_hi, n_grid)
    dz = gaussian_kde(post_draws)(x_zoom)
    ax_zoom.fill_between(x_zoom, dz, color=post_color, alpha=0.50, lw=0)  # cosmetic
    hdi_lo, hdi_hi = az.hdi(post_draws, hdi_prob=hdi_prob)
    hdi_lo, hdi_hi = float(hdi_lo), float(hdi_hi)
    m = (x_zoom >= hdi_lo) & (x_zoom <= hdi_hi)
    if m.any():
        ax_zoom.fill_between(x_zoom[m], dz[m], color=post_color, alpha=0.28, lw=0)  # cosmetic
    ax_zoom.axvline(true_value, color=truth_color, lw=1.6, ls="--", zorder=5)  # cosmetic
    ax_zoom.set_xlim(z_lo, z_hi)
    ax_zoom.set_yticks([])
    # Fewer ticks and shared scientific notation: in the zoom window the values have 5-6 significant
    # decimals and with the default they overlapped.
    ax_zoom.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))  # cosmetic
    ax_zoom.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useOffset=True)  # cosmetic

    mass_in_zoom = float(((prior_draws >= z_lo) & (prior_draws <= z_hi)).mean())
    return {
        "xlim_lo": z_lo, "xlim_hi": z_hi,
        "hdi_lo": hdi_lo, "hdi_hi": hdi_hi,
        "contains_truth": hdi_lo <= true_value <= hdi_hi,
        "wide_lo": lo - pad, "wide_hi": hi + pad,
        "prior_mass_in_zoom": mass_in_zoom,
    }
