"""THE PRIORS OF THE PRIOR CHECK IN 2a_.

They are not the priors of the model that is fitted (those are in mmm_dgp_config and log_regime):
they belong to the model that 2a_ builds only to derive the saturation_lam prior.
"""

# --- 2a_: where the saturation_lam prior comes from
LAM_PRIOR_DEFAULT_PYMC = {"alpha": 3.0, "beta": 1.0}   # the LogisticSaturation default in pymc-marketing
LAM_CHECK_BETA_SIGMA = 0.03                             # saturation ceiling of the 2a_ model
ACTIVE_EXPOSURE_THRESHOLD = 0.10      # weeks with adstock >= 10% of the peak count as active exposure
ACTIVE_RANGE_PERCENTILES = (10, 90)   # half saturation falls between these percentiles of active exposure
LOW_LAM_QUANTILE = 0.10               # the lowest lam of the range sits at this quantile of the derived prior
