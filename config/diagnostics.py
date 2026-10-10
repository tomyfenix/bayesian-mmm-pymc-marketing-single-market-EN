"""SAMPLING, FIGURES AND DIAGNOSTICS -- one single place.

How the prior and the posterior are sampled, which summary the figures draw and which threshold
decides each diagnostic verdict. The scripts import them; none of them defines them.
"""

# --- sampling
SAMPLING_SEED = 42                    # fit (3a_) and prior predictive (2a_, 4b_, 2b_, 4e_, 4h_)
N_PRIOR_SAMPLES = 2000                # prior draws in 4b_, 2b_, 4e_ and 4h_
N_PRIOR_SAMPLES_LAM_CHECK = 500       # prior draws in 2a_
FIGURE_SUBSAMPLE_SEED = 42            # which posterior draws the bands of 4f_, 4g_ and 4h_ draw
N_DRAWS_BANDS = 500                   # how many draws go into those bands

# --- the 3a_ sampler: the trial run estimates the time, the full one is the fit
SAMPLER = {"draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.9}
SAMPLER_TRIAL = {"draws": 200, "tune": 200, "chains": 1}
MAX_FIT_TIME_MIN = 30                 # if the trial extrapolates beyond this, 3a_ does not run the full fit

# --- diagnostic thresholds
R_HAT_THRESHOLD = 1.01
ESS_BULK_THRESHOLD = 400
COLLINEARITY_THRESHOLD = 0.7          # 1a_: correlation between regressors that gets reported
FOREST_ERROR_THRESHOLD = -0.50        # 4a_: relative error that flags a parameter in 4c1
INFORMED_CONTRACTION_THRESHOLD = 0.15     # 4b_: minimum contraction to say the data informed
INFORMED_DISPLACEMENT_THRESHOLD = 0.3     # 4b_: minimum displacement, same idea
QUADRANT_BIAS_THRESHOLD = 2.0         # 4e_: |bias| in % that separates the quadrants
PRIOR_PREDICTIVE_INNER_BAND = 0.50    # 2b1: the inner band (HDI50); the outer one is the repo-wide HDI_PROB
VARIANCE_DOMINANCE_THRESHOLD = 10     # 2b_: a source explaining more than 10x the smallest one dominates
TREND_GROWTH_THRESHOLDS = (-20, 40)   # 2b_: % growth over 3 years reported as unusual
SPLIT_TABLE_DECIMALS = 3              # 6b_ writes the table with these decimals and 6c_ checks it the same way

# --- hand-written reference: it does NOT come from this repo and is not recomputed. It does NOT come
# from a client's data: it comes from a diagnostic by the author on an earlier version of this same
# synthetic dataset, in the working repo where this one was developed (it did not travel with it).
# 2b2: the "real" variance profile of an OLS reconstruction of log(sales) from the true components of
# the generator, in %. The working repo does not keep the run that produced these numbers.
REFERENCE_VARIANCE_PROFILE = {"price": 56.0, "ACV": 20.8, "promo": 9.7, "seasonal": 4.8,
                              "noise": 3.8, "media": 2.9, "trend": 2.0}
