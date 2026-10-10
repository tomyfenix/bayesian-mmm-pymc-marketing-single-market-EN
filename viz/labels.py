"""The ONE vocabulary of the figures.
"""

from config.mmm_dgp_config import HDI_PROB

# --- Central quantities --------------------------------------------------
BIAS = "bias"
REL_BIAS = "relative bias"
REL_ERROR = "relative error"
TRUE_VALUE = "true value"
TRUTH = "truth"
POSTERIOR = "posterior"
PRIOR = "prior"
CONTRACTION = "prior→posterior contraction"
CONTRACTION_SHORT = "contraction"
HDI = f"HDI{round(HDI_PROB * 100)}"  # trivial: unit

# --- Phrases repeated verbatim across figures ------------------------------
# The axis of 4c1 and of any relative-error figure is the same one and has to be labeled the same.
AXIS_REL_ERROR = "relative error vs. true value (%)"
MISSES_HDI = f"{HDI} misses the {TRUE_VALUE}"
COVERS_HDI = f"{HDI} covers the {TRUE_VALUE}"

# The two criteria kept apart, which 4c1 has to name the same way in all its parts.
CRITERION_A = "CRITERION A (coverage)"
CRITERION_B = "CRITERION B (bias magnitude)"
CRITERION_A_RULE = f"0% = {TRUE_VALUE}. Fails when the {HDI} bar does not cross it."
CRITERION_B_CAVEAT = "true by construction: this is a scale, not evidence"

# --- Axes and dimensions -------------------------------------------------
WEEK = "week"
WEEKS = "weeks"
CHANNEL = "channel"
PARAMETER = "parameter"
DENSITY = "density"
LAG_WEEKS = "lag (weeks)"
ADSTOCKED_IMPRESSIONS = "adstocked impressions (thousands)"
LOG_SALES = "log sales"

# --- Recurring annotations -----------------------------------------------
OBSERVED_RANGE = "observed spend range"
EXTRAPOLATION = "extrapolation"
EXTRAPOLATION_NOTE = "extrapolation\n(axis continues\nunbounded →)"
POSTERIOR_WINDOW = "posterior window"
FULL_PRIOR = "full prior"
ZOOM_TO_WINDOW = f"zoom: {POSTERIOR_WINDOW}"
DATASET_LABEL_NOTE = "dataset_label"


# --- Sampling diagnostics ------------------------------------------------
DIVERGENCES = "divergences"
CONVERGENCE = "convergence"
RECOVERY = "parameter recovery"
ENERGY = "energy"
RANK_PLOTS = "rank plots"

# --- Display names of the controls ----------------------------------------
# They appear identical in 3b1, 4c1 and 4b1: if they diverge, the reader thinks they are
# different parameters.
CONTROL_DISPLAY = {
    "gamma_control[log_price_c]": "eta (price elasticity)",
    "gamma_control[promo_flag]": "gamma_control[promo] (uplift)",
    "gamma_control[log_acv_c]": "gamma (ACV elasticity)",
}
CONTROL_DISPLAY_SHORT = {
    "log_price_c": "eta (price)",
    "promo_flag": "promo",
    "log_acv_c": "gamma (ACV)",
}

# --- Business reading (5a1-5e1) ------------------------------------------
# The level / change-decomposition pair is the distinction these figures exist to keep: if the
# two are labeled the same, the reader adds up things that do not add up.
LOG_POINTS = "log-points"
YOY = "year over year"
CHANGE_DECOMP = "change decomposition"
CONTRIBUTION = "contribution"
INCREMENTAL = "incremental"
SPEND = "spend"
ROAS_AVG = "average ROAS"
ROAS_MARGINAL = "marginal ROAS"
CROSS_EFFECT = "cross-effect"
BUDGET_MULT = "budget multiplier"
REVENUE = "revenue"

# The most important warning of 5b1, and the reason there is no waterfall in levels.
DO_NOT_ADD_UP = ("channel contributions do NOT add up to the joint effect: in a log-additive model "
            "they compose multiplicatively")
CHANGE_DECOMP_EXACT = "exact by construction: the model is additive in logs, so this closes with no residual"
