"""THE BUDGET OPTIMIZATION SCENARIO -- one single place.

# ! WATCH OUT -- the priors were chosen by whoever wrote the generator: optimistic result by construction.
"""

from config.mmm_dgp_config import L_MAX

# The total budget. "historical_spend" = the sum of the four channels over the observed period, read
# from the CSV. That way the scenario is a REALLOCATION at constant budget and does not get mixed with
# the question of how much to spend in total, which is a different one.
BUDGET = "historical_spend"

LOWER_BOUND = 0.5
UPPER_BOUND = 1.5

# How many posterior draws go into the optimization. 400 is the same number 5d_ and 5e_ use, for the
# same reason: the cost is in rebuilding the adstock PER DRAW, and 400 already puts the Monte Carlo
# error of the optimum below the last decimal that is reported.
N_DRAWS = 400
# which posterior draws are sampled for the optimization (7a_, 7b_)
OPTIMIZATION_DRAWS_SEED = 42
# which posterior draws are sampled for the ROAS and budget sweeps (5d_, 5e_): a different use
SWEEP_DRAWS_SEED = 42
MARGINAL_ROAS_EPS = 0.01            # relative step of the finite difference of the marginal ROAS
SWEEP_GRID = (0.0, 1.5, 31)         # 5e_: sweep multipliers (start, end, points)
SWEEP_POINTS = (0.5, 1.0, 1.5)      # 5e_: where the deviations against the truth are reported

OBJECTIVE = "expected_sales_in_levels"

SATURATED_CHANNEL_CONTROL = "tv"
CONTROL_FACTOR = 2.0


# The four scenarios being compared. S0 is the observed plan.
SCENARIOS = ("S0_historical", "S1_cross_channel", "S2_even_pacing", "S3_temporal_optimum")

# WEEKLY cap, in IMPRESSIONS: 0 <= impressions_{k,t} <= WEEKLY_CAP_MAX * (maximum observed weekly
# impressions of channel k); in money, the cap of each week is that times its CPM. It exists so as not
# to evaluate the response curve at levels the model never saw: above the observed maximum, saturation
# is pure extrapolation.
WEEKLY_CAP_MAX = 1.0

N_DRAWS_OPTIMIZATION = 200
# The evaluation uses the draws that did NOT go into the optimization: 4000 - 200 = 3800. The value here
# is informative; 7b_ builds the complement and checks that they do not overlap.
N_DRAWS_EVALUATION = 3800

# Starts of the temporal optimizer. The problem is NOT concave --sales are exp of a sum of concave
# functions-- so a single start proves nothing. The three have to reach the same objective; if not,
# there are local optima and that is the result.
STARTS = ("S0", "S2", "random")
STARTS_TOLERANCE = 1e-4           # relative, between the best and the worst of the three
MAXITER_TEMPORAL = 1200
RANDOM_START_SEED = 49            # the random start of the temporal optimizer
FTOL_TEMPORAL = 1e-12
MAXITER_MULTIPLIERS = 300
FTOL_MULTIPLIERS = 1e-14
TEMPORAL_REBALANCE_ITER = 100     # rounds to get back to the exact budget
RELATIVE_BUDGET_TOL = 1e-3        # a plan that departs more than this from the budget is not feasible
# cross-channel optimization (7a_)
CROSS_CHANNEL_OPTIM = {"maxiter": 300, "ftol": 1e-12}
REBALANCE_ITER = 50
OPTIMIZATION_MARGINAL_EPS = 1e-3  # step of the finite difference of the marginal return
KKT_THRESHOLD = 1e-4              # maximum dispersion of the marginals at the optimum
OFF_THRESHOLD = 1e-4              # 7b_: a week with spend below this (relative) counts as switched off
EDGE_VERDICT_THRESHOLDS = (0.5, 0.85)   # 7b_: December ratio read as emptied / evened out

# The check that can fail: with a FLAT base, the optimum has to be ~even pacing except at the edges,
# where the adstock falls outside the period. The coefficient of variation of the optimal weekly spend
# is measured outside the first and last L_MAX weeks.
CONTROL_CV_MAX = 0.05

# Zero-spend weeks added at the end to collect the adstock carryover: as many as its memory, L_MAX.
CARRYOVER_TAIL = L_MAX
