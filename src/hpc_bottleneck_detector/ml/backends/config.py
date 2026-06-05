"""
Backend configuration constants
"""

# ---------------------------------------------------------------------------
# tsfresh feature-extraction parameter sets
# ---------------------------------------------------------------------------

# 1. Basic - lightweight descriptive statistics
BASIC_FC_PARAMETERS: dict = {
    "minimum": None,
    "maximum": None,
    "mean": None,
    "standard_deviation": None,
    "quantile": [
        {"q": 0.05}, {"q": 0.25}, {"q": 0.50}, {"q": 0.75}, {"q": 0.95},
    ],
    "skewness": None,
    "kurtosis": None,
    "agg_autocorrelation": [
        {"f_agg": "mean",   "maxlag": 3},
        {"f_agg": "median", "maxlag": 3},
        {"f_agg": "var",    "maxlag": 3},
    ],
    "agg_linear_trend": [
        {"attr": "slope",     "chunk_len": 5, "f_agg": "mean"},
        {"attr": "intercept", "chunk_len": 5, "f_agg": "mean"},
        {"attr": "rvalue",    "chunk_len": 5, "f_agg": "mean"},
        {"attr": "slope",     "chunk_len": 2, "f_agg": "mean"},
        {"attr": "intercept", "chunk_len": 2, "f_agg": "mean"},
    ],
    "absolute_sum_of_changes": None,
}

# 2. Basic + Advanced — adds thresholding, energy, complexity, and frequency
#    features on top of the basic set.
# BASIC_ADVANCED_FC_PARAMETERS: dict = {
#     **BASIC_FC_PARAMETERS,
#     "count_above_mean": None,
#     "count_below_mean": None,
#     "ratio_beyond_r_sigma": [
#         {"r": 0.5}, {"r": 1.0}, {"r": 1.5}, {"r": 2.0}, {"r": 2.5}, {"r": 3.0},
#     ],
#     "first_location_of_maximum": None,
#     "first_location_of_minimum": None,
#     "abs_energy": None,
#     "absolute_sum_of_changes": None,
#     "cid_ce": [{"normalize": True}, {"normalize": False}],
#     "c3": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
#     "ar_coefficient": [
#         {"coeff": 0, "k": 3}, {"coeff": 1, "k": 3}, {"coeff": 2, "k": 3},
#     ],
#     "fft_coefficient": [
#         {"coeff": 0, "attr": "real"},
#         {"coeff": 1, "attr": "real"}, {"coeff": 1, "attr": "imag"},
#         {"coeff": 2, "attr": "real"}, {"coeff": 2, "attr": "imag"},
#         {"coeff": 3, "attr": "real"}, {"coeff": 3, "attr": "imag"},
#     ],
# }

# ---------------------------------------------------------------------------
# Per-classifier hyperparameter grids for DefaultBackend.tune()
# ---------------------------------------------------------------------------

_PARAM_GRIDS: dict[str, dict] = {
    "RandomForestClassifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": ["balanced", "balanced_subsample"],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": ["balanced"],
        "criterion": ["gini", "entropy"],
    },
    "XGBClassifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "scale_pos_weight": [1, 5, 10],
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.5, 1.0, 1.5],
    },
}

