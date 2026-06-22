"""
Backend configuration constants
"""

from __future__ import annotations
from pathlib import Path

_CONFIGS_DIR = Path(__file__).parents[4] / "configs" / "classifiers"

_CLASSIFIER_FIXED_PARAMS = {
    "rf":      dict(random_state=42, n_jobs=-1),
    "xgboost": dict(random_state=42, n_jobs=-1, eval_metric="logloss"),
}


def build_classifier(name: str, config_path: str | Path | None = None):
    import yaml
    if config_path is None:
        config_path = _CONFIGS_DIR / f"{'rf' if name == 'rf' else 'xgboost'}_default.yaml"
    with open(config_path) as f:
        params = yaml.safe_load(f)
    params.update(_CLASSIFIER_FIXED_PARAMS[name])
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)
    from xgboost import XGBClassifier
    return XGBClassifier(**params)

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

# 2. Basic + Advanced - adds thresholding, energy, complexity, and frequency
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


