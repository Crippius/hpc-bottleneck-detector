"""
SHAP Explanation for a Single Held-Out Window

Retrains one LOO fold (held-out app, one bottleneck class) and computes SHAP
values for a single window, explaining that one prediction rather than the
model overall (unlike feature_importances_ used elsewhere).

Usage
-----
    python scripts/analysis/shap_window_explain.py \\
        --classifier xgboost --classifier-config configs/classifiers/xgboost.yaml \\
        --bottleneck-type COMPUTE_UNDERUTILIZATION \\
        --held-out-app 48283 --window-id 48283_w6 \\
        --output results/disagreement/shap_compute_underutilization_48283_w6.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from loo_cross_validation import _extract_test_features_and_labels, DATA_DIR
from hpc_bottleneck_detector.ml.backends.config import build_classifier
from hpc_bottleneck_detector.ml.backends.default_backend import _merge_app_y
from hpc_bottleneck_detector.ml.backends.default_trainer import DefaultTrainer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classifier", choices=["xgboost", "rf"], required=True)
    p.add_argument("--classifier-config", required=True, dest="classifier_config")
    p.add_argument("--bottleneck-type", required=True, dest="bottleneck_type")
    p.add_argument("--held-out-app", required=True, dest="held_out_app",
                    help="Job ID of the app to exclude from training and explain a window from.")
    p.add_argument("--window-id", required=True, dest="window_id",
                    help="Window ID to explain, e.g. 48283_w6.")
    p.add_argument("--window-size", type=int, default=12, dest="window_size")
    p.add_argument("--step-size", type=int, default=12, dest="step_size")
    p.add_argument("--severity-threshold", type=float, default=0.0, dest="severity_threshold")
    p.add_argument("--top-n", type=int, default=15, dest="top_n")
    p.add_argument("--output", default=None, help="Optional path to save the SHAP table as CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    csv_paths = sorted(DATA_DIR.rglob("*.csv"))
    held_out_path = next((p for p in csv_paths if p.stem == args.held_out_app), None)
    if held_out_path is None:
        raise SystemExit(f"No labelled CSV found for app '{args.held_out_app}' under {DATA_DIR}")
    train_paths = [p for p in csv_paths if p != held_out_path]

    print(f"[INFO] Training {args.classifier} on {len(train_paths)} apps "
          f"(held out: {args.held_out_app}), class={args.bottleneck_type}")

    all_X, all_y = [], []
    for p in train_paths:
        X, y_dict = _extract_test_features_and_labels(
            p, args.window_size, args.step_size, args.severity_threshold
        )
        all_X.append(X)
        all_y.append(y_dict)

    X_tr = pd.concat(all_X).fillna(0.0)
    y_tr = _merge_app_y(all_y)
    if args.bottleneck_type not in y_tr:
        raise SystemExit(f"No valid labels for {args.bottleneck_type} across training apps.")
    y_tr = {args.bottleneck_type: y_tr[args.bottleneck_type]}  # train only the target class

    trainer = DefaultTrainer(classifier=build_classifier(args.classifier, args.classifier_config))
    backend = trainer.from_preextracted_features(X_tr, y_tr)

    if args.bottleneck_type not in backend._models:
        raise SystemExit(f"No classifier trained for {args.bottleneck_type} (check labels/features).")
    clf = backend._models[args.bottleneck_type]
    feature_cols = backend._feature_cols[args.bottleneck_type]
    print(f"[INFO] Trained on {len(feature_cols)} selected features.")

    print(f"[INFO] Extracting features for held-out app {args.held_out_app}...")
    X_test, _ = _extract_test_features_and_labels(
        held_out_path, args.window_size, args.step_size, args.severity_threshold
    )
    if args.window_id not in X_test.index:
        raise SystemExit(
            f"Window '{args.window_id}' not found for app {args.held_out_app}. "
            f"Available: {list(X_test.index)[:10]}..."
        )
    x_window = X_test.reindex(columns=feature_cols, fill_value=0.0).loc[[args.window_id]]

    prob = float(clf.predict_proba(x_window)[0, 1])
    print(f"[INFO] Predicted probability for {args.window_id}: {prob:.4f}")

    print("[INFO] Computing SHAP values (TreeExplainer)...")
    import shap
    explainer = shap.TreeExplainer(clf)
    explanation = explainer(x_window)

    values = explanation.values[0]
    if values.ndim > 1:  # some sklearn RF/shap version combos return per-class columns
        values = values[:, -1]
    base_value = explanation.base_values[0]
    base_value = float(base_value[-1]) if hasattr(base_value, "__len__") else float(base_value)

    result = pd.DataFrame({
        "feature_name": feature_cols,
        "feature_value": x_window.iloc[0].values,
        "shap_value": values,
    })
    result["abs_shap"] = result["shap_value"].abs()
    result = result.sort_values("abs_shap", ascending=False).drop(columns="abs_shap").head(args.top_n)

    print(f"\n[INFO] Base value (expected prob. over training set): {base_value:.4f}")
    print(f"[INFO] Predicted probability: {prob:.4f}")
    print(f"\nTop {args.top_n} SHAP contributions for {args.window_id} "
          f"({args.classifier}, {args.bottleneck_type}):")
    print(result.to_string(index=False))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = pd.DataFrame([{
            "classifier": args.classifier, "bottleneck_type": args.bottleneck_type,
            "held_out_app": args.held_out_app, "window_id": args.window_id,
            "base_value": base_value, "predicted_prob": prob,
        }])
        meta.to_csv(out.with_name(f"{out.stem}_meta{out.suffix}"), index=False)
        result.to_csv(out, index=False)
        print(f"[INFO] Saved to: {out}")
