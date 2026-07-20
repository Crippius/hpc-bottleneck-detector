"""
Feature Importance Analysis

Checks for the following things:
  - Heuristic agreement of ML models
  - RF vs XGBoost feature agreement (Ruzicka)
  - Importance concentration
  - Statistic-family importance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpc_bottleneck_detector.strategies.strategy_tree import StrategyTree
from hpc_bottleneck_detector.strategies.property_node import PropertyNode
from hpc_bottleneck_detector.output.models import BottleneckType

RESULTS_DIR = ROOT / "results" / "feature_importance"
STRATEGY_DIR = ROOT / "configs" / "strategies" / "persyst_strategy"
MODEL_NAMES = ["rf", "xgboost"]
TOP_K_MASS = 10


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_importance(model_name: str) -> dict:
    path = RESULTS_DIR / f"feature_importance_stats_{model_name}.json"
    return json.loads(path.read_text())


def metric_key(group: str, metric: str, trace: str | None) -> str:
    """Mirrors DataManager's column-naming convention (data/manager.py)."""
    key = f"{group}_{metric}"
    if trace:
        key += f"_{trace.replace(' ', '_')}"
    return key


def feature_base(feature: str) -> str:
    return feature.split("__")[0]


KNOWN_GROUPS = ["load_imbalance", "cpu", "cache", "memory", "energy"]


def strip_group_prefix(name: str) -> str:
    """Presentation-only: drop the redundant leading '{group}_' (cpu_, cache_, ...)."""
    for group in KNOWN_GROUPS:
        prefix = f"{group}_"
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def truncate_label(text: str, max_len: int = 26) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def disambiguate_labels(names: list[str], max_len: int = 26) -> list[str]:
    truncated = [truncate_label(n, max_len) for n in names]
    groups: dict[str, set[str]] = {}
    for n, t in zip(names, truncated):
        groups.setdefault(t, set()).add(n)
    return [n if len(groups[t]) > 1 else t for n, t in zip(names, truncated)]


def short_stat_label(feature: str) -> str:
    parts = feature.split("__")[1:]
    if not parts:
        return feature
    stat = parts[0]
    rest = "__".join(parts[1:])

    if stat == "quantile":
        q = rest.replace("q_", "")
        return f"q{q}"
    if stat == "agg_linear_trend":
        chunk = rest.split("chunk_len_")[-1].split("__")[0] if "chunk_len_" in rest else "?"
        return f"trend(w={chunk})"
    if stat == "agg_autocorrelation":
        agg = "mean"
        for cand in ("mean", "median", "var"):
            if f'"{cand}"' in rest:
                agg = cand
                break
        lag = rest.split("maxlag_")[-1] if "maxlag_" in rest else "?"
        return f"autocorr({agg},lag{lag})"
    if stat == "absolute_sum_of_changes":
        return "abs_sum_changes"
    if stat == "standard_deviation":
        return "std"
    return stat


# ---------------------------------------------------------------------------
# Heuristic tree -> bottleneck_type / required-metric mapping
# ---------------------------------------------------------------------------

def _leaf_bottlenecks(node: PropertyNode) -> set[BottleneckType]:
    if node.is_leaf():
        diag = node.get_diagnosis(source="", triggered_metrics=[])
        if diag.bottleneck_type in (BottleneckType.NONE, BottleneckType.UNKNOWN):
            return set()
        return {diag.bottleneck_type}
    return _leaf_bottlenecks(node.get_child(True)) | _leaf_bottlenecks(node.get_child(False))


def build_heuristic_map() -> dict[str, dict]:
    """bottleneck_type.value -> {"trees": [...], "metrics": {metric_key, ...}}"""
    result: dict[str, dict] = {}
    for yaml_path in sorted(STRATEGY_DIR.glob("*.yaml")):
        if yaml_path.name == "families.yaml":
            continue
        tree = StrategyTree.load_from_yaml(str(yaml_path))
        bottlenecks = _leaf_bottlenecks(tree.root_node)
        if not bottlenecks:
            continue
        metrics = {
            metric_key(m.get("group", ""), m.get("metric", ""), m.get("trace"))
            for m in tree.get_required_metrics()
        }
        for bt in bottlenecks:
            entry = result.setdefault(bt.value, {"trees": [], "metrics": set()})
            entry["trees"].append(tree.tree_name)
            entry["metrics"] |= metrics
    return result


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def heuristic_agreement(models: dict[str, dict], heuristic_map: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("HEURISTIC AGREEMENT")
    print("=" * 78)

    bottlenecks = sorted(set().union(*(m["per_bottleneck"].keys() for m in models.values())))
    for bt in bottlenecks:
        h = heuristic_map.get(bt)
        print(f"\n--- {bt} ---")
        if not h:
            print("  (no heuristic tree assigns this bottleneck type)")
            continue
        print(f"  heuristic tree(s): {', '.join(h['trees'])}")
        print(f"  heuristic metric(s): {sorted(h['metrics'])}")

        for model_name in MODEL_NAMES:
            top = models[model_name]["per_bottleneck"].get(bt, {}).get("top_features", [])[:TOP_K_MASS]
            if not top:
                print(f"  {model_name}: no data")
                continue
            total_mass = sum(f["importance"] for f in top)
            matched_mass = sum(f["importance"] for f in top if feature_base(f["feature"]) in h["metrics"])
            pct = (matched_mass / total_mass * 100) if total_mass else 0.0
            print(f"  {model_name}: matched {matched_mass:.3f} of top-{TOP_K_MASS} mass {total_mass:.3f}  ({pct:.1f}%)")
            for f in top[:5]:
                mark = "OK " if feature_base(f["feature"]) in h["metrics"] else "-- "
                print(f"      {mark}{f['feature']:<75} {f['importance']:.4f}")


def metric_level_importance(top_features: list[dict]) -> dict[str, float]:
    agg: dict[str, float] = {}
    for f in top_features:
        base = feature_base(f["feature"])
        agg[base] = agg.get(base, 0.0) + f["importance"]
    return agg


def ruzicka(a: dict[str, float], b: dict[str, float]) -> float:
    """Weighted Jaccard: sum(min(a_i, b_i)) / sum(max(a_i, b_i)) over the key union."""
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return num / den if den else 0.0


def rf_xgb_agreement(models: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("RF vs XGBOOST FEATURE AGREEMENT (metric-level)")
    print("=" * 78)

    rf_bt = models["rf"]["per_bottleneck"]
    xgb_bt = models["xgboost"]["per_bottleneck"]
    bottlenecks = sorted(set(rf_bt) & set(xgb_bt))

    ruz_values = []
    for bt in bottlenecks:
        rf_imp = metric_level_importance(rf_bt[bt]["top_features"])
        xgb_imp = metric_level_importance(xgb_bt[bt]["top_features"])

        rf_ranked = sorted(rf_imp, key=rf_imp.get, reverse=True)
        xgb_ranked = sorted(xgb_imp, key=xgb_imp.get, reverse=True)

        ruz = ruzicka(rf_imp, xgb_imp)
        ruz_values.append(ruz)

        print(f"\n--- {bt} ---  Ruzicka={ruz:.2f}")
        merged = sorted(
            set(rf_ranked[:10]) | set(xgb_ranked[:10]),
            key=lambda m: -max(rf_imp.get(m, 0.0), xgb_imp.get(m, 0.0)),
        )
        for m in merged:
            print(f"    {m:<70} RF={rf_imp.get(m, 0.0):.3f}  XGB={xgb_imp.get(m, 0.0):.3f}")

    print(f"\nGlobal mean Ruzicka across {len(bottlenecks)} bottlenecks: "
          f"{sum(ruz_values) / len(ruz_values):.3f}")


def concentration(models: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("IMPORTANCE CONCENTRATION")
    print("=" * 78)

    for model_name in MODEL_NAMES:
        print(f"\n--- {model_name} ---")
        for bt, entry in models[model_name]["per_bottleneck"].items():
            imps = [f["importance"] for f in entry["top_features"]]
            top1 = imps[0] if imps else 0.0
            top5 = sum(imps[:5])
            hhi = sum(i ** 2 for i in imps)
            eff_n = (1 / hhi) if hhi else float("nan")
            flag = "  <-- single-feature dominance" if top1 > 0.5 else ""
            print(f"  {bt:<30} n_features={entry['n_features']:<4} "
                  f"top1={top1:.3f}  top5={top5:.3f}  eff_n(approx)={eff_n:.1f}{flag}")


def expansion_candidates(models: dict[str, dict], heuristic_map: dict[str, dict], top_n: int = 5) -> None:
    """Metrics with real importance that no heuristic tree for this bottleneck checks."""
    print("\n" + "=" * 78)
    print("CANDIDATE HEURISTIC EXTENSIONS (unmatched metrics, ranked by importance)")
    print("=" * 78)

    bottlenecks = sorted(set().union(*(m["per_bottleneck"].keys() for m in models.values())))
    for bt in bottlenecks:
        h = heuristic_map.get(bt, {"metrics": set()})
        print(f"\n--- {bt} ---")
        for model_name in MODEL_NAMES:
            imp = metric_level_importance(models[model_name]["per_bottleneck"].get(bt, {}).get("top_features", []))
            unmatched = sorted(
                ((m, v) for m, v in imp.items() if m not in h["metrics"]),
                key=lambda kv: -kv[1],
            )[:top_n]
            print(f"  {model_name}:")
            for m, v in unmatched:
                print(f"    {v:.3f}  {m}")


def stat_family_comparison(models: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("TSFRESH STATISTIC-FAMILY COMPARISON")
    print("=" * 78)

    rf_fam = {r["stat_base"]: r["mean_importance"] for r in models["rf"]["statistics"]["by_family"]}
    xgb_fam = {r["stat_base"]: r["mean_importance"] for r in models["xgboost"]["statistics"]["by_family"]}
    families = sorted(set(rf_fam) | set(xgb_fam), key=lambda f: -max(rf_fam.get(f, 0.0), xgb_fam.get(f, 0.0)))

    print()
    for fam in families:
        print(f"  {fam:<25} RF={rf_fam.get(fam, 0.0):.4f}  XGB={xgb_fam.get(fam, 0.0):.4f}")


def main() -> None:
    models = {name: load_importance(name) for name in MODEL_NAMES}
    heuristic_map = build_heuristic_map()

    heuristic_agreement(models, heuristic_map)
    rf_xgb_agreement(models)
    concentration(models)
    expansion_candidates(models, heuristic_map)
    stat_family_comparison(models)


if __name__ == "__main__":
    main()
