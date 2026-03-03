#!/usr/bin/env python3
"""
visualize_strategy_tree.py
==========================
Interactive GUI visualiser for HPC Bottleneck Detector strategy-tree YAML files.

Generates a self-contained HTML file and opens it in the default browser.

Usage
-----
  # Auto-detect all YAMLs under configs/strategies/
  python scripts/visualize_strategy_tree.py

  # Single file
  python scripts/visualize_strategy_tree.py configs/strategies/persyst_strategy/compute_bound_branch.yaml

  # A whole directory
  python scripts/visualize_strategy_tree.py configs/strategies/persyst_strategy/

  # Explicit list
  python scripts/visualize_strategy_tree.py a.yaml b.yaml c.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import textwrap
import webbrowser

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required: run  pip install pyyaml  then retry."
    )

# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _format_threshold(threshold) -> str:
    """Turn a scalar or benchmark-dict threshold into a human-readable string."""
    if isinstance(threshold, dict):
        bm   = threshold.get("benchmark", "?")
        frac = threshold.get("fraction", "?")
        fb   = threshold.get("fallback", "N/A")
        return f"{frac*100:.0f}% of benchmark({bm})  [fallback {fb}]"
    try:
        return str(round(float(threshold), 6))
    except (TypeError, ValueError):
        return str(threshold)


def _parse_node(raw: dict, parent_id: str | None, edge_label: str,
                nodes: list, edges: list) -> None:
    """Recursively walk a raw YAML node and populate *nodes* and *edges*."""

    node_id = raw.get("node_id", f"node_{len(nodes)}")

    diagnosis  = raw.get("diagnosis")
    is_leaf    = diagnosis is not None

    # ---- build the node record ----
    record: dict = {"id": node_id}

    if is_leaf:
        btype = diagnosis.get("bottleneck_type", "UNKNOWN")
        record["kind"]            = "leaf"
        record["bottleneck_type"] = btype
        record["severity_formula"] = diagnosis.get("severity_formula", "")
        record["threshold"]        = _format_threshold(diagnosis.get("threshold", ""))
        record["confidence"]       = diagnosis.get("confidence", "")
        rec = diagnosis.get("recommendation") or ""
        record["recommendation"]   = textwrap.dedent(rec).strip()
        record["label"]            = btype if btype != "NONE" else "✓  OK"
        record["description"]      = raw.get("description", "")
    else:
        metric    = raw.get("metric", {})
        threshold = raw.get("threshold")
        operator  = raw.get("operator", "")
        agg       = raw.get("aggregation", "mean")
        mtype     = metric.get("type", "simple")  # simple | sum | ratio
        record["kind"]               = "decision"
        record["description"]        = raw.get("description", "")
        record["metric_type"]        = mtype
        record["metric_description"] = _format_metric_description(metric)
        record["aggregation"]        = agg
        record["operator"]           = operator
        record["threshold"]          = _format_threshold(threshold)
        # Short label shown inside the node box
        record["label"] = _format_metric_label(metric) or node_id

    nodes.append(record)

    if parent_id is not None:
        edges.append({"source": parent_id, "target": node_id,
                      "label": edge_label})

    if not is_leaf:
        operator  = raw.get("operator", ">")
        threshold = raw.get("threshold")
        thresh_str = _format_threshold(threshold)

        true_label  = f"{operator} {thresh_str}"
        false_label = _negate_operator(operator) + f" {thresh_str}"

        if_true  = raw.get("if_true")
        if_false = raw.get("if_false")

        if if_true:
            _parse_node(if_true,  node_id, true_label,  nodes, edges)
        if if_false:
            _parse_node(if_false, node_id, false_label, nodes, edges)


def _format_metric_label(metric: dict) -> str:
    """Short one-line label for a metric (simple, sum, or ratio)."""
    mtype = metric.get("type")
    if not mtype:
        # Simple metric: use trace, fallback to metric name
        return metric.get("trace", metric.get("metric", "?"))
    if mtype == "sum":
        parts = [_format_metric_label(op) for op in metric.get("operands", [])]
        return " + ".join(parts)
    if mtype == "ratio":
        num = _format_metric_label(metric.get("numerator", {}))
        den = _format_metric_label(metric.get("denominator", {}))
        return f"({num}) / ({den})"
    return str(metric)


def _format_metric_description(metric: dict, indent: int = 0) -> str:
    """Multi-line human-readable description of a metric expression."""
    pad  = "  " * indent
    mtype = metric.get("type")
    if not mtype:
        g = metric.get("group", "")
        m = metric.get("metric", "")
        t = metric.get("trace", "")
        return f"{pad}{g} / {m} / {t}"
    if mtype == "sum":
        parts = [_format_metric_description(op, indent + 1)
                 for op in metric.get("operands", [])]
        return f"{pad}SUM(\n" + ",\n".join(parts) + f"\n{pad})"
    if mtype == "ratio":
        num = _format_metric_description(metric.get("numerator",  {}), indent + 1)
        den = _format_metric_description(metric.get("denominator", {}), indent + 1)
        return (
            f"{pad}RATIO(\n"
            f"{pad}  numerator:\n{num}\n"
            f"{pad}  denominator:\n{den}\n"
            f"{pad})"
        )
    return f"{pad}{metric}"


def _negate_operator(op: str) -> str:
    mapping = {">": "≤", ">=": "<", "<": "≥", "<=": ">", "==": "≠", "!=": "="}
    return mapping.get(op.strip(), f"NOT {op}")


def parse_tree_yaml(path: pathlib.Path) -> dict:
    """Return a dict with tree metadata + flat node/edge lists."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    nodes: list = []
    edges: list = []

    root = raw.get("root")
    if root:
        _parse_node(root, None, "", nodes, edges)

    return {
        "tree_name":        raw.get("tree_name", path.stem),
        "description":      raw.get("description", ""),
        "required_metrics": raw.get("required_metrics", []),
        "thresholds":       raw.get("thresholds", {}),
        "nodes":            nodes,
        "edges":            edges,
    }


# ---------------------------------------------------------------------------
# HTML template loader
# ---------------------------------------------------------------------------

_HTML_TEMPLATE_PATH = pathlib.Path(__file__).with_suffix(".html")


def _load_html_template() -> str:
    if not _HTML_TEMPLATE_PATH.is_file():
        sys.exit(
            f"HTML template not found: {_HTML_TEMPLATE_PATH}\n"
            "Make sure visualize_strategy_tree.html is in the same directory "
            "as this script."
        )
    return _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")


# Keep a sentinel we can find-and-replace for legacy reference
_TREES_PLACEHOLDER = "/*TREES_JSON*/null/*END*/"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_yaml_files(args: list[str]) -> list[pathlib.Path]:
    """Resolve CLI args to a sorted list of YAML paths."""
    paths: list[pathlib.Path] = []

    if not args:
        # Auto-detect: scan configs/strategies/
        base = _REPO_ROOT / "configs" / "strategies"
        if base.is_dir():
            paths = sorted(base.rglob("*.yaml")) + sorted(base.rglob("*.yml"))
        if not paths:
            sys.exit(
                "No YAML files found under configs/strategies/. "
                "Please pass file paths explicitly."
            )
        return paths

    for arg in args:
        p = pathlib.Path(arg)
        if not p.is_absolute():
            p = pathlib.Path.cwd() / p
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.yaml")) + sorted(p.rglob("*.yml")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"Warning: {arg!r} is neither a file nor a directory — skipped.",
                  file=sys.stderr)

    if not paths:
        sys.exit("No YAML files found.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive viewer for HPC Bottleneck Detector strategy trees."
    )
    parser.add_argument(
        "paths", nargs="*",
        help="YAML file(s) or director(ies) to visualise. "
             "Defaults to configs/strategies/ relative to the repo root."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Write the HTML to this file instead of a temp file."
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Generate the HTML without opening a browser."
    )
    ns    = parser.parse_args()
    files = collect_yaml_files(ns.paths)

    print(f"Loading {len(files)} strategy tree(s)…")
    trees = []
    for f in files:
        try:
            trees.append(parse_tree_yaml(f))
            print(f"  ✓  {f.name}  ({len(trees[-1]['nodes'])} nodes)")
        except Exception as exc:
            print(f"  ✗  {f.name}: {exc}", file=sys.stderr)

    if not trees:
        sys.exit("No trees could be parsed.")

    # Load template and inject JSON
    trees_json = json.dumps(trees, ensure_ascii=False, indent=2)
    html = _load_html_template().replace(
        "/*TREES_JSON*/null/*END*/",
        f"/*TREES_JSON*/{trees_json}/*END*/"
    )

    if ns.output:
        out_path = pathlib.Path(ns.output)
        out_path.write_text(html, encoding="utf-8")
        print(f"\nHTML written to: {out_path.resolve()}")
        if not ns.no_browser:
            webbrowser.open(out_path.resolve().as_uri())
    else:
        # Write to a temp file so the browser can load local assets
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", prefix="strategy_tree_",
            delete=False, mode="w", encoding="utf-8"
        )
        tmp.write(html)
        tmp.close()
        print(f"\nTemp file: {tmp.name}")
        if not ns.no_browser:
            webbrowser.open(pathlib.Path(tmp.name).as_uri())
            print("Opened in browser.")


if __name__ == "__main__":
    main()
