#!/usr/bin/env python3
"""
export_trees_pdf.py
===================
Export HPC Bottleneck Detector strategy trees as PDFs using headless Chrome.
Renders the existing HTML viewer as-is (same style, same LR layout).
Produces one PDF per group:
  memory_bound_analysis.pdf
  compute_bound_analysis.pdf
  load_imbalance_analysis.pdf

Usage
-----
  python scripts/analysis/export_trees_pdf.py [output_dir]
  # Default output_dir: results/trees/
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from visualize_strategy_tree import (
    collect_yaml_files,
    parse_tree_yaml,
    group_and_merge_trees,
)

_REPO_ROOT      = pathlib.Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUTPUT = _REPO_ROOT / "results" / "trees"
_HTML_TEMPLATE  = pathlib.Path(__file__).with_name("visualize_strategy_tree.html")

# A3 landscape in pixels at 96 dpi: 420mm × 297mm
_PAGE_W_PX = 1587
_PAGE_H_PX = 1122

_PRINT_CSS = f"""
<style id="print-overrides">
  @page {{ size: A3 landscape; margin: 0mm; }}
  #tab-bar, #detail {{ display: none !important; }}
  #main {{ height: 100vh !important; }}
  html, body {{ overflow: hidden !important; }}
</style>
<script id="print-fit">
window.addEventListener('load', function() {{
  var pad = 20;
  var svgEl = document.getElementById('tree-svg');
  var W = svgEl.clientWidth, H = svgEl.clientHeight;
  var nodeEls = Array.from(document.querySelectorAll('g.canvas g.node'));
  if (!nodeEls.length) return;
  var xs = [], ys = [];
  nodeEls.forEach(function(el) {{
    var t = el.getAttribute('transform') || '';
    var i = t.indexOf('('), j = t.indexOf(')');
    if (i < 0 || j < 0) return;
    var parts = t.slice(i + 1, j).split(',');
    xs.push(+parts[0]);
    ys.push(+parts[1]);
  }});
  if (!xs.length) return;
  var minX = Math.min.apply(null, xs);
  var maxX = Math.max.apply(null, xs) + NODE_W;
  var minY = Math.min.apply(null, ys);
  var maxY = Math.max.apply(null, ys) + NODE_H;
  var sc = Math.min((W - pad * 2) / (maxX - minX), (H - pad * 2) / (maxY - minY));
  var tx = pad - minX * sc + (W - pad * 2 - (maxX - minX) * sc) / 2;
  var ty = pad - minY * sc + (H - pad * 2 - (maxY - minY) * sc) / 2;
  d3.select('g.canvas').attr('transform', 'translate(' + tx + ',' + ty + ') scale(' + sc + ')');
}});
</script>"""


def make_print_html(tree: dict) -> str:
    html = _HTML_TEMPLATE.read_text(encoding="utf-8")
    html = html.replace(
        "/*TREES_JSON*/null/*END*/",
        f"/*TREES_JSON*/{json.dumps([tree], ensure_ascii=False)}/*END*/",
    )
    html = html.replace("</head>", _PRINT_CSS + "\n</head>")
    return html


def export_pdf(tree: dict, out_dir: pathlib.Path) -> pathlib.Path | None:
    slug     = tree["tree_name"].lower().replace(" ", "_").replace("/", "_")
    pdf_path = out_dir / f"{slug}.pdf"

    with tempfile.NamedTemporaryFile(
        suffix=".html", prefix="tree_print_",
        delete=False, mode="w", encoding="utf-8",
    ) as tmp:
        tmp.write(make_print_html(tree))
        tmp_path = pathlib.Path(tmp.name)

    try:
        r = subprocess.run(
            [
                "google-chrome", "--headless", "--disable-gpu",
                "--no-sandbox", "--disable-dev-shm-usage",
                f"--print-to-pdf={pdf_path}",
                "--no-pdf-header-footer",
                f"--window-size={_PAGE_W_PX},{_PAGE_H_PX}",
                tmp_path.as_uri(),
            ],
            capture_output=True, text=True, timeout=90,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if r.returncode != 0:
        print(f"  ERROR ({tree['tree_name']}): {r.stderr.strip()}", file=sys.stderr)
        return None
    return pdf_path


def main() -> None:
    out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_yaml_files([])
    print(f"Loading {len(files)} strategy tree(s)…")
    parsed = []
    for f in files:
        try:
            t = parse_tree_yaml(f)
            if not t["nodes"]:
                continue
            parsed.append((f, t))
            print(f"  +  {f.name}")
        except Exception as exc:
            print(f"  X  {f.name}: {exc}", file=sys.stderr)

    if not parsed:
        sys.exit("No trees could be parsed.")

    trees = group_and_merge_trees(parsed)
    print(f"\nExporting {len(trees)} PDF(s) → {out_dir}/")

    for tree in trees:
        pdf = export_pdf(tree, out_dir)
        if pdf:
            print(f"  +  {pdf.name}  ({len(tree['nodes'])} nodes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
