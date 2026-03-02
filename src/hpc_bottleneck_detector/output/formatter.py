"""
Output Formatter

Renders a list of :class:`~hpc_bottleneck_detector.output.models.WindowDiagnosis`
objects to the console, a JSON file, or a CSV file.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import List, Optional

from .models import WindowDiagnosis


def format_results(
    window_diagnoses: List[WindowDiagnosis],
    fmt: str = "print",
    save_path: Optional[str] = None,
) -> str:
    """
    Render *window_diagnoses* according to *fmt*.

    Args:
        window_diagnoses: Results produced by the orchestrator.
        fmt:              ``'print'`` | ``'json'`` | ``'csv'``
        save_path:        When provided the output is also written to this path.

    Returns:
        The rendered string (useful for testing / logging).
    """
    fmt = fmt.lower().strip()

    if fmt == "json":
        output = _to_json(window_diagnoses)
    elif fmt == "csv":
        output = _to_csv(window_diagnoses)
    else:
        output = _to_print(window_diagnoses)

    # ── write to file if requested ─────────────────────────────────────────
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")

    # ── print to stdout for 'print' fmt ───────────────────────────────────
    if fmt == "print":
        sys.stdout.write(output)

    return output


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_print(windows: List[WindowDiagnosis]) -> str:
    lines: List[str] = []

    lines.append("=" * 70)
    lines.append(f"  HPC BOTTLENECK DETECTOR — {len(windows)} window(s)")
    lines.append("=" * 70)

    for wd in windows:
        status = "BOTTLENECK" if wd.has_bottlenecks() else "HEALTHY"
        lines.append(
            f"\n[Window {wd.window_index:>3}]  "
            f"intervals {wd.start_interval}–{wd.end_interval}  "
            f"({status})"
        )
        if not wd.diagnoses:
            lines.append("  (no diagnoses)")
            continue

        for diag in wd.diagnoses:
            bt = diag.bottleneck_type.value
            sev = f"{diag.severity_score:.2f}"
            conf = f"{diag.confidence:.2f}"
            src = f"[{diag.source}]" if diag.source else ""
            lines.append(f"  • {bt:<35} severity={sev}  confidence={conf}  {src}")
            if diag.recommendation:
                # indent multi-line recommendations
                rec_lines = diag.recommendation.strip().splitlines()
                lines.append(f"    Recommendation: {rec_lines[0]}")
                for rl in rec_lines[1:]:
                    lines.append(f"    {rl}")

    lines.append("\n" + "=" * 70 + "\n")
    return "\n".join(lines)


def _to_json(windows: List[WindowDiagnosis]) -> str:
    data = [wd.to_dict() for wd in windows]
    return json.dumps(data, indent=2, ensure_ascii=False)


def _to_csv(windows: List[WindowDiagnosis]) -> str:
    buf = io.StringIO()
    fieldnames = [
        "window_index", "start_interval", "end_interval",
        "has_bottlenecks", "bottleneck_type",
        "severity_score", "confidence", "source", "recommendation",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for wd in windows:
        if not wd.diagnoses:
            writer.writerow({
                "window_index": wd.window_index,
                "start_interval": wd.start_interval,
                "end_interval": wd.end_interval,
                "has_bottlenecks": wd.has_bottlenecks(),
                "bottleneck_type": "",
                "severity_score": "",
                "confidence": "",
                "source": "",
                "recommendation": "",
            })
        else:
            for diag in wd.diagnoses:
                writer.writerow({
                    "window_index": wd.window_index,
                    "start_interval": wd.start_interval,
                    "end_interval": wd.end_interval,
                    "has_bottlenecks": wd.has_bottlenecks(),
                    "bottleneck_type": diag.bottleneck_type.value,
                    "severity_score": round(diag.severity_score, 4),
                    "confidence": round(diag.confidence, 4),
                    "source": diag.source,
                    "recommendation": (diag.recommendation or "").strip(),
                })

    return buf.getvalue()
