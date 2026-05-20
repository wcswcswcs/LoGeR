#!/usr/bin/env python3
"""Audit available historical post-zeropower basis traces for ACL2 v19 Track D.

This is a lightweight linear-basis diagnostic. It summarizes landed
`basis_projection_coefficients.csv` trace artifacts, computes cosine-style
similarities between available action summaries, and records whether there is
enough run diversity to justify building learned-basis candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


NUMERIC_COLUMNS = (
    "action_delta_norm",
    "committed_delta_norm",
    "cos_action_to_continuity_basis",
    "cos_committed_to_continuity_basis",
    "cos_short_to_long",
    "native_delta_norm",
    "short_delta_norm",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: object) -> float:
    try:
        if value in ("", None, "nan", "NaN", "None"):
            return float("nan")
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _json_default(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _trace_id(path: Path) -> str:
    return path.parent.name


def _candidate_from_trace(trace_id: str) -> str:
    for marker in ("PZBASIS_", "AUXGEO_TRUE_", "DLTRUE_"):
        idx = trace_id.find(marker)
        if idx >= 0:
            parts = trace_id[idx:].split("_chunk", 1)
            return parts[0]
    return trace_id


def _vector(rows: List[Dict[str, str]]) -> np.ndarray:
    values: List[float] = []
    for row in sorted(rows, key=lambda r: (int(float(r.get("layer", 0))), str(r.get("branch", "")))):
        for col in NUMERIC_COLUMNS:
            value = _to_float(row.get(col))
            values.append(0.0 if not math.isfinite(value) else value)
    return np.asarray(values, dtype=np.float64)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float("nan")
    aa = a[:n]
    bb = b[:n]
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-diverse-traces", type=int, default=6)
    args = parser.parse_args()

    trace_root = Path(args.trace_root)
    out_dir = Path(args.out_dir)
    coeff_files = sorted(trace_root.glob("*/basis_projection_coefficients.csv"))

    basis_rows: List[Dict[str, object]] = []
    vectors: Dict[str, np.ndarray] = {}
    for coeff in coeff_files:
        rows = _read_csv(coeff)
        if not rows:
            continue
        trace = _trace_id(coeff)
        candidate = _candidate_from_trace(trace)
        vec = _vector(rows)
        vectors[trace] = vec
        layer_branch_count = len(rows)
        action_norms = [_to_float(row.get("action_delta_norm")) for row in rows]
        committed_norms = [_to_float(row.get("committed_delta_norm")) for row in rows]
        continuity = [_to_float(row.get("cos_action_to_continuity_basis")) for row in rows]
        basis_rows.append(
            {
                "trace_id": trace,
                "candidate_id": candidate,
                "basis_source_file": str(coeff),
                "layer_branch_count": layer_branch_count,
                "vector_dim": int(vec.shape[0]),
                "action_delta_norm_mean": float(np.nanmean(action_norms)) if action_norms else float("nan"),
                "action_delta_norm_max": float(np.nanmax(action_norms)) if action_norms else float("nan"),
                "committed_delta_norm_mean": float(np.nanmean(committed_norms)) if committed_norms else float("nan"),
                "cos_action_to_continuity_mean": float(np.nanmean(continuity)) if continuity else float("nan"),
                "usable_for_learned_basis": True,
            }
        )

    cos_rows: List[Dict[str, object]] = []
    traces = sorted(vectors)
    for i, left in enumerate(traces):
        for right in traces[i:]:
            cos_rows.append({"trace_a": left, "trace_b": right, "cosine": _cos(vectors[left], vectors[right])})

    enough_diversity = len(traces) >= int(args.min_diverse_traces)
    summary = {
        "trace_count": len(traces),
        "basis_row_count": len(basis_rows),
        "min_diverse_traces": int(args.min_diverse_traces),
        "enough_diversity_for_pls_basis": enough_diversity,
        "candidate_builder_allowed": enough_diversity,
        "diagnostic_only": True,
        "decision": (
            "enough traces for learned-basis candidate construction"
            if enough_diversity
            else "insufficient landed true-action trace diversity; do not fabricate learned basis candidates"
        ),
    }

    _write_csv(out_dir / "historical_basis_summary.csv", basis_rows)
    _write_json(out_dir / "historical_basis_summary.json", basis_rows)
    _write_csv(out_dir / "basis_cosine_matrix.csv", cos_rows)
    _write_json(out_dir / "basis_cosine_matrix.json", cos_rows)
    _write_json(out_dir / "historical_basis_fit_summary.json", summary)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if cos_rows:
            mat = np.zeros((len(traces), len(traces)), dtype=np.float64)
            for row in cos_rows:
                i = traces.index(str(row["trace_a"]))
                j = traces.index(str(row["trace_b"]))
                mat[i, j] = mat[j, i] = _to_float(row["cosine"])
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(traces)))
            ax.set_yticks(range(len(traces)))
            ax.set_xticklabels([_candidate_from_trace(t) for t in traces], rotation=30, ha="right")
            ax.set_yticklabels([_candidate_from_trace(t) for t in traces])
            fig.colorbar(im, ax=ax, label="cosine")
            fig.tight_layout()
            fig.savefig(out_dir / "basis_cosine_matrix.png", dpi=160)
            plt.close(fig)
    except Exception as exc:  # pragma: no cover
        _write_json(out_dir / "plot_error.json", {"error": repr(exc)})

    lines = [
        "# ACL2 v19 Historical Basis Fit",
        "",
        "This is an offline diagnostic over landed true-action trace artifacts.",
        "",
        f"- Trace count: `{summary['trace_count']}`",
        f"- Minimum diverse traces requested: `{summary['min_diverse_traces']}`",
        f"- Enough diversity for PLS/learned-basis candidate construction: `{str(summary['enough_diversity_for_pls_basis']).lower()}`",
        f"- Decision: `{summary['decision']}`",
        "",
        "No learned-basis candidate is generated unless trace diversity is sufficient.",
        "",
    ]
    (out_dir / "historical_basis_fit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
