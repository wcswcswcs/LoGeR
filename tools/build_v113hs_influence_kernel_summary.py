#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v113-HS influence-kernel summary from trace CSVs.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--smoke-trace-root", required=True)
    parser.add_argument("--bounded-gla-root", required=True)
    parser.add_argument("--full-mrt-roots", required=True, help="Comma-separated trace dirs.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float | None:
    text = row.get(key, "")
    if text in ("", "None", "nan"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def corr(rows: list[dict[str, str]], x_key: str, y_key: str) -> float | None:
    pairs = []
    for row in rows:
        x = as_float(row, x_key)
        y = as_float(row, y_key)
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 3:
        return None
    arr = np.asarray(pairs, dtype=np.float64)
    if float(np.std(arr[:, 0])) == 0.0 or float(np.std(arr[:, 1])) == 0.0:
        return None
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def tag_rows(rows: list[dict[str, str]], scope: str) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        out = dict(row)
        out["trace_scope"] = scope
        tagged.append(out)
    return tagged


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    diag_dir = results_root / "diagnostics"

    smoke_root = Path(args.smoke_trace_root)
    bounded_gla_root = Path(args.bounded_gla_root)
    full_mrt_roots = [Path(p) for p in args.full_mrt_roots.split(",") if p.strip()]

    gla_rows = tag_rows(read_rows(smoke_root / "hs_gla_state_probe_rows.csv"), "smoke_exact_max12")
    gla_rows += tag_rows(read_rows(bounded_gla_root / "hs_gla_state_probe_rows.csv"), "bounded_max31_localoff")
    local_rows = tag_rows(read_rows(smoke_root / "hs_local_head_semantic_attention_rows.csv"), "smoke_exact_max12")
    mrt_rows: list[dict[str, Any]] = []
    for root in full_mrt_roots:
        mrt_rows += tag_rows(read_rows(root / "hs_mrt_readout_probe_rows.csv"), f"full_mrt:{root.name}")

    write_rows(diag_dir / "hs_gla_state_probe_rows.csv", gla_rows)
    write_rows(diag_dir / "hs_local_head_semantic_attention_rows.csv", local_rows)
    write_rows(diag_dir / "hs_mrt_readout_probe_rows.csv", mrt_rows)

    summary = {
        "trace_sources": {
            "smoke_exact": str(smoke_root),
            "bounded_gla": str(bounded_gla_root),
            "full_mrt": [str(p) for p in full_mrt_roots],
        },
        "row_counts": {
            "gla": len(gla_rows),
            "local": len(local_rows),
            "mrt": len(mrt_rows),
        },
        "gla": {
            "scope": "smoke exact max12 plus bounded max31 local-off; full GLA norm trace OOM on 22GB GPUs",
            "state_new_norm": stats([v for v in (as_float(r, "state_new_norm") for r in gla_rows) if v is not None]),
            "state_delta_norm": stats([v for v in (as_float(r, "state_delta_norm") for r in gla_rows) if v is not None]),
            "corr_dynamic_state_delta": corr(gla_rows, "chunk_dynamic_mass_mean", "state_delta_norm"),
            "corr_stable_state_delta": corr(gla_rows, "chunk_stable_mass_mean", "state_delta_norm"),
            "corr_boundary_state_delta": corr(gla_rows, "chunk_boundary_mass_mean", "state_delta_norm"),
        },
        "local": {
            "scope": "smoke exact max12 only; F=21 local trace OOM even with sample64 on 22GB GPUs",
            "pose_norm_before": stats([v for v in (as_float(r, "pose_token_output_norm_before") for r in local_rows) if v is not None]),
            "pose_norm_after": stats([v for v in (as_float(r, "pose_token_output_norm_after") for r in local_rows) if v is not None]),
            "kv_value_norm_dynamic": stats([v for v in (as_float(r, "local_kv_value_norm_dynamic") for r in local_rows) if v is not None]),
            "kv_value_norm_stable": stats([v for v in (as_float(r, "local_kv_value_norm_stable") for r in local_rows) if v is not None]),
        },
        "mrt": {
            "scope": "full KITTI 00/02 trace-only with GLA/local probes disabled",
            "predicted_metric_scale": stats([v for v in (as_float(r, "predicted_metric_scale") for r in mrt_rows) if v is not None]),
            "metric_readout_feature_norm": stats([v for v in (as_float(r, "metric_readout_feature_norm") for r in mrt_rows) if v is not None]),
            "corr_semantic_risk_scale": corr(mrt_rows, "chunk_semantic_risk", "predicted_metric_scale"),
            "corr_stable_mass_scale": corr(mrt_rows, "chunk_stable_mass", "predicted_metric_scale"),
            "corr_semantic_risk_feature_norm": corr(mrt_rows, "chunk_semantic_risk", "metric_readout_feature_norm"),
        },
        "decision_hint": {
            "hs_l": "local path is real but full local trace is memory-blocked; use HS-L action only with careful no-extra-norm implementation and full pilot metrics.",
            "hs_a": "GLA state path is traceable on bounded runs; full GLA norm trace is memory-blocked, so HS-A should start with pre-GLA input scaling rather than state-delta norm-dependent control.",
            "hs_m": "MRT has full trace rows and is safe to analyze, but actions remain high-risk because they touch metric scale readout.",
        },
    }
    with (diag_dir / "hs_influence_kernel_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
