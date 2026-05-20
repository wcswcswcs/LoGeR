#!/usr/bin/env python3
"""Create ACL2 v17 Phase 2 basis-proxy audit artifacts.

This is intentionally named "proxy": the landed controller exposes native
delta/post-zeropower routing primitives, but not historical per-tensor
contrastive anchor bases. The report records what was actually run and which
true Phase 2 tensor-basis requirements remain unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List


COEFFICIENTS = {
    "BASIS_01_PROXY_HARM_W0": {
        "basis_name": "BASIS_01_PROXY",
        "coeff_alpha": 1.0,
        "coeff_beta": 0.0,
        "coeff_lambda": 0.0,
        "proxy_primitive": "orthogonal_suppress_w0_rho0",
        "true_tensor_basis": False,
    },
    "BASIS_02_PROXY_HARM_W0_EMA090": {
        "basis_name": "BASIS_02_PROXY",
        "coeff_alpha": 1.0,
        "coeff_beta": 0.0,
        "coeff_lambda": 0.10,
        "proxy_primitive": "orthogonal_suppress_w0_rho0_plus_commit_ema090",
        "true_tensor_basis": False,
    },
    "BASIS_03_PROXY_RHO025_W0": {
        "basis_name": "BASIS_03_PROXY",
        "coeff_alpha": 0.75,
        "coeff_beta": 0.0,
        "coeff_lambda": 0.0,
        "proxy_primitive": "orthogonal_suppress_w0_rho025",
        "true_tensor_basis": False,
    },
    "BASIS_04_PROXY_CONFLICT_COMMIT_W0": {
        "basis_name": "BASIS_04_PROXY",
        "coeff_alpha": 1.0,
        "coeff_beta": 0.0,
        "coeff_lambda": 0.0,
        "proxy_primitive": "orthogonal_suppress_w0_rho0_plus_update_conflict_commit_filter",
        "true_tensor_basis": False,
    },
    "BASIS_05_PROXY_TTGR_ZERO_POSTZP_W0": {
        "basis_name": "BASIS_05_PROXY",
        "coeff_alpha": 1.0,
        "coeff_beta": 0.0,
        "coeff_lambda": 0.0,
        "proxy_primitive": "ttgr_zero_current_chunk_plus_orthogonal_suppress_w0",
        "true_tensor_basis": False,
    },
    "BASIS_06_PROXY_BRANCHSEP_W0W2": {
        "basis_name": "BASIS_06_PROXY",
        "coeff_alpha": 0.75,
        "coeff_beta": 0.0,
        "coeff_lambda": 0.25,
        "proxy_primitive": "orthogonal_suppress_all_rho025_plus_w2_native_mix075",
        "true_tensor_basis": False,
    },
}


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _post_zp_rows(metrics_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for metric in metrics_rows:
        run_dir = Path(metric.get("run_dir", ""))
        path = run_dir / "post_zeropower_delta_norm.csv"
        if not path.exists():
            continue
        for row in _read_csv(path):
            out.append({
                "candidate_id": metric.get("candidate_id"),
                "chunk_id": metric.get("chunk_id"),
                "horizon": metric.get("horizon"),
                "basis_name": COEFFICIENTS.get(metric.get("candidate_id", ""), {}).get("basis_name", ""),
                "layer_id": row.get("layer"),
                "branch_id": row.get("branch"),
                "mode": row.get("mode"),
                "coeff_alpha": COEFFICIENTS.get(metric.get("candidate_id", ""), {}).get("coeff_alpha", ""),
                "coeff_beta": COEFFICIENTS.get(metric.get("candidate_id", ""), {}).get("coeff_beta", ""),
                "delta_norm_before": _to_float(row.get("pre_delta_norm_mean")),
                "delta_norm_after": _to_float(row.get("post_delta_norm_mean")),
                "pre_post_cos_mean": _to_float(row.get("pre_post_cos_mean")),
                "norm_restore_ratio_mean": _to_float(row.get("norm_restore_ratio_mean")),
                "horizon_ATE_delta": _to_float(metric.get("ATE_delta_vs_H9")),
                "segment_delta_200_300": _to_float(metric.get("intersection_200_300_delta_vs_H9")),
                "downstream_delta": _to_float(metric.get("intersection_400_600_delta_vs_H9")),
                "run_dir": metric.get("run_dir"),
            })
    return out


def _summary_rows(metrics_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in metrics_rows:
        coeff = COEFFICIENTS.get(row.get("candidate_id", ""), {})
        out.append({
            "candidate_id": row.get("candidate_id"),
            "basis_name": coeff.get("basis_name", ""),
            "chunk_id": row.get("chunk_id"),
            "horizon": row.get("horizon"),
            "coeff_alpha": coeff.get("coeff_alpha", ""),
            "coeff_beta": coeff.get("coeff_beta", ""),
            "coeff_lambda": coeff.get("coeff_lambda", ""),
            "proxy_primitive": coeff.get("proxy_primitive", ""),
            "true_tensor_basis": coeff.get("true_tensor_basis", False),
            "horizon_ATE_delta": _to_float(row.get("ATE_delta_vs_H9")),
            "segment_delta_200_300": _to_float(row.get("intersection_200_300_delta_vs_H9")),
            "downstream_delta": _to_float(row.get("intersection_400_600_delta_vs_H9")),
            "raw_trans_max_diff": _to_float(row.get("raw_trans_max_diff")),
            "hmc_hash_mismatch": row.get("hmc_hash_mismatch"),
            "merge_hash_mismatch": row.get("merge_hash_mismatch"),
            "run_dir": row.get("run_dir"),
        })
    return out


def _write_markdown(path: Path, summary_rows: List[Dict[str, object]], gate_rows: List[Dict[str, str]]) -> None:
    gate = gate_rows[0] if gate_rows else {}
    best_ate = min(
        [row for row in summary_rows if int(row.get("horizon", 0)) in {8, 10}],
        key=lambda row: _to_float(row.get("horizon_ATE_delta")),
        default=None,
    )
    best_seg = min(
        [
            row for row in summary_rows
            if int(row.get("horizon", 0)) in {8, 10}
            and math.isfinite(_to_float(row.get("segment_delta_200_300")))
        ],
        key=lambda row: _to_float(row.get("segment_delta_200_300")),
        default=None,
    )
    best_h5_seg = min(
        [
            row for row in summary_rows
            if int(row.get("horizon", 0)) == 5
            and math.isfinite(_to_float(row.get("segment_delta_200_300")))
        ],
        key=lambda row: _to_float(row.get("segment_delta_200_300")),
        default=None,
    )
    lines = [
        "# ACL2 v17 Phase 2 Basis Proxy Audit",
        "",
        "This phase used sandbox-only proxy primitives for BASIS_01-06. It did not use true historical tensor basis projection.",
        "",
        "## Gate",
        "",
        f"Status: `{gate.get('status', '')}`",
        "",
        f"Best h8/h10 ATE delta: `{best_ate.get('horizon_ATE_delta') if best_ate else ''}` "
        f"({best_ate.get('candidate_id') if best_ate else ''} chunk {best_ate.get('chunk_id') if best_ate else ''} h{best_ate.get('horizon') if best_ate else ''})",
        "",
        f"Best h8/h10 [200,300) delta: `{best_seg.get('segment_delta_200_300') if best_seg else ''}` "
        f"({best_seg.get('candidate_id') if best_seg else ''} chunk {best_seg.get('chunk_id') if best_seg else ''} h{best_seg.get('horizon') if best_seg else ''})",
        "",
        f"Best h5 [200,300) local delta: `{best_h5_seg.get('segment_delta_200_300') if best_h5_seg else ''}` "
        f"({best_h5_seg.get('candidate_id') if best_h5_seg else ''} chunk {best_h5_seg.get('chunk_id') if best_h5_seg else ''})",
        "",
        "## Boundary",
        "",
        "- These rows are diagnostic short rollouts only.",
        "- `true_tensor_basis=false` for all BASIS proxy rows.",
        "- `post_zp_delta_before_after.pt` was not produced; landed artifacts contain aggregate post-zp norm/cos rows, not full per-tensor anchor bases.",
        "- No no-GT selector or full online validation is authorized.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--gate-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    metrics = Path(args.metrics)
    out_dir = Path(args.out_dir)
    rows = _read_csv(metrics)
    gate_rows = _read_csv(Path(args.gate_summary))
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(metrics, out_dir / "short_rollout_h5_h8_h10.csv")
    summary = _summary_rows(rows)
    norm_rows = _post_zp_rows(rows)
    _write_csv(out_dir / "basis_projection_summary.csv", summary)
    _write_csv(out_dir / "basis_norm_by_layer_branch.csv", norm_rows)
    _write_csv(out_dir / "norm_restore_ratio_before_after.csv", norm_rows)
    _write_jsonl(out_dir / "candidate_basis_coefficients.jsonl", summary)
    _write_markdown(out_dir / "acl2_v17_basis_proxy_audit.md", summary, gate_rows)


if __name__ == "__main__":
    main()
