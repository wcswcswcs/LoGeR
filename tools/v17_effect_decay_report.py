#!/usr/bin/env python3
"""Summarize ACL2 v17 horizon effect decay and local-regularizer diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


NUMERIC_HMC_FIELDS = [
    "pass1_pass2_pose_t_mean",
    "pass1_pass2_pose_t_max",
    "pass1_pass2_pose_r_deg_mean",
    "pass1_pass2_pose_r_deg_max",
    "memory_ttt_mean_rel_diff",
    "memory_ttt_max_rel_diff",
    "memory_ttt_w0_mean_rel_diff",
    "memory_ttt_w1_mean_rel_diff",
    "memory_ttt_w2_mean_rel_diff",
    "prior_hmc_write_score_mean",
]

NUMERIC_RAW_FIELDS = [
    "transform_scale_value",
    "transform_trans_norm",
    "transform_rot_trace",
]


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


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


def _json_default(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _write_json(path: Path, rows: List[Dict[str, object]]) -> None:
    serial = [
        {key: _json_default(value) for key, value in row.items()}
        for row in rows
    ]
    path.write_text(json.dumps(serial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _summarize_hmc(run_dir: Path, chunk_id: int, horizon: int) -> Dict[str, object]:
    rows = [
        row for row in _read_jsonl(run_dir / "hmc_state_hash.jsonl")
        if chunk_id <= int(row.get("chunk_idx", -1)) < chunk_id + horizon
    ]
    out: Dict[str, object] = {
        "hmc_eval_rows": len(rows),
    }
    for field in NUMERIC_HMC_FIELDS:
        out[f"{field}_mean"] = _mean(_to_float(row.get(field)) for row in rows)
        out[f"{field}_max"] = max(
            [_to_float(row.get(field)) for row in rows if math.isfinite(_to_float(row.get(field)))] or [float("nan")]
        )
    zp_rows = [row.get("v13_post_zeropower_delta_summary") for row in rows]
    zp_maps = [row for row in zp_rows if isinstance(row, Mapping)]
    for field in ["pre_delta_norm_mean", "post_delta_norm_mean", "norm_restore_ratio_mean", "pre_post_cos_mean"]:
        out[f"post_zp_{field}_mean"] = _mean(_to_float(row.get(field)) for row in zp_maps)
    return out


def _summarize_raw(run_dir: Path, chunk_id: int, horizon: int) -> Dict[str, object]:
    rows = [
        row for row in _read_jsonl(run_dir / "raw_prediction_buffer_summary.jsonl")
        if chunk_id <= int(row.get("chunk_idx", -1)) < chunk_id + horizon
    ]
    out: Dict[str, object] = {
        "raw_summary_eval_rows": len(rows),
    }
    for field in NUMERIC_RAW_FIELDS:
        values = [_to_float(row.get(field)) for row in rows]
        out[f"{field}_mean"] = _mean(values)
        out[f"{field}_last"] = values[-1] if values else float("nan")
    reasons = sorted({str(row.get("transform_reason")) for row in rows if row.get("transform_reason") is not None})
    out["transform_reasons"] = ",".join(reasons)
    return out


def _load_h3_rows(paths: Iterable[Path]) -> Dict[tuple[str, int], Dict[str, object]]:
    out: Dict[tuple[str, int], Dict[str, object]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _read_csv(path):
            candidate_id = row.get("candidate_id", "")
            if not candidate_id or candidate_id == "K1_H9":
                continue
            try:
                chunk_id = int(row.get("chunk_id", ""))
            except ValueError:
                continue
            key = (candidate_id, chunk_id)
            delta = _to_float(row.get("future_h3_ATE_delta_vs_H9"))
            current = out.get(key)
            if current is None or delta < _to_float(current.get("ATE_delta_vs_H9")):
                out[key] = {
                    "candidate_id": candidate_id,
                    "chunk_id": chunk_id,
                    "horizon": 3,
                    "ATE_delta_vs_H9": delta,
                    "intersection_200_300_delta_vs_H9": _to_float(row.get("future_h3_seg_200_300_delta_vs_H9")),
                    "intersection_400_600_delta_vs_H9": _to_float(row.get("future_h3_seg_400_600_delta_vs_H9")),
                    "source": str(path),
                }
    return out


def _write_markdown(path: Path, decay_rows: List[Dict[str, object]], gate_summary: Mapping[str, str]) -> None:
    best_h8h10 = min(
        [
            row for row in decay_rows
            if row.get("source") == "v17_horizon" and int(row.get("horizon", 0)) in {8, 10}
        ],
        key=lambda row: _to_float(row.get("ATE_delta_vs_H9")),
        default=None,
    )
    best_seg = min(
        [
            row for row in decay_rows
            if row.get("source") == "v17_horizon" and int(row.get("horizon", 0)) in {8, 10}
            and math.isfinite(_to_float(row.get("intersection_200_300_delta_vs_H9")))
        ],
        key=lambda row: _to_float(row.get("intersection_200_300_delta_vs_H9")),
        default=None,
    )
    lines = [
        "# ACL2 v17 Effect Decay Diagnostics",
        "",
        "This report audits whether v16 h3 gains survive longer h5/h8/h10 causal-fork horizons.",
        "",
        "## Gate Context",
        "",
        f"Phase 1 corrected gate status: `{gate_summary.get('status', '')}`",
        "",
        f"Best h8/h10 ATE delta: `{best_h8h10.get('ATE_delta_vs_H9') if best_h8h10 else ''}` "
        f"({best_h8h10.get('candidate_id') if best_h8h10 else ''} chunk {best_h8h10.get('chunk_id') if best_h8h10 else ''} h{best_h8h10.get('horizon') if best_h8h10 else ''})",
        "",
        f"Best h8/h10 [200,300) delta: `{best_seg.get('intersection_200_300_delta_vs_H9') if best_seg else ''}` "
        f"({best_seg.get('candidate_id') if best_seg else ''} chunk {best_seg.get('chunk_id') if best_seg else ''} h{best_seg.get('horizon') if best_seg else ''})",
        "",
        "## Interpretation",
        "",
        "- The longer-horizon rows do not reach the v17 Phase 1 weak gate.",
        "- h3-local improvements therefore behave like local regularizers, not durable target-window correction.",
        "- No selector or full online validation is authorized by this diagnostic.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--gate-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--h3-oracle", action="append", default=[])
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.out_dir)
    rows = _read_csv(metrics_path)
    gate_rows = _read_csv(Path(args.gate_summary))
    gate_summary = gate_rows[0] if gate_rows else {}
    h3_rows = _load_h3_rows(Path(path) for path in args.h3_oracle)

    decay_rows: List[Dict[str, object]] = []
    apply_rows: List[Dict[str, object]] = []
    scale_rows: List[Dict[str, object]] = []
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id == "K1_H9":
            continue
        chunk_id = int(row["chunk_id"])
        horizon = int(row["horizon"])
        base = {
            "candidate_id": candidate_id,
            "chunk_id": chunk_id,
            "horizon": horizon,
            "source": "v17_horizon",
            "ATE_delta_vs_H9": _to_float(row.get("ATE_delta_vs_H9")),
            "intersection_200_300_delta_vs_H9": _to_float(row.get("intersection_200_300_delta_vs_H9")),
            "intersection_400_600_delta_vs_H9": _to_float(row.get("intersection_400_600_delta_vs_H9")),
            "alignment_scale_delta_vs_H9": _to_float(row.get("alignment_scale_delta_vs_H9")),
            "raw_trans_max_diff": _to_float(row.get("raw_trans_max_diff")),
            "run_dir": row.get("run_dir", ""),
        }
        decay_rows.append(base)
        run_dir = Path(row.get("run_dir", ""))
        apply_rows.append({
            **base,
            **_summarize_hmc(run_dir, chunk_id, horizon),
        })
        scale_rows.append({
            **base,
            **_summarize_raw(run_dir, chunk_id, horizon),
        })

    for key, h3 in sorted(h3_rows.items()):
        if any(row["candidate_id"] == key[0] and row["chunk_id"] == key[1] for row in decay_rows):
            decay_rows.append(h3)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "candidate_effect_decay_curve.csv", decay_rows)
    _write_json(out_dir / "candidate_effect_decay_curve.json", decay_rows)
    _write_csv(out_dir / "apply_mismatch_over_horizon.csv", apply_rows)
    _write_json(out_dir / "apply_mismatch_over_horizon.json", apply_rows)
    _write_csv(out_dir / "scale_proxy_over_horizon.csv", scale_rows)
    _write_json(out_dir / "scale_proxy_over_horizon.json", scale_rows)
    _write_markdown(out_dir / "acl2_v17_effect_decay_report.md", decay_rows, gate_summary)


if __name__ == "__main__":
    main()
