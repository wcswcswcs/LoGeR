#!/usr/bin/env python3
"""Create ACL2 v17 Phase 4 dual-bank proxy audit artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


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
            clean = {
                key: (None if isinstance(value, float) and math.isnan(value) else value)
                for key, value in row.items()
            }
            handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")


def _debug_rows(metrics: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for metric in metrics:
        run_dir = Path(metric.get("run_dir", ""))
        chunk_id = int(float(metric.get("chunk_id", "0") or 0))
        horizon = int(float(metric.get("horizon", "0") or 0))
        rows = [
            row for row in _read_jsonl(run_dir / "hmc_state_hash.jsonl")
            if chunk_id <= int(row.get("chunk_idx", -1)) < chunk_id + horizon
        ]
        for row in rows:
            short_norm = _mean([
                _to_float(row.get("dlbank_w0_short_norm_mean")),
                _to_float(row.get("dlbank_w1_short_norm_mean")),
                _to_float(row.get("dlbank_w2_short_norm_mean")),
            ])
            long_scale = _to_float(row.get("dlbank_transient_long_scale"))
            apply_scale = _to_float(row.get("dlbank_transient_apply_scale"))
            out.append({
                "candidate_id": metric.get("candidate_id"),
                "chunk_id": chunk_id,
                "horizon": horizon,
                "eval_chunk_idx": row.get("chunk_idx"),
                "run_done": metric.get("run_done"),
                "dual_bank_state_hash": row.get("hash_H_next"),
                "short_mode": row.get("dlbank_transient_mode"),
                "short_apply_scale": apply_scale,
                "short_decay_tau": row.get("dlbank_short_ttl_out"),
                "long_scale": long_scale,
                "W_short_norm": short_norm,
                "W_short_w0_norm": _to_float(row.get("dlbank_w0_short_norm_mean")),
                "W_short_w1_norm": _to_float(row.get("dlbank_w1_short_norm_mean")),
                "W_short_w2_norm": _to_float(row.get("dlbank_w2_short_norm_mean")),
                "W_long_norm": _to_float(row.get("memory_ttt_mean_rel_diff")),
                "cos_Wshort_Wlong": float("nan"),
                "cos_source": "unavailable_not_exported",
                "short_present_prev": row.get("dlbank_prev_short_present"),
                "short_carry_prev": row.get("dlbank_prev_short_carry"),
                "short_stored": row.get("dlbank_short_stored"),
                "short_applied_layer_count": row.get("dlbank_transient_applied_layer_count"),
                "pos_mass": row.get("auxgeo_tri_replay_pos_mass_mean"),
                "neutral_mass": row.get("auxgeo_tri_replay_neu_mass_mean"),
                "correction_mass": row.get("auxgeo_tri_replay_neg_mass_mean"),
                "horizon_ATE_delta": _to_float(metric.get("ATE_delta_vs_H9")),
                "segment_delta": _to_float(metric.get("intersection_200_300_delta_vs_H9")),
                "downstream_delta": _to_float(metric.get("intersection_400_600_delta_vs_H9")),
                "run_dir": metric.get("run_dir"),
            })
    return out


def _summary_rows(metrics: List[Dict[str, str]], debug: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in debug:
        key = (str(row.get("candidate_id")), str(row.get("chunk_id")), str(row.get("horizon")))
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, object]] = []
    for metric in metrics:
        key = (metric.get("candidate_id", ""), metric.get("chunk_id", ""), metric.get("horizon", ""))
        rows = grouped.get(key, [])
        out.append({
            "candidate_id": metric.get("candidate_id"),
            "chunk_id": metric.get("chunk_id"),
            "horizon": metric.get("horizon"),
            "run_done": metric.get("run_done"),
            "mean_W_short_norm": _mean(_to_float(row.get("W_short_norm")) for row in rows),
            "mean_W_long_norm": _mean(_to_float(row.get("W_long_norm")) for row in rows),
            "mean_short_apply_scale": _mean(_to_float(row.get("short_apply_scale")) for row in rows),
            "mean_short_decay_tau": _mean(_to_float(row.get("short_decay_tau")) for row in rows),
            "horizon_ATE_delta": _to_float(metric.get("ATE_delta_vs_H9")),
            "segment_delta": _to_float(metric.get("intersection_200_300_delta_vs_H9")),
            "downstream_delta": _to_float(metric.get("intersection_400_600_delta_vs_H9")),
            "run_dir": metric.get("run_dir"),
        })
    return out


def _write_markdown(path: Path, summary: List[Dict[str, object]], gate_rows: List[Dict[str, str]]) -> None:
    gate = gate_rows[0] if gate_rows else {}
    h8_h10 = [row for row in summary if int(float(row.get("horizon", 0) or 0)) in {8, 10}]
    best_ate = min(h8_h10, key=lambda row: _to_float(row.get("horizon_ATE_delta")), default=None)
    best_seg = min(
        [row for row in h8_h10 if math.isfinite(_to_float(row.get("segment_delta")))],
        key=lambda row: _to_float(row.get("segment_delta")),
        default=None,
    )
    lines = [
        "# ACL2 v17 Phase 4 Dual-Bank Proxy Audit",
        "",
        "This report uses the landed transient/dual-lifetime fast-weight path as a proxy for W_long/W_short separation.",
        "",
        "## Gate",
        "",
        f"Status: `{gate.get('status', '')}`",
        "",
        f"Best h8/h10 ATE delta: `{best_ate.get('horizon_ATE_delta') if best_ate else ''}` "
        f"({best_ate.get('candidate_id') if best_ate else ''} chunk {best_ate.get('chunk_id') if best_ate else ''} h{best_ate.get('horizon') if best_ate else ''})",
        "",
        f"Best h8/h10 [200,300) delta: `{best_seg.get('segment_delta') if best_seg else ''}` "
        f"({best_seg.get('candidate_id') if best_seg else ''} chunk {best_seg.get('chunk_id') if best_seg else ''} h{best_seg.get('horizon') if best_seg else ''})",
        "",
        "## Boundary",
        "",
        "- These rows are sandbox-only dual-bank proxy diagnostics.",
        "- `cos_Wshort_Wlong` is unavailable unless the runtime exports full tensor short/long states.",
        "- No no-GT selector or full online validation is authorized from this report alone.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--gate-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    gate_path = Path(args.gate_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _read_csv(metrics_path)
    gate_rows = _read_csv(gate_path)
    debug = _debug_rows(metrics)
    summary = _summary_rows(metrics, debug)

    shutil.copyfile(metrics_path, out_dir / "short_rollout_h5_h8_h10.csv")
    _write_jsonl(out_dir / "dual_bank_state_hash.jsonl", debug)
    _write_csv(out_dir / "W_long_norm.csv", summary)
    _write_csv(out_dir / "W_short_norm.csv", summary)
    _write_csv(out_dir / "short_to_long_cosine.csv", debug)
    _write_csv(out_dir / "short_decay_curve.csv", debug)
    _write_csv(out_dir / "apply_W_short_mass.csv", debug)
    _write_markdown(out_dir / "acl2_v17_dual_bank_proxy_audit.md", summary, gate_rows)


if __name__ == "__main__":
    main()
