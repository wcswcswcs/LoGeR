#!/usr/bin/env python3
"""Create ACL2 v17 Phase 3 AUXGEO proxy audit artifacts.

The landed controller exposes overlap pseudo-replay diagnostics, but not a
semantic structure-token oracle. This report writes the values that are
actually present in landed rollouts and marks unavailable fields explicitly.
"""

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


def _mean(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _max(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return max(finite)


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


def _candidate_role(candidate_id: str) -> str:
    if "KV" in candidate_id:
        return "overlap_pseudo_kv"
    if "V_" in candidate_id or candidate_id.endswith("_V_W0"):
        return "overlap_pseudo_v"
    return "overlap_geometry_proxy"


def _metric_rows_with_debug(metrics: List[Dict[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for metric in metrics:
        run_dir = Path(metric.get("run_dir", ""))
        chunk_id = int(float(metric.get("chunk_id", "0") or 0))
        horizon = int(float(metric.get("horizon", "0") or 0))
        hmc_rows = [
            row for row in _read_jsonl(run_dir / "hmc_state_hash.jsonl")
            if chunk_id <= int(row.get("chunk_idx", -1)) < chunk_id + horizon
        ]
        for row in hmc_rows:
            mode = row.get("auxgeo_replay_feature_gate_mode") or _candidate_role(metric.get("candidate_id", ""))
            token_count = _to_float(row.get("auxgeo_overlap_replay_token_count_mean"))
            token_count_max = _to_float(row.get("auxgeo_overlap_replay_token_count_max"))
            token_filter_after = _to_float(row.get("auxgeo_token_filter_tokens_after_mean"))
            v_norm = _to_float(row.get("auxgeo_pseudo_v_delta_norm_mean"))
            k_norm = _to_float(row.get("auxgeo_pseudo_k_delta_norm_mean"))
            single_norm = _to_float(row.get("auxgeo_pseudo_single_delta_norm_mean"))
            pseudo_norm = _mean([v_norm, k_norm, single_norm])
            if not math.isfinite(pseudo_norm):
                pseudo_norm = _max([v_norm, k_norm, single_norm])
            out.append({
                "candidate_id": metric.get("candidate_id"),
                "chunk_id": chunk_id,
                "horizon": horizon,
                "eval_chunk_idx": row.get("chunk_idx"),
                "run_done": metric.get("run_done"),
                "run_dir": metric.get("run_dir"),
                "replay_mode": mode,
                "replay_targets": row.get("auxgeo_replay_feature_gate_targets"),
                "branch_mask": row.get("auxgeo_replay_feature_branch_mask"),
                "rho": row.get("auxgeo_replay_feature_gate_rho"),
                "applied_layer_count": row.get("auxgeo_replay_applied_layer_count"),
                "num_overlap_tokens": token_count,
                "num_overlap_tokens_max_layer": token_count_max,
                "num_structure_tokens": token_filter_after if math.isfinite(token_filter_after) else float("nan"),
                "structure_token_source": (
                    "overlap_static_topk_filter"
                    if math.isfinite(token_filter_after)
                    else "unavailable_not_exported"
                ),
                "pseudo_target_norm": pseudo_norm,
                "pseudo_target_v_delta_norm": v_norm,
                "pseudo_target_k_delta_norm": k_norm,
                "pseudo_target_single_delta_norm": single_norm,
                "pseudo_target_cosine_to_native_v": float("nan"),
                "pseudo_target_cosine_to_original_v": float("nan"),
                "pseudo_target_cosine_source": "unavailable_not_exported",
                "branch_w0_update_rel_diff": _to_float(row.get("memory_ttt_w0_mean_rel_diff")),
                "branch_w1_update_rel_diff": _to_float(row.get("memory_ttt_w1_mean_rel_diff")),
                "branch_w2_update_rel_diff": _to_float(row.get("memory_ttt_w2_mean_rel_diff")),
                "layer_update_norm": float("nan"),
                "layer_update_norm_source": "unavailable_not_exported",
                "horizon_ATE_delta": _to_float(metric.get("ATE_delta_vs_H9")),
                "segment_200_300_delta": _to_float(metric.get("intersection_200_300_delta_vs_H9")),
                "downstream_delta": _to_float(metric.get("intersection_400_600_delta_vs_H9")),
                "raw_trans_max_diff": _to_float(metric.get("raw_trans_max_diff")),
            })
    return out


def _summary_rows(metrics: List[Dict[str, str]], debug_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in debug_rows:
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
            "replay_mode": rows[0].get("replay_mode") if rows else _candidate_role(metric.get("candidate_id", "")),
            "mean_overlap_tokens": _mean(_to_float(row.get("num_overlap_tokens")) for row in rows),
            "mean_structure_tokens": _mean(_to_float(row.get("num_structure_tokens")) for row in rows),
            "mean_pseudo_target_norm": _mean(_to_float(row.get("pseudo_target_norm")) for row in rows),
            "mean_branch_w0_update_rel_diff": _mean(_to_float(row.get("branch_w0_update_rel_diff")) for row in rows),
            "horizon_ATE_delta": _to_float(metric.get("ATE_delta_vs_H9")),
            "segment_200_300_delta": _to_float(metric.get("intersection_200_300_delta_vs_H9")),
            "downstream_delta": _to_float(metric.get("intersection_400_600_delta_vs_H9")),
            "run_dir": metric.get("run_dir"),
        })
    return out


def _write_markdown(path: Path, summary: List[Dict[str, object]], gate_rows: List[Dict[str, str]]) -> None:
    gate = gate_rows[0] if gate_rows else {}
    h8_h10 = [row for row in summary if int(float(row.get("horizon", 0) or 0)) in {8, 10}]
    best_ate = min(h8_h10, key=lambda row: _to_float(row.get("horizon_ATE_delta")), default=None)
    best_seg = min(
        [row for row in h8_h10 if math.isfinite(_to_float(row.get("segment_200_300_delta")))],
        key=lambda row: _to_float(row.get("segment_200_300_delta")),
        default=None,
    )
    lines = [
        "# ACL2 v17 Phase 3 AUXGEO Proxy Audit",
        "",
        "This report uses landed overlap pseudo-replay debug fields. Semantic structure-token grouping and pseudo-target cosine were not exported by the runtime.",
        "",
        "## Gate",
        "",
        f"Status: `{gate.get('status', '')}`",
        "",
        f"Best h8/h10 ATE delta: `{best_ate.get('horizon_ATE_delta') if best_ate else ''}` "
        f"({best_ate.get('candidate_id') if best_ate else ''} chunk {best_ate.get('chunk_id') if best_ate else ''} h{best_ate.get('horizon') if best_ate else ''})",
        "",
        f"Best h8/h10 [200,300) delta: `{best_seg.get('segment_200_300_delta') if best_seg else ''}` "
        f"({best_seg.get('candidate_id') if best_seg else ''} chunk {best_seg.get('chunk_id') if best_seg else ''} h{best_seg.get('horizon') if best_seg else ''})",
        "",
        "## Boundary",
        "",
        "- These rows are sandbox-only AUXGEO proxy diagnostics.",
        "- `pseudo_target_cosine_to_native_v` and semantic per-group norms are marked unavailable when the runtime did not export them.",
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
    debug_rows = _metric_rows_with_debug(metrics)
    summary = _summary_rows(metrics, debug_rows)

    shutil.copyfile(metrics_path, out_dir / "short_rollout_h5_h8_h10.csv")
    _write_jsonl(out_dir / "overlap_geometry_replay_debug.jsonl", debug_rows)
    _write_csv(out_dir / "pseudo_target_norm.csv", summary)
    _write_csv(out_dir / "pseudo_target_cos_to_native.csv", debug_rows)
    _write_csv(out_dir / "structure_token_count.csv", summary)
    _write_csv(out_dir / "per_group_update_norm.csv", debug_rows)
    _write_markdown(out_dir / "acl2_v17_auxgeo_proxy_audit.md", summary, gate_rows)


if __name__ == "__main__":
    main()
