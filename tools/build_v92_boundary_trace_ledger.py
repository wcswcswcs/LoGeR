#!/usr/bin/env python3
"""Build v92 Phase2 boundary trace ledger from v91 policy rows and route traces."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from v86_soft_latent_utils import safe_float, write_csv, write_json
from v92_semantic_policy_carrier_utils import ROOT, V91_PHASE7, pair_id, read_jsonl


DEFAULT_PHASE1 = ROOT / "phase1_semantic_policy_row_bank"
DEFAULT_OUT = ROOT / "phase2_boundary_trace_ledger"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--v91-phase7-dir", type=Path, default=V91_PHASE7)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _float(value: Any) -> float | None:
    out = safe_float(value)
    return None if out is None else float(out)


def _route_pair_from_root(path: Path) -> str:
    name = path.name
    if not name.startswith("seq") or "_chunk" not in name:
        return ""
    seq_part, rest = name.split("_chunk", 1)
    digits = ""
    for char in rest:
        if char.isdigit():
            digits += char
        else:
            break
    if not digits:
        return ""
    curr = int(digits)
    return pair_id(seq_part.replace("seq", ""), curr - 1, curr)


def _trace_for_pair(run_dir: Path, curr_chunk: int) -> dict[str, Any]:
    trace_path = run_dir / "merge_state_trace.jsonl"
    rows = read_jsonl(trace_path)
    if not rows:
        return {}
    chosen = None
    for row in rows:
        if int(row.get("chunk_idx") or -1) == int(curr_chunk):
            chosen = row
    if chosen is None:
        chosen = rows[-1]
    scale = _float(chosen.get("transform_scale_value"))
    trans = _float(chosen.get("transform_trans_norm"))
    scale_term = abs(math.log(max(scale or 1.0, 1e-12)))
    update_norm = float(scale_term + (trans or 0.0))
    return {
        "true_trace_path": str(trace_path),
        "true_trace_rows": len(rows),
        "trace_schema": chosen.get("schema", ""),
        "trace_chunk_idx": chosen.get("chunk_idx"),
        "transform_kind": chosen.get("transform_kind", ""),
        "transform_reason": chosen.get("transform_reason", ""),
        "transform_scale_value": scale,
        "transform_trans_norm": trans,
        "transform_rot_trace": chosen.get("transform_rot_trace"),
        "boundary_update_norm": update_norm,
        "boundary_update_direction": chosen.get("transform_reason", ""),
        "premerge_camera_pose_hash": chosen.get("premerge_camera_pose_hash", ""),
        "postmerge_camera_pose_hash": chosen.get("postmerge_camera_pose_hash", ""),
        "state_hash": chosen.get("state_hash", ""),
    }


def _route_trace_map(route_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not route_root.exists():
        return out
    for seq_root in sorted(route_root.iterdir()):
        if not seq_root.is_dir():
            continue
        pid = _route_pair_from_root(seq_root)
        if not pid:
            continue
        curr = int(pid.split("_")[-1])
        metric_path = seq_root / "phase9_swa_cache_value_metrics.csv"
        metrics: dict[str, Any] = {}
        if metric_path.exists():
            mdf = pd.read_csv(metric_path)
            by_run = {str(row.get("run")): row for _, row in mdf.iterrows()}
            actual = by_run.get("P9_48_ATTENTION_BIAS_V84_EXTERNAL_ANCHOR_MASS_AUDIT_LAST")
            random = by_run.get("P9_49_ATTENTION_BIAS_V84_EXTERNAL_ANCHOR_RANDOM_SAME_MASS_MASS_AUDIT_LAST")
            if actual is not None:
                metrics["route_actual_selected_lift"] = actual.get("phase9_swa_attention_mass_selected_lift", "")
                metrics["route_actual_headmax_lift"] = actual.get("phase9_swa_attention_mass_selected_head_max_lift", "")
                metrics["route_actual_metrics_path"] = str(metric_path)
            if random is not None:
                metrics["route_random_selected_lift"] = random.get("phase9_swa_attention_mass_selected_lift", "")
                metrics["route_random_headmax_lift"] = random.get("phase9_swa_attention_mass_selected_head_max_lift", "")
            a_sel = _float(metrics.get("route_actual_selected_lift"))
            r_sel = _float(metrics.get("route_random_selected_lift"))
            a_head = _float(metrics.get("route_actual_headmax_lift"))
            r_head = _float(metrics.get("route_random_headmax_lift"))
            if a_sel is not None and r_sel is not None:
                metrics["route_actual_minus_random_selected_lift"] = float(a_sel - r_sel)
            if a_head is not None and r_head is not None:
                metrics["route_actual_minus_random_headmax_lift"] = float(a_head - r_head)
        actual_dir = next(seq_root.glob("chunk*/P9_48_ATTENTION_BIAS_V84_EXTERNAL_ANCHOR_MASS_AUDIT_LAST"), None)
        random_dir = next(seq_root.glob("chunk*/P9_49_ATTENTION_BIAS_V84_EXTERNAL_ANCHOR_RANDOM_SAME_MASS_MASS_AUDIT_LAST"), None)
        actual_trace = _trace_for_pair(actual_dir, curr) if actual_dir else {}
        random_trace = _trace_for_pair(random_dir, curr) if random_dir else {}
        merged = {
            "route_smoke_root": str(seq_root),
            "actual_run_dir": str(actual_dir) if actual_dir else "",
            "random_run_dir": str(random_dir) if random_dir else "",
            **metrics,
            **actual_trace,
        }
        if random_trace:
            merged["random_transform_scale_value"] = random_trace.get("transform_scale_value")
            merged["random_boundary_update_norm"] = random_trace.get("boundary_update_norm")
            merged["random_true_trace_path"] = random_trace.get("true_trace_path")
        out[pid] = merged
    return out


def main() -> None:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.phase1_dir / "semantic_policy_rows.csv")
    rows["seq"] = rows["seq"].astype(str).str.zfill(2)
    direct_proxy_path = args.v91_phase7_dir / "direct_boundary_update_trace_proxy.csv"
    proxy = pd.read_csv(direct_proxy_path) if direct_proxy_path.exists() else pd.DataFrame()
    proxy_by_pair = {str(row.get("pair_id")): row for _, row in proxy.iterrows()} if len(proxy) else {}
    route_by_pair = _route_trace_map(args.v91_phase7_dir / "route_dump_smoke")
    ledger: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        pid = str(row.get("pair_id", ""))
        proxy_row = proxy_by_pair.get(pid)
        route = route_by_pair.get(pid, {})
        true_trace = bool(route.get("true_trace_path"))
        proxy_available = proxy_row is not None
        trace_scope = "true_trace_smoke_partial" if true_trace else ("auditable_policy_proxy_only" if proxy_available else "unavailable")
        boundary_update_norm = route.get("boundary_update_norm", "")
        boundary_scale_proxy = row.get("B_proxy", "")
        merge_residual_delta = ""
        item = {
            "seq": row.get("seq", ""),
            "prev_chunk": row.get("prev_chunk", ""),
            "curr_chunk": row.get("curr_chunk", ""),
            "pair_id": pid,
            "policy_state": row.get("policy_state", ""),
            "policy_risk_positive": row.get("policy_risk_positive", ""),
            "regime": row.get("regime", ""),
            "base_case_type": row.get("base_case_type", ""),
            "trace_scope": trace_scope,
            "true_boundary_trace_available": true_trace,
            "auditable_policy_proxy_available": proxy_available,
            "boundary_update_norm": boundary_update_norm,
            "boundary_update_direction": route.get("boundary_update_direction", ""),
            "boundary_update_norm_proxy_policy_mass": proxy_row.get("boundary_update_eligible_mass", "") if proxy_available else "",
            "invalid_rejected_mass_proxy": proxy_row.get("invalid_rejected_mass", "") if proxy_available else "",
            "context_only_mass_proxy": proxy_row.get("context_only_mass", "") if proxy_available else "",
            "merge_residual_before": "",
            "merge_residual_after": "",
            "merge_residual_delta": merge_residual_delta,
            "boundary_scale_proxy": boundary_scale_proxy,
            "native_boundary_jump": "",
            "postmerge_boundary_jump": "",
            "gauge_refresh_proxy": proxy_row.get("boundary_update_eligible_mass", "") if proxy_available else "",
            "gauge_hold_proxy": proxy_row.get("context_only_mass", "") if proxy_available else "",
            "overlap_support_count": row.get("raw_overlap_support_count", ""),
            "feature_match_support_count": row.get("feature_match_support_count", ""),
            "semantic_conflict_mass": row.get("S_invalid", ""),
            "semantic_context_mass": row.get("S_context", ""),
            "component_consistency": row.get("component_consistency", ""),
            "route_smoke_available": bool(pid in route_by_pair),
            "route_actual_minus_random_selected_lift": route.get("route_actual_minus_random_selected_lift", ""),
            "route_actual_minus_random_headmax_lift": route.get("route_actual_minus_random_headmax_lift", ""),
            "true_trace_path": route.get("true_trace_path", ""),
            "proxy_trace_path": str(direct_proxy_path) if proxy_available else "",
            "route_smoke_root": route.get("route_smoke_root", ""),
            "trace_schema": route.get("trace_schema", ""),
            "transform_kind": route.get("transform_kind", ""),
            "transform_reason": route.get("transform_reason", ""),
            "transform_scale_value": route.get("transform_scale_value", ""),
            "transform_trans_norm": route.get("transform_trans_norm", ""),
            "premerge_camera_pose_hash": route.get("premerge_camera_pose_hash", ""),
            "postmerge_camera_pose_hash": route.get("postmerge_camera_pose_hash", ""),
            "phase1_row_source": str(args.phase1_dir / "semantic_policy_rows.csv"),
        }
        ledger.append(item)
    write_csv(out / "boundary_trace_rows.csv", ledger)
    by_pair_fields = [
        "seq",
        "prev_chunk",
        "curr_chunk",
        "pair_id",
        "policy_state",
        "regime",
        "trace_scope",
        "true_boundary_trace_available",
        "auditable_policy_proxy_available",
        "boundary_update_norm",
        "boundary_update_norm_proxy_policy_mass",
        "invalid_rejected_mass_proxy",
        "boundary_scale_proxy",
        "route_actual_minus_random_selected_lift",
        "true_trace_path",
        "proxy_trace_path",
        "route_smoke_root",
    ]
    write_csv(out / "boundary_trace_by_pair.csv", ledger, by_pair_fields)
    true_count = sum(1 for row in ledger if row["true_boundary_trace_available"])
    proxy_count = sum(1 for row in ledger if row["auditable_policy_proxy_available"])
    summary = {
        "phase": "Phase2_boundary_trace_ledger_build",
        "row_count": len(ledger),
        "phase1_row_count": int(len(rows)),
        "sequence_coverage": int(rows["seq"].nunique()),
        "true_trace_rows": int(true_count),
        "auditable_policy_proxy_rows": int(proxy_count),
        "route_smoke_pair_count": int(len(route_by_pair)),
        "direct_proxy_path": str(direct_proxy_path),
        "route_dump_root": str(args.v91_phase7_dir / "route_dump_smoke"),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(out / "boundary_trace_availability.json", summary)
    (out / "schema_report.md").write_text(
        "\n".join(
            [
                "# v92 Boundary Trace Ledger Schema",
                "",
                "- `true_trace_smoke_partial`: row has landed `merge_state_trace.jsonl` from v91 route smoke.",
                "- `auditable_policy_proxy_only`: row has v91 `direct_boundary_update_trace_proxy.csv` but no true merge trace.",
                "- `boundary_update_norm` is derived only from true trace `abs(log(transform_scale_value)) + transform_trans_norm`.",
                "- `boundary_update_norm_proxy_policy_mass` is kept separate and must not be claimed as true boundary update norm.",
                "- `merge_residual_*` fields are blank unless an artifact explicitly exposes residuals.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    missing = [row for row in ledger if not row["true_boundary_trace_available"]]
    (out / "missing_trace_report.md").write_text(
        "\n".join(
            [
                "# Missing True Boundary Trace Report",
                "",
                f"- total_rows: `{len(ledger)}`",
                f"- true_trace_rows: `{true_count}`",
                f"- rows_without_true_trace: `{len(missing)}`",
                "",
                "Rows without true trace still have policy proxy when `auditable_policy_proxy_available=true`, but proxy rows are not runtime carrier evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"row_count={summary['row_count']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"true_trace_rows={summary['true_trace_rows']}")
    print(f"auditable_policy_proxy_rows={summary['auditable_policy_proxy_rows']}")
    print(f"route_smoke_pair_count={summary['route_smoke_pair_count']}")


if __name__ == "__main__":
    main()
