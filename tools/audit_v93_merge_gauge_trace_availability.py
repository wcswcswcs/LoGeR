#!/usr/bin/env python3
"""Audit v93 Phase3 merge/gauge true trace availability across target rows."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from tools.v86_soft_latent_utils import safe_float, write_csv, write_json  # noqa: E402
from tools.v93_semantic_object_identity_utils import ROOT, V91_ROOT, V92_ROOT, pair_id, seq_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=ROOT / "phase2_object_topology_policy")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase3_merge_gauge_trace_audit")
    parser.add_argument("--v91-route-root", type=Path, default=V91_ROOT / "phase7_carrier_attribution_or_blocked/route_dump_smoke")
    parser.add_argument(
        "--v92-noop-root",
        type=Path,
        default=V92_ROOT / "phase2_boundary_trace_ledger/noop_trace_smoke",
    )
    parser.add_argument("--v93-smoke-root", type=Path, default=ROOT / "phase3_merge_gauge_trace_smoke")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def pair_from_path(path: Path) -> tuple[str, int, int, str]:
    text = path.as_posix()
    match = re.search(r"seq(?P<seq>\d+)_chunk(?P<chunk>\d+)", text)
    if match:
        seq = seq_text(match.group("seq"))
        curr = int(match.group("chunk"))
        prev = curr - 1
        return seq, prev, curr, pair_id(seq, prev, curr)
    seq_match = re.search(r"(?:^|/)seq(?P<seq>\d+)(?:[_/]|$)", text)
    chunk_match = re.search(r"(?:^|/)chunk_?(?P<chunk>\d+)(?:[_/]|$)", text)
    if not seq_match or not chunk_match:
        return "", -1, -1, ""
    seq = seq_text(seq_match.group("seq"))
    curr = int(chunk_match.group("chunk"))
    prev = curr - 1
    return seq, prev, curr, pair_id(seq, prev, curr)


def trace_priority(path: Path, source: str) -> int:
    score = {"v93_smoke": 300, "v92_noop_smoke": 200, "v91_route_dump": 100}.get(source, 0)
    text = path.as_posix()
    if "online_semantic" in text:
        score += 60
    if "thingstuff_radio_qscale" in text or "radio_qscale" in text or "radio_component" in text:
        score += 20
    if "geometry_only" in text:
        score += 10
    if "P9_48_ATTENTION_BIAS_V84_EXTERNAL_ANCHOR_MASS_AUDIT_LAST" in text:
        score += 20
    if "P9_49_ATTENTION_BIAS_V84_EXTERNAL_ANCHOR_RANDOM" in text:
        score -= 20
    if "P9_0_NATIVE" in text:
        score += 10
    return score


def collect_traces(roots: list[tuple[str, Path]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    inventory: list[dict[str, Any]] = []
    for source, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/merge_state_trace.jsonl")):
            seq, prev, curr, pid = pair_from_path(path)
            rows = read_jsonl(path)
            chosen = None
            for row in rows:
                try:
                    if int(row.get("chunk_idx")) == curr:
                        chosen = row
                except Exception:
                    continue
            if chosen is None and rows:
                chosen = rows[-1]
            inv = {
                "source": source,
                "path": str(path),
                "pair_id": pid,
                "seq": seq,
                "prev_chunk": prev,
                "curr_chunk": curr,
                "row_count": len(rows),
                "priority": trace_priority(path, source),
                "chosen_chunk_idx": chosen.get("chunk_idx") if isinstance(chosen, dict) else "",
                "schema": chosen.get("schema") if isinstance(chosen, dict) else "",
            }
            inventory.append(inv)
            if pid and chosen:
                candidates.setdefault(pid, []).append({"path": path, "source": source, "row": chosen, "priority": inv["priority"]})
    best: dict[str, dict[str, Any]] = {}
    for pid, items in candidates.items():
        best[pid] = sorted(items, key=lambda item: item["priority"], reverse=True)[0]
    return best, inventory


def num(value: Any) -> float | None:
    out = safe_float(value)
    return None if out is None else float(out)


def derived_boundary(row: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    scale = num(row.get("transform_scale", row.get("transform_scale_value")))
    trans = num(row.get("transform_translation_norm", row.get("transform_trans_norm")))
    if scale is None and trans is None:
        return None, None, None
    scale_component = abs(math.log(max(abs(scale if scale is not None else 1.0), 1e-12)))
    trans = trans or 0.0
    return float(scale_component + trans), float(math.sqrt(scale_component * scale_component + trans * trans)), float(scale_component)


def has_number(value: Any) -> bool:
    return num(value) is not None


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = pd.read_csv(args.phase2_dir / "object_topology_policy_rows.csv")
    target["seq"] = target["seq"].map(seq_text)
    best, inventory = collect_traces(
        [
            ("v93_smoke", args.v93_smoke_root),
            ("v92_noop_smoke", args.v92_noop_root),
            ("v91_route_dump", args.v91_route_root),
        ]
    )
    ledger = []
    for _, item in target.iterrows():
        pid = str(item.get("pair_id"))
        trace = best.get(pid, {})
        row = trace.get("row", {}) if trace else {}
        derived_norm, derived_l2, derived_scale_component = derived_boundary(row) if row else (None, None, None)
        direct_norm = num(row.get("boundary_update_norm")) if row else None
        direct_l2 = num(row.get("boundary_update_l2")) if row else None
        direct_scale_component = num(row.get("boundary_update_scale_component")) if row else None
        non_identity_direct = row.get("non_identity_transform_flag") if row else None
        if non_identity_direct is None and row:
            non_identity = bool((derived_norm or 0.0) > 1e-9)
            non_identity_source = "derived_from_transform"
        elif non_identity_direct is not None:
            non_identity = str(non_identity_direct).lower() in {"true", "1"}
            non_identity_source = "direct"
        else:
            non_identity = ""
            non_identity_source = ""
        ledger.append(
            {
                "pair_id": pid,
                "seq": seq_text(item.get("seq")),
                "prev_chunk": item.get("prev_chunk"),
                "curr_chunk": item.get("curr_chunk"),
                "policy_state": item.get("p0_v92_policy_baseline"),
                "object_policy_state": item.get("p5_combined_object_policy"),
                "base_case_type": item.get("base_case_type"),
                "labelled": str(item.get("base_case_type")) in {"bad", "good"},
                "merge_state_trace_path": str(trace.get("path", "")) if trace else "",
                "trace_source": trace.get("source", "") if trace else "",
                "hmc_state_hash_path": str(trace.get("path", "")).replace("merge_state_trace.jsonl", "hmc_state_hash.jsonl") if trace else "",
                "hook_effect_summary_path": str(trace.get("path", "")).replace("merge_state_trace.jsonl", "hook_effect_summary.jsonl") if trace else "",
                "boundary_update_norm": direct_norm if direct_norm is not None else derived_norm,
                "boundary_update_norm_source": "direct" if direct_norm is not None else ("derived_from_transform" if derived_norm is not None else ""),
                "boundary_update_l2": direct_l2 if direct_l2 is not None else derived_l2,
                "boundary_update_scale_component": direct_scale_component if direct_scale_component is not None else derived_scale_component,
                "merge_residual_before": row.get("merge_residual_before", "") if row else "",
                "merge_residual_after": row.get("merge_residual_after", "") if row else "",
                "merge_residual_delta": row.get("merge_residual_delta", "") if row else "",
                "boundary_scale_proxy": row.get("boundary_scale_proxy", derived_scale_component if row else ""),
                "transform_matrix": row.get("transform_matrix", "") if row else "",
                "transform_scale": row.get("transform_scale", row.get("transform_scale_value", "")) if row else "",
                "transform_rotation_angle": row.get("transform_rotation_angle", "") if row else "",
                "transform_translation_norm": row.get("transform_translation_norm", row.get("transform_trans_norm", "")) if row else "",
                "non_identity_transform_flag": non_identity,
                "non_identity_transform_flag_source": non_identity_source,
                "selected_overlap_pair_count": row.get("selected_overlap_pair_count", "") if row else "",
                "rejected_overlap_pair_count": row.get("rejected_overlap_pair_count", "") if row else "",
                "semantic_policy_weight_mass": row.get("semantic_policy_weight_mass", "") if row else "",
                "object_identity_support_mass": row.get("object_identity_support_mass", "") if row else "",
                "cross_object_reject_mass": row.get("cross_object_reject_mass", "") if row else "",
                "lowobs_hold_mass": row.get("lowobs_hold_mass", "") if row else "",
                "native_update_hash": row.get("native_state_hash", "") if row else "",
                "no_op_update_hash": row.get("state_hash", "") if row else "",
                "trace_provenance": row.get("trace_provenance", "pre_v93_transform_trace") if row else "",
                "trace_schema": row.get("schema", "") if row else "",
            }
        )
    frame = pd.DataFrame(ledger)
    labelled = frame["labelled"].astype(str).str.lower().isin(["true", "1"])
    has_trace = frame["merge_state_trace_path"].astype(str).str.len() > 0
    norm_available = frame["boundary_update_norm"].map(has_number)
    norm_direct = frame["boundary_update_norm_source"].astype(str).eq("direct")
    residual_available = frame["merge_residual_delta"].map(has_number)
    non_identity_available = frame["non_identity_transform_flag_source"].astype(str).str.len() > 0
    provenance = frame["trace_provenance"].astype(str).str.len() > 0
    summary = {
        "phase": "Phase3_merge_gauge_trace_availability",
        "row_count": int(len(frame)),
        "labelled_row_count": int(labelled.sum()),
        "trace_row_count": int(has_trace.sum()),
        "row_coverage": float(has_trace.mean()) if len(frame) else 0.0,
        "labelled_trace_coverage": float(has_trace[labelled].mean()) if labelled.any() else 0.0,
        "trace_seq_coverage": int(frame.loc[has_trace, "seq"].nunique()),
        "boundary_update_norm_available_ratio": float(norm_available.mean()) if len(frame) else 0.0,
        "boundary_update_norm_direct_ratio": float(norm_direct.mean()) if len(frame) else 0.0,
        "merge_residual_delta_available_ratio": float(residual_available.mean()) if len(frame) else 0.0,
        "non_identity_or_explicit_noop_status_available_ratio": float(non_identity_available.mean()) if len(frame) else 0.0,
        "trace_provenance_ratio": float(provenance.mean()) if len(frame) else 0.0,
        "trace_sources": frame.loc[has_trace, "trace_source"].value_counts().to_dict(),
        "phase3_trace_availability_gate_pass": bool(
            (float(has_trace.mean()) if len(frame) else 0.0) >= 0.80
            and (float(has_trace[labelled].mean()) if labelled.any() else 0.0) >= 0.80
            and int(frame.loc[has_trace, "seq"].nunique()) >= 4
            and (float(norm_available.mean()) if len(frame) else 0.0) >= 0.80
            and (float(residual_available.mean()) if len(frame) else 0.0) >= 0.60
            and (float(non_identity_available.mean()) if len(frame) else 0.0) >= 0.80
            and (float(provenance.mean()) if len(frame) else 0.0) >= 0.95
        ),
        "blocker": "",
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    blockers = []
    if summary["row_coverage"] < 0.80:
        blockers.append("trace_row_coverage_insufficient")
    if summary["merge_residual_delta_available_ratio"] < 0.60:
        blockers.append("merge_residual_delta_unavailable")
    if summary["boundary_update_norm_direct_ratio"] < 0.80:
        blockers.append("direct_boundary_update_norm_instrumentation_insufficient")
    if summary["trace_provenance_ratio"] < 0.95:
        blockers.append("trace_provenance_coverage_insufficient")
    summary["blocker"] = "" if summary["phase3_trace_availability_gate_pass"] else ";".join(blockers)
    write_csv(args.out_dir / "merge_gauge_trace_ledger.csv", ledger)
    write_csv(args.out_dir / "merge_gauge_trace_file_inventory.csv", inventory)
    write_json(args.out_dir / "phase3_trace_availability_summary.json", summary)
    print(f"phase3_trace_availability_gate_pass={summary['phase3_trace_availability_gate_pass']}")
    print(f"row_coverage={summary['row_coverage']}")
    print(f"boundary_update_norm_available_ratio={summary['boundary_update_norm_available_ratio']}")
    print(f"boundary_update_norm_direct_ratio={summary['boundary_update_norm_direct_ratio']}")
    print(f"merge_residual_delta_available_ratio={summary['merge_residual_delta_available_ratio']}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
