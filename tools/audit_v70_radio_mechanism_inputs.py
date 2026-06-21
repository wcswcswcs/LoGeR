#!/usr/bin/env python3
"""Audit whether v70 RADIO R5/R6 mechanism inputs are actually available."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from v70_radio_sidecar_common import utc_now, write_json, write_text


REQUIRED_CONTROL_TYPES = {
    "geometry_only",
    "current_label_confidence_only",
    "radio",
    "radio_feature_shuffle",
    "radio_component_shuffle",
    "radio_confidence_temporal_shuffle",
    "same_entropy_random_proxy_attention",
    "same_degree_random_affinity_graph",
    "current_label_shuffle",
    "current_confidence_shuffle",
}

REQUIRED_MECHANISM_METRICS = {
    "J_v70",
    "future_after_overlap",
    "head_to_tail",
    "scale_cv",
    "intra_scale_variance",
    "raw_overlap_fit_residual",
    "raw_overlap_validation_residual",
    "object_internal_validation_residual",
    "object_cross_boundary_residual",
    "boundary_jump",
    "ATE_proxy_regression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--radio-sidecar-dir", type=Path, required=True)
    parser.add_argument("--path-taps-dir", type=Path, required=True)
    parser.add_argument("--attention-oracle-dir", type=Path, required=True)
    parser.add_argument("--merge-oracle-dir", type=Path, action="append", default=None)
    parser.add_argument("--swa-oracle-dir", type=Path, action="append", default=None)
    parser.add_argument("--read-oracle-dir", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalise_control(value: str) -> str:
    if value in {"radio_only", "radio_semantic_correspondence", "radio_robust_kernel", "radio_object_internal_validation", "radio_weighted_geometry", "radio_internal_ranked_geometry"}:
        return "radio"
    if value in {"radio_feature_spatial_shuffle", "radio_weighted_geometry_feature_shuffle"}:
        return "radio_feature_shuffle"
    if value in {"radio_component_id_shuffle", "radio_weighted_geometry_component_shuffle"}:
        return "radio_component_shuffle"
    if value in {"radio_confidence_temporal_shuffle", "radio_weighted_geometry_confidence_temporal_shuffle"}:
        return "radio_confidence_temporal_shuffle"
    return value


def _merge_metric_names(row_keys: set[str]) -> set[str]:
    aliases = {
        "J_v70": {"J_v70_offline_merge_proxy"},
        "future_after_overlap": {"future_after_overlap_mean", "future_after_overlap_mean_improvement_vs_baseline"},
        "head_to_tail": {"head_to_tail_transfer_ratio_mean", "head_to_tail_transfer_ratio_mean_improvement_vs_baseline"},
        "scale_cv": {"intra_scale_variance_mean"},
        "intra_scale_variance": {"intra_scale_variance_mean", "intra_scale_variance_mean_improvement_vs_baseline"},
        "raw_overlap_fit_residual": {"raw_overlap_fit_after_m", "raw_overlap_fit_improvement_ratio"},
        "raw_overlap_validation_residual": {"raw_overlap_validation_after_m", "raw_overlap_validation_improvement_ratio"},
        "object_internal_validation_residual": {"object_internal_validation_residual_m"},
        "object_cross_boundary_residual": {"object_cross_boundary_residual_m"},
        "boundary_jump": {"boundary_jump_after_m", "boundary_jump_delta_m"},
        "ATE_proxy_regression": {"delta_vs_baseline_global_ate"},
    }
    return {name for name, keys in aliases.items() if keys & row_keys}


def main() -> None:
    args = parse_args()
    repo = args.repo_root
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tool_paths = {
        "attention_oracle": repo / "tools/diagnose_v70_radio_attention_oracle.py",
        "merge_oracle": repo / "tools/diagnose_v70_radio_merge_oracle.py",
        "swa_oracle": repo / "tools/diagnose_v70_radio_swa_oracle.py",
        "read_oracle": repo / "tools/diagnose_v70_radio_read_oracle.py",
    }
    tool_status = {name: path.exists() for name, path in tool_paths.items()}

    sidecars = sorted(args.radio_sidecar_dir.glob("chunk_*/radio_sidecar.pt"))
    path_taps_manifest = _load_json(args.path_taps_dir / "path_taps_manifest.json")
    tap_entries = path_taps_manifest.get("entries", []) if isinstance(path_taps_manifest, dict) else []
    available_taps = [row for row in tap_entries if row.get("available")]

    attention_summary = _load_json(args.attention_oracle_dir / "radio_attention_oracle_summary.json")
    proxy_rows = _csv_rows(args.attention_oracle_dir / "radio_attention_oracle_proxy_rows.csv")
    row_keys = set(proxy_rows[0].keys()) if proxy_rows else set()
    present_controls = {row.get("control_type") for row in proxy_rows if row.get("control_type")}
    metric_keys_present = REQUIRED_MECHANISM_METRICS.intersection(row_keys)
    merge_oracle_dirs = args.merge_oracle_dir or []
    swa_oracle_dirs = args.swa_oracle_dir or []
    merge_summaries: list[dict[str, Any]] = []
    merge_rows: list[dict[str, str]] = []
    merge_controls_present: set[str] = set()
    merge_metric_keys_present: set[str] = set()
    for merge_dir in merge_oracle_dirs:
        summary = _load_json(merge_dir / "radio_merge_oracle_summary.json")
        if summary:
            merge_summaries.append({
                "dir": str(merge_dir),
                "r5_merge_oracle_gate_pass": bool(summary.get("r5_merge_oracle_gate_pass")),
                "r6_online_allowed_by_this_oracle": bool(summary.get("r6_online_allowed_by_this_oracle")),
                "decision": summary.get("decision"),
                "rows": summary.get("rows"),
                "counts": summary.get("counts", {}),
                "radio_gate_chunks": summary.get("radio_gate_chunks", []),
                "radio_beats_controls_chunks": summary.get("radio_beats_controls_chunks", []),
                "median_best_radio_gate_mechanism_improvement": summary.get("median_best_radio_gate_mechanism_improvement"),
            })
        rows = _csv_rows(merge_dir / "radio_merge_oracle_results.csv")
        merge_rows.extend(rows)
        if rows:
            merge_metric_keys_present.update(_merge_metric_names(set(rows[0].keys())))
            merge_controls_present.update(
                _normalise_control(row.get("candidate_type", ""))
                for row in rows
                if row.get("candidate_type")
            )
    present_controls |= merge_controls_present
    metric_keys_present |= merge_metric_keys_present
    merge_gate_pass_any = any(bool(row.get("r5_merge_oracle_gate_pass")) for row in merge_summaries)
    swa_summaries: list[dict[str, Any]] = []
    swa_rows: list[dict[str, str]] = []
    swa_controls_present: set[str] = set()
    for swa_dir in swa_oracle_dirs:
        summary = _load_json(swa_dir / "radio_swa_oracle_summary.json")
        if summary:
            swa_summaries.append({
                "dir": str(swa_dir),
                "swa_residual_proxy_gate_pass": bool(summary.get("swa_residual_proxy_gate_pass")),
                "r5_swa_oracle_gate_pass": bool(summary.get("r5_swa_oracle_gate_pass")),
                "r6_online_allowed_by_this_oracle": bool(summary.get("r6_online_allowed_by_this_oracle")),
                "decision": summary.get("decision"),
                "rows": summary.get("rows"),
                "radio_proxy_gate_rows": summary.get("radio_proxy_gate_rows"),
                "control_proxy_gate_rows": summary.get("control_proxy_gate_rows"),
                "radio_proxy_gate_chunks": summary.get("radio_proxy_gate_chunks", []),
                "radio_beats_controls_chunks": summary.get("radio_beats_controls_chunks", []),
                "median_best_radio_proxy_improvement": summary.get("median_best_radio_proxy_improvement"),
            })
        rows = _csv_rows(swa_dir / "radio_swa_oracle_results.csv")
        swa_rows.extend(rows)
        for row in rows:
            ctype = row.get("candidate_type", "")
            if ctype == "swa_geometry_only":
                swa_controls_present.add("geometry_only")
            elif ctype == "swa_current_label_trust":
                swa_controls_present.add("current_label_confidence_only")
            elif ctype == "swa_current_label_shuffle":
                swa_controls_present.add("current_label_shuffle")
            elif ctype == "swa_current_confidence_shuffle":
                swa_controls_present.add("current_confidence_shuffle")
            elif ctype in {"swa_radio_kv_protect", "swa_radio_risky_cross_object_gate", "swa_radio_proxy_replace", "swa_radio_geometry_blend"}:
                swa_controls_present.add("radio")
            elif ctype == "swa_radio_component_shuffle":
                swa_controls_present.add("radio_component_shuffle")
            elif ctype == "swa_radio_feature_shuffle":
                swa_controls_present.add("radio_feature_shuffle")
            elif ctype == "swa_radio_confidence_temporal_shuffle":
                swa_controls_present.add("radio_confidence_temporal_shuffle")
            elif ctype == "swa_same_entropy_random_proxy":
                swa_controls_present.add("same_entropy_random_proxy_attention")
            elif ctype == "swa_same_count_random_components":
                swa_controls_present.add("same_degree_random_affinity_graph")
    present_controls |= swa_controls_present
    swa_gate_pass_any = any(bool(row.get("r5_swa_oracle_gate_pass")) for row in swa_summaries)
    read_oracle_dirs = args.read_oracle_dir or []
    read_summaries: list[dict[str, Any]] = []
    read_rows: list[dict[str, str]] = []
    read_controls_present: set[str] = set()
    for read_dir in read_oracle_dirs:
        summary = _load_json(read_dir / "radio_read_oracle_summary.json")
        if summary:
            read_summaries.append({
                "dir": str(read_dir),
                "read_attention_proxy_gate_pass": bool(summary.get("read_attention_proxy_gate_pass")),
                "r5_read_oracle_gate_pass": bool(summary.get("r5_read_oracle_gate_pass")),
                "r6_online_allowed_by_this_oracle": bool(summary.get("r6_online_allowed_by_this_oracle")),
                "decision": summary.get("decision"),
                "rows": summary.get("rows"),
                "radio_proxy_gate_rows": summary.get("radio_proxy_gate_rows"),
                "control_proxy_gate_rows": summary.get("control_proxy_gate_rows"),
                "radio_proxy_gate_chunks": summary.get("radio_proxy_gate_chunks", []),
                "radio_beats_controls_chunks": summary.get("radio_beats_controls_chunks", []),
                "median_best_radio_proxy_score": summary.get("median_best_radio_proxy_score"),
                "gate_rule": summary.get("gate_rule", {}),
            })
        rows = _csv_rows(read_dir / "radio_read_oracle_results.csv")
        read_rows.extend(rows)
        for row in rows:
            ctype = row.get("candidate_type", "")
            if ctype == "READ_R1_label_only":
                read_controls_present.add("current_label_confidence_only")
            elif ctype == "READ_label_shuffle":
                read_controls_present.add("current_label_shuffle")
            elif ctype == "READ_confidence_shuffle":
                read_controls_present.add("current_confidence_shuffle")
            elif ctype in {"READ_R3_object_interior_floor", "READ_R4_cross_object_risk_veto"}:
                read_controls_present.add("radio")
            elif ctype == "READ_radio_component_shuffle":
                read_controls_present.add("radio_component_shuffle")
            elif ctype == "READ_radio_feature_shuffle":
                read_controls_present.add("radio_feature_shuffle")
            elif ctype == "READ_radio_risk_shuffle":
                read_controls_present.add("radio_confidence_temporal_shuffle")
            elif ctype == "READ_same_entropy_random_proxy":
                read_controls_present.add("same_entropy_random_proxy_attention")
    present_controls |= read_controls_present
    read_gate_pass_any = any(bool(row.get("r5_read_oracle_gate_pass")) for row in read_summaries)

    mechanism_gate_pass = bool(
        tool_status.get("merge_oracle")
        and tool_status.get("swa_oracle")
        and tool_status.get("read_oracle")
        and metric_keys_present == REQUIRED_MECHANISM_METRICS
        and REQUIRED_CONTROL_TYPES.issubset(present_controls)
        and (merge_gate_pass_any or swa_gate_pass_any or read_gate_pass_any)
    )
    blockers: list[str] = []
    if not tool_status.get("merge_oracle"):
        blockers.append("missing_tools/diagnose_v70_radio_merge_oracle.py")
    if not tool_status.get("swa_oracle"):
        blockers.append("missing_tools/diagnose_v70_radio_swa_oracle.py")
    if not tool_status.get("read_oracle"):
        blockers.append("missing_tools/diagnose_v70_radio_read_oracle.py")
    if metric_keys_present != REQUIRED_MECHANISM_METRICS:
        blockers.append("missing_required_mechanism_metrics")
    missing_controls = sorted(REQUIRED_CONTROL_TYPES - present_controls)
    if missing_controls:
        blockers.append("missing_required_controls")
    if attention_summary.get("mechanism_metrics_available") is not True:
        blockers.append("attention_oracle_is_feature_proxy_only")
    if merge_summaries and not merge_gate_pass_any:
        blockers.append("merge_oracle_r5_gate_failed")
    if not merge_summaries:
        blockers.append("missing_merge_oracle_results")
    if swa_summaries and not swa_gate_pass_any:
        blockers.append("swa_oracle_r5_gate_failed")
    if not swa_summaries:
        blockers.append("missing_swa_oracle_results")
    if read_summaries and not read_gate_pass_any:
        blockers.append("read_oracle_r5_gate_failed")
    if not read_summaries:
        blockers.append("missing_read_oracle_results")

    audit = {
        "created_at": utc_now(),
        "radio_sidecar_dir": str(args.radio_sidecar_dir),
        "sidecar_count": len(sidecars),
        "path_taps_dir": str(args.path_taps_dir),
        "path_taps_manifest_exists": bool(path_taps_manifest),
        "path_taps_available_entries": len(available_taps),
        "attention_oracle_dir": str(args.attention_oracle_dir),
        "attention_oracle_gate_pass": attention_summary.get("gate_pass"),
        "attention_oracle_blocker": attention_summary.get("blocker"),
        "attention_proxy_rows": len(proxy_rows),
        "attention_mechanism_metrics_available": attention_summary.get("mechanism_metrics_available"),
        "merge_oracle_dirs": [str(x) for x in merge_oracle_dirs],
        "merge_oracle_summaries": merge_summaries,
        "merge_oracle_rows": len(merge_rows),
        "merge_oracle_gate_pass_any": merge_gate_pass_any,
        "merge_controls_present": sorted(merge_controls_present),
        "merge_required_mechanism_metrics_present": sorted(merge_metric_keys_present),
        "swa_oracle_dirs": [str(x) for x in swa_oracle_dirs],
        "swa_oracle_summaries": swa_summaries,
        "swa_oracle_rows": len(swa_rows),
        "swa_oracle_gate_pass_any": swa_gate_pass_any,
        "swa_controls_present": sorted(swa_controls_present),
        "read_oracle_dirs": [str(x) for x in read_oracle_dirs],
        "read_oracle_summaries": read_summaries,
        "read_oracle_rows": len(read_rows),
        "read_oracle_gate_pass_any": read_gate_pass_any,
        "read_controls_present": sorted(read_controls_present),
        "J_v70_note": "J_v70 present here is J_v70_offline_merge_proxy, not official online J_v70.",
        "tool_status": tool_status,
        "present_controls": sorted(present_controls),
        "missing_required_controls": missing_controls,
        "present_required_mechanism_metrics": sorted(metric_keys_present),
        "missing_required_mechanism_metrics": sorted(REQUIRED_MECHANISM_METRICS - metric_keys_present),
        "mechanism_gate_pass": mechanism_gate_pass,
        "r6_online_allowed": mechanism_gate_pass,
        "blockers": blockers,
        "decision": "diagnostic_only_no_online" if blockers else "mechanism_inputs_available",
    }
    write_json(args.out_dir / "radio_mechanism_input_audit.json", audit)
    report = [
        "# v70 RADIO Mechanism Input Audit",
        "",
        f"- sidecar_count: `{audit['sidecar_count']}`",
        f"- path_taps_available_entries: `{audit['path_taps_available_entries']}`",
        f"- attention_proxy_rows: `{audit['attention_proxy_rows']}`",
        f"- attention_mechanism_metrics_available: `{audit['attention_mechanism_metrics_available']}`",
        f"- merge_oracle_rows: `{audit['merge_oracle_rows']}`",
        f"- merge_oracle_gate_pass_any: `{audit['merge_oracle_gate_pass_any']}`",
        f"- swa_oracle_rows: `{audit['swa_oracle_rows']}`",
        f"- swa_oracle_gate_pass_any: `{audit['swa_oracle_gate_pass_any']}`",
        f"- read_oracle_rows: `{audit['read_oracle_rows']}`",
        f"- read_oracle_gate_pass_any: `{audit['read_oracle_gate_pass_any']}`",
        f"- mechanism_gate_pass: `{audit['mechanism_gate_pass']}`",
        f"- r6_online_allowed: `{audit['r6_online_allowed']}`",
        f"- decision: `{audit['decision']}`",
        "",
        "## Missing Inputs",
        "",
        f"- blockers: `{', '.join(blockers) if blockers else ''}`",
        f"- missing_required_controls: `{', '.join(missing_controls)}`",
        f"- missing_required_mechanism_metrics: `{', '.join(audit['missing_required_mechanism_metrics'])}`",
    ]
    write_text(args.out_dir / "radio_mechanism_input_audit.md", "\n".join(report) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "mechanism_gate_pass": mechanism_gate_pass, "blockers": blockers}, indent=2))


if __name__ == "__main__":
    main()
