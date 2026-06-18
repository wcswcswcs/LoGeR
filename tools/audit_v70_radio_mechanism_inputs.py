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


def main() -> None:
    args = parse_args()
    repo = args.repo_root
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tool_paths = {
        "attention_oracle": repo / "tools/diagnose_v70_radio_attention_oracle.py",
        "merge_oracle": repo / "tools/diagnose_v70_radio_merge_oracle.py",
        "swa_oracle": repo / "tools/diagnose_v70_radio_swa_oracle.py",
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

    mechanism_gate_pass = bool(
        tool_status.get("merge_oracle")
        and tool_status.get("swa_oracle")
        and metric_keys_present == REQUIRED_MECHANISM_METRICS
        and REQUIRED_CONTROL_TYPES.issubset(present_controls)
        and attention_summary.get("mechanism_metrics_available") is True
    )
    blockers: list[str] = []
    if not tool_status.get("merge_oracle"):
        blockers.append("missing_tools/diagnose_v70_radio_merge_oracle.py")
    if not tool_status.get("swa_oracle"):
        blockers.append("missing_tools/diagnose_v70_radio_swa_oracle.py")
    if metric_keys_present != REQUIRED_MECHANISM_METRICS:
        blockers.append("missing_required_mechanism_metrics")
    missing_controls = sorted(REQUIRED_CONTROL_TYPES - present_controls)
    if missing_controls:
        blockers.append("missing_required_controls")
    if attention_summary.get("mechanism_metrics_available") is not True:
        blockers.append("attention_oracle_is_feature_proxy_only")

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
