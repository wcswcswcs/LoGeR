#!/usr/bin/env python3
"""Summarize Stream4D v79 repair sweep outputs into an auditable decision artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _suffix_from_pipeline_dir(path: Path, prefix: str) -> str:
    name = path.name
    return "" if name == prefix else name[len(prefix) :]


def collect(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = ROOT / args.audit_root
    output_root = ROOT / args.output_root
    prefix = args.pipeline_prefix
    rows: list[dict[str, Any]] = []
    for pipeline_dir in sorted(audit_root.glob(f"{prefix}*")):
        summary_path = pipeline_dir / "pipeline_summary.json"
        if not summary_path.exists():
            continue
        payload = _read_json(summary_path)
        summaries = payload.get("summaries", {})
        phase1 = summaries.get("phase1", {})
        phase2 = summaries.get("phase2", {})
        phase3 = summaries.get("phase3", {})
        phase4 = summaries.get("phase4", {})
        phase5 = summaries.get("phase5", {})
        suffix = _suffix_from_pipeline_dir(pipeline_dir, prefix)
        rows.append(
            {
                "variant": suffix or "_baseline",
                "pipeline_root": pipeline_dir.relative_to(ROOT).as_posix(),
                "final_decision": payload.get("decision", ""),
                "phase1_decision": phase1.get("decision", ""),
                "phase2_decision": phase2.get("decision", ""),
                "phase4_decision": phase4.get("decision", ""),
                "phase5_decision": phase5.get("decision", ""),
                "local_SF50": _float(phase4.get("local_SF50")),
                "local_AP50": _float(phase4.get("local_AP50")),
                "local_AP25": _float(phase4.get("local_AP25")),
                "GT_best_IoU_mean": _float(phase4.get("GT_best_IoU_mean")),
                "broad_mask_contribution_ratio": _float(phase1.get("broad_mask_contribution_ratio")),
                "cosine_approx_error_p95": _float(phase1.get("cosine_approx_error_p95")),
                "largest_connected_component_ratio": _float(phase2.get("largest_connected_component_ratio")),
                "heldout_same_mask_AUC_sampled": _float(phase2.get("heldout_same_mask_AUC_sampled")),
                "real_minus_shuffled_heldout": _float(phase2.get("real_minus_shuffled_heldout")),
                "real_minus_no_temporal_heldout": _float(phase2.get("real_minus_no_temporal_heldout")),
                "unary_semantic_control_available": bool(phase2.get("unary_semantic_control_available")),
                "unary_semantic_control_type": phase2.get("unary_semantic_control_type", ""),
                "unary_semantic_vs_affinity_gap": _float(phase2.get("unary_semantic_vs_affinity_gap")),
                "unary_semantic_AUC_vs_affinity_gap": _float(phase2.get("unary_semantic_AUC_vs_affinity_gap")),
                "semantic_prototype_unary_heldout": _float(phase2.get("semantic_prototype_unary_heldout")),
                "semantic_prototype_unary_AUC_sampled": _float(phase2.get("semantic_prototype_unary_AUC_sampled")),
                "semantic_prototype_unary_carrier_coverage_rate": _float(phase2.get("semantic_prototype_unary_carrier_coverage_rate")),
                "cluster_count_per_chunk": _float(phase3.get("cluster_count_per_chunk")),
                "small_cluster_merge_count": phase3.get("small_cluster_merge_count", ""),
                "no_temporal_proxy_SF50": _float(phase5.get("no_temporal_proxy_SF50")),
                "risk_count_matched_area_control_SF50": _float(phase5.get("risk_count_matched_area_control_SF50")),
                "runtime_sec": _float(payload.get("runtime_sec")),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["local_SF50"]),
            not bool(row.get("unary_semantic_control_available")),
            float(row["runtime_sec"]),
            str(row["variant"]),
        )
    )
    _write_csv(output_root / "variant_comparison_rows.csv", rows)

    best = rows[0] if rows else {}
    v77_sf50 = _float(best.get("risk_count_matched_area_control_SF50"), 0.0)
    # The v77 baseline is stored in each final summary; use the authoritative best row payload if present.
    if best:
        best_payload = _read_json(ROOT / best["pipeline_root"] / "pipeline_summary.json")
        v77_sf50 = _float(best_payload.get("summaries", {}).get("final", {}).get("v77_M0_SF50"), v77_sf50)

    local_sf50 = _float(best.get("local_SF50"))
    first_stage_pass = local_sf50 >= 0.40 and local_sf50 >= v77_sf50 + 0.05
    strict_control_pass = local_sf50 >= _float(best.get("risk_count_matched_area_control_SF50")) + 0.03
    final_decision = (
        "GO_LOCAL_AFFINITY_FEATURE_METHOD"
        if first_stage_pass and strict_control_pass and bool(best.get("unary_semantic_control_available"))
        else "NO_GO_AFFINITY_FEATURE_BELOW_V77_AFTER_REPAIRS"
    )
    summary = {
        "phase": "v79_repair_sweep_summary",
        "schema": "stream4d_v79_repair_sweep_summary_v1",
        "variant_count": len(rows),
        "best_variant": best,
        "v77_M0_SF50": v77_sf50,
        "first_stage_pass": first_stage_pass,
        "strict_control_pass": strict_control_pass,
        "unary_control_available_for_best": bool(best.get("unary_semantic_control_available")) if best else False,
        "unary_semantic_control_type_for_best": best.get("unary_semantic_control_type", "") if best else "",
        "unary_semantic_vs_affinity_gap_for_best": best.get("unary_semantic_vs_affinity_gap", "") if best else "",
        "unary_semantic_AUC_vs_affinity_gap_for_best": best.get("unary_semantic_AUC_vs_affinity_gap", "") if best else "",
        "can_enter_local2history": False,
        "final_decision": final_decision,
        "primary_blocker": (
            "best_local_SF50_below_v77_and_first_stage_target"
            if not first_stage_pass
            else "strict_or_unary_control_missing"
        ),
        "notes": [
            "This summary compares measured v79 probe artifacts only; it does not fabricate missing unary semantic controls.",
            "local2history remains blocked unless local first-stage and attribution gates pass.",
        ],
    }
    _write_json(output_root / "sweep_summary.json", summary)
    _write_json(output_root / "best_final_decision.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", default="outputs/audit")
    parser.add_argument("--pipeline-prefix", default="v79_cmap_af_l2h_pipeline")
    parser.add_argument("--output-root", default="outputs/audit/v79_repair_sweep_summary")
    args = parser.parse_args()
    print(json.dumps(collect(args), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
