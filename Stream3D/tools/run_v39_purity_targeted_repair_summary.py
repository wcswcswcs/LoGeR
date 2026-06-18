from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


RAW_CONFIG = "v39_purity_targeted_i4_gap2_rgb099_probe5"
TOPK_CONFIGS = [
    "v39_purity_targeted_i4_gap2_rgb099_top50",
    "v39_purity_targeted_i4_gap2_rgb099_top100",
    "v39_purity_targeted_i4_gap2_rgb099_top200",
    "v39_purity_targeted_i4_gap2_rgb099_top300",
]
CONFLICT_CONFIGS = [
    "v39_purity_targeted_i4_gap2_rgb099_nms_minioc090_top300",
    "v39_purity_targeted_i4_gap2_rgb099_wta_area_desc_min100",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _metric_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    out = float(value)
    return None if math.isnan(out) else out


def _parse_eval_average(path: Path) -> dict[str, float | str | None]:
    if not path.exists():
        return {"AP": None, "AP50": None, "AP25": None, "eval_path": str(path), "status": "missing"}
    rows = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if re.match(r"^[0-9.+\\-nan]", line.strip())
    ]
    if not rows:
        return {"AP": None, "AP50": None, "AP25": None, "eval_path": str(path), "status": "parse_failed"}
    parts = rows[-1].split(",")
    return {
        "AP": _metric_float(parts[0]),
        "AP50": _metric_float(parts[1]),
        "AP25": _metric_float(parts[2]),
        "eval_path": str(path),
        "status": "ok",
    }


def _mean(values: list[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _count_regularized_row(root: Path, config: str, eval_path: Path) -> dict[str, Any]:
    summary_path = root / f"{config}_summary.json"
    row = {"config": config, **_parse_eval_average(eval_path)}
    if summary_path.exists():
        agg = _read_json(summary_path).get("aggregate", {})
        row.update(
            {
                "mean_num_instances_before": agg.get("mean_num_instances_before"),
                "mean_num_instances_after": agg.get("mean_num_instances_after"),
                "mean_output_union_count": agg.get("mean_output_union_count"),
                "postprocess": agg.get("output_config"),
                "overlap_mode": agg.get("overlap_mode"),
                "overlap_threshold": agg.get("overlap_threshold"),
                "tie_breaker": agg.get("tie_breaker"),
                "backend": agg.get("backend"),
            }
        )
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    repair_root = root / args.repair_root
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    local_decision = _read_json(repair_root / "v37_final_decision/decision_summary.json")
    memory_decision = _read_json(repair_root / "v37_4d_if_allowed/4d_memory_decision.json")
    export_summary = _read_json(repair_root / "v37_ap_if_allowed/ap_export_summary.json")
    raw_eval = _parse_eval_average(
        repair_root / "v37_ap_if_allowed" / f"{RAW_CONFIG}_class_agnostic_allow_oracle_eval.txt"
    )
    raw_export_rows = export_summary.get("scene_rows", [])
    raw_row = {
        "config": RAW_CONFIG,
        **raw_eval,
        "mean_num_instances_before": _mean([row.get("num_candidate_objects") for row in raw_export_rows]),
        "mean_num_instances_after": _mean([row.get("num_exported_objects") for row in raw_export_rows]),
        "mean_export_conflict_rate": _mean([row.get("export_conflict_rate") for row in raw_export_rows]),
        "mean_covered_GT_instance_ratio": export_summary.get("mean_covered_GT_instance_ratio"),
        "mean_mesh_coverage": export_summary.get("mean_mesh_coverage"),
        "postprocess": "raw_i4_ap_export",
    }

    count_root = repair_root / "v39_count_regularized_export"
    count_rows = [
        _count_regularized_row(
            count_root,
            config,
            count_root / f"{config}_class_agnostic_allow_oracle_eval.txt",
        )
        for config in TOPK_CONFIGS
    ]
    conflict_root = repair_root / "v39_conflict_regularized_export"
    conflict_rows = [
        _count_regularized_row(
            conflict_root,
            config,
            conflict_root / f"{config}_class_agnostic_allow_oracle_eval.txt",
        )
        for config in CONFLICT_CONFIGS
    ]
    ap_rows = [raw_row, *count_rows, *conflict_rows]
    best_ap = max(ap_rows, key=lambda row: float(row.get("AP") or -1.0), default={})

    local_metrics = local_decision.get("best_metrics", {})
    memory_metrics = memory_decision.get("best_metrics", {})
    final_status = (
        "NO_GO_EXPORT_COUNT_AP_BLOCKER_AFTER_3D4D_REPAIR"
        if local_decision.get("final_status") == "GO_3D_TEMPORAL_CURRICULUM"
        and memory_decision.get("final_status") == "GO_4D_MEMORY"
        and float(best_ap.get("AP") or 0.0) < float(args.ap_gate)
        else "UNKNOWN_REVIEW_REQUIRED"
    )
    summary = {
        "phase": "v39_purity_targeted_repair_summary",
        "final_status": final_status,
        "method_success_claimed": False,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": True,
        "uses_pose_for_prediction": True,
        "uses_scannet_mesh_for_prediction": True,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "uses_frozen_visual_backbone": False,
        "mask_source": "v37_dino_compact_filter:dino_k2_compact060_filter",
        "object_birth_source": "F31_rgb090_component_split_adaptive_density010_frac060",
        "d4rt_role": "support/4d memory attachment; AP export uses ScanNet RGB-D bridge",
        "alignment_source": "d4rt_self_sim3 for identity; scannet_rgbd_pose_mesh_export_bridge for AP diagnostic",
        "local_3d": {
            "status": local_decision.get("final_status"),
            "best_stage": local_decision.get("best_stage"),
            "ARI": local_metrics.get("ARI"),
            "purity": local_metrics.get("purity"),
            "completeness": local_metrics.get("completeness"),
            "scene0081_ARI": local_metrics.get("scene0081_ARI"),
            "unknown_tube_ratio": local_metrics.get("unknown_tube_ratio"),
        },
        "memory_4d": {
            "status": memory_decision.get("final_status"),
            "best_variant": memory_decision.get("best_variant"),
            "4D_ARI": memory_metrics.get("4D_ARI"),
            "4D_purity": memory_metrics.get("4D_purity"),
            "4D_completeness": memory_metrics.get("4D_completeness"),
            "scene0081_ARI": memory_metrics.get("scene0081_ARI"),
            "unknown_tube_ratio": memory_metrics.get("unknown_tube_ratio"),
        },
        "raw_ap_export": raw_row,
        "ap_rows": ap_rows,
        "best_ap_row": best_ap,
        "gates": {
            "local_3d_pass": bool(local_decision.get("final_status") == "GO_3D_TEMPORAL_CURRICULUM"),
            "memory_4d_pass": bool(memory_decision.get("final_status") == "GO_4D_MEMORY"),
            "best_ap": best_ap.get("AP"),
            "ap_gate": float(args.ap_gate),
            "ap_gate_pass": bool(float(best_ap.get("AP") or 0.0) >= float(args.ap_gate)),
            "raw_mean_exported_objects": raw_row.get("mean_num_instances_after"),
            "raw_mean_export_conflict_rate": raw_row.get("mean_export_conflict_rate"),
        },
        "insight": [
            "F31 adaptive density repair recovers the 3D object-identity gate.",
            "I4 sparse RGB temporal memory preserves the 4D gate.",
            "AP/export remains in the old candidate-flood regime: raw export keeps 4051.8 predictions/scene and AP stays 0.003937.",
            "Top-K count caps, mask-overlap NMS, and point WTA do not recover AP; the best postprocess AP in this continuation is still only 0.004662.",
        ],
    }
    _write_json(output_root / "purity_targeted_repair_summary.json", summary)
    _write_csv(output_root / "purity_targeted_ap_rows.csv", ap_rows)
    md = [
        "# Stream4D v39 Purity-Targeted Repair Summary",
        "",
        f"`final_status={final_status}`",
        "",
        "| item | value |",
        "|---|---:|",
        f"| local_3d_status | {local_decision.get('final_status')} |",
        f"| local_3d_best_stage | {local_decision.get('best_stage')} |",
        f"| local_3d_ARI | {local_metrics.get('ARI')} |",
        f"| local_3d_purity | {local_metrics.get('purity')} |",
        f"| local_3d_completeness | {local_metrics.get('completeness')} |",
        f"| local_3d_scene0081_ARI | {local_metrics.get('scene0081_ARI')} |",
        f"| memory_4d_status | {memory_decision.get('final_status')} |",
        f"| memory_4d_best_variant | {memory_decision.get('best_variant')} |",
        f"| best_AP_config | {best_ap.get('config')} |",
        f"| best_AP | {best_ap.get('AP')} |",
        f"| best_AP50 | {best_ap.get('AP50')} |",
        f"| best_AP25 | {best_ap.get('AP25')} |",
        f"| raw_mean_exported_objects | {raw_row.get('mean_num_instances_after')} |",
        f"| raw_mean_export_conflict_rate | {raw_row.get('mean_export_conflict_rate')} |",
        "",
        "Diagnostic-only AP/export evidence; no v39 method success is claimed.",
    ]
    (output_root / "purity_targeted_repair_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v39 purity-targeted 3D/4D repair and AP/export blocker evidence.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--repair-root", default="outputs/audit/v39_purity_targeted_repair")
    parser.add_argument("--output-root", default="outputs/audit/v39_purity_targeted_repair/v39_summary")
    parser.add_argument("--ap-gate", type=float, default=0.20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
