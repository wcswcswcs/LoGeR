from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .v47_common import ROOT, parse_float, parse_int, read_csv, read_json, utc_now, write_csv, write_json
from .v53_local_objectlets import weighted_partition_metrics


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _load_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in text.split(";") if item.strip()]
    return [str(item) for item in payload]


def _method_variant(value: Any) -> bool:
    text = str(value)
    return text.startswith("L4_") or text.startswith("L6_") or text.startswith("L11_") or text.startswith("L12_")


def _component_maps(objectlet_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = defaultdict(dict)
    for row in objectlet_rows:
        variant = str(row.get("variant"))
        objectlet_id = str(row.get("objectlet_id"))
        for component_id in _load_json_list(row.get("component_ids")):
            maps[variant][component_id] = objectlet_id
    return maps


def _support_stats(support_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    stats: dict[str, dict[str, Any]] = {}
    scenes: set[str] = set()
    gt_counts: dict[str, Counter[str]] = defaultdict(Counter)
    support_counts: Counter[str] = Counter()
    scene_by_component: dict[str, str] = {}
    for row in support_rows:
        component = str(row.get("component_id"))
        scene = str(row.get("scene"))
        scenes.add(scene)
        scene_by_component[component] = scene
        support = parse_int(row.get("support_count"), 1)
        support_counts[component] += support
        gt = str(row.get("diagnostic_gt_instance") or "")
        if gt:
            gt_counts[component][f"{scene}|{gt}"] += support
    for component, count in support_counts.items():
        gt = gt_counts[component].most_common(1)[0][0] if gt_counts[component] else ""
        stats[component] = {
            "component_id": component,
            "scene": scene_by_component.get(component, ""),
            "support_count": int(count),
            "diagnostic_gt": gt,
        }
    return stats, sorted(scenes)


def _assignments_for_map(
    support_rows: list[dict[str, str]],
    component_to_object: dict[str, str],
    *,
    scene_filter: str | None = None,
) -> list[tuple[str, str, float]]:
    assignments: list[tuple[str, str, float]] = []
    for row in support_rows:
        scene = str(row.get("scene"))
        if scene_filter is not None and scene != scene_filter:
            continue
        gt = str(row.get("diagnostic_gt_instance") or "")
        if not gt:
            continue
        component = str(row.get("component_id"))
        pred = component_to_object.get(component, f"{scene}|unknown:{component}")
        assignments.append((pred, f"{scene}|{gt}", float(parse_int(row.get("support_count"), 1))))
    return assignments


def _metrics_for_map(
    support_rows: list[dict[str, str]],
    component_to_object: dict[str, str],
    scenes: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    overall = weighted_partition_metrics(_assignments_for_map(support_rows, component_to_object))
    per_scene = {
        scene: weighted_partition_metrics(_assignments_for_map(support_rows, component_to_object, scene_filter=scene))
        for scene in scenes
    }
    return overall, per_scene


def build_local_control_gap_audit(
    *,
    support_rows_path: str | Path = "outputs/audit/v53_mask_component_support_tau005/mask_component_support_rows.csv",
    support_variant: str = "R0_visible_tau0.05",
    local_summary_path: str | Path = "outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000_l11_dynamic/local_objectlet_summary.json",
    objectlet_rows_path: str | Path = "outputs/audit/v53_local_objectlets_k0_conflict_veto025_max4000_l11_dynamic/objectlet_rows.csv",
) -> dict[str, Any]:
    support_rows = [
        row for row in read_csv(_project(support_rows_path)) if str(row.get("variant")) == str(support_variant)
    ]
    local_summary = read_json(_project(local_summary_path))
    objectlet_rows = read_csv(_project(objectlet_rows_path))
    component_stats, scenes = _support_stats(support_rows)
    maps = _component_maps(objectlet_rows)
    variant_rows = local_summary.get("variant_rows", [])
    row_by_variant = {str(row.get("variant")): row for row in variant_rows if isinstance(row, dict)}
    mask_variant = "L9_mask_only_representative_support"
    mask_map = maps.get(mask_variant, {})
    mask_metrics, mask_scene_metrics = _metrics_for_map(support_rows, mask_map, scenes)

    variant_gap_rows: list[dict[str, Any]] = []
    scene_gap_rows: list[dict[str, Any]] = []
    for variant, component_map in sorted(maps.items()):
        if not (_method_variant(variant) or variant == mask_variant):
            continue
        metrics, scene_metrics = _metrics_for_map(support_rows, component_map, scenes)
        row = row_by_variant.get(variant, {})
        variant_gap_rows.append(
            {
                "variant": variant,
                "is_method_variant": _method_variant(variant),
                "4D_ARI": metrics["ARI"],
                "4D_purity": metrics["purity"],
                "4D_completeness": metrics["completeness"],
                "reported_4D_ARI": row.get("4D_ARI"),
                "reported_4D_purity": row.get("4D_purity"),
                "reported_4D_completeness": row.get("4D_completeness"),
                "component_coverage_ratio": row.get("component_coverage_ratio"),
                "mean_predictions_per_scene": row.get("mean_predictions_per_scene"),
                "conflict_rate": row.get("conflict_rate"),
                "outside_residual_mean": row.get("outside_residual_mean"),
                "real_minus_mask_only_ARI": float(metrics["ARI"] - mask_metrics["ARI"]),
                "real_minus_mask_only_purity": float(metrics["purity"] - mask_metrics["purity"]),
                "real_minus_mask_only_completeness": float(metrics["completeness"] - mask_metrics["completeness"]),
                "success_gate_pass": bool(row.get("success_gate", {}).get("pass")) if isinstance(row.get("success_gate"), dict) else False,
            }
        )
        for scene in scenes:
            scene_gap_rows.append(
                {
                    "variant": variant,
                    "scene": scene,
                    "ARI": scene_metrics[scene]["ARI"],
                    "purity": scene_metrics[scene]["purity"],
                    "completeness": scene_metrics[scene]["completeness"],
                    "mask_only_ARI": mask_scene_metrics[scene]["ARI"],
                    "mask_only_purity": mask_scene_metrics[scene]["purity"],
                    "mask_only_completeness": mask_scene_metrics[scene]["completeness"],
                    "real_minus_mask_only_ARI": scene_metrics[scene]["ARI"] - mask_scene_metrics[scene]["ARI"],
                    "real_minus_mask_only_completeness": scene_metrics[scene]["completeness"]
                    - mask_scene_metrics[scene]["completeness"],
                }
            )

    method_rows = [row for row in variant_gap_rows if row["is_method_variant"]]
    best_method = max(method_rows, key=lambda row: (parse_float(row.get("4D_ARI")), parse_float(row.get("4D_completeness"))), default={})
    best_l11 = max(
        [row for row in method_rows if str(row.get("variant")).startswith("L11_")],
        key=lambda row: (parse_float(row.get("4D_ARI")), parse_float(row.get("4D_completeness"))),
        default={},
    )
    best_map = maps.get(str(best_method.get("variant")), {})
    uncovered_rows: list[dict[str, Any]] = []
    for component_id, stats in component_stats.items():
        in_best = component_id in best_map
        in_mask = component_id in mask_map
        if in_best == in_mask:
            continue
        uncovered_rows.append(
            {
                "component_id": component_id,
                "scene": stats["scene"],
                "support_count": stats["support_count"],
                "diagnostic_gt": stats["diagnostic_gt"],
                "in_best_method": in_best,
                "in_mask_only": in_mask,
                "gap_type": "mask_only_only" if in_mask and not in_best else "method_only",
            }
        )
    uncovered_rows.sort(key=lambda row: (-parse_int(row.get("support_count")), str(row.get("scene")), str(row.get("component_id"))))

    summary = {
        "phase": "v53_local_control_gap_audit",
        "created_at": utc_now(),
        "support_rows_path": str(support_rows_path),
        "local_summary_path": str(local_summary_path),
        "objectlet_rows_path": str(objectlet_rows_path),
        "support_variant": support_variant,
        "best_method_variant": best_method.get("variant"),
        "best_method_ARI": best_method.get("4D_ARI"),
        "best_method_purity": best_method.get("4D_purity"),
        "best_method_completeness": best_method.get("4D_completeness"),
        "mask_only_ARI": mask_metrics["ARI"],
        "mask_only_purity": mask_metrics["purity"],
        "mask_only_completeness": mask_metrics["completeness"],
        "best_method_real_minus_mask_only_ARI": best_method.get("real_minus_mask_only_ARI"),
        "best_method_success_gate_pass": best_method.get("success_gate_pass"),
        "best_l11_variant": best_l11.get("variant"),
        "best_l11_ARI": best_l11.get("4D_ARI"),
        "best_l11_real_minus_mask_only_ARI": best_l11.get("real_minus_mask_only_ARI"),
        "mask_only_only_component_count": sum(1 for row in uncovered_rows if row["gap_type"] == "mask_only_only"),
        "method_only_component_count": sum(1 for row in uncovered_rows if row["gap_type"] == "method_only"),
        "top_mask_only_only_support_count": sum(
            parse_int(row.get("support_count")) for row in uncovered_rows if row["gap_type"] == "mask_only_only"
        ),
        "repair_conclusion": (
            "selection_repair_did_not_close_control_gap"
            if parse_float(best_method.get("real_minus_mask_only_ARI"), -9999.0) < 0.10
            else "selection_repair_closed_local_control_gap"
        ),
        "blocker_location": "selection_objective_tradeoff_after_support_and_reprojection_pass",
        "evidence_chain": [
            "Phase1 support tau0.05 passed incidence gate.",
            "Phase4 K0 conflict-veto 0.25 passed reprojection gate.",
            "Lowering min_new to 0.10/0.00 increases coverage but lowers ARI and purity.",
            "Dynamic uncovered-gain beam lowers conflict/outside but does not recover completeness enough.",
        ],
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {
        "summary": summary,
        "variant_gap_rows": variant_gap_rows,
        "scene_gap_rows": scene_gap_rows,
        "component_gap_rows": uncovered_rows[:2000],
    }


def write_local_control_gap_audit(output_root: str | Path, payload: dict[str, Any]) -> None:
    out = _project(output_root)
    write_json(out / "local_control_gap_summary.json", payload["summary"])
    write_csv(out / "variant_gap_rows.csv", payload["variant_gap_rows"])
    write_csv(out / "scene_gap_rows.csv", payload["scene_gap_rows"])
    write_csv(out / "component_gap_rows.csv", payload["component_gap_rows"])
