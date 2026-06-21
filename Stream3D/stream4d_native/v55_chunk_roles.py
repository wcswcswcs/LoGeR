from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import ROOT, parse_bool, parse_float, parse_int, read_csv, read_json, utc_now, write_csv, write_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _minmax(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return 1.0
    return float((value - lo) / (hi - lo))


def _component_sets(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        out[str(row.get("chunk_id"))].add(str(row.get("component_id")))
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return float(len(left & right) / max(len(left | right), 1))


def _representative_coverage_by_chunk(rows: list[dict[str, Any]], preferred_variant: str = "K8_underseg_capped_partial_repair") -> dict[str, float]:
    best: dict[str, float] = {}
    for row in rows:
        if str(row.get("variant")) == preferred_variant:
            best[str(row.get("chunk_id"))] = parse_float(row.get("component_coverage"))
    if best:
        return best
    for row in rows:
        chunk_id = str(row.get("chunk_id"))
        coverage = parse_float(row.get("component_coverage"))
        best[chunk_id] = max(best.get(chunk_id, 0.0), coverage)
    return best


def _local_completeness_by_chunk(rows: list[dict[str, Any]], variant: str) -> dict[str, float]:
    return {
        str(row.get("chunk_id")): parse_float(row.get("local_completeness"))
        for row in rows
        if str(row.get("variant")) == variant
    }


def _role_for_row(
    row: dict[str, Any],
    *,
    scene_mask_counts: list[float],
    scene_mask_frames: list[float],
    scene_role_index: int,
    anchor_threshold: float,
) -> tuple[str, float, str]:
    mask_count = float(row["mask_count"])
    mask_frames = float(row["mask_measurement_frame_count"])
    component_coverage = float(row["component_coverage_by_masks"])
    representative_coverage = float(row["representative_coverage"])
    boundary_prev = float(row["boundary_overlap_prev"])
    boundary_next = float(row["boundary_overlap_next"])
    boundary = max(boundary_prev, boundary_next)
    scale_ok = bool(row["scale_guard_status"])
    mask_density_score = _minmax(mask_count, scene_mask_counts)
    mask_frame_score = _minmax(mask_frames, scene_mask_frames)
    anchor_score = 0.35 * mask_density_score + 0.20 * mask_frame_score + 0.25 * representative_coverage + 0.20 * component_coverage
    if (
        scale_ok
        and anchor_score >= anchor_threshold
        and representative_coverage >= 0.60
        and component_coverage >= 0.80
    ):
        return "anchor", anchor_score, "mask_density_representative_coverage"
    if not scale_ok or (component_coverage < 0.35 and representative_coverage < 0.35):
        return "uncertain", anchor_score, "weak_scale_or_low_coverage"
    if boundary_prev >= 0.20 and boundary_next >= 0.20 and scene_role_index not in {0}:
        return "bridge", anchor_score, "strong_adjacent_component_overlap"
    if boundary >= 0.08 or component_coverage >= 0.70:
        return "update", anchor_score, "history_update_evidence_without_anchor_density"
    return "uncertain", anchor_score, "insufficient_update_evidence"


def _ensure_one_anchor_per_scene(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    repairs: list[str] = []
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_scene[str(row["scene"])].append(row)
    for scene, scene_rows in rows_by_scene.items():
        if any(row["role"] == "anchor" for row in scene_rows):
            continue
        candidates = [
            row
            for row in scene_rows
            if bool(row["scale_guard_status"])
            and float(row["representative_coverage"]) >= 0.50
            and float(row["component_coverage_by_masks"]) >= 0.70
        ]
        if not candidates:
            repairs.append(f"{scene}:no_anchor_candidate_after_widening")
            continue
        chosen = max(candidates, key=lambda row: (float(row["role_confidence"]), float(row["mask_count"])))
        chosen["role"] = "anchor"
        chosen["role_rule"] = "widened_anchor_top_mask_density_per_scene"
        chosen["role_repair_applied"] = True
        repairs.append(f"{scene}:promoted_{chosen['chunk_id']}_as_anchor")
    return rows, repairs


def infer_chunk_roles_from_features(features: list[dict[str, Any]], *, anchor_threshold: float = 0.62) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        by_scene[str(row["scene"])].append(row)
    for scene, scene_rows in sorted(by_scene.items()):
        scene_rows = sorted(scene_rows, key=lambda row: int(row["chunk_index"]))
        mask_counts = [float(row["mask_count"]) for row in scene_rows]
        mask_frames = [float(row["mask_measurement_frame_count"]) for row in scene_rows]
        for scene_role_index, row in enumerate(scene_rows):
            role, confidence, rule = _role_for_row(
                row,
                scene_mask_counts=mask_counts,
                scene_mask_frames=mask_frames,
                scene_role_index=scene_role_index,
                anchor_threshold=anchor_threshold,
            )
            out = dict(row)
            out["role_variant"] = "R3_rule_with_boundary_evidence"
            out["role"] = role
            out["role_confidence"] = confidence
            out["role_rule"] = rule
            out["role_repair_applied"] = False
            out["uses_gt_for_prediction"] = False
            out["uses_gt_for_diagnostic_labels"] = True
            rows.append(out)
    return _ensure_one_anchor_per_scene(rows)


def build_v55_chunk_roles(
    *,
    phase0_rows_path: str | Path = "outputs/audit/v55_phase0_fact_lock/v54_failure_decomposition_rows.csv",
    representative_variant_rows_path: str | Path = "outputs/audit/v54_representative_observations_k8_stride1_probe5_q4096_notopup/representative_variant_rows.csv",
    chunk_component_rows_path: str | Path = "outputs/audit/v54_chunk_universe_stride1_probe5_q4096_notopup/chunk_component_rows.csv",
    local_metric_rows_path: str | Path = "outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_metric_rows.csv",
    local_summary_path: str | Path = "outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_reproduction_summary.json",
    anchor_threshold: float = 0.62,
) -> dict[str, Any]:
    phase0_rows = read_csv(_project(phase0_rows_path))
    representative_rows = read_csv(_project(representative_variant_rows_path))
    chunk_component_rows = read_csv(_project(chunk_component_rows_path))
    local_metric_rows = read_csv(_project(local_metric_rows_path))
    local_summary = read_json(_project(local_summary_path))
    best_variant = str(local_summary.get("best_method_variant") or "")
    representative_coverage = _representative_coverage_by_chunk(representative_rows)
    component_sets = _component_sets(chunk_component_rows)
    local_completeness = _local_completeness_by_chunk(local_metric_rows, best_variant)

    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in phase0_rows:
        rows_by_scene[str(row.get("scene"))].append(row)

    features: list[dict[str, Any]] = []
    for scene, rows in sorted(rows_by_scene.items()):
        ordered = sorted(rows, key=lambda row: parse_int(row.get("chunk_index")))
        for idx, row in enumerate(ordered):
            chunk_id = str(row.get("chunk_id"))
            prev_set = component_sets.get(str(ordered[idx - 1].get("chunk_id")), set()) if idx > 0 else set()
            next_set = component_sets.get(str(ordered[idx + 1].get("chunk_id")), set()) if idx + 1 < len(ordered) else set()
            cur_set = component_sets.get(chunk_id, set())
            features.append(
                {
                    "scene": scene,
                    "chunk_id": chunk_id,
                    "chunk_index": parse_int(row.get("chunk_index")),
                    "mask_measurement_frame_count": parse_int(row.get("mask_measurement_frame_count")),
                    "mask_count": parse_int(row.get("mask_count")),
                    "component_count": parse_int(row.get("component_count")),
                    "component_coverage_by_masks": parse_float(row.get("component_coverage_by_masks")),
                    "representative_coverage": representative_coverage.get(chunk_id, 0.0),
                    "reprojection_success_rate": parse_float(row.get("v54_reprojection_success_rate_by_chunk")),
                    "boundary_overlap_prev": _jaccard(cur_set, prev_set),
                    "boundary_overlap_next": _jaccard(cur_set, next_set),
                    "scale_guard_status": not parse_bool(row.get("weak_scale_chunk")),
                    "anchor_local_completeness": local_completeness.get(chunk_id),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )

    role_rows, repairs = infer_chunk_roles_from_features(features, anchor_threshold=anchor_threshold)
    role_counts = Counter(str(row["role"]) for row in role_rows)
    scenes = sorted({str(row["scene"]) for row in role_rows})
    anchors = [row for row in role_rows if row["role"] == "anchor"]
    updates = [row for row in role_rows if row["role"] == "update"]
    anchor_mask = [float(row["mask_count"]) for row in anchors]
    update_mask = [float(row["mask_count"]) for row in updates]
    all_mask = [float(row["mask_count"]) for row in role_rows]
    role_separation_score = 0.0
    if anchor_mask and update_mask and all_mask:
        role_separation_score = float((np.mean(anchor_mask) - np.mean(update_mask)) / max(np.mean(all_mask), 1.0))
    anchor_local = [float(row["anchor_local_completeness"]) for row in anchors if row.get("anchor_local_completeness") not in ("", None)]
    update_local = [float(row["anchor_local_completeness"]) for row in updates if row.get("anchor_local_completeness") not in ("", None)]
    anchor_local_mean = _mean(anchor_local)
    update_local_mean = _mean(update_local)
    role_variant_summaries = [
        {
            "variant": "R0_uniform_chunks_control",
            "anchor_chunk_count": len(role_rows),
            "update_chunk_count": 0,
            "note": "control only; treats all chunks as anchor",
        },
        {
            "variant": "R3_rule_with_boundary_evidence",
            "anchor_chunk_count": int(role_counts.get("anchor", 0)),
            "update_chunk_count": int(role_counts.get("update", 0)),
            "bridge_chunk_count": int(role_counts.get("bridge", 0)),
            "uncertain_chunk_count": int(role_counts.get("uncertain", 0)),
            "repair_events": repairs,
        },
    ]
    gate = {
        "anchor_chunk_count_ge_scene_count": int(role_counts.get("anchor", 0)) >= len(scenes),
        "anchor_chunk_ratio_ge_0.20": float(role_counts.get("anchor", 0) / max(len(role_rows), 1)) >= 0.20,
        "update_chunk_ratio_ge_0.20": float(role_counts.get("update", 0) / max(len(role_rows), 1)) >= 0.20,
        "role_separation_score_ge_0.30": role_separation_score >= 0.30,
        "anchor_local_completeness_ge_update_plus_0.15_diagnostic": (
            anchor_local_mean is not None
            and update_local_mean is not None
            and anchor_local_mean >= update_local_mean + 0.15
        ),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "phase": "v55_chunk_roles",
        "created_at": utc_now(),
        "input_paths": {
            "phase0_rows_path": _rel(phase0_rows_path),
            "representative_variant_rows_path": _rel(representative_variant_rows_path),
            "chunk_component_rows_path": _rel(chunk_component_rows_path),
            "local_metric_rows_path": _rel(local_metric_rows_path),
            "local_summary_path": _rel(local_summary_path),
        },
        "role_variant": "R3_rule_with_boundary_evidence",
        "role_variant_summaries": role_variant_summaries,
        "scene_count": len(scenes),
        "chunk_count": len(role_rows),
        "anchor_chunk_count": int(role_counts.get("anchor", 0)),
        "update_chunk_count": int(role_counts.get("update", 0)),
        "bridge_chunk_count": int(role_counts.get("bridge", 0)),
        "uncertain_chunk_count": int(role_counts.get("uncertain", 0)),
        "anchor_chunk_ratio": float(role_counts.get("anchor", 0) / max(len(role_rows), 1)),
        "update_chunk_ratio": float(role_counts.get("update", 0) / max(len(role_rows), 1)),
        "role_separation_score": role_separation_score,
        "anchor_local_completeness": anchor_local_mean,
        "update_local_completeness": update_local_mean,
        "GT_object_coverage_by_anchor_chunks": None,
        "GT_object_coverage_by_anchor_chunks_status": "not_computed_in_phase1_no_oracle_role_prediction",
        "role_repairs_applied": repairs,
        "gate": gate,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return {"summary": summary, "role_rows": role_rows}


def _write_visualizations(out: Path, vis_root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        write_json(out / "visualization_error.json", {"error": repr(exc)})
        return [{"path": str(out / "visualization_error.json"), "status": "matplotlib_unavailable"}]

    colors = {"anchor": "tab:green", "update": "tab:blue", "bridge": "tab:orange", "uncertain": "tab:gray"}
    vis_root.mkdir(parents=True, exist_ok=True)
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["role_rows"]:
        rows_by_scene[str(row["scene"])].append(row)
    for scene, rows in sorted(rows_by_scene.items()):
        rows = sorted(rows, key=lambda row: int(row["chunk_index"]))
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.4), 2.8))
        ax.bar(x, [1.0 for _ in rows], color=[colors.get(str(row["role"]), "black") for row in rows])
        ax.set_xticks(x)
        ax.set_xticklabels([f"c{int(row['chunk_index'])}\n{row['role']}" for row in rows])
        ax.set_yticks([])
        ax.set_title(f"{scene} chunk role timeline")
        path = vis_root / f"chunk_role_timeline_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "chunk_role_timeline", "scene": scene})

        fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.8), 4.0))
        width = 0.18
        ax.bar(x - 1.5 * width, [float(row["mask_count"]) for row in rows], width=width, label="mask_count")
        ax.bar(x - 0.5 * width, [float(row["component_coverage_by_masks"]) * 300.0 for row in rows], width=width, label="coverage*300")
        ax.bar(x + 0.5 * width, [float(row["representative_coverage"]) * 300.0 for row in rows], width=width, label="rep_cov*300")
        ax.bar(x + 1.5 * width, [max(float(row["boundary_overlap_prev"]), float(row["boundary_overlap_next"])) * 300.0 for row in rows], width=width, label="boundary*300")
        ax.set_xticks(x)
        ax.set_xticklabels([f"c{int(row['chunk_index'])}" for row in rows])
        ax.set_title(f"{scene} role feature radar proxy")
        ax.legend(fontsize=8)
        path = vis_root / f"role_feature_radar_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "role_feature_radar", "scene": scene})

        fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.8), 4.0))
        ax.scatter(
            [float(row["mask_count"]) for row in rows],
            [float(row["representative_coverage"]) for row in rows],
            c=[colors.get(str(row["role"]), "black") for row in rows],
            s=[70 + 80 * float(row["role_confidence"]) for row in rows],
        )
        for row in rows:
            ax.text(float(row["mask_count"]), float(row["representative_coverage"]), f"c{int(row['chunk_index'])}", fontsize=8)
        ax.set_xlabel("mask_count")
        ax.set_ylabel("representative_coverage")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{scene} anchor/update feature examples")
        path = vis_root / f"anchor_update_examples_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "anchor_update_examples", "scene": scene})
    return manifest


def write_v55_chunk_roles(
    output_root: str | Path,
    payload: dict[str, Any],
    visualization_root: str | Path = "outputs/audit/v55_visualizations/chunk_roles",
) -> None:
    out = _project(output_root)
    vis = _project(visualization_root)
    write_json(out / "chunk_role_summary.json", payload["summary"])
    write_csv(out / "chunk_role_rows.csv", payload["role_rows"])
    manifest = _write_visualizations(out, vis, payload)
    write_json(out / "visualization_manifest.json", {"phase": "v55_chunk_roles", "files": manifest})


__all__ = ["build_v55_chunk_roles", "infer_chunk_roles_from_features", "write_v55_chunk_roles"]
