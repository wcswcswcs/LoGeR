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


def _safe_get(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _chunk_index(chunk_id: str, fallback: int = -1) -> int:
    if "chunk" not in chunk_id:
        return fallback
    try:
        return int(chunk_id.rsplit("chunk", 1)[1])
    except ValueError:
        return fallback


def _count_mask_frames(chunk_mask_rows: list[dict[str, Any]]) -> dict[str, int]:
    frames_by_chunk: dict[str, set[int]] = defaultdict(set)
    for row in chunk_mask_rows:
        frames_by_chunk[str(row.get("chunk_id"))].add(parse_int(row.get("raw_frame_id")))
    return {chunk_id: len(frames) for chunk_id, frames in frames_by_chunk.items()}


def _reprojection_success_by_chunk(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        chunk_id = str(row.get("chunk_id"))
        values[chunk_id].append(1.0 if parse_bool(row.get("reprojection_success")) else 0.0)
    return {chunk_id: _mean(scores) for chunk_id, scores in values.items()}


def _local_metric_by_chunk(rows: list[dict[str, Any]], variant: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("variant")) == variant:
            out[str(row.get("chunk_id"))] = row
    return out


def _component_occurrence_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    occurrences = len(rows)
    unique_by_scene: set[tuple[str, str]] = set()
    chunks_by_component: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (str(row.get("scene")), str(row.get("component_id")))
        unique_by_scene.add(key)
        chunks_by_component[key].add(str(row.get("chunk_id")))
    unique_count = len(unique_by_scene)
    repeats = [len(chunks) for chunks in chunks_by_component.values()]
    return {
        "chunk_component_occurrence_count": int(occurrences),
        "unique_scene_component_count": int(unique_count),
        "component_occurrence_to_unique_ratio": float(occurrences / max(unique_count, 1)),
        "mean_chunks_per_unique_component": _mean([float(value) for value in repeats]),
    }


def build_v55_fact_lock(
    *,
    v53_final_decision_path: str | Path = "outputs/audit/v53_full_stage1/final_decision.json",
    v54_final_decision_path: str | Path = "outputs/audit/v54_final_decision/final_decision.json",
    v54_chunk_summary_path: str | Path = "outputs/audit/v54_chunk_universe_stride1_probe5_q4096_notopup/chunk_summary.json",
    v54_chunk_rows_path: str | Path = "outputs/audit/v54_chunk_universe_stride1_probe5_q4096_notopup/chunk_rows.csv",
    v54_chunk_mask_rows_path: str | Path = "outputs/audit/v54_chunk_universe_stride1_probe5_q4096_notopup/chunk_mask_rows.csv",
    v54_chunk_component_rows_path: str | Path = "outputs/audit/v54_chunk_universe_stride1_probe5_q4096_notopup/chunk_component_rows.csv",
    v54_support_summary_path: str | Path = "outputs/audit/v54_mask_component_support_tau005_stride1_probe5_q4096_notopup/support_summary.json",
    v54_reprojection_summary_path: str | Path = "outputs/audit/v54_reprojection_ledger_k0all_conflict_veto018_skip_repeated_sig_stride1_probe5_q4096_notopup_max4000_skip/reprojection_summary.json",
    v54_reprojection_rows_path: str | Path = "outputs/audit/v54_reprojection_ledger_k0all_conflict_veto018_skip_repeated_sig_stride1_probe5_q4096_notopup_max4000_skip/reprojection_ledger_rows.csv",
    v54_local_summary_path: str | Path = "outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_reproduction_summary.json",
    v54_local_metric_rows_path: str | Path = "outputs/audit/v54_local_reproduction_stride1_probe5_q4096_notopup_k0all_max4000_veto018_w123_fullchunk/local_metric_rows.csv",
) -> dict[str, Any]:
    v53_final = read_json(_project(v53_final_decision_path))
    v54_final = read_json(_project(v54_final_decision_path))
    v54_chunk_summary = read_json(_project(v54_chunk_summary_path))
    v54_support_summary = read_json(_project(v54_support_summary_path))
    v54_reprojection_summary = read_json(_project(v54_reprojection_summary_path))
    v54_local_summary = read_json(_project(v54_local_summary_path))
    chunk_rows = read_csv(_project(v54_chunk_rows_path))
    chunk_mask_rows = read_csv(_project(v54_chunk_mask_rows_path))
    chunk_component_rows = read_csv(_project(v54_chunk_component_rows_path))
    reprojection_rows = read_csv(_project(v54_reprojection_rows_path))
    local_metric_rows = read_csv(_project(v54_local_metric_rows_path))

    best_method_variant = str(v54_local_summary.get("best_method_variant") or "")
    mask_only_variant = "L9_mask_only_representative_support"
    best_by_chunk = _local_metric_by_chunk(local_metric_rows, best_method_variant)
    mask_only_by_chunk = _local_metric_by_chunk(local_metric_rows, mask_only_variant)
    mask_frame_counts = _count_mask_frames(chunk_mask_rows)
    reprojection_by_chunk = _reprojection_success_by_chunk(reprojection_rows)
    component_stats = _component_occurrence_stats(chunk_component_rows)

    decomposition_rows: list[dict[str, Any]] = []
    later_mask_frame_counts: list[float] = []
    chunk0_local_completeness: list[float] = []
    chunk0_mask_only_completeness: list[float] = []
    scenes = sorted({str(row.get("scene")) for row in chunk_rows})
    chunk_counts_by_scene = Counter(str(row.get("scene")) for row in chunk_rows)
    for row in chunk_rows:
        chunk_id = str(row.get("chunk_id"))
        idx = parse_int(row.get("chunk_index"), _chunk_index(chunk_id))
        local_row = best_by_chunk.get(chunk_id, {})
        mask_only_row = mask_only_by_chunk.get(chunk_id, {})
        if idx == 0 and local_row:
            chunk0_local_completeness.append(parse_float(local_row.get("local_completeness")))
        if idx == 0 and mask_only_row:
            chunk0_mask_only_completeness.append(parse_float(mask_only_row.get("local_completeness")))
        if idx > 0:
            later_mask_frame_counts.append(float(mask_frame_counts.get(chunk_id, 0)))
        decomposition_rows.append(
            {
                "scene": row.get("scene"),
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "raw_frame_start": parse_int(row.get("raw_frame_start")),
                "raw_frame_end": parse_int(row.get("raw_frame_end")),
                "mask_measurement_frame_count": int(mask_frame_counts.get(chunk_id, 0)),
                "mask_count": parse_int(row.get("mask_count")),
                "component_count": parse_int(row.get("component_count")),
                "component_coverage_by_masks": parse_float(row.get("chunk_component_coverage")),
                "chunk_mask_coverage": parse_float(row.get("chunk_mask_coverage")),
                "component_visibility_frame_count_mean": parse_float(row.get("component_visibility_frame_count_mean")),
                "weak_scale_chunk": parse_bool(row.get("weak_scale_chunk")),
                "allow_metric_relation": parse_bool(row.get("allow_metric_relation")),
                "v54_best_method_variant": best_method_variant,
                "v54_best_method_local_ARI": parse_float(local_row.get("local_ARI")) if local_row else None,
                "v54_best_method_local_purity": parse_float(local_row.get("local_purity")) if local_row else None,
                "v54_best_method_local_completeness": parse_float(local_row.get("local_completeness")) if local_row else None,
                "v54_best_method_conflict_rate": parse_float(local_row.get("conflict_rate")) if local_row else None,
                "v54_mask_only_restitution_local_completeness": parse_float(mask_only_row.get("local_completeness")) if mask_only_row else None,
                "v54_reprojection_success_rate_by_chunk": reprojection_by_chunk.get(chunk_id),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

    v53_local = _safe_get(v53_final, ["key_evidence", "phase6_k0_best_local"], {})
    v53_native = _safe_get(v53_final, ["key_evidence", "phase11_ap_diagnostic"], {})
    v54_readiness = v54_final.get("multi_chunk_readiness", {})
    v54_best = v54_final.get("best_formal_5scene_attempt", {})

    summary: dict[str, Any] = {
        "phase": "v55_phase0_fact_lock",
        "created_at": utc_now(),
        "input_paths": {
            "v53_final_decision_path": _rel(v53_final_decision_path),
            "v54_final_decision_path": _rel(v54_final_decision_path),
            "v54_chunk_summary_path": _rel(v54_chunk_summary_path),
            "v54_chunk_rows_path": _rel(v54_chunk_rows_path),
            "v54_chunk_mask_rows_path": _rel(v54_chunk_mask_rows_path),
            "v54_chunk_component_rows_path": _rel(v54_chunk_component_rows_path),
            "v54_support_summary_path": _rel(v54_support_summary_path),
            "v54_reprojection_summary_path": _rel(v54_reprojection_summary_path),
            "v54_reprojection_rows_path": _rel(v54_reprojection_rows_path),
            "v54_local_summary_path": _rel(v54_local_summary_path),
            "v54_local_metric_rows_path": _rel(v54_local_metric_rows_path),
        },
        "v53_best_local_variant": v53_local.get("variant"),
        "v53_4D_ARI": v53_local.get("4D_ARI"),
        "v53_purity": v53_local.get("4D_purity"),
        "v53_completeness": v53_local.get("4D_completeness"),
        "v53_real_minus_mask_only_ARI": v53_local.get("real_minus_mask_only_ARI"),
        "v53_native_carrier_materialization_available": bool(v53_native.get("method_safe_native_support_available")),
        "v54_multi_chunk_scene_count": int(v54_readiness.get("multi_chunk_scene_count") or len([s for s, n in chunk_counts_by_scene.items() if n >= 3])),
        "v54_scene_count": len(scenes),
        "v54_chunk_count": int(v54_readiness.get("chunk_count") or len(chunk_rows)),
        "v54_chunks_per_scene_mean": float(v54_readiness.get("chunks_per_scene_mean") or _mean([float(n) for n in chunk_counts_by_scene.values()]) or 0.0),
        "v54_q4096_notopup_component_count": int(v54_support_summary.get("component_count", 0)),
        "v54_chunk_summary_component_count": int(v54_chunk_summary.get("chunk_count", 0)),
        "v54_local_best_variant": v54_best.get("best_method_variant"),
        "v54_local_ARI_mean": v54_best.get("local_ARI_mean"),
        "v54_local_purity_mean": v54_best.get("local_purity_mean"),
        "v54_local_completeness_mean": v54_best.get("local_completeness_mean"),
        "v54_final_label": v54_final.get("final_label"),
        "v54_chunk0_local_completeness": _mean(chunk0_local_completeness),
        "v54_chunk0_mask_only_restitution_completeness": _mean(chunk0_mask_only_completeness),
        "v54_later_chunk_mask_frame_count_mean": _mean(later_mask_frame_counts),
        "v54_mask_measurement_frame_count_by_chunk": mask_frame_counts,
        "v54_reprojection_success_rate": v54_reprojection_summary.get("reprojection_success_rate"),
        "v54_reprojection_rows_available": bool(reprojection_rows),
        "v54_chunk_level_mask_sparsity_rows_available": bool(chunk_mask_rows),
        "component_fragmentation_observed": component_stats,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    gate = {
        "v54_multi_chunk_scene_count_ge_3": int(summary["v54_multi_chunk_scene_count"]) >= 3,
        "v54_chunks_per_scene_mean_ge_3": float(summary["v54_chunks_per_scene_mean"]) >= 3.0,
        "v54_final_label_is_no_go_local_reproduction": summary["v54_final_label"] == "NO_GO_LOCAL_REPRODUCTION",
        "v54_reprojection_success_rate_available": summary["v54_reprojection_success_rate"] is not None,
        "v54_chunk_level_mask_sparsity_rows_available": bool(summary["v54_chunk_level_mask_sparsity_rows_available"]),
    }
    gate["pass"] = bool(all(gate.values()))
    summary["gate"] = gate
    return {"summary": summary, "decomposition_rows": decomposition_rows}


def _write_visualizations(out: Path, vis_root: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        write_json(out / "visualization_error.json", {"error": repr(exc)})
        return [{"path": str(out / "visualization_error.json"), "status": "matplotlib_unavailable"}]

    vis_root.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    labels = ["v53 ARI", "v53 comp", "v54 ARI", "v54 comp", "v54 chunk0 comp"]
    values = [
        summary.get("v53_4D_ARI"),
        summary.get("v53_completeness"),
        summary.get("v54_local_ARI_mean"),
        summary.get("v54_local_completeness_mean"),
        summary.get("v54_chunk0_local_completeness"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, [0.0 if value is None else float(value) for value in values])
    ax.axhline(0.70, color="tab:red", linestyle="--", linewidth=1, label="local completeness gate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("v55 Phase 0 v53/v54 failure dashboard")
    ax.legend(loc="lower right")
    path = vis_root / "v55_phase0_v54_failure_dashboard.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    manifest.append({"path": str(path), "kind": "phase0_dashboard"})

    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["decomposition_rows"]:
        rows_by_scene[str(row["scene"])].append(row)
    for scene, rows in sorted(rows_by_scene.items()):
        rows = sorted(rows, key=lambda row: int(row["chunk_index"]))
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.5), 4))
        ax.bar(x, [float(row["mask_measurement_frame_count"]) for row in rows], label="mask measurement frames")
        ax.plot(x, [float(row["mask_count"]) for row in rows], color="tab:orange", marker="o", label="mask count")
        ax.set_xticks(x)
        ax.set_xticklabels([f"c{int(row['chunk_index'])}" for row in rows])
        ax.set_title(f"{scene} chunk mask density")
        ax.legend()
        path = vis_root / f"chunk_mask_density_timeline_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "chunk_mask_density_timeline", "scene": scene})

        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.5), 4))
        ax.bar(x, [float(row["component_count"]) for row in rows], label="chunk component count")
        ax.plot(x, [float(row["component_coverage_by_masks"]) for row in rows], color="tab:green", marker="o", label="mask coverage")
        ax.set_xticks(x)
        ax.set_xticklabels([f"c{int(row['chunk_index'])}" for row in rows])
        ax.set_title(f"{scene} component fragmentation proxy")
        ax.legend()
        path = vis_root / f"component_fragmentation_growth_{scene}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        manifest.append({"path": str(path), "kind": "component_fragmentation_growth", "scene": scene})
    return manifest


def write_v55_fact_lock(
    output_root: str | Path,
    payload: dict[str, Any],
    visualization_root: str | Path = "outputs/audit/v55_visualizations/phase0",
) -> None:
    out = _project(output_root)
    vis = _project(visualization_root)
    write_json(out / "fact_lock.json", payload["summary"])
    write_csv(out / "v54_failure_decomposition_rows.csv", payload["decomposition_rows"])
    manifest = _write_visualizations(out, vis, payload)
    write_json(out / "visualization_manifest.json", {"phase": "v55_phase0_fact_lock", "files": manifest})


__all__ = ["build_v55_fact_lock", "write_v55_fact_lock"]
