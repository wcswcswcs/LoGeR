from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stream4d_native.v47_common import ROOT, read_json, utc_now, write_csv, write_json


def _load(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {"missing": True, "path": str(path)}


def _best_v37() -> dict[str, Any]:
    path = ROOT / "outputs/audit/v44_native_full_probe5_core_first_l034/v44_native_typed_summary.json"
    payload = _load(path)
    return {
        "source": str(path.relative_to(ROOT)),
        "status": payload.get("baseline", {}).get("v37_status"),
        "final_status": payload.get("baseline", {}).get("v37_final_status"),
        "metrics": payload.get("baseline", {}).get("v37_best_metrics", {}),
    }


def build_fact_lock(*, carrier_cache_root: Path, scenes: list[str]) -> dict[str, Any]:
    v37 = _best_v37()
    v41 = _load(ROOT / "outputs/audit/v41_1_native_support_metrics_probe5/native_support_metrics_summary.json")
    v44 = _load(ROOT / "outputs/audit/v44_native_full_probe5_core_first_l034/v44_native_typed_summary.json")
    v46_fact = _load(ROOT / "outputs/audit/v46_fact_lock/fact_lock.json")
    v46_final = _load(ROOT / "outputs/audit/v46_final_decision/v46_final_decision.json")
    carrier_scene_dirs = {scene: (carrier_cache_root / scene).exists() for scene in scenes}
    stride_rows: list[dict[str, Any]] = []
    for scene in scenes:
        for manifest in sorted((carrier_cache_root / scene).glob("carriers_window*_manifest.json")):
            data = read_json(manifest)
            frames = [int(value) for value in data.get("frame_ids", [])]
            diffs = [b - a for a, b in zip(frames, frames[1:])]
            stride_rows.append(
                {
                    "scene": scene,
                    "manifest": str(manifest.relative_to(ROOT) if manifest.is_relative_to(ROOT) else manifest),
                    "frame_count": len(frames),
                    "first_frame": frames[0] if frames else None,
                    "last_frame": frames[-1] if frames else None,
                    "all_frame_diffs_eq_1": bool(diffs and all(diff == 1 for diff in diffs)),
                    "variant": data.get("variant"),
                    "uses_gt_for_prediction": bool(data.get("uses_gt_for_prediction", False)),
                    "uses_pose_for_prediction": bool(data.get("uses_pose_for_prediction", False)),
                    "uses_rgbd_for_prediction": bool(data.get("uses_rgbd_for_prediction", False)),
                    "uses_scannet_mesh_for_prediction": bool(data.get("uses_scannet_mesh_for_prediction", False)),
                }
            )
    fact_rows = [
        {"key": "carrier_cache_root", "source": "current_run_arg", "value": str(carrier_cache_root.relative_to(ROOT) if carrier_cache_root.is_relative_to(ROOT) else carrier_cache_root)},
        {"key": "scenes", "source": "current_run_arg", "value": ",".join(scenes)},
        {"key": "v37_4D_ARI", "source": v37["source"], "value": v37["metrics"].get("4D_ARI")},
        {"key": "v37_4D_purity", "source": v37["source"], "value": v37["metrics"].get("4D_purity")},
        {"key": "v37_4D_completeness", "source": v37["source"], "value": v37["metrics"].get("4D_completeness")},
        {"key": "v37_temporal_span_mean", "source": v37["source"], "value": v37["metrics"].get("temporal_span_mean")},
        {"key": "v41_1_4D_ARI", "source": "outputs/audit/v41_1_native_support_metrics_probe5/native_support_metrics_summary.json", "value": v41.get("aggregate_metrics", {}).get("4D_ARI")},
        {"key": "v44_best_ARI", "source": "outputs/audit/v44_native_full_probe5_core_first_l034/v44_native_typed_summary.json", "value": v44.get("aggregate_metrics", {}).get("4D_ARI")},
        {"key": "v44_best_purity", "source": "outputs/audit/v44_native_full_probe5_core_first_l034/v44_native_typed_summary.json", "value": v44.get("aggregate_metrics", {}).get("4D_purity")},
        {"key": "v46_final_label", "source": "outputs/audit/v46_final_decision/v46_final_decision.json", "value": v46_final.get("final_label")},
        {"key": "v46_positive_edge_status", "source": "outputs/audit/v46_final_decision/v46_final_decision.json", "value": v46_final.get("positive_gate", {}).get("pass")},
        {"key": "v46_supporter_reliability_status", "source": "outputs/audit/v46_final_decision/v46_final_decision.json", "value": v46_final.get("supporter_gate", {}).get("pass")},
        {"key": "v46_solver_status", "source": "outputs/audit/v46_final_decision/v46_final_decision.json", "value": v46_final.get("solver_gate", {}).get("pass")},
        {"key": "scale_guard_pass", "source": "outputs/audit/v46_fact_lock/fact_lock.json", "value": v46_fact.get("gate", {}).get("scale_guard_pass")},
        {"key": "cross_chunk_local_metric_reads", "source": "outputs/audit/v46_fact_lock/fact_lock.json", "value": 0 if v46_fact.get("gate", {}).get("cross_chunk_local_metric_reads_eq_0") else None},
        {"key": "cross_chunk_eval_reads", "source": "outputs/audit/v46_fact_lock/fact_lock.json", "value": 0 if v46_fact.get("gate", {}).get("cross_chunk_eval_reads_eq_0") else None},
        {"key": "D4RT_encoder_stride", "source": "current_carrier_cache_manifests", "value": 1 if stride_rows and all(row["all_frame_diffs_eq_1"] for row in stride_rows) else None},
        {"key": "prepared_masks_available", "source": "v46_fact_lock/current_data", "value": v46_fact.get("gate", {}).get("prepared_masks_available")},
        {"key": "carrier_cache_available", "source": "current_carrier_cache_root", "value": all(carrier_scene_dirs.values())},
    ]
    gate = {
        "v37_v41_v44_v46_facts_loaded": not v41.get("missing") and not v44.get("missing") and not v46_fact.get("missing") and not v46_final.get("missing"),
        "D4RT_encoder_stride_eq_1": bool(stride_rows and all(row["all_frame_diffs_eq_1"] for row in stride_rows)),
        "temporal_chunk_size_le_checkpoint_clip_frames": bool(stride_rows and all(int(row["frame_count"]) <= 32 for row in stride_rows)),
        "prepared_masks_available": bool(v46_fact.get("gate", {}).get("prepared_masks_available", False)),
        "carrier_cache_available": all(carrier_scene_dirs.values()),
        "scale_guard_pass": bool(v46_fact.get("gate", {}).get("scale_guard_pass", False)),
        "cross_chunk_local_metric_reads_eq_0": bool(v46_fact.get("gate", {}).get("cross_chunk_local_metric_reads_eq_0", False)),
        "cross_chunk_eval_reads_eq_0": bool(v46_fact.get("gate", {}).get("cross_chunk_eval_reads_eq_0", False)),
        "forbidden_prediction_sources_absent": bool(stride_rows and not any(row["uses_gt_for_prediction"] or row["uses_pose_for_prediction"] or row["uses_rgbd_for_prediction"] or row["uses_scannet_mesh_for_prediction"] for row in stride_rows)),
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "phase": "v47_fact_lock",
        "created_at": utc_now(),
        "carrier_cache_root": str(carrier_cache_root.relative_to(ROOT) if carrier_cache_root.is_relative_to(ROOT) else carrier_cache_root),
        "scenes": scenes,
        "carrier_scene_dirs": carrier_scene_dirs,
        "fact_rows": fact_rows,
        "stride_rows": stride_rows,
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v47 fact lock and input contract.")
    parser.add_argument("--carrier-cache-root", default="outputs/stream4d_debug_v47_stride1_d5_probe5_mf32")
    parser.add_argument("--scenes", default="scene0011_00,scene0030_00,scene0050_00,scene0081_01,scene0591_00")
    parser.add_argument("--output-root", default="outputs/audit/v47_fact_lock")
    args = parser.parse_args()
    scenes = [item.strip() for item in str(args.scenes).split(",") if item.strip()]
    payload = build_fact_lock(carrier_cache_root=ROOT / str(args.carrier_cache_root), scenes=scenes)
    out = ROOT / str(args.output_root)
    write_json(out / "fact_lock.json", payload)
    write_csv(out / "fact_lock_rows.csv", payload["fact_rows"])
    write_csv(out / "stride_rows.csv", payload["stride_rows"])
    print({"summary": str(out / "fact_lock.json"), "gate": payload["gate"]})


if __name__ == "__main__":
    main()

