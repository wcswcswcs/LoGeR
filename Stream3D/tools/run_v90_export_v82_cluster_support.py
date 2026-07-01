from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_v80_cmap_af_l2h_pipeline as v80  # noqa: E402
from tools import run_v82_revised_causal_tracklet_memory as v82  # noqa: E402


DEFAULT_OUT = ROOT / "outputs/audit/v90_phase3_v82_full_support"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _make_v82_args(args: argparse.Namespace, out_root: Path) -> argparse.Namespace:
    parser = v82.build_arg_parser()
    v82_args = parser.parse_args(
        [
            "--phase",
            "phase1",
            "--split",
            args.split,
            "--run-tag",
            args.run_tag,
            "--pipeline-root",
            str((out_root / "replay_pipeline").relative_to(ROOT)),
            "--phase1-output-root",
            str((out_root / "replay_phase1").relative_to(ROOT)),
            "--local-shadow-root",
            str((out_root / "replay_local_shadow").relative_to(ROOT)),
            "--appearance-feature-mode",
            "dino_csv",
            "--appearance-feature-rows",
            "outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv",
        ]
    )
    return v82_args


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    v82_args = _make_v82_args(args, out_root)
    local_args = v82._v81_phase1_args(v82_args)
    incidence = v80._load_incidence(local_args)
    feature_summary, bundles = v80._run_phase1(local_args, incidence)
    signed_summary, signed_graphs = v80._run_phase2(local_args, bundles)
    cluster_summary, clusters = v80._run_phase4(local_args, signed_graphs, bundles)

    support_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    for (scene, chunk), graph in sorted(clusters.items()):
        data = graph["data"]
        carriers_all = graph["carriers"]
        for label, indices in sorted(graph["label_to_indices"].items(), key=lambda item: int(item[0])):
            cluster_id = int(label)
            local_slot_id = f"V80_object:c{int(chunk)}:cluster{cluster_id}"
            cluster_carriers = [carriers_all[int(idx)] for idx in indices]
            frame_ids: set[int] = set()
            mask_keys: set[tuple[int, int]] = set()
            support_observation_count = 0
            for carrier in cluster_carriers:
                for obs in data["carrier_obs"][carrier]:
                    frame_id = int(obs["frame"])
                    mask_id = int(obs["mask"])
                    frame_ids.add(frame_id)
                    mask_keys.add((frame_id, mask_id))
                    support_observation_count += 1
                    support_rows.append(
                        {
                            "scene_id": scene,
                            "chunk_id": int(chunk),
                            "frame_id": frame_id,
                            "mask_id": mask_id,
                            "history_id": "",
                            "local_slot_id": local_slot_id,
                            "cluster_id": cluster_id,
                            "native_carrier_global_id": carrier,
                            "carrier_uv_x": float(obs.get("uv_x", 0.0)),
                            "carrier_uv_y": float(obs.get("uv_y", 0.0)),
                            "confidence": float(obs.get("confidence", 0.0)),
                            "visibility_prob": 1.0 if obs.get("visible") else 0.0,
                            "observed_mask_support_density": float(obs.get("support_density", 0.0)),
                            "source_observation_table": "v75_soft_incidence_replayed_v82_phase1",
                            "native_support_kind": "v82_replayed_local_object_cluster_carrier_uv",
                            "native_support_allowed": True,
                            "is_scannet_ap_export": False,
                            "uses_gt_for_prediction": False,
                            "uses_rgbd_pose_mesh_for_export": False,
                            "method_uses_gt": False,
                            "uses_future": False,
                        }
                    )
            slot_rows.append(
                {
                    "scene_id": scene,
                    "chunk_id": int(chunk),
                    "local_slot_id": local_slot_id,
                    "cluster_id": cluster_id,
                    "carrier_count": len(cluster_carriers),
                    "support_observation_count": support_observation_count,
                    "support_frame_count": len(frame_ids),
                    "support_mask_count": len(mask_keys),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    _write_csv(out_root / "native_carrier_support_rows.csv", support_rows)
    _write_csv(out_root / "local_slot_support_rows.csv", slot_rows)
    summary = {
        "phase": "v90_phase3_v82_full_support_export",
        "schema": "stream4d_v90_phase3_v82_full_support_export_v1",
        "split": args.split,
        "support_row_count": len(support_rows),
        "slot_count": len(slot_rows),
        "scene_count": len({row["scene_id"] for row in slot_rows}),
        "feature_summary_decision": feature_summary.get("decision", ""),
        "signed_summary_decision": signed_summary.get("decision", ""),
        "cluster_summary_decision": cluster_summary.get("decision", ""),
        "source": {
            "v82_run_tag": args.run_tag,
            "incidence_root": local_args.v75_phase1_root,
            "incidence_variant": local_args.incidence_variant,
            "semantic_feature_rows": local_args.semantic_feature_rows,
            "appearance_feature_rows": local_args.appearance_feature_rows,
        },
        "outputs": {
            "native_carrier_support_rows": _rel(out_root / "native_carrier_support_rows.csv"),
            "local_slot_support_rows": _rel(out_root / "local_slot_support_rows.csv"),
        },
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "runtime_sec": time.time() - started,
    }
    _write_json(out_root / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay v82 Phase1 local clusters and export full V80_object carrier UV support for v90 Phase3.")
    parser.add_argument("--run-tag", default="dev_v82_phase1_b0")
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT.relative_to(ROOT)))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
