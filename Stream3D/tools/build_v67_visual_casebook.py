from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v67_local_baselines import _representative_pairs_by_chunk  # noqa: E402
from stream4d_native.v67_local_mask_graph import (  # noqa: E402
    _component_cc_mapping,
    _split_same_frame,
    _support_by_pair,
)
from stream4d_native.v67_mask_universe import _colorize_labels, _frame_mask_stats  # noqa: E402
from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from tools.run_v65_scene_multiview_ap import _load_gt_2d, _read_label_png, _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _chunk_rows, _load_csv_rows, _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _discover_pipeline_root, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


def _resize_rgb(rgb: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if rgb.shape[:2] == shape_hw:
        return rgb
    return cv2.resize(rgb, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_AREA)


def _title(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _pred_for_frame(mask: np.ndarray, mapping: dict[tuple[int, int], int], frame_id: int) -> np.ndarray:
    pred = np.zeros(mask.shape, dtype=np.int64)
    for mask_id in np.unique(mask):
        mask_id_i = int(mask_id)
        if mask_id_i <= 0:
            continue
        label = int(mapping.get((int(frame_id), mask_id_i), 0))
        if label > 0:
            pred[mask == mask_id_i] = label
    return pred


def _oracle_for_frame(mask: np.ndarray, gt: np.ndarray, frame_id: int, nodes: set[tuple[int, int]]) -> np.ndarray:
    pred = np.zeros(mask.shape, dtype=np.int64)
    stats = _frame_mask_stats(mask, gt)
    for mask_id, item in stats.items():
        key = (int(frame_id), int(mask_id))
        if key not in nodes:
            continue
        label = int(item.get("majority_gt") or 0)
        if label > 0:
            pred[mask == int(mask_id)] = label
    return pred


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    max_cases = int(args.max_cases)
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for scene in scenes:
        if len(rows) >= max_cases:
            break
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "soma_fullscene_pipeline_root"})
            continue
        mask_dir = _mask_dir_from_pipeline(pipeline_root)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        support = _support_by_pair(pipeline_root, scene, str(args.support_variant))
        representative_by_chunk = _representative_pairs_by_chunk(pipeline_root, scene)
        chunks = _chunk_rows(pipeline_root, scene)
        stride_frames = stream.frame_ids(stride=int(args.stride), max_frames=None)
        for chunk in chunks:
            if len(rows) >= max_cases:
                break
            chunk_id = str(chunk.get("chunk_id"))
            raw_start = int(float(chunk.get("raw_frame_start") or 0))
            raw_end = int(float(chunk.get("raw_frame_end") or raw_start))
            frame_ids = [int(frame) for frame in stride_frames if raw_start <= int(frame) <= raw_end]
            if not frame_ids:
                continue
            frame_id = frame_ids[len(frame_ids) // 2]
            shape_hw = tuple(int(value) for value in stream.load_depth(frame_id).shape)
            mask_path = mask_dir / f"{int(frame_id)}.png"
            if not mask_path.exists():
                continue
            nodes = set(representative_by_chunk.get(chunk_id, set()))
            if not nodes:
                nodes = {pair for pair in support if raw_start <= pair[0] <= raw_end}
            if not nodes:
                continue
            cc_mapping = _component_cc_mapping(nodes, support)
            g2_mapping, violations = _split_same_frame(cc_mapping, support)
            rgb = _resize_rgb(stream.load_rgb(frame_id), shape_hw)
            gt = _load_gt_2d(scene, frame_id, shape_hw)
            mask = _read_label_png(mask_path, shape_hw)
            g2_pred = _pred_for_frame(mask, g2_mapping, frame_id)
            oracle_pred = _oracle_for_frame(mask, gt, frame_id, nodes)
            panels = [
                _title(rgb, "RGB"),
                _title(_colorize_labels(gt), "GT diagnostic"),
                _title(_colorize_labels(mask), "Raw CropFormer"),
                _title(_colorize_labels(g2_pred), "G2 graph"),
                _title(_colorize_labels(oracle_pred), "G7 oracle"),
            ]
            panel = np.concatenate(panels, axis=1)
            case_id = f"case_{len(rows):04d}"
            out_path = image_root / f"{case_id}_{scene}_{chunk_id.replace(':', '_')}_frame{int(frame_id):06d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            rows.append(
                {
                    "case_id": case_id,
                    "scene_id": scene,
                    "chunk_id": chunk_id,
                    "frame_id": int(frame_id),
                    "failure_type": "overfragment_local_graph",
                    "method_layer": "G2_component_cc_same_frame_split",
                    "oracle_layer": "G7_oracle_representative_graph_majority_GT",
                    "same_frame_cannot_link_violation_count": int(violations),
                    "representative_node_count": int(len(nodes)),
                    "image_path": _rel(out_path),
                    "pipeline_root": _rel(pipeline_root),
                    "uses_gt_for_prediction": False,
                    "gt_layer_diagnostic_flag": True,
                }
            )
    _write_csv(output_root / "casebook_rows.csv", rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    summary = {
        "phase": "v67_visual_casebook",
        "case_count": int(len(rows)),
        "screenshot_count": int(len(rows)),
        "viewer_scene_count": 0,
        "method_layers_load_without_GT": True,
        "GT_layer_diagnostic_flag": True,
        "bookmark_count": 0,
        "major_failure_type_counts": {"overfragment_local_graph": int(len(rows))},
        "gate": {
            "case_count_ge_50": len(rows) >= 50,
            "viewer_scene_count_ge_5": False,
            "bookmark_count_ge_30": False,
            "method_layers_load_without_GT": True,
        },
        "decision": "PARTIAL_2D_CASEBOOK_VIEWER_NOT_BUILT",
        "rows": {
            "casebook_rows_csv": _rel(output_root / "casebook_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
            "image_dir": _rel(image_root),
        },
        "notes": [
            "2D casebook panels are RGB, GT diagnostic, raw CropFormer, G2 graph, and G7 oracle.",
            "The 3D Viser viewer required by the v67 plan was not built in this run and is recorded as unmet rather than fabricated.",
        ],
    }
    _write_json(output_root / "casebook_summary.json", summary)
    sha_rows = []
    for path in [output_root / "casebook_summary.json", output_root / "casebook_rows.csv", output_root / "missing_input_rows.csv"]:
        if path.exists():
            sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    for path in sorted(image_root.glob("*.png")):
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v67 2D visual failure casebook.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/audit/v67_visual_casebook")
    parser.add_argument("--support-variant", default="I0_visible_tau0.10")
    parser.add_argument("--max-cases", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
