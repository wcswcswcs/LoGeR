from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.v71_representative_setcover import _load_pipeline_roots  # noqa: E402
from tools.run_v66_local_chunk_eval import _frame_data  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    sel = mask.astype(bool)
    out[sel] = out[sel] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text[:80], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _load_proposal_rows(path: Path, scenes: set[str]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    broad: list[dict[str, Any]] = []
    sp4_by_source: dict[str, list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene_id") or "")
            if scenes and scene not in scenes:
                continue
            if row.get("variant") == "SP0_existing_masks_baseline" and str(row.get("source_broad_large_risk")) == "True":
                broad.append(row)
            if row.get("variant") == "SP4_same_frame_mask_constrained_cut":
                for source in str(row.get("source_mask_ids") or "").split("|"):
                    sp4_by_source.setdefault(source, []).append(row)
    broad = sorted(broad, key=lambda row: _float(row.get("majority_iou_diagnostic")), reverse=True)
    return broad, sp4_by_source


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = set(_parse_csv_list(args.scenes))
    output_root = _rooted(args.output_root)
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    broad_rows, sp4_by_source = _load_proposal_rows(_rooted(args.proposal_rows), scenes)
    pipeline_roots = _load_pipeline_roots(_rooted(args.witness_summary), list(scenes or DEFAULT_SCENES))

    case_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for broad in broad_rows:
        if len(case_rows) >= int(args.case_count):
            break
        scene = str(broad.get("scene_id") or "")
        frame_id = int(float(broad.get("frame_id") or -1))
        mask_id = int(float(broad.get("target_mask_id") or -1))
        key = (scene, frame_id, mask_id)
        if key in seen or scene not in pipeline_roots:
            continue
        seen.add(key)
        stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
        rgb = stream.load_rgb(frame_id)
        mask_dir = _mask_dir_from_pipeline(pipeline_roots[scene])
        fd = _frame_data(scene, [frame_id], mask_dir)[0]
        pred_mask = fd["mask"]
        gt_mask = fd["gt"]
        if rgb.shape[:2] != pred_mask.shape[:2]:
            rgb = cv2.resize(rgb, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_AREA)
        source_sel = pred_mask == mask_id
        gt_id = int(float(broad.get("majority_gt_id_diagnostic") or 0))
        gt_sel = gt_mask == gt_id if gt_id > 0 else np.zeros_like(gt_mask, dtype=bool)

        sp4_rows_raw = sorted(
            sp4_by_source.get(str(broad.get("target_mask_observation_id") or ""), []),
            key=lambda row: _float(row.get("majority_iou_diagnostic")),
            reverse=True,
        )
        sp4_rows = []
        sp4_seen: set[str] = set()
        for row in sp4_rows_raw:
            obs_id = str(row.get("target_mask_observation_id") or "")
            if obs_id in sp4_seen:
                continue
            sp4_seen.add(obs_id)
            sp4_rows.append(row)
            if len(sp4_rows) >= 3:
                break
        sub_sel = np.zeros_like(source_sel, dtype=bool)
        sub_ids = []
        for row in sp4_rows:
            sub_id = int(float(row.get("target_mask_id") or -1))
            if sub_id >= 0:
                sub_sel |= pred_mask == sub_id
                sub_ids.append(str(row.get("target_mask_observation_id") or ""))

        source_panel = _label(_overlay(rgb, source_sel, (255, 64, 64), 0.48), "source broad/underseg mask")
        sub_panel = _label(_overlay(rgb, sub_sel, (64, 255, 96), 0.50), "SP4 clean subproposal proxies")
        gt_panel = _label(_overlay(rgb, gt_sel, (80, 160, 255), 0.48), "diagnostic GT overlay")
        rgb_panel = _label(rgb, f"RGB {scene} frame {frame_id}")
        panel = np.concatenate([rgb_panel, source_panel, sub_panel, gt_panel], axis=1)
        out_path = image_dir / f"case_{len(case_rows):03d}_{scene}_f{frame_id}_m{mask_id}_diagnostic_gt.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        case_rows.append(
            {
                "case_id": len(case_rows),
                "scene_id": scene,
                "chunk_id": broad.get("chunk_id"),
                "frame_id": frame_id,
                "source_mask_id": mask_id,
                "source_mask_observation_id": broad.get("target_mask_observation_id"),
                "source_majority_iou_diagnostic": broad.get("majority_iou_diagnostic"),
                "source_area_ratio": broad.get("proposal_area_ratio"),
                "source_semantic_entropy": broad.get("semantic_entropy"),
                "sp4_subproposal_count": len(sp4_rows),
                "sp4_subproposal_ids": "|".join(sub_ids),
                "image_path": _rel(out_path),
                "uses_gt_for_visual_annotation": True,
                "diagnostic_only": True,
                "forbidden_for_method_table": True,
            }
        )

    _write_csv(output_root / "casebook_rows.csv", case_rows)
    summary = {
        "phase": "v72_phase2_broad_decomposition_case_inspection",
        "decision": "BROAD_DECOMPOSITION_CASE_INSPECTION_DONE",
        "case_count": len(case_rows),
        "requested_case_count": int(args.case_count),
        "uses_gt_for_visual_annotation": True,
        "diagnostic_only": True,
        "forbidden_for_method_table": True,
        "notes": [
            "Cases are selected from SP0 broad/underseg source masks with highest diagnostic IoU.",
            "GT overlay is for diagnosis only and appears in file names and rows.",
        ],
    }
    _write_json(output_root / "casebook_summary.json", summary)
    sha_rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export v72 broad decomposition diagnostic cases.")
    parser.add_argument("--proposal-rows", default="outputs/audit/v72_phase2_semantic_proposals/proposal_rows.csv")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v72_visualizations/semantic_proposals_broad_cases")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--case-count", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
