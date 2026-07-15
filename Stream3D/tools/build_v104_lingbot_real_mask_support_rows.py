#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v104_lingbot_map_only_phase7_real_mask_support_rows"
DEFAULT_OUT = AUDIT_ROOT / PHASE_ID
DEFAULT_SELECTED_ROWS = STREAM3D_ROOT / "outputs/audit/v87_phase1_mv_input_generation/frame_mask_selected_rows.csv"
DEFAULT_SCENE_ALIGNMENT_ROWS = AUDIT_ROOT / "v104_lingbot_map_only_phase4_scene_alignment_audit/scene_alignment_rows.csv"
DEFAULT_LINGBOT_CANDIDATE_ROWS = AUDIT_ROOT / "v104_lingbot_map_only_phase4_scene_alignment_audit/lingbot_candidate_roots.csv"


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(str(value)))
    except Exception:
        return default


def _load_mask(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return np.empty((0, 0), dtype=np.int64)
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int64)


def _sampling_source_to_bss(lingbot_root: Path) -> dict[int, int]:
    payload = _read_json(lingbot_root.parent / "gt" / "sampling.json")
    out: dict[int, int] = {}
    for idx, value in enumerate(payload.get("frames", [])):
        try:
            out[int(value)] = int(idx)
        except Exception:
            continue
    return out


def _best_lingbot_roots(rows: list[dict[str, str]]) -> dict[str, Path]:
    best: dict[str, tuple[int, Path]] = {}
    for row in rows:
        scene_id = row.get("scene_id", "")
        root = _project(row.get("lingbot_root", ""))
        if not scene_id or not root.exists():
            continue
        count = _as_int(row.get("sampling_source_frame_count") or row.get("complete_num_frames"), 0)
        if scene_id not in best or count > best[scene_id][0]:
            best[scene_id] = (count, root)
    return {scene_id: item[1] for scene_id, item in best.items()}


def _mask_dirs(rows: list[dict[str, str]]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for row in rows:
        scene_id = row.get("scene_id", "")
        mask_dir = row.get("mask_dir", "")
        if scene_id and mask_dir:
            out[scene_id] = _project(mask_dir)
    return out


def _finite_points_and_xy(points: np.ndarray, xy: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if xy is None:
        return points[:0], np.empty((0, 2), dtype=np.float32)
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    n = min(points.shape[0], xy.shape[0])
    points = points[:n]
    xy = xy[:n]
    finite = np.isfinite(points).all(axis=1) & np.isfinite(xy).all(axis=1)
    return points[finite], xy[finite]


def build(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(STREAM3D_ROOT))
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    from geometry_provider.lingbot_map_provider import LingBotMapGeometryProvider

    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    target_scenes = [part.strip() for part in str(args.target_scenes).split(",") if part.strip()]
    selected_rows = [
        row
        for row in _read_csv(_project(args.selected_rows))
        if row.get("scene_id") in target_scenes and str(row.get("selected_flag", "True")).lower() == "true"
    ]
    scene_rows = _read_csv(_project(args.scene_alignment_rows))
    candidate_rows = _read_csv(_project(args.lingbot_candidate_rows))
    mask_dirs = _mask_dirs(scene_rows)
    lingbot_roots = _best_lingbot_roots(candidate_rows)

    providers: dict[str, LingBotMapGeometryProvider] = {}
    source_to_bss: dict[str, dict[int, int]] = {}
    for scene_id, root in lingbot_roots.items():
        providers[scene_id] = LingBotMapGeometryProvider(
            geometry_root=root,
            max_points_per_frame=int(args.max_points_per_frame),
            min_confidence=args.min_confidence,
        )
        source_to_bss[scene_id] = _sampling_source_to_bss(root)

    frame_cache: dict[tuple[str, int], dict[str, Any]] = {}
    support_id_dir = out / "support_point_ids"
    support_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    def load_frame(scene_id: str, source_frame_id: int) -> dict[str, Any] | None:
        key = (scene_id, source_frame_id)
        if key in frame_cache:
            return frame_cache[key]
        provider = providers.get(scene_id)
        root = lingbot_roots.get(scene_id)
        bss_frame_id = source_to_bss.get(scene_id, {}).get(source_frame_id)
        mask_dir = mask_dirs.get(scene_id)
        if provider is None or root is None:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_real_mask_support_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "source_frame_id": source_frame_id,
                    "failure_id": "MISSING_LINGBOT_ROOT",
                    "severity": "blocking",
                }
            )
            return None
        if bss_frame_id is None:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_real_mask_support_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "source_frame_id": source_frame_id,
                    "lingbot_root": _rel(root),
                    "failure_id": "MISSING_BSS_FRAME_MAPPING",
                    "severity": "blocking",
                }
            )
            return None
        if mask_dir is None:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_real_mask_support_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "source_frame_id": source_frame_id,
                    "failure_id": "MISSING_MASK_DIR",
                    "severity": "blocking",
                }
            )
            return None
        mask_path = mask_dir / f"{source_frame_id}.png"
        mask = _load_mask(mask_path)
        samples = provider.load_frame_samples(bss_frame_id)
        points, xy = _finite_points_and_xy(samples.points, samples.xy)
        if mask.size == 0:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_real_mask_support_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "source_frame_id": source_frame_id,
                    "bss_frame_id": bss_frame_id,
                    "mask_path": _rel(mask_path),
                    "failure_id": "MISSING_MASK_RASTER",
                    "severity": "blocking",
                }
            )
            return None
        if xy.shape[0] == 0:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_real_mask_support_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "source_frame_id": source_frame_id,
                    "bss_frame_id": bss_frame_id,
                    "mask_path": _rel(mask_path),
                    "failure_id": "NO_PIXEL_SAMPLES",
                    "severity": "blocking",
                }
            )
            return None
        mask_ids = LingBotMapGeometryProvider._mask_ids_for_xy(mask, xy, samples.image_shape)
        frame_payload = {
            "scene_id": scene_id,
            "source_frame_id": source_frame_id,
            "bss_frame_id": bss_frame_id,
            "lingbot_root": root,
            "mask_path": mask_path,
            "mask_shape": list(mask.shape[:2]),
            "image_shape": list(samples.image_shape) if samples.image_shape is not None else [],
            "num_frame_samples": int(points.shape[0]),
            "num_pixel_samples": int(xy.shape[0]),
            "mask_ids_at_samples": mask_ids,
        }
        frame_cache[key] = frame_payload
        return frame_payload

    for row in selected_rows:
        scene_id = row.get("scene_id", "")
        source_frame_id = _as_int(row.get("frame_id"))
        mask_id = _as_int(row.get("mask_id"))
        candidate_row_id = row.get("candidate_row_id", "")
        frame_payload = load_frame(scene_id, source_frame_id)
        if frame_payload is None:
            continue
        mask_path = Path(frame_payload["mask_path"])
        mask = _load_mask(mask_path)
        mask_positive_pixels = int(np.count_nonzero(mask == mask_id)) if mask.size else 0
        if mask_positive_pixels == 0:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_real_mask_support_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "source_frame_id": source_frame_id,
                    "bss_frame_id": frame_payload["bss_frame_id"],
                    "mask_id": mask_id,
                    "candidate_row_id": candidate_row_id,
                    "mask_path": _rel(mask_path),
                    "failure_id": "MASK_ID_NO_PIXELS",
                    "severity": "blocking",
                }
            )
        support_ids = np.flatnonzero(frame_payload["mask_ids_at_samples"] == mask_id).astype(np.int32)
        support_path = support_id_dir / scene_id / f"frame{source_frame_id:06d}_mask{mask_id:06d}_row{candidate_row_id}.npy"
        support_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(support_path, support_ids)
        if support_ids.size == 0 and mask_positive_pixels > 0:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v104_lingbot_real_mask_support_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "source_frame_id": source_frame_id,
                    "bss_frame_id": frame_payload["bss_frame_id"],
                    "mask_id": mask_id,
                    "candidate_row_id": candidate_row_id,
                    "mask_path": _rel(mask_path),
                    "failure_id": "EMPTY_LINGBOT_SUPPORT_FOR_REAL_MASK",
                    "severity": "diagnostic",
                    "mask_positive_pixels": mask_positive_pixels,
                }
            )
        support_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_real_mask_support_row_v1",
                "phase_id": PHASE_ID,
                "candidate_row_id": candidate_row_id,
                "scene_id": scene_id,
                "chunk_id": row.get("chunk_id", ""),
                "source_frame_id": source_frame_id,
                "bss_frame_id": frame_payload["bss_frame_id"],
                "mask_id": mask_id,
                "history_id": row.get("history_id", ""),
                "variant": row.get("variant", ""),
                "variant_family": row.get("variant_family", ""),
                "support_scope": "real_stream4d_selected_mask_on_full_lingbot_bss",
                "support_point_count": int(support_ids.size),
                "support_point_ids_path": _rel(support_path),
                "support_point_ids_preview": ";".join(str(int(v)) for v in support_ids[:32]),
                "mask_positive_pixels": mask_positive_pixels,
                "num_frame_samples": frame_payload["num_frame_samples"],
                "num_pixel_samples": frame_payload["num_pixel_samples"],
                "mask_path": _rel(mask_path),
                "lingbot_root": _rel(frame_payload["lingbot_root"]),
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_prediction": False,
                "stream4d_metric_ready": False,
            }
        )

    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in support_rows:
        by_frame.setdefault((row["scene_id"], int(row["source_frame_id"])), []).append(row)
    for (scene_id, source_frame_id), rows in sorted(by_frame.items()):
        payload = frame_cache.get((scene_id, source_frame_id), {})
        frame_rows.append(
            {
                "schema_version": "stream4d_v104_lingbot_real_mask_support_frame_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene_id,
                "source_frame_id": source_frame_id,
                "bss_frame_id": payload.get("bss_frame_id", ""),
                "selected_mask_rows": len(rows),
                "nonempty_support_rows": sum(int(row["support_point_count"]) > 0 for row in rows),
                "empty_support_rows": sum(int(row["support_point_count"]) == 0 for row in rows),
                "total_support_assignments": sum(int(row["support_point_count"]) for row in rows),
                "num_frame_samples": payload.get("num_frame_samples", ""),
                "num_pixel_samples": payload.get("num_pixel_samples", ""),
                "mask_shape": payload.get("mask_shape", []),
                "image_shape": payload.get("image_shape", []),
                "uses_d4rt_for_prediction": False,
                "uses_da3_for_prediction": False,
                "uses_gt_for_prediction": False,
            }
        )

    blocking_failures = [row for row in failure_rows if row.get("severity") == "blocking"]
    diagnostic_failures = [row for row in failure_rows if row.get("severity") != "blocking"]
    selected_count = len(selected_rows)
    support_count = len(support_rows)
    nonempty_count = sum(int(row["support_point_count"]) > 0 for row in support_rows)
    support_coverage_ratio = float(nonempty_count) / float(support_count) if support_count else 0.0
    materialization_pass = selected_count > 0 and support_count == selected_count and not blocking_failures
    all_nonempty = materialization_pass and nonempty_count == support_count
    gate_rows = [
        {
            "schema_version": "stream4d_v104_lingbot_real_mask_support_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "all_selected_rows_processed",
            "pass": support_count == selected_count and selected_count > 0,
            "observed": {"selected_rows": selected_count, "support_rows": support_count},
            "required": "one support row per selected real Stream4D mask row",
        },
        {
            "schema_version": "stream4d_v104_lingbot_real_mask_support_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "no_blocking_mapping_or_mask_failures",
            "pass": not blocking_failures,
            "observed": {"blocking_failure_count": len(blocking_failures)},
            "required": "no missing BSS mapping, LingBot root, mask dir, mask raster, or mask pixels",
        },
        {
            "schema_version": "stream4d_v104_lingbot_real_mask_support_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "all_real_masks_have_positive_lingbot_support",
            "pass": all_nonempty,
            "observed": {
                "nonempty_support_rows": nonempty_count,
                "support_rows": support_count,
                "support_coverage_ratio": support_coverage_ratio,
                "empty_support_diagnostic_count": len(diagnostic_failures),
            },
            "required": "every selected real mask has at least one LingBot pixel sample support point",
        },
    ]
    summary = {
        "schema_version": "stream4d_v104_lingbot_real_mask_support_summary_v1",
        "phase_id": PHASE_ID,
        "materialization_pass": materialization_pass,
        "all_real_masks_have_positive_lingbot_support": all_nonempty,
        "taxonomy": (
            "LINGBOT_REAL_MASK_SUPPORT_ROWS_ALL_NONEMPTY_PASS"
            if all_nonempty
            else "LINGBOT_REAL_MASK_SUPPORT_ROWS_MATERIALIZED_WITH_EMPTY_SUPPORTS"
            if materialization_pass
            else "LINGBOT_REAL_MASK_SUPPORT_ROWS_FAIL"
        ),
        "blocker": "" if materialization_pass else "REAL_MASK_SUPPORT_ROWS_NOT_MATERIALIZED",
        "target_scenes": target_scenes,
        "selected_row_count": selected_count,
        "support_row_count": support_count,
        "frame_row_count": len(frame_rows),
        "nonempty_support_row_count": nonempty_count,
        "empty_support_row_count": support_count - nonempty_count,
        "support_coverage_ratio": support_coverage_ratio,
        "blocking_failure_count": len(blocking_failures),
        "diagnostic_failure_count": len(diagnostic_failures),
        "max_points_per_frame": int(args.max_points_per_frame),
        "stream4d_metric_ready": False,
        "stream4d_metric_note": "Real mask support rows only; no mask-pair affinity or AP/MV_AP metric is produced.",
        "outputs": {
            "support_rows": _rel(out / "real_mask_support_rows.csv"),
            "frame_rows": _rel(out / "frame_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "support_point_ids_dir": _rel(support_id_dir),
            "summary": _rel(out / "summary.json"),
        },
        "runtime_sec": round(time.time() - t0, 3),
    }
    _write_csv(out / "real_mask_support_rows.csv", support_rows)
    _write_csv(out / "frame_rows.csv", frame_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize LingBot support rows for real selected Stream4D masks.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--selected-rows", default=str(DEFAULT_SELECTED_ROWS))
    parser.add_argument("--scene-alignment-rows", default=str(DEFAULT_SCENE_ALIGNMENT_ROWS))
    parser.add_argument("--lingbot-candidate-rows", default=str(DEFAULT_LINGBOT_CANDIDATE_ROWS))
    parser.add_argument("--target-scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--max-points-per-frame", type=int, default=20000)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
