from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _write_csv, _write_json  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import (  # noqa: E402
    DEFAULT_SCENES,
    _discover_pipeline_root,
    _mask_dir_from_pipeline,
    _parse_csv_list,
)


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return REPO_ROOT / path_obj
    return ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        try:
            return str(path_obj.relative_to(REPO_ROOT))
        except ValueError:
            return str(path_obj)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_project(path).read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path_obj = _project(path)
    if not path_obj.exists():
        return []
    with path_obj.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_obs(value: Any) -> tuple[str, int, int] | None:
    parts = str(value or "").split(":")
    if len(parts) != 3:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _best_variant(pipeline_root: Path) -> str:
    summary = _read_json(pipeline_root / "local_objectlets/local_objectlet_summary.json")
    return str(summary.get("best_real_variant") or summary.get("best_real_row", {}).get("variant") or "").strip()


def _mask_ids(path: Path) -> set[int] | None:
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    return {int(value) for value in np.unique(image) if int(value) > 0}


def _collision_count(keys: list[Any]) -> int:
    counter = Counter(keys)
    return sum(1 for value in counter.values() if value > 1)


def _selected_objectlets(objectlet_rows: list[dict[str, str]], scene: str, variant: str) -> list[dict[str, str]]:
    return [row for row in objectlet_rows if row.get("scene") == scene and row.get("variant") == variant]


def _duplicate_conflict_rate(
    *,
    scene: str,
    selected_rows: list[dict[str, str]],
    ledger_rows: list[dict[str, str]],
) -> tuple[int, int, float]:
    selected_candidates = {str(row.get("candidate_id") or ""): str(row.get("objectlet_id") or "") for row in selected_rows}
    owners: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in ledger_rows:
        candidate_id = str(row.get("candidate_id") or "")
        objectlet_id = selected_candidates.get(candidate_id)
        if not objectlet_id or not _parse_bool(row.get("reprojection_success")):
            continue
        parsed = _parse_obs(row.get("best_mask_observation_id"))
        if parsed is None or parsed[0] != scene:
            continue
        owners[(int(parsed[1]), int(parsed[2]))].add(objectlet_id)
    duplicate = sum(1 for values in owners.values() if len(values) > 1)
    return int(duplicate), int(len(owners)), float(duplicate / max(1, len(owners)))


def _audit_scene(scene: str, pipeline_root: Path) -> dict[str, Any]:
    variant = _best_variant(pipeline_root)
    mask_dir = _mask_dir_from_pipeline(pipeline_root)
    candidate_rows = _read_csv(pipeline_root / "reprojection_ledger/candidate_rows.csv")
    ledger_rows = _read_csv(pipeline_root / "reprojection_ledger/reprojection_ledger_rows.csv")
    objectlet_rows = _read_csv(pipeline_root / "local_objectlets/objectlet_rows.csv")
    selected_rows = _selected_objectlets(objectlet_rows, scene, variant)
    candidate_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    candidate_id_scene_keys: list[tuple[str, str]] = []
    candidate_id_chunk_keys: list[tuple[str, str, str]] = []
    error_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for row in candidate_rows:
        row_scene = str(row.get("scene") or "")
        chunk_id = str(row.get("chunk_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        candidate_id_scene_keys.append((row_scene, candidate_id))
        candidate_id_chunk_keys.append((row_scene, chunk_id, candidate_id))
        candidate_by_key[(row_scene, chunk_id, candidate_id)] = row

    candidate_id_collision_count_scene = _collision_count(candidate_id_scene_keys)
    candidate_id_collision_count_chunk = _collision_count(candidate_id_chunk_keys)
    candidate_id_collision_count_across_chunk = 0
    by_scene_candidate: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row_scene, chunk_id, candidate_id in candidate_id_chunk_keys:
        by_scene_candidate[(row_scene, candidate_id)].add(chunk_id)
    candidate_id_collision_count_across_chunk = sum(1 for chunks in by_scene_candidate.values() if len(chunks) > 1)

    objectlet_keys = [
        (str(row.get("scene") or ""), str(row.get("variant") or ""), str(row.get("objectlet_id") or ""))
        for row in objectlet_rows
        if str(row.get("objectlet_id") or "")
    ]
    objectlet_id_collision_count = _collision_count(objectlet_keys)

    mask_id_cache: dict[int, set[int] | None] = {}
    missing_mask_png_count = 0
    missing_mask_id_count = 0
    best_mask_parse_error_count = 0
    mismatched_scene_chunk_count = 0
    stale_path_count = 0 if mask_dir.exists() else 1

    ledger_by_candidate: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ledger_rows:
        candidate_id = str(row.get("candidate_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        ledger_by_candidate[(chunk_id, candidate_id)].append(row)
        parsed = _parse_obs(row.get("best_mask_observation_id"))
        if parsed is None:
            best_mask_parse_error_count += 1
            error_rows.append({"scene_id": scene, "error_type": "best_mask_observation_parse_error", "candidate_id": candidate_id, "value": row.get("best_mask_observation_id")})
            continue
        obs_scene, frame_id, mask_id = parsed
        if obs_scene != scene:
            mismatched_scene_chunk_count += 1
            error_rows.append({"scene_id": scene, "error_type": "best_mask_scene_mismatch", "candidate_id": candidate_id, "value": row.get("best_mask_observation_id")})
            continue
        if frame_id not in mask_id_cache:
            mask_id_cache[frame_id] = _mask_ids(mask_dir / f"{int(frame_id)}.png")
        ids = mask_id_cache[frame_id]
        if ids is None:
            missing_mask_png_count += 1
            error_rows.append({"scene_id": scene, "error_type": "missing_mask_png", "candidate_id": candidate_id, "frame_id": frame_id, "mask_path": _rel(mask_dir / f"{int(frame_id)}.png")})
            continue
        if mask_id > 0 and mask_id not in ids:
            missing_mask_id_count += 1
            error_rows.append({"scene_id": scene, "error_type": "missing_mask_id", "candidate_id": candidate_id, "frame_id": frame_id, "mask_id": mask_id})

    missing_ledger_candidate_count = 0
    mask_only_objectlet_count = 0
    mask_only_missing_ledger_candidate_count = 0
    source_mask_ledger_disagreement_count = 0
    for row in objectlet_rows:
        if row.get("scene") != scene:
            continue
        candidate_id = str(row.get("candidate_id") or "")
        chunk_id = str(row.get("chunk_id") or "")
        objectlet_id = str(row.get("objectlet_id") or "")
        objectlet_source = str(row.get("objectlet_source") or "")
        is_mask_only_objectlet = objectlet_source == "L9_mask_only_representative_support" or candidate_id.startswith("maskonly")
        candidate_row = candidate_by_key.get((scene, chunk_id, candidate_id))
        candidate_exists = candidate_row is not None
        if not candidate_exists:
            if is_mask_only_objectlet:
                mask_only_objectlet_count += 1
                mask_only_missing_ledger_candidate_count += 1
            else:
                missing_ledger_candidate_count += 1
                error_rows.append(
                    {
                        "scene_id": scene,
                        "error_type": "objectlet_candidate_missing_from_candidate_rows",
                        "variant": row.get("variant"),
                        "chunk_id": chunk_id,
                        "candidate_id": candidate_id,
                        "objectlet_id": objectlet_id,
                    }
                )
        elif is_mask_only_objectlet:
            mask_only_objectlet_count += 1
        if candidate_exists and str(candidate_row.get("chunk_id") or "") != chunk_id:
            mismatched_scene_chunk_count += 1
        source_obs = str(row.get("source_mask_observation_id") or "")
        ledger_for_candidate = ledger_by_candidate.get((chunk_id, candidate_id), [])
        source_agrees = any(str(item.get("best_mask_observation_id") or "") == source_obs for item in ledger_for_candidate)
        if source_obs and ledger_for_candidate and not source_agrees:
            source_mask_ledger_disagreement_count += 1
            error_rows.append(
                {
                    "scene_id": scene,
                    "error_type": "source_mask_not_seen_as_ledger_best_mask",
                    "variant": row.get("variant"),
                    "chunk_id": chunk_id,
                    "candidate_id": candidate_id,
                    "objectlet_id": objectlet_id,
                    "source_mask_observation_id": source_obs,
                }
            )
        if row.get("variant") == variant:
            provenance_rows.append(
                {
                    "scene_id": scene,
                    "variant": variant,
                    "chunk_id": chunk_id,
                    "objectlet_id": objectlet_id,
                    "candidate_id": candidate_id,
                    "candidate_exists": candidate_exists,
                    "source_mask_observation_id": source_obs,
                    "ledger_row_count_for_candidate": len(ledger_for_candidate),
                    "source_mask_agrees_with_some_ledger_best_mask": bool(source_agrees),
                    "candidate_source": candidate_row.get("candidate_source") if candidate_row else "",
                    "candidate_source_mask_observation_id": candidate_row.get("source_mask_observation_id") if candidate_row else "",
                }
            )

    duplicate_conflicts, support_pair_count, duplicate_rate = _duplicate_conflict_rate(
        scene=scene,
        selected_rows=selected_rows,
        ledger_rows=ledger_rows,
    )
    scene_summary = {
        "scene_id": scene,
        "pipeline_root": _rel(pipeline_root),
        "mask_dir": _rel(mask_dir),
        "best_objectlet_variant": variant,
        "candidate_row_count": int(len(candidate_rows)),
        "ledger_row_count": int(len(ledger_rows)),
        "objectlet_row_count": int(len(objectlet_rows)),
        "selected_objectlet_count": int(len(selected_rows)),
        "candidate_id_collision_count_scene": int(candidate_id_collision_count_scene),
        "candidate_id_collision_count_chunk": int(candidate_id_collision_count_chunk),
        "candidate_id_collision_count_across_chunk": int(candidate_id_collision_count_across_chunk),
        "objectlet_id_collision_count": int(objectlet_id_collision_count),
        "missing_ledger_candidate_count": int(missing_ledger_candidate_count),
        "mask_only_objectlet_count": int(mask_only_objectlet_count),
        "mask_only_missing_ledger_candidate_count": int(mask_only_missing_ledger_candidate_count),
        "best_mask_parse_error_count": int(best_mask_parse_error_count),
        "missing_mask_png_count": int(missing_mask_png_count),
        "missing_mask_id_count": int(missing_mask_id_count),
        "mismatched_scene_chunk_count": int(mismatched_scene_chunk_count),
        "source_mask_ledger_disagreement_count": int(source_mask_ledger_disagreement_count),
        "stale_path_count": int(stale_path_count),
        "duplicate_frame_mask_conflict_count": int(duplicate_conflicts),
        "selected_support_pair_count": int(support_pair_count),
        "duplicate_frame_mask_conflict_rate": float(duplicate_rate),
    }
    return {"summary": scene_summary, "errors": error_rows, "provenance": provenance_rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = _parse_csv_list(args.scenes)
    scene_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    pipeline_roots: dict[str, str] = {}
    for scene in scenes:
        pipeline_root = _discover_pipeline_root(scene)
        if pipeline_root is None:
            missing_rows.append({"scene_id": scene, "missing": "pipeline_root"})
            continue
        pipeline_roots[scene] = _rel(pipeline_root)
        result = _audit_scene(scene, pipeline_root)
        scene_rows.append(result["summary"])
        error_rows.extend(result["errors"])
        provenance_rows.extend(result["provenance"])

    def _sum(field: str) -> int:
        return int(sum(int(row.get(field) or 0) for row in scene_rows))

    duplicate_rates = [float(row["duplicate_frame_mask_conflict_rate"]) for row in scene_rows]
    summary_counts = {
        "candidate_id_collision_count_scene": _sum("candidate_id_collision_count_scene"),
        "candidate_id_collision_count_chunk": _sum("candidate_id_collision_count_chunk"),
        "candidate_id_collision_count_across_chunk": _sum("candidate_id_collision_count_across_chunk"),
        "objectlet_id_collision_count": _sum("objectlet_id_collision_count"),
        "missing_ledger_candidate_count": _sum("missing_ledger_candidate_count"),
        "mask_only_objectlet_count": _sum("mask_only_objectlet_count"),
        "mask_only_missing_ledger_candidate_count": _sum("mask_only_missing_ledger_candidate_count"),
        "best_mask_parse_error_count": _sum("best_mask_parse_error_count"),
        "missing_mask_png_count": _sum("missing_mask_png_count"),
        "missing_mask_id_count": _sum("missing_mask_id_count"),
        "mismatched_scene_chunk_count": _sum("mismatched_scene_chunk_count"),
        "source_mask_ledger_disagreement_count": _sum("source_mask_ledger_disagreement_count"),
        "stale_path_count": _sum("stale_path_count"),
        "duplicate_frame_mask_conflict_rate_mean": float(np.mean(duplicate_rates)) if duplicate_rates else None,
    }
    gate = {
        "all_pipeline_roots_available": len(pipeline_roots) == len(scenes),
        "candidate_id_collision_count_chunk_eq_0": summary_counts["candidate_id_collision_count_chunk"] == 0,
        "missing_ledger_candidate_count_eq_0": summary_counts["missing_ledger_candidate_count"] == 0,
        "missing_mask_id_count_eq_0": summary_counts["missing_mask_id_count"] == 0,
        "mismatched_scene_chunk_count_eq_0": summary_counts["mismatched_scene_chunk_count"] == 0,
        "stale_path_count_eq_0": summary_counts["stale_path_count"] == 0,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    _write_csv(output_root / "scene_join_rows.csv", scene_rows)
    _write_csv(output_root / "join_error_rows.csv", error_rows)
    _write_csv(output_root / "candidate_provenance_rows.csv", provenance_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing_rows)
    payload = {
        "phase": "v67_ledger_join",
        "diagnostic_only": True,
        "scenes": scenes,
        "pipeline_roots": pipeline_roots,
        "summary_counts": summary_counts,
        "gate": gate,
        "decision": "PASS_JOIN_ENGINEERING" if gate["pass"] else "FAIL_FIX_JOIN_BEFORE_ALGORITHM",
        "rows": {
            "scene_join_rows_csv": _rel(output_root / "scene_join_rows.csv"),
            "join_error_rows_csv": _rel(output_root / "join_error_rows.csv"),
            "candidate_provenance_rows_csv": _rel(output_root / "candidate_provenance_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "candidate_id collisions across chunks are recorded separately; chunk-scoped keys are the required engineering key.",
            "missing_mask_id_count and mismatched_scene_chunk_count are hard join blockers per v67 plan.",
            "source_mask_ledger_disagreement is recorded as evidence; it is not used as a hard gate unless it coincides with missing candidates or scene/chunk mismatch.",
        ],
    }
    _write_json(output_root / "ledger_join_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v67 Phase 2 ledger join/provenance audit.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--output-root", default="outputs/audit/v67_ledger_join")
    return parser.parse_args()
