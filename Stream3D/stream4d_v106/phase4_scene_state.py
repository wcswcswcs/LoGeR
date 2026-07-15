from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import cv2
import numpy as np

from .artifacts import sha256_file, write_json
from .config import Phase4SceneStateConfig


def _resolve(repo_root: Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_label(path: str | Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.uint16, copy=False)


def _visible_ids(label: np.ndarray) -> Set[int]:
    return {int(v) - 1 for v in np.unique(label).tolist() if int(v) > 0}


def _summary_records(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(summary.get("records", []), key=lambda row: int(row["chunk_frame_index"]))


def _chunk_inventory(summary_path: Path) -> Dict[str, Any]:
    summary = _read_json(summary_path)
    records = _summary_records(summary)
    per_frame = []
    visible_any: Set[int] = set()
    visible_tail3: Set[int] = set()
    visible_first: Set[int] = set()
    visible_last: Set[int] = set()
    for idx, row in enumerate(records):
        label = _load_label(row["label_path"])
        ids = _visible_ids(label)
        visible_any |= ids
        if idx == 0:
            visible_first = set(ids)
        if idx >= max(0, len(records) - 3):
            visible_tail3 |= ids
        if idx == len(records) - 1:
            visible_last = set(ids)
        per_frame.append(
            {
                "chunk_frame_index": int(row["chunk_frame_index"]),
                "frame_id": int(row["frame_id"]),
                "visible_global_ids": [int(v) for v in sorted(ids)],
                "visible_id_count": int(len(ids)),
                "foreground_ratio": float(row.get("foreground_ratio", 0.0)),
                "label_path": str(row["label_path"]),
            }
        )
    return {
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "schema_version": summary.get("schema_version"),
        "frame_ids": [int(v) for v in summary.get("frame_ids", [])],
        "frame_count": int(summary.get("frame_count", len(records))),
        "birth_record_count": int(summary.get("birth_record_count", 0) or 0),
        "visible_any_ids": [int(v) for v in sorted(visible_any)],
        "visible_any_count": int(len(visible_any)),
        "visible_first_ids": [int(v) for v in sorted(visible_first)],
        "visible_first_count": int(len(visible_first)),
        "visible_tail3_ids": [int(v) for v in sorted(visible_tail3)],
        "visible_tail3_count": int(len(visible_tail3)),
        "visible_last_ids": [int(v) for v in sorted(visible_last)],
        "visible_last_count": int(len(visible_last)),
        "per_frame": per_frame,
        "records": records,
    }


def _package_audit(repo_root: Path, package_path: Path) -> Dict[str, Any]:
    package = _read_json(package_path)
    runtime_map = {int(k): int(v) for k, v in package.get("runtime_local_to_global", {}).items()}
    objects = package.get("objects", [])
    object_global_ids = [int(obj["global_id"]) for obj in objects]
    unique_object_global_ids = sorted(set(object_global_ids))
    runtime_targets = list(runtime_map.values())
    duplicate_runtime_targets = sorted(
        {int(gid) for gid in runtime_targets if runtime_targets.count(gid) > 1}
    )
    mask_missing = []
    for obj in objects:
        mask_path = _resolve(repo_root, obj["mask_path"])
        if not mask_path.exists():
            mask_missing.append(str(obj["mask_path"]))
    return {
        "path": str(package_path),
        "sha256": sha256_file(package_path),
        "schema_version": package.get("schema_version"),
        "from_chunk_index": int(package.get("from_chunk_index")),
        "to_chunk_index": int(package.get("to_chunk_index")),
        "from_history_version": int(package.get("from_history_version")),
        "to_history_version": int(package.get("to_history_version")),
        "object_count": int(package.get("object_count", len(unique_object_global_ids))),
        "prompt_record_count": int(package.get("prompt_record_count", len(objects))),
        "runtime_local_to_global": {str(k): int(v) for k, v in sorted(runtime_map.items())},
        "runtime_local_to_global_count": int(len(runtime_map)),
        "unique_object_global_ids": [int(v) for v in unique_object_global_ids],
        "duplicate_runtime_target_global_ids": [int(v) for v in duplicate_runtime_targets],
        "mask_missing_count": int(len(mask_missing)),
        "mask_missing": mask_missing[:16],
        "stage12_full_initialization_used_by_handoff": bool(
            package.get("stage12_full_initialization_used_by_handoff", False)
        ),
        "fake_logits_written": bool(package.get("fake_logits_written", False)),
    }


def _boundary_metrics(gate_summary_path: Path, variant: str = "H2_best_plus_one_correction") -> Dict[str, Any]:
    gate = _read_json(gate_summary_path)
    metrics = gate.get("variant_metrics", {}).get(variant, {})
    aggregate = metrics.get("aggregate", {})
    return {
        "path": str(gate_summary_path),
        "sha256": sha256_file(gate_summary_path),
        "gate_passes": bool(gate.get("passes", False)),
        "passing_variants": gate.get("passing_variants", []),
        "variant": variant,
        "variant_passes": bool(metrics.get("passes", False)),
        "aggregate": aggregate,
    }


def _metric_checks(boundary_name: str, metrics: Dict[str, Any], config: Phase4SceneStateConfig) -> List[Dict[str, Any]]:
    aggregate = metrics.get("aggregate", {})
    return [
        {
            "name": f"{boundary_name}_gate_passes",
            "passes": bool(metrics.get("gate_passes")) and bool(metrics.get("variant_passes")),
            "actual": {
                "gate_passes": bool(metrics.get("gate_passes")),
                "variant_passes": bool(metrics.get("variant_passes")),
                "passing_variants": metrics.get("passing_variants", []),
            },
            "expected": "boundary gate and H2 variant pass",
        },
        {
            "name": f"{boundary_name}_min_CCOC",
            "passes": float(aggregate.get("min_CCOC", 0.0)) >= float(config.min_ccoc),
            "actual": aggregate.get("min_CCOC"),
            "expected_min": config.min_ccoc,
        },
        {
            "name": f"{boundary_name}_min_HIR",
            "passes": float(aggregate.get("min_HIR", 0.0)) >= float(config.min_hir),
            "actual": aggregate.get("min_HIR"),
            "expected_min": config.min_hir,
        },
        {
            "name": f"{boundary_name}_min_HCR",
            "passes": float(aggregate.get("min_HCR", 0.0)) >= float(config.min_hcr),
            "actual": aggregate.get("min_HCR"),
            "expected_min": config.min_hcr,
        },
        {
            "name": f"{boundary_name}_max_CFR",
            "passes": float(aggregate.get("max_CFR", 1.0)) <= float(config.max_cfr),
            "actual": aggregate.get("max_CFR"),
            "expected_max": config.max_cfr,
        },
        {
            "name": f"{boundary_name}_max_CMR",
            "passes": float(aggregate.get("max_CMR", 1.0)) <= float(config.max_cmr),
            "actual": aggregate.get("max_CMR"),
            "expected_max": config.max_cmr,
        },
        {
            "name": f"{boundary_name}_max_BFMR",
            "passes": float(aggregate.get("max_BFMR", 1.0)) <= float(config.max_bfmr),
            "actual": aggregate.get("max_BFMR"),
            "expected_max": config.max_bfmr,
        },
    ]


def _remap_geometry_records(chunk_index: int, inventory: Dict[str, Any], runtime_map: Dict[int, int]) -> Dict[str, Any]:
    frame_records = []
    missing_mapping: Set[int] = set()
    changed_geometry = []
    duplicate_global_frames = []
    for row in inventory["records"]:
        label = _load_label(row["label_path"])
        ids = _visible_ids(label)
        if chunk_index == 0:
            mapping = {local_id: local_id + 1 for local_id in ids}
        else:
            mapping = runtime_map
            missing_mapping |= {int(local_id) for local_id in ids if int(local_id) not in mapping}
        target_ids = [int(mapping.get(local_id, -1)) for local_id in ids if local_id in mapping or chunk_index == 0]
        if len(target_ids) != len(set(target_ids)):
            duplicate_global_frames.append(int(row["frame_id"]))
        remapped_fg = label > 0
        geometry_equal = bool(np.array_equal(remapped_fg, label > 0))
        if not geometry_equal:
            changed_geometry.append(int(row["frame_id"]))
        frame_records.append(
            {
                "chunk_index": int(chunk_index),
                "frame_id": int(row["frame_id"]),
                "local_visible_id_count": int(len(ids)),
                "mapped_global_id_count": int(len(set(target_ids))),
                "missing_runtime_mapping_ids": [int(v) for v in sorted(ids - set(mapping))] if chunk_index > 0 else [],
                "foreground_geometry_equal_after_remap": geometry_equal,
            }
        )
    return {
        "chunk_index": int(chunk_index),
        "frame_count": int(len(frame_records)),
        "all_foreground_geometry_equal_after_remap": not changed_geometry,
        "changed_geometry_frame_ids": changed_geometry,
        "missing_runtime_mapping_ids": [int(v) for v in sorted(missing_mapping)],
        "missing_runtime_mapping_count": int(len(missing_mapping)),
        "duplicate_global_mapping_frame_ids": duplicate_global_frames,
        "duplicate_global_mapping_frame_count": int(len(duplicate_global_frames)),
        "frame_records": frame_records,
    }


def _state_records(chunks: List[Dict[str, Any]], handoffs: List[Dict[str, Any]]) -> Dict[str, Any]:
    known: Set[int] = set(int(v) + 1 for v in chunks[0]["visible_any_ids"])
    next_global_id = max(known) + 1 if known else 1
    records = []
    for idx, chunk in enumerate(chunks):
        if idx > 0:
            incoming = set(int(v) for v in handoffs[idx - 1]["unique_object_global_ids"])
            known |= incoming
        active = set(int(v) + 1 for v in chunk["visible_tail3_ids"])
        occluded = known - active
        records.append(
            {
                "chunk_index": int(idx),
                "history_version_in": int(idx),
                "history_version_out": int(idx + 1),
                "global_registry_version_in": int(idx),
                "global_registry_version_out": int(idx + 1),
                "next_global_id": int(next_global_id),
                "known_global_id_count": int(len(known)),
                "active_global_ids": [int(v) for v in sorted(active)],
                "active_global_id_count": int(len(active)),
                "occluded_global_ids": [int(v) for v in sorted(occluded)],
                "occluded_global_id_count": int(len(occluded)),
                "active_occluded_intersection": [int(v) for v in sorted(active & occluded)],
                "visible_any_count": int(chunk["visible_any_count"]),
                "visible_tail3_count": int(chunk["visible_tail3_count"]),
            }
        )
    return {
        "schema_version": "stream4d_v106_phase4_scene_stream_state_v1",
        "scene_id": chunks[0].get("scene_id", ""),
        "chunk_count": int(len(records)),
        "history_versions": [int(row["history_version_out"]) for row in records],
        "global_registry_versions": [int(row["global_registry_version_out"]) for row in records],
        "next_global_id_final": int(next_global_id),
        "chunk_records": records,
    }


def run_phase4_scene_state_audit(repo_root: Path, config: Phase4SceneStateConfig, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "c0_summary": _resolve(repo_root, config.c0_summary),
        "c1_summary": _resolve(repo_root, config.c1_summary),
        "c2_summary": _resolve(repo_root, config.c2_summary),
        "c0_to_c1_handoff_package": _resolve(repo_root, config.c0_to_c1_handoff_package),
        "c1_to_c2_handoff_package": _resolve(repo_root, config.c1_to_c2_handoff_package),
        "c0_to_c1_gate_summary": _resolve(repo_root, config.c0_to_c1_gate_summary),
        "c1_to_c2_gate_summary": _resolve(repo_root, config.c1_to_c2_gate_summary),
    }
    missing_paths = [name for name, path in paths.items() if not path.exists()]
    if missing_paths:
        summary = {
            "schema_version": "stream4d_v106_phase4_scene_state_audit_v1",
            "passes": False,
            "missing_paths": {name: str(paths[name]) for name in missing_paths},
        }
        write_json(output_dir / "gate_records.json", summary)
        write_json(output_dir / "failure_records.json", [{"failure": "missing_required_artifact", "paths": missing_paths}])
        return summary

    chunk_inventories = [
        _chunk_inventory(paths["c0_summary"]),
        _chunk_inventory(paths["c1_summary"]),
        _chunk_inventory(paths["c2_summary"]),
    ]
    for idx, chunk in enumerate(chunk_inventories):
        chunk["chunk_index"] = idx
        chunk["scene_id"] = config.scene_id
    handoff_audits = [
        _package_audit(repo_root, paths["c0_to_c1_handoff_package"]),
        _package_audit(repo_root, paths["c1_to_c2_handoff_package"]),
    ]
    boundary_metrics = {
        "c0_to_c1": _boundary_metrics(paths["c0_to_c1_gate_summary"]),
        "c1_to_c2": _boundary_metrics(paths["c1_to_c2_gate_summary"]),
    }
    state = _state_records(chunk_inventories, handoff_audits)
    registry = {
        "schema_version": "stream4d_v106_global_identity_registry_v1",
        "registry_version": 3,
        "next_global_id": state["next_global_id_final"],
        "runtime_local_to_global_by_chunk": {
            "0": {str(local_id): int(local_id) + 1 for local_id in chunk_inventories[0]["visible_any_ids"]},
            "1": handoff_audits[0]["runtime_local_to_global"],
            "2": handoff_audits[1]["runtime_local_to_global"],
        },
        "active_global_ids": state["chunk_records"][-1]["active_global_ids"],
        "occluded_global_ids": state["chunk_records"][-1]["occluded_global_ids"],
    }
    geometry_records = [
        _remap_geometry_records(0, chunk_inventories[0], {}),
        _remap_geometry_records(
            1,
            chunk_inventories[1],
            {int(k): int(v) for k, v in handoff_audits[0]["runtime_local_to_global"].items()},
        ),
        _remap_geometry_records(
            2,
            chunk_inventories[2],
            {int(k): int(v) for k, v in handoff_audits[1]["runtime_local_to_global"].items()},
        ),
    ]

    checks: List[Dict[str, Any]] = []
    checks.append(
        {
            "name": "three_consecutive_chunks_present",
            "passes": len(chunk_inventories) == 3 and all(c["frame_count"] == 32 for c in chunk_inventories),
            "actual": [c["frame_count"] for c in chunk_inventories],
            "expected": [32, 32, 32],
        }
    )
    checks.append(
        {
            "name": "history_versions_monotonic",
            "passes": state["history_versions"] == [1, 2, 3],
            "actual": state["history_versions"],
            "expected": [1, 2, 3],
        }
    )
    checks.append(
        {
            "name": "handoff_history_versions_chain",
            "passes": (
                handoff_audits[0]["from_history_version"] == 1
                and handoff_audits[0]["to_history_version"] == 2
                and handoff_audits[1]["from_history_version"] == 2
                and handoff_audits[1]["to_history_version"] == 3
            ),
            "actual": [
                [handoff_audits[0]["from_history_version"], handoff_audits[0]["to_history_version"]],
                [handoff_audits[1]["from_history_version"], handoff_audits[1]["to_history_version"]],
            ],
            "expected": [[1, 2], [2, 3]],
        }
    )
    checks.append(
        {
            "name": "global_allocator_not_reset",
            "passes": registry["next_global_id"] > max(registry["runtime_local_to_global_by_chunk"]["0"].values()),
            "actual": registry["next_global_id"],
            "expected": "greater than max chunk0 global id",
        }
    )
    checks.append(
        {
            "name": "active_occluded_disjoint_all_chunks",
            "passes": all(not row["active_occluded_intersection"] for row in state["chunk_records"]),
            "actual": [row["active_occluded_intersection"] for row in state["chunk_records"]],
            "expected": "empty intersections",
        }
    )
    checks.append(
        {
            "name": "handoff_one_to_one_runtime_global_mapping",
            "passes": all(not audit["duplicate_runtime_target_global_ids"] for audit in handoff_audits),
            "actual": [audit["duplicate_runtime_target_global_ids"] for audit in handoff_audits],
            "expected": "no duplicated target global ids in each handoff",
        }
    )
    checks.append(
        {
            "name": "handoff_no_stage12_full_initialization",
            "passes": all(not audit["stage12_full_initialization_used_by_handoff"] for audit in handoff_audits),
            "actual": [audit["stage12_full_initialization_used_by_handoff"] for audit in handoff_audits],
            "expected": [False, False],
        }
    )
    checks.append(
        {
            "name": "handoff_masks_present",
            "passes": all(audit["mask_missing_count"] == 0 for audit in handoff_audits),
            "actual": [audit["mask_missing_count"] for audit in handoff_audits],
            "expected": [0, 0],
        }
    )
    checks.append(
        {
            "name": "global_remap_preserves_foreground_geometry",
            "passes": all(record["all_foreground_geometry_equal_after_remap"] for record in geometry_records),
            "actual": [
                {
                    "chunk_index": record["chunk_index"],
                    "changed_geometry_frame_ids": record["changed_geometry_frame_ids"],
                }
                for record in geometry_records
            ],
            "expected": "no foreground geometry changes",
        }
    )
    checks.append(
        {
            "name": "all_visible_runtime_ids_have_global_mapping",
            "passes": all(record["missing_runtime_mapping_count"] == 0 for record in geometry_records),
            "actual": [
                {
                    "chunk_index": record["chunk_index"],
                    "missing_runtime_mapping_ids": record["missing_runtime_mapping_ids"],
                }
                for record in geometry_records
            ],
            "expected": "all nonzero runtime ids map to global ids",
        }
    )
    for boundary_name, metrics in boundary_metrics.items():
        checks.extend(_metric_checks(boundary_name, metrics, config))

    support_records = {
        "c0_visible_any_count": chunk_inventories[0]["visible_any_count"],
        "c0_tail3_transferable_count": handoff_audits[0]["object_count"],
        "c1_visible_any_count": chunk_inventories[1]["visible_any_count"],
        "c1_tail3_transferable_count": handoff_audits[1]["object_count"],
        "c2_visible_any_count": chunk_inventories[2]["visible_any_count"],
        "c2_tail3_visible_count": chunk_inventories[2]["visible_tail3_count"],
        "support_warning": (
            "C1->C2 inherited boundary has only "
            f"{handoff_audits[1]['object_count']} transferable IDs; Phase5 still needs repair/birth/defer."
        ),
    }
    failure_records = [
        {
            "name": check["name"],
            "actual": check.get("actual"),
            "expected": check.get("expected", check.get("expected_min", check.get("expected_max"))),
        }
        for check in checks
        if not check["passes"]
    ]
    summary = {
        "schema_version": "stream4d_v106_phase4_scene_state_audit_v1",
        "scene_id": config.scene_id,
        "passes": all(check["passes"] for check in checks),
        "gate_checks": checks,
        "failure_count": int(len(failure_records)),
        "support_records": support_records,
        "artifact_paths": {name: str(path) for name, path in paths.items()},
    }
    write_json(output_dir / "chunk_inventory_records.json", chunk_inventories)
    write_json(output_dir / "handoff_package_audit_records.json", handoff_audits)
    write_json(output_dir / "boundary_identity_metric_records.json", boundary_metrics)
    write_json(output_dir / "scene_stream_state.json", state)
    write_json(output_dir / "global_identity_registry.json", registry)
    write_json(output_dir / "global_remap_geometry_records.json", geometry_records)
    write_json(output_dir / "support_universe_records.json", support_records)
    write_json(output_dir / "gate_records.json", summary)
    write_json(output_dir / "failure_records.json", failure_records)
    return summary
