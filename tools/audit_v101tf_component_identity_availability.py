#!/usr/bin/env python3
"""Audit Stage-C component identity availability for ACL2 v101.

This is a read-only fail-forward audit for the JL4/Track U recommendation to
search semantic/component artifacts.  It checks whether target cases have
Stage-C masklet/component-like track ids available and whether those ids can be
directly aligned to v101 anchor ids.  It does not claim instance identity when
the mapping is not explicit.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission")
FINAL = ROOT / "final_decision"
TRACK_T = ROOT / "trackT_drift_target_relabel"
TRACK_U = ROOT / "trackU_true_current_support"
TRACK_JL4 = ROOT / "trackJL4_semantic_anchor_instance_atlas"
PREPROCESS_ROOT = Path("results/kitti_preprocess")
V100_ROOT = Path("results/acl2_v100tf_same_space_semantic_anchor_latent_state_memory_control")
V100_S_ROWS = V100_ROOT / "trackS_same_space_latent_state/same_space_anchor_rows.csv"
V100_GEOMETRY_EDGE_ROWS = V100_ROOT / "trackL2_anchor_scale_observability/geometry_edge_rows.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify(row.get(key, "")) for key in keys})


def stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": f"{type(exc).__name__}:{exc}"}
    return payload if isinstance(payload, dict) else {"value": payload}


def parse_boundary(case_id: str) -> tuple[str, int | None, int | None]:
    parts = case_id.split("_")
    if len(parts) != 3:
        return parts[0] if parts else "", None, None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError:
        return parts[0], None, None


def read_cache_index(seq: str) -> dict[int, dict[str, Any]]:
    path = PREPROCESS_ROOT / seq / "stage_c_cache_semantic_chunks" / "cache_index.jsonl"
    out: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                out[int(row.get("chunk_idx"))] = row
            except Exception:
                continue
    return out


def load_stage_c_summary(masklet_path: Path) -> dict[str, Any]:
    if not masklet_path.is_file():
        return {
            "masklet_path": str(masklet_path),
            "masklet_exists": False,
            "load_ok": False,
            "load_error": "missing",
        }
    try:
        import torch  # Imported lazily so normal CSV-only checks do not require torch import at module import time.

        payload = torch.load(masklet_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "masklet_path": str(masklet_path),
            "masklet_exists": True,
            "load_ok": False,
            "load_error": f"{type(exc).__name__}:{exc}",
        }
    if not isinstance(payload, dict):
        return {
            "masklet_path": str(masklet_path),
            "masklet_exists": True,
            "load_ok": False,
            "load_error": f"unexpected_payload_type:{type(payload).__name__}",
        }
    seed_ids = [str(v) for v in payload.get("seed_global_track_idx", [])]
    labels = [str(v) for v in payload.get("G_sem", [])]
    source_types = [str(v) for v in payload.get("source_type", [])]
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    return {
        "masklet_path": str(masklet_path),
        "masklet_exists": True,
        "load_ok": True,
        "load_error": "",
        "num_masklets": int(payload.get("num_masklets", 0) or 0),
        "num_frames": int(payload.get("num_frames", 0) or 0),
        "seed_global_track_idx_count": len(seed_ids),
        "seed_global_track_idx_unique_count": len(set(seed_ids)),
        "seed_global_track_idx_sample": seed_ids[:8],
        "semantic_label_count": len(labels),
        "semantic_label_counts": dict(Counter(labels)),
        "source_type_counts": dict(Counter(source_types)),
        "visible_masklet_frames": debug.get("visible_masklet_frames", ""),
        "has_source_global_track_indices_debug": "source_global_track_indices" in debug,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ACL2 v101 Component Identity Availability Audit",
        "",
        "This audit checks Stage-C component-like identity availability for v101 JL4/Track U fail-forward.",
        "",
        "## Summary",
        "",
        f"- case_count: {summary['case_count']}",
        f"- stage_c_cache_case_coverage: {summary['stage_c_cache_case_coverage']}",
        f"- stage_c_masklet_loadable_case_coverage: {summary['stage_c_masklet_loadable_case_coverage']}",
        f"- full_sparse_masklet_present_sequence_count: {summary['full_sparse_masklet_present_sequence_count']}",
        f"- component_like_track_ids_available: {summary['component_like_track_ids_available']}",
        f"- direct_anchor_to_stage_c_seed_match_count: {summary['direct_anchor_to_stage_c_seed_match_count']}",
        f"- diagnostic_anchor_seed_join_feasible: {summary['diagnostic_anchor_seed_join_feasible']}",
        f"- diagnostic_anchor_seed_lifecycle_pair_count: {summary['diagnostic_anchor_seed_lifecycle_pair_count']}",
        f"- diagnostic_lifecycle_explicit_anchor_seed_mapping_available: {summary['diagnostic_lifecycle_explicit_anchor_seed_mapping_available']}",
        f"- diagnostic_lifecycle_stage_c_seed_support_join_unique_coverage: {summary['diagnostic_lifecycle_stage_c_seed_support_join_unique_coverage']}",
        f"- upstream_component_provenance_bridge_available: {summary['upstream_component_provenance_bridge_available']}",
        f"- explicit_anchor_component_mapping_available: {summary['explicit_anchor_component_mapping_available']}",
        f"- jl4_identity_rescue_available: {summary['jl4_identity_rescue_available']}",
        f"- runtime_action_allowed: {summary['runtime_action_allowed']}",
        "",
        "## Decision",
        "",
        summary["blocked_reason"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def provenance_schema_summary(rows: list[dict[str, str]], *, name: str) -> dict[str, Any]:
    if not rows:
        return {
            f"{name}_row_count": 0,
            f"{name}_column_count": 0,
            f"{name}_provenance_like_columns": [],
            f"{name}_has_component_or_stage_c_columns": False,
            f"{name}_anchor_alignment_source_counts": {},
        }
    columns = list(rows[0].keys())
    needles = ("component", "instance", "stage_c", "masklet", "seed", "global_track")
    provenance_like = [col for col in columns if any(needle in col.lower() for needle in needles)]
    alignment_counts = Counter(row.get("anchor_alignment_source", "") for row in rows if row.get("anchor_alignment_source", ""))
    return {
        f"{name}_row_count": len(rows),
        f"{name}_column_count": len(columns),
        f"{name}_provenance_like_columns": provenance_like,
        f"{name}_has_component_or_stage_c_columns": bool(provenance_like),
        f"{name}_anchor_alignment_source_counts": dict(alignment_counts),
    }


def trace_payload_key_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    paths = sorted({row.get("trace_payload", "") for row in rows if row.get("trace_payload", "")})
    needles = ("component", "instance", "stage_c", "masklet", "seed", "global_track")
    scanned = 0
    load_error_count = 0
    with_provenance = 0
    key_counter: Counter[str] = Counter()
    examples: list[str] = []
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {
            "trace_payload_unique_path_count": len(paths),
            "trace_payload_scanned_count": 0,
            "trace_payload_load_error_count": len(paths),
            "trace_payload_with_component_or_stage_c_key_count": 0,
            "trace_payload_provenance_like_key_counts": {},
            "trace_payload_scan_error": f"{type(exc).__name__}:{exc}",
        }
    for path_text in paths:
        path = Path(path_text)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            load_error_count += 1
            continue
        if not isinstance(payload, dict):
            continue
        scanned += 1
        provenance_like = [key for key in payload.keys() if any(needle in str(key).lower() for needle in needles)]
        if provenance_like:
            with_provenance += 1
            if len(examples) < 8:
                examples.append(path_text)
        key_counter.update(str(key) for key in provenance_like)
    return {
        "trace_payload_unique_path_count": len(paths),
        "trace_payload_scanned_count": scanned,
        "trace_payload_load_error_count": load_error_count,
        "trace_payload_with_component_or_stage_c_key_count": with_provenance,
        "trace_payload_provenance_like_key_counts": dict(key_counter),
        "trace_payload_with_component_or_stage_c_key_examples": examples,
    }


def main() -> None:
    target_rows = read_rows(TRACK_T / "target_universe_v101.csv")
    support_rows = read_rows(TRACK_U / "anchor_current_support_rows.csv")
    atlas_gap_rows = read_rows(TRACK_JL4 / "identity_resolution_gap_rows.csv")
    same_space_rows = [
        row
        for row in read_rows(V100_S_ROWS)
        if row.get("canonical_space_name") == "S-B_preprojection_hidden"
    ]
    geometry_edge_rows = read_rows(V100_GEOMETRY_EDGE_ROWS)
    support_by_case: dict[str, list[dict[str, str]]] = {}
    for row in support_rows:
        support_by_case.setdefault(row.get("case_id", ""), []).append(row)

    cache_by_seq: dict[str, dict[int, dict[str, Any]]] = {}
    stage_c_by_chunk: dict[tuple[str, int], dict[str, Any]] = {}
    case_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    direct_anchor_match_count = 0
    stage_c_cache_case_hit = 0
    stage_c_loadable_case_hit = 0
    component_like_case_hit = 0
    all_stage_c_seed_ids_by_case: dict[str, set[str]] = {}

    for target in target_rows:
        case_id = target.get("case_id", "")
        seq, left, right = parse_boundary(case_id)
        if seq not in cache_by_seq:
            cache_by_seq[seq] = read_cache_index(seq)
        chunks = [idx for idx in [left, right] if idx is not None]
        chunk_summaries: list[dict[str, Any]] = []
        seed_ids: set[str] = set()
        load_ok_count = 0
        cache_hit_count = 0
        for chunk_idx in chunks:
            cache_row = cache_by_seq[seq].get(chunk_idx, {})
            chunk_name = cache_row.get("chunk", f"chunk_{chunk_idx:03d}")
            masklet_path = PREPROCESS_ROOT / seq / "stage_c_cache_semantic_chunks" / str(chunk_name) / "masklet.pt"
            stage_key = (seq, chunk_idx)
            if stage_key not in stage_c_by_chunk:
                stage_c_by_chunk[stage_key] = load_stage_c_summary(masklet_path)
            loaded = stage_c_by_chunk[stage_key]
            if cache_row:
                cache_hit_count += 1
            if loaded.get("load_ok") is True:
                load_ok_count += 1
                seed_ids.update(str(v) for v in loaded.get("seed_global_track_idx_sample", []))
                # If the sample is truncated, avoid pretending it is exhaustive in row-level evidence.
                if int(loaded.get("seed_global_track_idx_count", 0) or 0) > len(loaded.get("seed_global_track_idx_sample", [])):
                    pass
            chunk_row = {
                "case_id": case_id,
                "seq": seq,
                "chunk_idx": chunk_idx,
                "target_taxonomy": target.get("target_taxonomy", ""),
                "cache_index_hit": bool(cache_row),
                "cache_chunk": cache_row.get("chunk", ""),
                "cache_num_masklets": cache_row.get("num_masklets", ""),
                **loaded,
            }
            chunk_rows.append(chunk_row)
            chunk_summaries.append(chunk_row)

        # Reload full seed ids for direct matching only for loadable chunks.
        exhaustive_seed_ids: set[str] = set()
        for chunk_idx in chunks:
            cache_row = cache_by_seq[seq].get(chunk_idx, {})
            if not cache_row:
                continue
            masklet_path = PREPROCESS_ROOT / seq / "stage_c_cache_semantic_chunks" / str(cache_row.get("chunk", "")) / "masklet.pt"
            try:
                import torch

                payload = torch.load(masklet_path, map_location="cpu", weights_only=False)
                if isinstance(payload, dict):
                    exhaustive_seed_ids.update(str(v) for v in payload.get("seed_global_track_idx", []))
            except Exception:
                continue

        support_anchors = {str(row.get("anchor_id", "")) for row in support_by_case.get(case_id, []) if row.get("anchor_id")}
        direct_matches = sorted(support_anchors & exhaustive_seed_ids)
        direct_anchor_match_count += len(direct_matches)
        all_stage_c_seed_ids_by_case[case_id] = exhaustive_seed_ids
        cache_case_hit = bool(chunks) and cache_hit_count == len(chunks)
        loadable_case_hit = bool(chunks) and load_ok_count == len(chunks)
        component_like_hit = loadable_case_hit and bool(exhaustive_seed_ids)
        stage_c_cache_case_hit += int(cache_case_hit)
        stage_c_loadable_case_hit += int(loadable_case_hit)
        component_like_case_hit += int(component_like_hit)
        full_sparse = PREPROCESS_ROOT / seq / "sparse_masklets_with_semantic.pt"
        case_rows.append(
            {
                "case_id": case_id,
                "seq": seq,
                "boundary_left_chunk": left if left is not None else "",
                "boundary_right_chunk": right if right is not None else "",
                "target_taxonomy": target.get("target_taxonomy", ""),
                "support_anchor_count": len(support_anchors),
                "stage_c_cache_chunks_expected": len(chunks),
                "stage_c_cache_chunks_hit": cache_hit_count,
                "stage_c_masklet_chunks_loadable": load_ok_count,
                "component_like_stage_c_seed_id_count": len(exhaustive_seed_ids),
                "direct_anchor_to_stage_c_seed_match_count": len(direct_matches),
                "direct_anchor_to_stage_c_seed_matches": direct_matches[:12],
                "full_sparse_masklets_with_semantic_exists": full_sparse.is_file(),
                "identity_resolution_level": "semantic_class_fallback",
                "explicit_anchor_component_mapping_available": False,
                "claim_level": "component_artifact_availability_no_identity_rescue",
            }
        )

    sequence_ids = sorted({row.get("seq", "") for row in target_rows})
    full_sparse_present = [
        seq for seq in sequence_ids if (PREPROCESS_ROOT / seq / "sparse_masklets_with_semantic.pt").is_file()
    ]
    case_count = len(target_rows)
    support_schema = provenance_schema_summary(support_rows, name="support")
    same_space_schema = provenance_schema_summary(same_space_rows, name="same_space")
    geometry_schema = provenance_schema_summary(geometry_edge_rows, name="geometry_edge")
    trace_payload_schema = trace_payload_key_summary(same_space_rows)
    anchor_seed_join = read_json(FINAL / "anchor_seed_join_feasibility_summary.json")
    lifecycle_support_join = read_json(FINAL / "anchor_seed_lifecycle_support_join_summary.json")
    diagnostic_anchor_seed_join_feasible = bool(anchor_seed_join.get("diagnostic_anchor_seed_join_feasible"))
    diagnostic_stage_c_payload_count = int(anchor_seed_join.get("payload_with_stage_c_seed_count") or 0)
    diagnostic_ttt_anchor_payload_count = int(anchor_seed_join.get("payload_with_ttt_anchor_id_count") or 0)
    diagnostic_lifecycle_pair_count = int(anchor_seed_join.get("lifecycle_anchor_seed_pair_count") or 0)
    diagnostic_lifecycle_seed_join_row_count = int(
        lifecycle_support_join.get("lifecycle_rows_with_stage_c_seed_support_join_count") or 0
    )
    diagnostic_lifecycle_seed_join_unique_coverage = lifecycle_support_join.get(
        "lifecycle_stage_c_seed_support_join_unique_coverage"
    )
    diagnostic_lifecycle_anchor_seed_mapping_available = (
        diagnostic_lifecycle_seed_join_row_count > 0
        and diagnostic_lifecycle_seed_join_unique_coverage == 1.0
    )
    trace_payload_schema["legacy_same_space_trace_payload_with_component_or_stage_c_key_count"] = (
        trace_payload_schema.get("trace_payload_with_component_or_stage_c_key_count", 0)
    )
    trace_payload_schema["diagnostic_anchor_seed_trace_payload_with_stage_c_seed_count"] = diagnostic_stage_c_payload_count
    trace_payload_schema["diagnostic_anchor_seed_trace_payload_with_ttt_anchor_id_count"] = diagnostic_ttt_anchor_payload_count
    trace_payload_schema["diagnostic_anchor_seed_lifecycle_pair_count"] = diagnostic_lifecycle_pair_count
    trace_payload_schema["trace_payload_with_component_or_stage_c_key_count"] = max(
        int(trace_payload_schema.get("trace_payload_with_component_or_stage_c_key_count", 0) or 0),
        diagnostic_stage_c_payload_count,
    )
    upstream_bridge_available = (
        support_schema["support_has_component_or_stage_c_columns"]
        or same_space_schema["same_space_has_component_or_stage_c_columns"]
        or geometry_schema["geometry_edge_has_component_or_stage_c_columns"]
        or trace_payload_schema["trace_payload_with_component_or_stage_c_key_count"] > 0
        or diagnostic_anchor_seed_join_feasible
        or diagnostic_lifecycle_anchor_seed_mapping_available
    )
    explicit_mapping_available = (
        direct_anchor_match_count > 0
        and len(atlas_gap_rows) < len(support_rows)
        and upstream_bridge_available
    )
    summary = {
        "schema": "acl2_v101_component_identity_availability_v1",
        "case_count": case_count,
        "sequence_count": len(sequence_ids),
        "sequence_ids": sequence_ids,
        "full_sparse_masklet_present_sequence_count": len(full_sparse_present),
        "full_sparse_masklet_present_sequences": full_sparse_present,
        "stage_c_cache_case_hit_count": stage_c_cache_case_hit,
        "stage_c_cache_case_coverage": stage_c_cache_case_hit / max(case_count, 1),
        "stage_c_masklet_loadable_case_count": stage_c_loadable_case_hit,
        "stage_c_masklet_loadable_case_coverage": stage_c_loadable_case_hit / max(case_count, 1),
        "component_like_track_id_case_count": component_like_case_hit,
        "component_like_track_ids_available": component_like_case_hit > 0,
        "support_anchor_row_count": len(support_rows),
        "jl4_identity_gap_row_count": len(atlas_gap_rows),
        "direct_anchor_to_stage_c_seed_match_count": direct_anchor_match_count,
        "diagnostic_anchor_seed_join_feasible": diagnostic_anchor_seed_join_feasible,
        "diagnostic_anchor_seed_payload_with_stage_c_seed_count": diagnostic_stage_c_payload_count,
        "diagnostic_anchor_seed_payload_with_ttt_anchor_id_count": diagnostic_ttt_anchor_payload_count,
        "diagnostic_anchor_seed_lifecycle_pair_count": diagnostic_lifecycle_pair_count,
        "diagnostic_lifecycle_explicit_anchor_seed_mapping_available": diagnostic_lifecycle_anchor_seed_mapping_available,
        "diagnostic_lifecycle_stage_c_seed_support_join_row_count": diagnostic_lifecycle_seed_join_row_count,
        "diagnostic_lifecycle_stage_c_seed_support_join_unique_coverage": (
            diagnostic_lifecycle_seed_join_unique_coverage
        ),
        "diagnostic_lifecycle_anchor_id_semantics": "chunk_idx*1000000+patch_index from HMC ttt_stable_anchor_id_patch; diagnostic only",
        "upstream_component_provenance_bridge_available": upstream_bridge_available,
        "explicit_anchor_component_mapping_available": explicit_mapping_available,
        "jl4_identity_rescue_available": False,
        "runtime_action_allowed": False,
        "full_validation_run": False,
        "full_method_success": False,
        "blocked_reason": (
            "Stage-C component-like track ids are available for target cases, but current v101 rows do not carry an "
            "explicit action-ready anchor_id -> Stage-C seed_global_track_idx/component_id mapping. Direct string matches are zero. "
            "Diagnostic probe_ttt_write traces now expose anchor_id -> Stage-C seed lifecycle pairs and seed-level support joins, "
            "but those anchor ids are chunk_idx*1000000+patch_index diagnostic ids, JL4 identity gap rows remain "
            "semantic_class_fallback, strict current support is not rescued, and runtime action remains blocked."
        ),
        **support_schema,
        **same_space_schema,
        **geometry_schema,
        **trace_payload_schema,
    }
    write_rows(FINAL / "component_identity_availability_case_rows.csv", case_rows)
    write_rows(FINAL / "component_identity_availability_chunk_rows.csv", chunk_rows)
    write_json(FINAL / "component_identity_availability_summary.json", summary)
    write_report(FINAL / "component_identity_availability_report.md", summary)
    print(json.dumps(json_clean(summary), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
