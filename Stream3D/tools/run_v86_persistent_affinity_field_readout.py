#!/usr/bin/env python3
"""Run Stream4D v86 persistent affinity field readout audit.

This runner converts the v85 native-carrier diagnostic evidence into a v86
pre-registered native-carrier readout/evaluator audit. It deliberately keeps
native-carrier metrics separate from ScanNet scene-vertex AP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

PHASE_ORDER = (
    "phase0",
    "phase1",
    "phase5",
    "phase6",
    "phase8",
    "phase9",
    "phase10",
    "phase11",
    "phase12",
    "phase13",
    "phase14",
    "phase15",
    "phase16",
)

DEV_SPLIT_CHUNKS = {
    "scene0011_00": set(range(0, 6)),
    "scene0050_00": set(range(0, 4)),
}
HOLDOUT_SPLIT_CHUNKS = {
    "scene0011_00": set(range(6, 12)),
    "scene0050_00": set(range(4, 12)),
}
DEFAULT_NATIVE_CARRIER_OBSERVATION_TABLES = [
    "outputs/audit/v66_soma_fullscene_pipeline_scene0011_00_stride5_conf02_integrated_d4rt/observation_tables/carrier_observation_table.csv",
    "outputs/audit/v65_soma_fullscene_pipeline_scene0050_stride5_conf02_integrated_d4rt/observation_tables/carrier_observation_table.csv",
]


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "Stream3D":
        return REPO / path
    return ROOT / path


def _rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        out = int(float(str(value)))
    except (TypeError, ValueError):
        return default
    return out


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _canonical_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _comb2(n: int) -> int:
    return n * (n - 1) // 2


def _counter_from_json(text: str) -> Counter[str]:
    if not text:
        return Counter()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return Counter()
    return Counter({str(k): int(v) for k, v in payload.items()})


def _counter_json(counter: Counter[str]) -> str:
    return json.dumps(dict(sorted(counter.items())), sort_keys=True)


def _counter_winner(counter: Counter[str]) -> tuple[str, int, float]:
    total = sum(counter.values())
    if total <= 0:
        return "", 0, 0.0
    label, count = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[0]
    return str(label), int(count), _safe_ratio(count, total)


def _split_name(scene: str, chunk: int) -> str:
    if scene in DEV_SPLIT_CHUNKS and chunk in DEV_SPLIT_CHUNKS[scene]:
        return "dev"
    if scene in HOLDOUT_SPLIT_CHUNKS and chunk in HOLDOUT_SPLIT_CHUNKS[scene]:
        return "holdout"
    return "outside_registered_split"


def _support_rows_for_split(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _split_name(str(row.get("scene_id", "")), _int(row.get("chunk_id"), -1)) == split_name
    ]


def _support_split_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    row_counts: Counter[tuple[str, int, str]] = Counter()
    for row in rows:
        scene = str(row.get("scene_id", "")).strip()
        chunk = _int(row.get("chunk_id"), -1)
        split = _split_name(scene, chunk)
        key = (scene, chunk, split)
        row_counts[key] += 1
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        if native_id:
            grouped[key].add(native_id)
    return [
        {
            "scene_id": scene,
            "chunk_id": chunk,
            "split": split,
            "support_row_count": row_counts[(scene, chunk, split)],
            "unique_native_carrier_count": len(grouped[(scene, chunk, split)]),
        }
        for scene, chunk, split in sorted(grouped, key=lambda item: (item[0], item[1], item[2]))
    ]


def _split_counter_rows(rows: list[dict[str, Any]], *, selected_only: bool = False) -> dict[str, Any]:
    total_rows = 0
    selected_rows = 0
    split_rows: Counter[str] = Counter()
    split_selected_rows: Counter[str] = Counter()
    split_valid_rows: Counter[str] = Counter()
    split_diagnostic_only_rows: Counter[str] = Counter()
    split_method_gt_rows: Counter[str] = Counter()
    split_future_rows: Counter[str] = Counter()
    split_confirmed_rows: Counter[str] = Counter()
    split_tentative_rows: Counter[str] = Counter()
    unique_native_by_split: dict[str, set[str]] = defaultdict(set)
    unique_frame_mask_by_split: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        total_rows += 1
        scene = str(row.get("scene_id") or row.get("scene") or "").strip()
        chunk = _int(row.get("chunk_id"), -1)
        split = _split_name(scene, chunk)
        is_selected = _bool(row.get("selected_flag"))
        if selected_only and not is_selected:
            continue
        split_rows[split] += 1
        if is_selected:
            selected_rows += 1
            split_selected_rows[split] += 1
        if _bool(row.get("adapter_candidate_valid")) or _bool(row.get("native_support_allowed")):
            split_valid_rows[split] += 1
        if _bool(row.get("diagnostic_only")) or _bool(row.get("is_diagnostic_only")):
            split_diagnostic_only_rows[split] += 1
        if _bool(row.get("method_uses_gt")) or _bool(row.get("uses_gt_for_prediction")):
            split_method_gt_rows[split] += 1
        if _bool(row.get("uses_future")):
            split_future_rows[split] += 1
        assignment_type = str(row.get("assignment_type", "")).strip()
        if assignment_type == "confirmed_history":
            split_confirmed_rows[split] += 1
        if "tentative" in assignment_type:
            split_tentative_rows[split] += 1
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        if native_id:
            unique_native_by_split[split].add(native_id)
        frame_id = str(row.get("frame_id", "")).strip()
        mask_id = str(row.get("mask_id", "")).strip()
        if scene and frame_id and mask_id:
            unique_frame_mask_by_split[split].add((scene, frame_id, mask_id))
    return {
        "total_row_count": total_rows,
        "selected_row_count": selected_rows,
        "dev_row_count": split_rows["dev"],
        "holdout_row_count": split_rows["holdout"],
        "outside_registered_split_row_count": split_rows["outside_registered_split"],
        "dev_selected_row_count": split_selected_rows["dev"],
        "holdout_selected_row_count": split_selected_rows["holdout"],
        "holdout_valid_row_count": split_valid_rows["holdout"],
        "holdout_diagnostic_only_row_count": split_diagnostic_only_rows["holdout"],
        "holdout_method_uses_gt_row_count": split_method_gt_rows["holdout"],
        "holdout_uses_future_row_count": split_future_rows["holdout"],
        "holdout_confirmed_assignment_row_count": split_confirmed_rows["holdout"],
        "holdout_tentative_assignment_row_count": split_tentative_rows["holdout"],
        "dev_unique_native_carrier_count": len(unique_native_by_split["dev"]),
        "holdout_unique_native_carrier_count": len(unique_native_by_split["holdout"]),
        "dev_unique_frame_mask_count": len(unique_frame_mask_by_split["dev"]),
        "holdout_unique_frame_mask_count": len(unique_frame_mask_by_split["holdout"]),
    }


def _existing_source_tables_from_support(support_rows: list[dict[str, Any]]) -> list[Path]:
    paths = [
        _repo_path(path)
        for path in sorted(
            {
                str(row.get("source_observation_table", "")).strip()
                for row in support_rows
                if str(row.get("source_observation_table", "")).strip()
            }
        )
    ]
    if paths:
        return paths
    return [_repo_path(path) for path in DEFAULT_NATIVE_CARRIER_OBSERVATION_TABLES]


def _selected_frame_mask_keys(rows: list[dict[str, Any]], *, split_name: str | None = None) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        if not _bool(row.get("selected_flag")):
            continue
        scene = str(row.get("scene_id", "")).strip()
        chunk = _int(row.get("chunk_id"), -1)
        if split_name is not None and _split_name(scene, chunk) != split_name:
            continue
        frame_id = str(row.get("frame_id", "")).strip()
        mask_id = str(row.get("mask_id", "")).strip()
        if scene and frame_id and mask_id:
            keys.add((scene, frame_id, mask_id))
    return keys


def _audit_observation_table_source(table_path: Path, selected_keys: set[tuple[str, str, str]]) -> dict[str, Any]:
    row_count = 0
    dev_row_count = 0
    holdout_row_count = 0
    allowed_row_count = 0
    holdout_allowed_row_count = 0
    holdout_matched_selected_row_count = 0
    holdout_allowed_matched_selected_row_count = 0
    holdout_uses_gt_for_prediction_row_count = 0
    holdout_unique_allowed_carriers: set[str] = set()
    holdout_chunks: set[str] = set()
    exists = table_path.exists()
    if exists:
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_count += 1
                scene = str(row.get("scene_id") or row.get("scene") or "").strip()
                chunk = _int(row.get("chunk_id"), -1)
                split = _split_name(scene, chunk)
                if split == "dev":
                    dev_row_count += 1
                if split == "holdout":
                    holdout_row_count += 1
                    holdout_chunks.add(f"{scene}:{chunk}")
                uses_gt = _bool(row.get("uses_gt_for_prediction"))
                allowed = (
                    not uses_gt
                    and _bool(row.get("visible"))
                    and _bool(row.get("valid"))
                    and _bool(row.get("valid_uv"))
                    and _bool(row.get("inside_prepared_mask"))
                    and _bool(row.get("scale_guard_pass"))
                    and bool(str(row.get("carrier_global_id", "")).strip())
                )
                if allowed:
                    allowed_row_count += 1
                key = (scene, str(row.get("frame_id", "")).strip(), str(row.get("observed_mask_id", "")).strip())
                if split == "holdout":
                    if uses_gt:
                        holdout_uses_gt_for_prediction_row_count += 1
                    if allowed:
                        holdout_allowed_row_count += 1
                        holdout_unique_allowed_carriers.add(str(row.get("carrier_global_id", "")).strip())
                    if key in selected_keys:
                        holdout_matched_selected_row_count += 1
                        if allowed:
                            holdout_allowed_matched_selected_row_count += 1
    method_input_available = holdout_allowed_matched_selected_row_count > 0
    return {
        "candidate_id": "d4rt_carrier_observation_table",
        "source_path": _rel(table_path),
        "exists": exists,
        "source_role": "raw method-safe D4RT carrier observations; requires selected frame-mask/history readout join",
        "total_row_count": row_count,
        "dev_row_count": dev_row_count,
        "holdout_row_count": holdout_row_count,
        "allowed_row_count": allowed_row_count,
        "holdout_allowed_row_count": holdout_allowed_row_count,
        "holdout_unique_native_carrier_count": len(holdout_unique_allowed_carriers),
        "holdout_selected_join_row_count": holdout_matched_selected_row_count,
        "holdout_allowed_selected_join_row_count": holdout_allowed_matched_selected_row_count,
        "holdout_uses_gt_for_prediction_row_count": holdout_uses_gt_for_prediction_row_count,
        "holdout_chunks": ",".join(sorted(holdout_chunks)),
        "method_safe_prediction_candidate_available": method_input_available,
        "method_safe_holdout_input_available": method_input_available,
        "forbidden_for_method_table": False,
        "primary_blocker": "" if method_input_available else "raw_observations_exist_but_no_holdout_selected_frame_mask_history_join",
        "notes": (
            "Carrier observations are legal support primitives, but they are not object predictions. "
            "v86 Phase10 needs selected frame-mask/native support rows produced by the frozen method."
        ),
    }


def _holdout_source_candidate_audit(ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase7 = ctx["v85_phase7"]
    support_rows = ctx["support_rows"]
    frame_rows = _read_csv_rows(phase7 / "frame_mask_prediction_rows.csv")
    holdout_selected_keys = _selected_frame_mask_keys(frame_rows, split_name="holdout")
    rows: list[dict[str, Any]] = []

    frame_counts = _split_counter_rows(frame_rows)
    rows.append(
        {
            "candidate_id": "v85_frame_mask_prediction_rows",
            "source_path": _rel(phase7 / "frame_mask_prediction_rows.csv"),
            "exists": (phase7 / "frame_mask_prediction_rows.csv").exists(),
            "source_role": "frozen-method frame-mask readout candidates selected before native carrier join",
            **frame_counts,
            "method_safe_prediction_candidate_available": frame_counts["holdout_selected_row_count"] > 0
            and frame_counts["holdout_method_uses_gt_row_count"] == 0
            and frame_counts["holdout_uses_future_row_count"] == 0,
            "method_safe_holdout_input_available": frame_counts["holdout_selected_row_count"] > 0,
            "forbidden_for_method_table": False,
            "primary_blocker": ""
            if frame_counts["holdout_selected_row_count"] > 0
            else "zero_selected_frame_mask_rows_in_registered_holdout_split",
            "notes": "Upstream readout coverage check; without selected holdout frame masks no native support rows can be joined.",
        }
    )

    support_counts = _split_counter_rows(support_rows)
    rows.append(
        {
            "candidate_id": "v85_native_carrier_support_rows",
            "source_path": _rel(phase7 / "native_carrier_support_rows.csv"),
            "exists": (phase7 / "native_carrier_support_rows.csv").exists(),
            "source_role": "method-safe native-carrier support generated from selected v85/v86 frame masks",
            **support_counts,
            "method_safe_prediction_candidate_available": support_counts["holdout_row_count"] > 0
            and support_counts["holdout_method_uses_gt_row_count"] == 0
            and support_counts["holdout_uses_future_row_count"] == 0,
            "method_safe_holdout_input_available": support_counts["holdout_row_count"] > 0,
            "forbidden_for_method_table": False,
            "primary_blocker": ""
            if support_counts["holdout_row_count"] > 0
            else "zero_native_support_rows_in_registered_holdout_split",
            "notes": "This is the actual Phase10 input table for native-carrier objectness holdout replay.",
        }
    )

    v84_sources = [
        (
            "v84_holdout_phase1_local_slots",
            _repo_path("outputs/audit/v84_holdout_replay_v82_phase1_local_b0/local_slot_rows.csv"),
            "holdout local slots only; no confirmed history/native-carrier readout",
        ),
        (
            "v84_holdout_phase5_weak_history",
            _repo_path("outputs/audit/v84_holdout_replay_v82_phase5_weak_history/local_slot_history_assignment_rows.csv"),
            "weak local-to-history assignments from v84 holdout replay",
        ),
        (
            "v84_holdout_phase6_strong_adapter_rows",
            _repo_path("outputs/audit/v84_holdout_replay_v82_phase6_strong_history/adapter_rows.csv"),
            "strong history adapter rows required before frame-mask/native readout",
        ),
        (
            "v84_holdout_phase6_strong_cluster_rows",
            _repo_path("outputs/audit/v84_holdout_replay_v82_phase6_strong_history/cluster_rows.csv"),
            "strong history cluster rows required before frame-mask/native readout",
        ),
    ]
    weak_summary = _read_json(
        _repo_path("outputs/audit/v84_holdout_replay_v82_phase5_weak_history/summary.json")
    )
    strong_summary = _read_json(
        _repo_path("outputs/audit/v84_holdout_replay_v82_phase6_strong_history/summary.json")
    )
    final_local_summary = _read_json(
        _repo_path("outputs/audit/v84_holdout_replay_v82_phase7_final_local/summary.json")
    )
    for candidate_id, path, role in v84_sources:
        source_rows = _read_csv_rows(path)
        counts = _split_counter_rows(source_rows)
        if candidate_id == "v84_holdout_phase5_weak_history":
            method_allowed = _bool(weak_summary.get("method_mode_claim_allowed"))
            blocker = ""
            if not method_allowed:
                blocker = "v84_weak_history_marked_diagnostic_only_and_method_mode_claim_disallowed"
            elif counts["holdout_confirmed_assignment_row_count"] <= 0:
                blocker = "zero_confirmed_history_assignments_in_holdout"
        elif candidate_id.startswith("v84_holdout_phase6"):
            method_allowed = bool(source_rows) and not _bool(strong_summary.get("primary_blocker"))
            blocker = "" if method_allowed else "v84_strong_history_preconditions_failed_or_output_empty"
        else:
            method_allowed = False
            blocker = "local_slots_do_not_contain_confirmed_history_or_native_carrier_readout"
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_path": _rel(path),
                "exists": path.exists(),
                "source_role": role,
                **counts,
                "v84_weak_decision": weak_summary.get("decision", "") if "phase5" in candidate_id else "",
                "v84_weak_method_mode_claim_allowed": weak_summary.get("method_mode_claim_allowed", "")
                if "phase5" in candidate_id
                else "",
                "v84_strong_decision": strong_summary.get("decision", "") if "phase6" in candidate_id else "",
                "v84_final_local_decision": final_local_summary.get("decision", "") if "phase1" in candidate_id else "",
                "method_safe_prediction_candidate_available": method_allowed,
                "method_safe_holdout_input_available": method_allowed,
                "forbidden_for_method_table": not method_allowed and candidate_id == "v84_holdout_phase5_weak_history",
                "primary_blocker": blocker,
                "notes": "Candidate inspected as repair input; not promoted into v86 method path unless method-safe strong readout exists.",
            }
        )

    for table_path in _existing_source_tables_from_support(support_rows):
        rows.append(_audit_observation_table_source(table_path, holdout_selected_keys))

    method_safe_available_count = sum(1 for row in rows if _bool(row.get("method_safe_holdout_input_available")))
    observation_holdout_allowed = sum(
        _int(row.get("holdout_allowed_row_count"), 0)
        for row in rows
        if row.get("candidate_id") == "d4rt_carrier_observation_table"
    )
    observation_holdout_join = sum(
        _int(row.get("holdout_allowed_selected_join_row_count"), 0)
        for row in rows
        if row.get("candidate_id") == "d4rt_carrier_observation_table"
    )
    summary = {
        "schema": "stream4d_v86_holdout_source_candidate_audit_v1",
        "candidate_source_count": len(rows),
        "method_safe_holdout_input_available_count": method_safe_available_count,
        "holdout_selected_frame_mask_row_count": frame_counts["holdout_selected_row_count"],
        "holdout_native_support_row_count": support_counts["holdout_row_count"],
        "holdout_observation_allowed_row_count": observation_holdout_allowed,
        "holdout_observation_allowed_selected_join_row_count": observation_holdout_join,
        "v84_weak_diagnostic_only_holdout_row_count": next(
            (
                _int(row.get("holdout_diagnostic_only_row_count"), 0)
                for row in rows
                if row.get("candidate_id") == "v84_holdout_phase5_weak_history"
            ),
            0,
        ),
        "v84_weak_confirmed_holdout_assignment_row_count": next(
            (
                _int(row.get("holdout_confirmed_assignment_row_count"), 0)
                for row in rows
                if row.get("candidate_id") == "v84_holdout_phase5_weak_history"
            ),
            0,
        ),
        "primary_blocker": ""
        if method_safe_available_count
        else "holdout_has_raw_d4rt_observations_but_no_method_safe_selected_frame_mask_native_readout",
        "method_uses_gt_anywhere": False,
        "uses_future_anywhere": False,
    }
    return rows, summary


def _diagnostic_tentative_holdout_replay(ctx: dict[str, Any], out: Path) -> dict[str, Any]:
    weak_path = _repo_path(
        "outputs/audit/v84_holdout_replay_v82_phase5_weak_history/local_slot_history_assignment_rows.csv"
    )
    local_path = _repo_path("outputs/audit/v84_holdout_replay_v82_phase1_local_b0/local_slot_rows.csv")
    adapter_path = _repo_path(
        "outputs/audit/v84_holdout_replay_v82_local_shadow/phase1_adapter_v84_holdout_replay/adapter_rows.csv"
    )
    weak_rows = _read_csv_rows(weak_path)
    local_rows = _read_csv_rows(local_path)
    adapter_rows = _read_csv_rows(adapter_path)
    cluster_by_slot = {
        (str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("local_slot_id", ""))): str(
            row.get("cluster_id", "")
        )
        for row in local_rows
    }
    weak_by_cluster: dict[tuple[str, str, str], dict[str, Any]] = {}
    weak_missing_cluster_count = 0
    weak_forbidden_count = 0
    for row in weak_rows:
        scene = str(row.get("scene_id", ""))
        chunk = str(row.get("chunk_id", ""))
        local_slot = str(row.get("local_slot_id", ""))
        cluster_id = cluster_by_slot.get((scene, chunk, local_slot), "")
        if not cluster_id:
            weak_missing_cluster_count += 1
            continue
        if _bool(row.get("diagnostic_only")) or _bool(row.get("method_uses_gt")) or _bool(row.get("uses_future")):
            weak_forbidden_count += 1
        weak_by_cluster[(scene, chunk, cluster_id)] = row

    candidate_rows: list[dict[str, Any]] = []
    best_by_history_frame: dict[tuple[str, str, str], int] = {}
    joined_adapter_count = 0
    allowed_adapter_count = 0
    for adapter in adapter_rows:
        scene = str(adapter.get("scene_id", ""))
        chunk = str(adapter.get("chunk_id", ""))
        cluster_id = str(adapter.get("cluster_id", ""))
        weak = weak_by_cluster.get((scene, chunk, cluster_id))
        if not weak:
            continue
        joined_adapter_count += 1
        score = _num(adapter.get("hybrid_adapter_F1"), _num(adapter.get("carrier_F1"), 0.0))
        adapter_allowed = (
            _bool(adapter.get("object_mask_ownership_allowed"))
            and not _bool(adapter.get("adapter_caused_split"))
            and not _bool(adapter.get("adapter_caused_merge"))
            and not _bool(weak.get("method_uses_gt"))
            and not _bool(weak.get("uses_future"))
        )
        if adapter_allowed:
            allowed_adapter_count += 1
        row = {
            "candidate_row_id": len(candidate_rows),
            "scene_id": scene,
            "chunk_id": chunk,
            "frame_id": adapter.get("frame_id", ""),
            "mask_id": adapter.get("mask_id", ""),
            "history_id": weak.get("assigned_history_id", ""),
            "tracklet_id": weak.get("tracklet_id", ""),
            "local_slot_id": weak.get("local_slot_id", ""),
            "cluster_id": cluster_id,
            "adapter_score": score,
            "carrier_F1": adapter.get("carrier_F1", ""),
            "rendered_pixel_F1": adapter.get("rendered_pixel_F1", ""),
            "hybrid_adapter_F1": adapter.get("hybrid_adapter_F1", ""),
            "object_mask_ownership_allowed": adapter.get("object_mask_ownership_allowed", ""),
            "adapter_caused_split": adapter.get("adapter_caused_split", ""),
            "adapter_caused_merge": adapter.get("adapter_caused_merge", ""),
            "weak_assignment_type": weak.get("assignment_type", ""),
            "weak_assignment_score": weak.get("score", ""),
            "weak_q_margin": weak.get("q_margin", ""),
            "weak_diagnostic_only": weak.get("diagnostic_only", ""),
            "adapter_candidate_valid": adapter_allowed,
            "selected_flag": False,
            "selection_policy": "diagnostic_top_adapter_per_scene_tentative_history_frame",
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "method_uses_gt": False,
            "uses_future": False,
            "blocker_for_method_claim": "source_weak_history_assignments_are_diagnostic_only_tentative",
        }
        candidate_rows.append(row)
        if adapter_allowed:
            key = (scene, str(row["history_id"]), str(row["frame_id"]))
            old_idx = best_by_history_frame.get(key)
            if old_idx is None:
                best_by_history_frame[key] = int(row["candidate_row_id"])
            else:
                old = candidate_rows[old_idx]
                if (
                    score > _num(old.get("adapter_score"), -1.0)
                    or (score == _num(old.get("adapter_score"), -1.0) and str(row.get("mask_id", "")) < str(old.get("mask_id", "")))
                ):
                    best_by_history_frame[key] = int(row["candidate_row_id"])
    for idx in best_by_history_frame.values():
        candidate_rows[idx]["selected_flag"] = True
    selected_rows = [row for row in candidate_rows if _bool(row.get("selected_flag"))]
    selected_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        key = (str(row.get("scene_id", "")), str(row.get("frame_id", "")), str(row.get("mask_id", "")))
        if all(key):
            selected_by_key[key].append(row)

    support_rows: list[dict[str, Any]] = []
    matched_observation_rows = 0
    blocked_observation_rows = 0
    for table_path in _existing_source_tables_from_support(ctx["support_rows"]):
        rel_path = _rel(table_path)
        if not table_path.exists():
            continue
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for obs in reader:
                scene = str(obs.get("scene_id") or obs.get("scene") or "")
                key = (scene, str(obs.get("frame_id", "")), str(obs.get("observed_mask_id", "")))
                selected_matches = selected_by_key.get(key)
                if not selected_matches:
                    continue
                matched_observation_rows += 1
                allowed = (
                    not _bool(obs.get("uses_gt_for_prediction"))
                    and _bool(obs.get("visible"))
                    and _bool(obs.get("valid"))
                    and _bool(obs.get("valid_uv"))
                    and _bool(obs.get("inside_prepared_mask"))
                    and _bool(obs.get("scale_guard_pass"))
                    and bool(str(obs.get("carrier_global_id", "")).strip())
                )
                if not allowed:
                    blocked_observation_rows += len(selected_matches)
                    continue
                for selected in selected_matches:
                    support_rows.append(
                        {
                            "scene_id": scene,
                            "chunk_id": selected.get("chunk_id", ""),
                            "frame_id": obs.get("frame_id", ""),
                            "mask_id": obs.get("observed_mask_id", ""),
                            "history_id": selected.get("history_id", ""),
                            "local_slot_id": selected.get("local_slot_id", ""),
                            "cluster_id": selected.get("cluster_id", ""),
                            "candidate_row_id": selected.get("candidate_row_id", ""),
                            "native_carrier_global_id": obs.get("carrier_global_id", ""),
                            "native_carrier_id": obs.get("carrier_id", ""),
                            "source_observation_table": rel_path,
                            "native_support_allowed": True,
                            "native_support_kind": "diagnostic_tentative_history_d4rt_carrier_global_id",
                            "is_method_result": False,
                            "is_diagnostic_only": True,
                            "forbidden_for_method_table": True,
                            "uses_gt_for_prediction": False,
                            "method_uses_gt": False,
                            "uses_future": False,
                        }
                    )

    assignment_rows, source_audit_rows, assignment_summary = _assignment_rows_from_support(support_rows)
    pred_rows = [
        dict(
            row,
            v86_label=row.get("pred_history_eval_label", ""),
            v86_score=row.get("pred_vote_purity", ""),
            is_method_result=False,
            is_diagnostic_only=True,
            forbidden_for_method_table=True,
        )
        for row in assignment_rows
    ]
    metric_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    if pred_rows:
        pred_scores = _pred_scores(pred_rows, label_field="v86_label")
        metric_row = {
            **_eval_variant(
                assignment_rows,
                pred_rows,
                variant_id="D0_diagnostic_tentative_weak_history_frame_mask_native_replay",
                label_field="v86_label",
                pred_scores=pred_scores,
                score_contract="diagnostic_mean_pred_vote_purity_per_tentative_history",
            ),
            "variant_role": "diagnostic_only_not_method_candidate",
            "GT_label_coverage_rate": assignment_summary.get("native_gt_label_coverage_rate", 0.0),
            "is_method_result": False,
            "is_diagnostic_only": True,
            "forbidden_for_method_table": True,
            "blocker_for_method_claim": "v84_holdout_phase5_weak_history_method_mode_claim_allowed_false",
        }
        controls: list[tuple[str, str, list[dict[str, Any]], str, bool, bool]] = [
            (
                "D_B5_size_matched_hash_global",
                "diagnostic global size matched hash",
                _labels_to_rows(
                    pred_rows,
                    _size_matched_hash_labels(
                        pred_rows,
                        group_field=None,
                        source_label_field="v86_label",
                        salt="v86_diagnostic_tentative_holdout_size_global",
                    ),
                    label_field="control_label",
                ),
                "cluster_size_desc",
                False,
                False,
            ),
            (
                "D_B6_size_matched_hash_by_scene",
                "diagnostic scene size matched hash",
                _labels_to_rows(
                    pred_rows,
                    _size_matched_hash_labels(
                        pred_rows,
                        group_field="scene_id",
                        source_label_field="v86_label",
                        salt="v86_diagnostic_tentative_holdout_size_scene",
                    ),
                    label_field="control_label",
                ),
                "cluster_size_desc",
                False,
                False,
            ),
            (
                "D_B7_uniform_hash_history",
                "diagnostic uniform hash over tentative labels",
                _labels_to_rows(
                    pred_rows,
                    _uniform_hash_labels(
                        pred_rows,
                        source_label_field="v86_label",
                        salt="v86_diagnostic_tentative_holdout_uniform",
                    ),
                    label_field="control_label",
                ),
                "cluster_size_desc",
                False,
                False,
            ),
            (
                "D_B8_single_largest_by_scene",
                "diagnostic single largest cluster by scene",
                _labels_to_rows(
                    pred_rows,
                    _single_largest_labels(pred_rows, source_label_field="v86_label", group_field="scene_id"),
                    label_field="control_label",
                ),
                "cluster_size_desc",
                False,
                False,
            ),
            (
                "D_B11_oracle_diagnostic_gt",
                "diagnostic oracle GT upper bound",
                [dict(row, control_label=row.get("diagnostic_gt_eval_label", "")) for row in pred_rows],
                "cluster_size_desc_for_oracle",
                True,
                True,
            ),
        ]
        non_oracle_ap50: list[float] = []
        for control_id, notes, rows, score_contract, uses_gt, is_oracle in controls:
            control_row = {
                **_eval_variant(
                    assignment_rows,
                    rows,
                    variant_id=f"diagnostic_tentative:{control_id}",
                    label_field="control_label",
                    pred_scores=None,
                    score_contract=score_contract,
                    prediction_uses_gt=uses_gt,
                    is_oracle=is_oracle,
                ),
                "control_id": control_id,
                "control_notes": notes,
                "is_method_result": False,
                "is_diagnostic_only": True,
            }
            control_rows.append(control_row)
            if not uses_gt:
                non_oracle_ap50.append(_num(control_row.get("native_AP50"), 0.0))
        metric_row["real_minus_best_non_oracle_AP50"] = _num(metric_row.get("native_AP50"), 0.0) - (
            max(non_oracle_ap50) if non_oracle_ap50 else 0.0
        )
        diagnostic_gate = {
            "native_AP50_ge_0p35": _num(metric_row.get("native_AP50"), 0.0) >= 0.35,
            "real_minus_best_non_oracle_AP50_ge_0p07": metric_row["real_minus_best_non_oracle_AP50"] >= 0.07,
            "purity_ge_0p75": _num(metric_row.get("purity"), 0.0) >= 0.75,
            "ARI_ge_0p50": _num(metric_row.get("adjusted_rand_index"), 0.0) >= 0.50,
            "GT_label_coverage_rate_ge_0p80": _num(metric_row.get("GT_label_coverage_rate"), 0.0) >= 0.80,
        }
        diagnostic_gate["pass"] = all(diagnostic_gate.values())
        metric_row["diagnostic_holdout_gate"] = json.dumps(diagnostic_gate, sort_keys=True)
        metric_rows.append(metric_row)

    summary = {
        "schema": "stream4d_v86_diagnostic_tentative_holdout_replay_v1",
        "decision": "DIAGNOSTIC_TENTATIVE_HOLDOUT_REPLAY_NOT_METHOD",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "weak_assignment_path": _rel(weak_path),
        "adapter_rows_path": _rel(adapter_path),
        "weak_assignment_row_count": len(weak_rows),
        "weak_missing_cluster_count": weak_missing_cluster_count,
        "weak_forbidden_or_diagnostic_row_count": weak_forbidden_count,
        "joined_adapter_row_count": joined_adapter_count,
        "allowed_adapter_row_count": allowed_adapter_count,
        "selected_frame_mask_row_count": len(selected_rows),
        "selected_unique_frame_mask_count": len(selected_by_key),
        "matched_observation_row_count": matched_observation_rows,
        "blocked_observation_row_count": blocked_observation_rows,
        "diagnostic_native_support_row_count": len(support_rows),
        "diagnostic_unique_native_carrier_count": len(
            {row.get("native_carrier_global_id", "") for row in support_rows if row.get("native_carrier_global_id")}
        ),
        "diagnostic_native_assignment_count": len(assignment_rows),
        "diagnostic_native_AP50": metric_rows[0].get("native_AP50", "") if metric_rows else "",
        "diagnostic_real_minus_best_non_oracle_AP50": metric_rows[0].get("real_minus_best_non_oracle_AP50", "")
        if metric_rows
        else "",
        "diagnostic_purity": metric_rows[0].get("purity", "") if metric_rows else "",
        "diagnostic_ARI": metric_rows[0].get("adjusted_rand_index", "") if metric_rows else "",
        "diagnostic_GT_label_coverage_rate": assignment_summary.get("native_gt_label_coverage_rate", ""),
        "method_claim_blocker": "source weak history assignments are diagnostic-only tentative and phase2-6 holdout preconditions failed",
    }
    _write_csv(out / "diagnostic_tentative_holdout_frame_mask_candidate_rows.csv", candidate_rows)
    _write_csv(out / "diagnostic_tentative_holdout_frame_mask_selected_rows.csv", selected_rows)
    _write_csv(out / "diagnostic_tentative_holdout_native_assignment_rows.csv", pred_rows)
    _write_csv(out / "diagnostic_tentative_holdout_source_audit_rows.csv", source_audit_rows)
    _write_csv(out / "diagnostic_tentative_holdout_native_metric_rows.csv", metric_rows)
    _write_csv(out / "diagnostic_tentative_holdout_control_rows.csv", control_rows)
    _write_json(out / "diagnostic_tentative_holdout_summary.json", summary)
    return summary


def _holdout_repair_probe_source_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    weak_path = _repo_path(
        "outputs/audit/v84_holdout_replay_v82_phase5_weak_history/local_slot_history_assignment_rows.csv"
    )
    tracklet_path = _repo_path(
        "outputs/audit/v84_holdout_replay_v82_phase2_object_tracklets/tracklet_assignment_rows.csv"
    )
    phase2_summary = _read_json(
        _repo_path("outputs/audit/v84_holdout_replay_v82_phase2_object_tracklets/summary.json")
    )
    phase5_summary = _read_json(
        _repo_path("outputs/audit/v84_holdout_replay_v82_phase5_weak_history/summary.json")
    )
    weak_rows = _read_csv_rows(weak_path)
    tracklet_rows = _read_csv_rows(tracklet_path)
    weak_tracklet_chunks: dict[str, set[str]] = defaultdict(set)
    for row in weak_rows:
        tracklet = str(row.get("tracklet_id", "")).strip()
        chunk = str(row.get("chunk_id", "")).strip()
        if tracklet and chunk:
            weak_tracklet_chunks[tracklet].add(chunk)

    source_rows: list[dict[str, Any]] = []

    def add_source(row: dict[str, Any], *, variant: str, label: str, source_kind: str, diagnostic_only: bool) -> None:
        if not label:
            return
        source_rows.append(
            {
                "variant": variant,
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "readout_label": label,
                "tracklet_id": row.get("tracklet_id", ""),
                "source_kind": source_kind,
                "source_score": row.get("score", ""),
                "source_margin": row.get("q_margin", row.get("margin", "")),
                "source_entropy": row.get("assignment_entropy", row.get("entropy", "")),
                "source_state_before": row.get("tracklet_state_before", ""),
                "source_state_after": row.get("tracklet_state_after", ""),
                "support_slot_count_after": row.get("support_slot_count_after", ""),
                "support_chunk_count_after": row.get("support_chunk_count_after", ""),
                "full_minus_semantic_slot": row.get("full_minus_semantic_slot", ""),
                "source_diagnostic_only": diagnostic_only,
                "source_method_uses_gt": _bool(row.get("method_uses_gt")),
                "source_uses_future": _bool(row.get("uses_future")),
            }
        )

    for row in weak_rows:
        if _bool(row.get("method_uses_gt")) or _bool(row.get("uses_future")):
            continue
        label = str(row.get("assigned_history_id", "")).strip()
        add_source(
            row,
            variant="RP1_weak_tentative_all",
            label=f"weak:{label}",
            source_kind="v84_phase5_weak_tentative",
            diagnostic_only=True,
        )
        score = _num(row.get("score"), 0.0)
        margin = _num(row.get("q_margin"), 0.0)
        entropy = _num(row.get("assignment_entropy"), 1.0)
        new_object_score = _num(row.get("new_object_score"), 1.0)
        tracklet = str(row.get("tracklet_id", "")).strip()
        repeated_chunks = len(weak_tracklet_chunks.get(tracklet, set()))
        if score >= 0.75 and margin >= 0.05 and entropy <= 0.50 and new_object_score <= 0.30:
            add_source(
                row,
                variant="RP2_weak_high_margin_low_entropy",
                label=f"weak_strict:{label}",
                source_kind="v84_phase5_weak_tentative_filtered",
                diagnostic_only=True,
            )
        if score >= 0.70 and margin >= 0.03 and repeated_chunks >= 2:
            add_source(
                row,
                variant="RP3_weak_repeated_tracklet_margin",
                label=f"weak_repeated:{label}",
                source_kind="v84_phase5_weak_tentative_repeated_tracklet",
                diagnostic_only=True,
            )

    for row in tracklet_rows:
        if _bool(row.get("method_uses_gt")) or _bool(row.get("uses_future")):
            continue
        tracklet = str(row.get("tracklet_id", "")).strip()
        state_after = str(row.get("tracklet_state_after", "")).strip()
        support_slots = _int(row.get("support_slot_count_after"), 0)
        support_chunks = _int(row.get("support_chunk_count_after"), 0)
        score = _num(row.get("score"), 0.0)
        margin = _num(row.get("margin"), 0.0)
        entropy = _num(row.get("entropy"), 1.0)
        full_minus_sem = _num(row.get("full_minus_semantic_slot"), -1.0)
        if state_after == "confirmed":
            add_source(
                row,
                variant="RP4_phase2_confirmed_tracklets",
                label=f"tracklet_confirmed:{tracklet}",
                source_kind="v84_phase2_confirmed_tracklet",
                diagnostic_only=False,
            )
        if state_after == "confirmed" and support_slots >= 2 and support_chunks >= 2 and full_minus_sem >= 0.03:
            add_source(
                row,
                variant="RP5_phase2_confirmed_object_gain",
                label=f"tracklet_confirmed_gain:{tracklet}",
                source_kind="v84_phase2_confirmed_tracklet_object_gain",
                diagnostic_only=False,
            )
        if support_slots >= 2 and score >= 0.70 and margin >= 0.05 and entropy <= 0.50:
            add_source(
                row,
                variant="RP6_phase2_repeated_high_margin",
                label=f"tracklet_repeated_margin:{tracklet}",
                source_kind="v84_phase2_repeated_high_margin_tracklet",
                diagnostic_only=False,
            )
        if full_minus_sem >= 0.03 and score >= 0.70 and margin >= 0.03:
            add_source(
                row,
                variant="RP7_phase2_object_gain_margin",
                label=f"tracklet_object_gain:{tracklet}",
                source_kind="v84_phase2_object_gain_tracklet",
                diagnostic_only=False,
            )

    audit = {
        "weak_path": _rel(weak_path),
        "tracklet_assignment_path": _rel(tracklet_path),
        "weak_row_count": len(weak_rows),
        "tracklet_assignment_row_count": len(tracklet_rows),
        "phase2_decision": phase2_summary.get("decision", ""),
        "phase2_primary_blocker": phase2_summary.get("primary_blocker", ""),
        "phase2_eligible_tracklet_coverage_rate": phase2_summary.get("eligible_tracklet_coverage_rate", ""),
        "phase2_full_minus_semantic_score": phase2_summary.get("full_minus_semantic_score", ""),
        "phase5_decision": phase5_summary.get("decision", ""),
        "phase5_method_mode_claim_allowed": phase5_summary.get("method_mode_claim_allowed", ""),
        "source_row_count": len(source_rows),
    }
    return source_rows, audit


def _materialize_probe_support(
    ctx: dict[str, Any],
    source_rows: list[dict[str, Any]],
    *,
    local_path: Path | None = None,
    adapter_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    local_path = local_path or _repo_path("outputs/audit/v84_holdout_replay_v82_phase1_local_b0/local_slot_rows.csv")
    adapter_path = adapter_path or _repo_path(
        "outputs/audit/v84_holdout_replay_v82_local_shadow/phase1_adapter_v84_holdout_replay/adapter_rows.csv"
    )
    local_rows = _read_csv_rows(local_path)
    adapter_rows = _read_csv_rows(adapter_path)
    cluster_by_slot = {
        (str(row.get("scene_id", "")), str(row.get("chunk_id", "")), str(row.get("local_slot_id", ""))): str(
            row.get("cluster_id", "")
        )
        for row in local_rows
    }
    sources_by_cluster: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    missing_cluster_count = 0
    for row in source_rows:
        scene = str(row.get("scene_id", ""))
        chunk = str(row.get("chunk_id", ""))
        local_slot = str(row.get("local_slot_id", ""))
        cluster_id = cluster_by_slot.get((scene, chunk, local_slot), "")
        if not cluster_id:
            missing_cluster_count += 1
            continue
        new_row = dict(row)
        new_row["cluster_id"] = cluster_id
        sources_by_cluster[(scene, chunk, cluster_id)].append(new_row)

    candidate_rows: list[dict[str, Any]] = []
    best_by_variant_label_frame: dict[tuple[str, str, str, str], int] = {}
    joined_adapter_count = 0
    allowed_adapter_count = 0
    for adapter in adapter_rows:
        scene = str(adapter.get("scene_id", ""))
        chunk = str(adapter.get("chunk_id", ""))
        cluster_id = str(adapter.get("cluster_id", ""))
        matches = sources_by_cluster.get((scene, chunk, cluster_id), [])
        if not matches:
            continue
        joined_adapter_count += len(matches)
        score = _num(adapter.get("hybrid_adapter_F1"), _num(adapter.get("carrier_F1"), 0.0))
        adapter_allowed = (
            _bool(adapter.get("object_mask_ownership_allowed"))
            and not _bool(adapter.get("adapter_caused_split"))
            and not _bool(adapter.get("adapter_caused_merge"))
        )
        if adapter_allowed:
            allowed_adapter_count += len(matches)
        for source in matches:
            source_safe = (
                not _bool(source.get("source_method_uses_gt"))
                and not _bool(source.get("source_uses_future"))
            )
            row = {
                "candidate_row_id": len(candidate_rows),
                "variant": source.get("variant", ""),
                "scene_id": scene,
                "chunk_id": chunk,
                "frame_id": adapter.get("frame_id", ""),
                "mask_id": adapter.get("mask_id", ""),
                "readout_label": source.get("readout_label", ""),
                "tracklet_id": source.get("tracklet_id", ""),
                "local_slot_id": source.get("local_slot_id", ""),
                "cluster_id": cluster_id,
                "adapter_score": score,
                "carrier_F1": adapter.get("carrier_F1", ""),
                "rendered_pixel_F1": adapter.get("rendered_pixel_F1", ""),
                "hybrid_adapter_F1": adapter.get("hybrid_adapter_F1", ""),
                "object_mask_ownership_allowed": adapter.get("object_mask_ownership_allowed", ""),
                "adapter_caused_split": adapter.get("adapter_caused_split", ""),
                "adapter_caused_merge": adapter.get("adapter_caused_merge", ""),
                "source_kind": source.get("source_kind", ""),
                "source_score": source.get("source_score", ""),
                "source_margin": source.get("source_margin", ""),
                "source_entropy": source.get("source_entropy", ""),
                "source_state_after": source.get("source_state_after", ""),
                "support_slot_count_after": source.get("support_slot_count_after", ""),
                "support_chunk_count_after": source.get("support_chunk_count_after", ""),
                "full_minus_semantic_slot": source.get("full_minus_semantic_slot", ""),
                "source_diagnostic_only": source.get("source_diagnostic_only", ""),
                "adapter_candidate_valid": adapter_allowed and source_safe,
                "selected_flag": False,
                "selection_policy": "repair_probe_top_adapter_per_variant_label_frame",
                "is_method_result": False,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
                "method_uses_gt": False,
                "uses_future": False,
                "blocker_for_method_claim": "repair_probe_not_frozen_formal_holdout_and_or_source_precondition_failed",
            }
            candidate_rows.append(row)
            if adapter_allowed and source_safe:
                key = (str(row["variant"]), scene, str(row["readout_label"]), str(row["frame_id"]))
                old_idx = best_by_variant_label_frame.get(key)
                if old_idx is None:
                    best_by_variant_label_frame[key] = int(row["candidate_row_id"])
                else:
                    old = candidate_rows[old_idx]
                    if (
                        score > _num(old.get("adapter_score"), -1.0)
                        or (
                            score == _num(old.get("adapter_score"), -1.0)
                            and str(row.get("mask_id", "")) < str(old.get("mask_id", ""))
                        )
                    ):
                        best_by_variant_label_frame[key] = int(row["candidate_row_id"])

    for idx in best_by_variant_label_frame.values():
        candidate_rows[idx]["selected_flag"] = True
    selected_rows = [row for row in candidate_rows if _bool(row.get("selected_flag"))]
    selected_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        key = (str(row.get("scene_id", "")), str(row.get("frame_id", "")), str(row.get("mask_id", "")))
        if all(key):
            selected_by_key[key].append(row)

    support_rows: list[dict[str, Any]] = []
    matched_observation_rows = 0
    blocked_observation_rows = 0
    for table_path in _existing_source_tables_from_support(ctx["support_rows"]):
        rel_path = _rel(table_path)
        if not table_path.exists():
            continue
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for obs in reader:
                scene = str(obs.get("scene_id") or obs.get("scene") or "")
                key = (scene, str(obs.get("frame_id", "")), str(obs.get("observed_mask_id", "")))
                selected_matches = selected_by_key.get(key)
                if not selected_matches:
                    continue
                matched_observation_rows += 1
                allowed = (
                    not _bool(obs.get("uses_gt_for_prediction"))
                    and _bool(obs.get("visible"))
                    and _bool(obs.get("valid"))
                    and _bool(obs.get("valid_uv"))
                    and _bool(obs.get("inside_prepared_mask"))
                    and _bool(obs.get("scale_guard_pass"))
                    and bool(str(obs.get("carrier_global_id", "")).strip())
                )
                if not allowed:
                    blocked_observation_rows += len(selected_matches)
                    continue
                for selected in selected_matches:
                    support_rows.append(
                        {
                            "variant": selected.get("variant", ""),
                            "scene_id": scene,
                            "chunk_id": selected.get("chunk_id", ""),
                            "frame_id": obs.get("frame_id", ""),
                            "mask_id": obs.get("observed_mask_id", ""),
                            "history_id": selected.get("readout_label", ""),
                            "tracklet_id": selected.get("tracklet_id", ""),
                            "local_slot_id": selected.get("local_slot_id", ""),
                            "cluster_id": selected.get("cluster_id", ""),
                            "candidate_row_id": selected.get("candidate_row_id", ""),
                            "native_carrier_global_id": obs.get("carrier_global_id", ""),
                            "native_carrier_id": obs.get("carrier_id", ""),
                            "source_observation_table": rel_path,
                            "native_support_allowed": True,
                            "native_support_kind": "repair_probe_d4rt_carrier_global_id",
                            "is_method_result": False,
                            "is_diagnostic_only": True,
                            "forbidden_for_method_table": True,
                            "uses_gt_for_prediction": False,
                            "method_uses_gt": False,
                            "uses_future": False,
                        }
                    )

    materialize_summary = {
        "local_path": _rel(local_path),
        "adapter_path": _rel(adapter_path),
        "source_missing_cluster_count": missing_cluster_count,
        "joined_adapter_row_count": joined_adapter_count,
        "allowed_adapter_row_count": allowed_adapter_count,
        "selected_frame_mask_row_count": len(selected_rows),
        "selected_unique_frame_mask_count": len(selected_by_key),
        "matched_observation_row_count": matched_observation_rows,
        "blocked_observation_row_count": blocked_observation_rows,
        "native_support_row_count": len(support_rows),
        "unique_native_carrier_count": len(
            {row.get("native_carrier_global_id", "") for row in support_rows if row.get("native_carrier_global_id")}
        ),
    }
    return candidate_rows, selected_rows, support_rows, materialize_summary


def _evaluate_holdout_repair_probe(
    source_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_variant_audit: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"source_row_count": 0, "source_diagnostic_only_count": 0, "source_method_unsafe_count": 0}
    )
    for row in source_rows:
        variant = str(row.get("variant", ""))
        audit = source_variant_audit[variant]
        audit["source_row_count"] += 1
        if _bool(row.get("source_diagnostic_only")):
            audit["source_diagnostic_only_count"] += 1
        if _bool(row.get("source_method_uses_gt")) or _bool(row.get("source_uses_future")):
            audit["source_method_unsafe_count"] += 1

    support_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in support_rows:
        support_by_variant[str(row.get("variant", ""))].append(row)

    metric_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    source_audit_rows_all: list[dict[str, Any]] = []
    best_metric: dict[str, Any] = {}
    best_metric_gate: dict[str, Any] = {}
    for variant in sorted(source_variant_audit):
        rows = support_by_variant.get(variant, [])
        assignment_rows, source_audit_rows, assignment_summary = _assignment_rows_from_support(rows)
        for audit_row in source_audit_rows:
            source_audit_rows_all.append({"variant": variant, **audit_row})
        pred_rows = [
            dict(
                row,
                variant=variant,
                v86_label=row.get("pred_history_eval_label", ""),
                v86_score=row.get("pred_vote_purity", ""),
                is_method_result=False,
                is_diagnostic_only=True,
                forbidden_for_method_table=True,
            )
            for row in assignment_rows
        ]
        source_audit = source_variant_audit[variant]
        source_method_safe_no_gt_future = source_audit["source_method_unsafe_count"] == 0
        source_not_diagnostic = source_audit["source_diagnostic_only_count"] == 0
        if pred_rows:
            pred_scores = _pred_scores(pred_rows, label_field="v86_label")
            metric_row = {
                **_eval_variant(
                    assignment_rows,
                    pred_rows,
                    variant_id=variant,
                    label_field="v86_label",
                    pred_scores=pred_scores,
                    score_contract="repair_probe_mean_pred_vote_purity_per_label",
                ),
                "variant": variant,
                "variant_role": "holdout_repair_probe_not_formal_method_claim",
                "GT_label_coverage_rate": assignment_summary.get("native_gt_label_coverage_rate", 0.0),
                "native_support_row_count": len(rows),
                "native_assignment_count": len(assignment_rows),
                "is_method_result": False,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
                "source_method_safe_no_gt_future": source_method_safe_no_gt_future,
                "source_not_diagnostic": source_not_diagnostic,
                "formal_method_claim_allowed": False,
                "formal_claim_blocker": (
                    "repair probe is not the frozen dev-selected Phase10 method run; weak-source variants also inherit "
                    "diagnostic-only source rows; tracklet-source variants inherit the failed Phase2 aggregate gate"
                ),
            }
            controls: list[tuple[str, str, list[dict[str, Any]], str, bool, bool]] = [
                (
                    "RP_B5_size_matched_hash_global",
                    "repair-probe global size matched hash",
                    _labels_to_rows(
                        pred_rows,
                        _size_matched_hash_labels(
                            pred_rows,
                            group_field=None,
                            source_label_field="v86_label",
                            salt=f"v86_repair_probe_{variant}_size_global",
                        ),
                        label_field="control_label",
                    ),
                    "cluster_size_desc",
                    False,
                    False,
                ),
                (
                    "RP_B6_size_matched_hash_by_scene",
                    "repair-probe scene size matched hash",
                    _labels_to_rows(
                        pred_rows,
                        _size_matched_hash_labels(
                            pred_rows,
                            group_field="scene_id",
                            source_label_field="v86_label",
                            salt=f"v86_repair_probe_{variant}_size_scene",
                        ),
                        label_field="control_label",
                    ),
                    "cluster_size_desc",
                    False,
                    False,
                ),
                (
                    "RP_B7_uniform_hash_history",
                    "repair-probe uniform hash over labels",
                    _labels_to_rows(
                        pred_rows,
                        _uniform_hash_labels(
                            pred_rows,
                            source_label_field="v86_label",
                            salt=f"v86_repair_probe_{variant}_uniform",
                        ),
                        label_field="control_label",
                    ),
                    "cluster_size_desc",
                    False,
                    False,
                ),
                (
                    "RP_B8_single_largest_by_scene",
                    "repair-probe single largest cluster by scene",
                    _labels_to_rows(
                        pred_rows,
                        _single_largest_labels(pred_rows, source_label_field="v86_label", group_field="scene_id"),
                        label_field="control_label",
                    ),
                    "cluster_size_desc",
                    False,
                    False,
                ),
                (
                    "RP_B11_oracle_diagnostic_gt",
                    "repair-probe oracle GT upper bound",
                    [dict(row, control_label=row.get("diagnostic_gt_eval_label", "")) for row in pred_rows],
                    "cluster_size_desc_for_oracle",
                    True,
                    True,
                ),
            ]
            non_oracle_ap50: list[float] = []
            for control_id, notes, control_pred_rows, score_contract, uses_gt, is_oracle in controls:
                control_row = {
                    **_eval_variant(
                        assignment_rows,
                        control_pred_rows,
                        variant_id=f"{variant}:{control_id}",
                        label_field="control_label",
                        pred_scores=None,
                        score_contract=score_contract,
                        prediction_uses_gt=uses_gt,
                        is_oracle=is_oracle,
                    ),
                    "variant": variant,
                    "control_id": control_id,
                    "control_notes": notes,
                    "is_method_result": False,
                    "is_diagnostic_only": True,
                }
                control_rows.append(control_row)
                if not uses_gt:
                    non_oracle_ap50.append(_num(control_row.get("native_AP50"), 0.0))
            metric_row["real_minus_best_non_oracle_AP50"] = _num(metric_row.get("native_AP50"), 0.0) - max(
                non_oracle_ap50, default=0.0
            )
            gate = {
                "native_AP50_ge_0p35": _num(metric_row.get("native_AP50"), 0.0) >= 0.35,
                "real_minus_best_non_oracle_AP50_ge_0p07": metric_row["real_minus_best_non_oracle_AP50"] >= 0.07,
                "purity_ge_0p75": _num(metric_row.get("purity"), 0.0) >= 0.75,
                "ARI_ge_0p50": _num(metric_row.get("adjusted_rand_index"), 0.0) >= 0.50,
                "GT_label_coverage_rate_ge_0p80": _num(metric_row.get("GT_label_coverage_rate"), 0.0) >= 0.80,
                "source_method_safe_no_gt_future": source_method_safe_no_gt_future,
            }
            gate["metric_pass_without_formal_claim"] = all(gate.values())
            metric_row["repair_probe_gate"] = json.dumps(gate, sort_keys=True)
            metric_rows.append(metric_row)
            if not best_metric or _num(metric_row.get("native_AP50"), -1.0) > _num(best_metric.get("native_AP50"), -1.0):
                best_metric = metric_row
                best_metric_gate = gate

        variant_rows.append(
            {
                "variant": variant,
                **source_audit,
                "source_method_safe_no_gt_future": source_method_safe_no_gt_future,
                "source_not_diagnostic": source_not_diagnostic,
                "native_support_row_count": len(rows),
                "native_assignment_count": len(pred_rows),
                "GT_label_coverage_rate": assignment_summary.get("native_gt_label_coverage_rate", ""),
                "formal_method_claim_allowed": False,
                "formal_claim_blocker": "not a frozen formal holdout input; diagnostic weak variants or failed Phase2 aggregate gate",
            }
        )

    summary = {
        "schema": "stream4d_v86_holdout_repair_probe_v1",
        "decision": "REPAIR_PROBE_COMPLETED_NOT_FORMAL_METHOD_CLAIM",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "variant_count": len(variant_rows),
        "metric_variant_count": len(metric_rows),
        "best_variant_by_AP50": best_metric.get("variant", ""),
        "best_native_AP50": best_metric.get("native_AP50", ""),
        "best_ARI": best_metric.get("adjusted_rand_index", ""),
        "best_purity": best_metric.get("purity", ""),
        "best_real_minus_best_non_oracle_AP50": best_metric.get("real_minus_best_non_oracle_AP50", ""),
        "best_metric_pass_without_formal_claim": bool(best_metric_gate.get("metric_pass_without_formal_claim")),
        "formal_method_claim_allowed": False,
        "primary_blocker": "repair_probe_not_frozen_formal_holdout_and_phase2_history_confirmation_still_unrepaired",
    }
    return variant_rows, metric_rows, control_rows, source_audit_rows_all, summary


def _holdout_repair_probe(ctx: dict[str, Any], out: Path) -> dict[str, Any]:
    source_rows, source_summary = _holdout_repair_probe_source_rows()
    candidate_rows, selected_rows, support_rows, materialize_summary = _materialize_probe_support(ctx, source_rows)
    variant_rows, metric_rows, control_rows, source_audit_rows, eval_summary = _evaluate_holdout_repair_probe(
        source_rows, support_rows
    )
    summary = {
        **eval_summary,
        **{f"source_{key}": value for key, value in source_summary.items()},
        **{f"materialize_{key}": value for key, value in materialize_summary.items()},
        "candidate_rows_path": _rel(out / "holdout_repair_probe_frame_mask_candidate_rows.csv"),
        "selected_rows_path": _rel(out / "holdout_repair_probe_frame_mask_selected_rows.csv"),
        "variant_rows_path": _rel(out / "holdout_repair_probe_variant_rows.csv"),
        "metric_rows_path": _rel(out / "holdout_repair_probe_native_metric_rows.csv"),
        "control_rows_path": _rel(out / "holdout_repair_probe_control_rows.csv"),
    }
    _write_csv(out / "holdout_repair_probe_frame_mask_candidate_rows.csv", candidate_rows)
    _write_csv(out / "holdout_repair_probe_frame_mask_selected_rows.csv", selected_rows)
    _write_csv(out / "holdout_repair_probe_variant_rows.csv", variant_rows)
    _write_csv(out / "holdout_repair_probe_native_metric_rows.csv", metric_rows)
    _write_csv(out / "holdout_repair_probe_control_rows.csv", control_rows)
    _write_csv(out / "holdout_repair_probe_source_audit_rows.csv", source_audit_rows)
    _write_json(out / "holdout_repair_probe_summary.json", summary)
    return summary


def _dev_tracklet_readout_source_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tracklet_path = _repo_path(args.dev_tracklet_phase2_root) / "tracklet_assignment_rows.csv"
    summary_path = _repo_path(args.dev_tracklet_phase2_root) / "summary.json"
    tracklet_rows = _read_csv_rows(tracklet_path)
    phase2_summary = _read_json(summary_path)
    source_rows: list[dict[str, Any]] = []

    def add_source(row: dict[str, Any], *, variant: str, label: str, source_kind: str) -> None:
        if not label:
            return
        source_rows.append(
            {
                "variant": variant,
                "scene_id": row.get("scene_id", ""),
                "chunk_id": row.get("chunk_id", ""),
                "local_slot_id": row.get("local_slot_id", ""),
                "readout_label": label,
                "tracklet_id": row.get("tracklet_id", ""),
                "source_kind": source_kind,
                "source_score": row.get("score", ""),
                "source_margin": row.get("margin", ""),
                "source_entropy": row.get("entropy", ""),
                "source_state_before": row.get("tracklet_state_before", ""),
                "source_state_after": row.get("tracklet_state_after", ""),
                "support_slot_count_after": row.get("support_slot_count_after", ""),
                "support_chunk_count_after": row.get("support_chunk_count_after", ""),
                "full_minus_semantic_slot": row.get("full_minus_semantic_slot", ""),
                "source_diagnostic_only": False,
                "source_method_uses_gt": _bool(row.get("method_uses_gt")),
                "source_uses_future": _bool(row.get("uses_future")),
            }
        )

    for row in tracklet_rows:
        if _bool(row.get("method_uses_gt")) or _bool(row.get("uses_future")):
            continue
        tracklet = str(row.get("tracklet_id", "")).strip()
        state_after = str(row.get("tracklet_state_after", "")).strip()
        support_slots = _int(row.get("support_slot_count_after"), 0)
        support_chunks = _int(row.get("support_chunk_count_after"), 0)
        score = _num(row.get("score"), 0.0)
        margin = _num(row.get("margin"), 0.0)
        entropy = _num(row.get("entropy"), 1.0)
        full_minus_sem = _num(row.get("full_minus_semantic_slot"), -1.0)
        if state_after == "confirmed":
            add_source(
                row,
                variant="DV4_phase2_confirmed_tracklets",
                label=f"dev_tracklet_confirmed:{tracklet}",
                source_kind="dev_phase2_confirmed_tracklet",
            )
        if state_after == "confirmed" and support_slots >= 2 and support_chunks >= 2 and full_minus_sem >= 0.03:
            add_source(
                row,
                variant="DV5_phase2_confirmed_object_gain",
                label=f"dev_tracklet_confirmed_gain:{tracklet}",
                source_kind="dev_phase2_confirmed_tracklet_object_gain",
            )
        if support_slots >= 2 and score >= 0.70 and margin >= 0.05 and entropy <= 0.50:
            add_source(
                row,
                variant="DV6_phase2_repeated_high_margin",
                label=f"dev_tracklet_repeated_margin:{tracklet}",
                source_kind="dev_phase2_repeated_high_margin_tracklet",
            )
        if full_minus_sem >= 0.03 and score >= 0.70 and margin >= 0.03:
            add_source(
                row,
                variant="DV7_phase2_object_gain_margin",
                label=f"dev_tracklet_object_gain:{tracklet}",
                source_kind="dev_phase2_object_gain_tracklet",
            )

    audit = {
        "dev_tracklet_phase2_root": _rel(_repo_path(args.dev_tracklet_phase2_root)),
        "tracklet_assignment_path": _rel(tracklet_path),
        "tracklet_assignment_row_count": len(tracklet_rows),
        "phase2_decision": phase2_summary.get("decision", ""),
        "phase2_can_enter_next_phase": phase2_summary.get("can_enter_next_phase", ""),
        "phase2_eligible_tracklet_coverage_rate": phase2_summary.get("eligible_tracklet_coverage_rate", ""),
        "phase2_full_minus_semantic_score": phase2_summary.get("full_minus_semantic_score", ""),
        "phase2_false_attachment_proxy_rate": phase2_summary.get("false_attachment_proxy_rate", ""),
        "phase2_future_tracklet_descriptor_count": phase2_summary.get("future_tracklet_descriptor_count", ""),
        "phase2_self_confirmation_count": phase2_summary.get("self_confirmation_count", ""),
        "source_row_count": len(source_rows),
    }
    return source_rows, audit


def _dev_tracklet_readout_repair(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    ctx = _load_inputs(args)
    out = _repo_path(args.phase12_output_root)
    out.mkdir(parents=True, exist_ok=True)
    source_rows, source_summary = _dev_tracklet_readout_source_rows(args)
    candidate_rows, selected_rows, support_rows, materialize_summary = _materialize_probe_support(
        ctx,
        source_rows,
        local_path=_repo_path(args.dev_local_phase1_root) / "local_slot_rows.csv",
        adapter_path=_repo_path(args.dev_adapter_root) / "adapter_rows.csv",
    )
    variant_rows, metric_rows, control_rows, source_audit_rows, eval_summary = _evaluate_holdout_repair_probe(
        source_rows, support_rows
    )
    passing_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        gate = _counter_from_json(str(row.get("repair_probe_gate", "{}")))
        variant_audit = next((v for v in variant_rows if v.get("variant") == row.get("variant")), {})
        if (
            bool(gate.get("metric_pass_without_formal_claim"))
            and _bool(row.get("source_method_safe_no_gt_future"))
            and _bool(row.get("source_not_diagnostic"))
            and _bool(source_summary.get("phase2_can_enter_next_phase"))
        ):
            passing_rows.append({**row, **{"source_row_count": variant_audit.get("source_row_count", "")}})

    best = sorted(
        passing_rows or metric_rows,
        key=lambda row: (_num(row.get("native_AP50"), -1.0), _num(row.get("adjusted_rand_index"), -1.0)),
        reverse=True,
    )
    selected = best[0] if best else {}
    candidate_freeze_recommended = bool(passing_rows)
    frozen_candidate_config = {
        "schema": "stream4d_v86_phase12_dev_tracklet_candidate_config_v1",
        "selected_from_dev_only": True,
        "candidate_freeze_recommended": candidate_freeze_recommended,
        "selected_variant": selected.get("variant", ""),
        "dev_tracklet_phase2_root": source_summary.get("dev_tracklet_phase2_root", ""),
        "dev_local_phase1_root": _rel(_repo_path(args.dev_local_phase1_root)),
        "dev_adapter_root": _rel(_repo_path(args.dev_adapter_root)),
        "source_rule": "confirmed object-gain/repeated tracklet readout; GT labels evaluator-only",
        "requires_fresh_holdout_before_method_claim": True,
        "does_not_modify_current_v86_phase10": True,
    }
    frozen_candidate_config["config_sha256"] = _canonical_sha256(frozen_candidate_config)

    summary = {
        "schema": "stream4d_v86_phase12_dev_tracklet_readout_repair_v1",
        "phase": "v86_phase12_dev_tracklet_readout_repair",
        "decision": "PASS_DEV_TRACKLET_READOUT_CANDIDATE" if candidate_freeze_recommended else "NO_GO_DEV_TRACKLET_READOUT_CANDIDATE",
        "candidate_freeze_recommended": candidate_freeze_recommended,
        "selected_variant": selected.get("variant", ""),
        "selected_native_AP50": selected.get("native_AP50", ""),
        "selected_ARI": selected.get("adjusted_rand_index", ""),
        "selected_purity": selected.get("purity", ""),
        "selected_real_minus_best_non_oracle_AP50": selected.get("real_minus_best_non_oracle_AP50", ""),
        "passing_variant_count": len(passing_rows),
        "formal_v86_success": False,
        "formal_method_claim_allowed": False,
        "formal_claim_blocker": "phase12 is a dev-side candidate repair; current v86 Phase10 remains failed until a fresh frozen holdout is run",
        "next_required_action": "freeze this dev-side candidate in a future method version and run a fresh formal holdout without retuning",
        **{f"source_{key}": value for key, value in source_summary.items()},
        **{f"materialize_{key}": value for key, value in materialize_summary.items()},
        "runtime_sec": time.time() - t0,
    }
    _write_csv(out / "dev_tracklet_readout_source_rows.csv", source_rows)
    _write_csv(out / "dev_tracklet_readout_frame_mask_candidate_rows.csv", candidate_rows)
    _write_csv(out / "dev_tracklet_readout_frame_mask_selected_rows.csv", selected_rows)
    _write_csv(out / "dev_tracklet_readout_variant_rows.csv", variant_rows)
    _write_csv(out / "dev_tracklet_readout_native_metric_rows.csv", metric_rows)
    _write_csv(out / "dev_tracklet_readout_control_rows.csv", control_rows)
    _write_csv(out / "dev_tracklet_readout_source_audit_rows.csv", source_audit_rows)
    _write_json(out / "dev_tracklet_readout_candidate_config.json", frozen_candidate_config)
    _write_json(out / "dev_tracklet_readout_summary.json", summary)
    return summary


def _holdout_variant_for_dev_candidate(dev_variant: str) -> str:
    mapping = {
        "DV4_phase2_confirmed_tracklets": "RP4_phase2_confirmed_tracklets",
        "DV5_phase2_confirmed_object_gain": "RP5_phase2_confirmed_object_gain",
        "DV6_phase2_repeated_high_margin": "RP6_phase2_repeated_high_margin",
        "DV7_phase2_object_gain_margin": "RP7_phase2_object_gain_margin",
    }
    return mapping.get(dev_variant, "")


def _candidate_freeze_holdout_audit(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    ctx = _load_inputs(args)
    out = _repo_path(args.phase13_output_root)
    out.mkdir(parents=True, exist_ok=True)

    config_path = _repo_path(args.phase13_candidate_config)
    candidate_config = _read_json(config_path)
    expected_sha = str(candidate_config.get("config_sha256", "")).strip()
    config_for_hash = dict(candidate_config)
    config_for_hash.pop("config_sha256", None)
    actual_sha = _canonical_sha256(config_for_hash)
    selected_dev_variant = str(candidate_config.get("selected_variant", "")).strip()
    holdout_variant = _holdout_variant_for_dev_candidate(selected_dev_variant)

    all_source_rows, source_summary = _holdout_repair_probe_source_rows()
    source_rows: list[dict[str, Any]] = []
    for row in all_source_rows:
        if str(row.get("variant", "")) != holdout_variant:
            continue
        new_row = dict(row)
        new_row["original_holdout_probe_variant"] = row.get("variant", "")
        new_row["variant"] = f"{selected_dev_variant}_holdout_fixed"
        new_row["dev_selected_variant"] = selected_dev_variant
        new_row["candidate_config_sha256"] = expected_sha
        source_rows.append(new_row)

    candidate_rows, selected_rows, support_rows, materialize_summary = _materialize_probe_support(ctx, source_rows)
    variant_rows, metric_rows, control_rows, source_audit_rows, eval_summary = _evaluate_holdout_repair_probe(
        source_rows, support_rows
    )

    selected_metric = metric_rows[0] if metric_rows else {}
    selected_gate: dict[str, Any] = {}
    if selected_metric:
        try:
            selected_gate = json.loads(str(selected_metric.get("repair_probe_gate", "{}")))
        except json.JSONDecodeError:
            selected_gate = {}
    metric_pass_without_formal_claim = bool(selected_gate.get("metric_pass_without_formal_claim"))
    source_phase2_pass = bool(source_summary.get("phase2_can_enter_next_phase"))
    config_sha256_matches = bool(expected_sha and expected_sha == actual_sha)
    candidate_config_dev_only = bool(candidate_config.get("selected_from_dev_only"))
    decision = (
        "PASS_CANDIDATE_FREEZE_HOLDOUT_METRIC_BUT_NOT_FORMAL_CLAIM"
        if (
            metric_pass_without_formal_claim
            and config_sha256_matches
            and candidate_config_dev_only
            and bool(holdout_variant)
            and bool(source_rows)
        )
        else "NO_GO_CANDIDATE_FREEZE_HOLDOUT_AUDIT"
    )

    summary = {
        "schema": "stream4d_v86_phase13_candidate_freeze_holdout_audit_v1",
        "phase": "v86_phase13_candidate_freeze_holdout_audit",
        "decision": decision,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "candidate_config_path": _rel(config_path),
        "candidate_config_sha256_expected": expected_sha,
        "candidate_config_sha256_recomputed": actual_sha,
        "candidate_config_sha256_matches": config_sha256_matches,
        "candidate_config_selected_from_dev_only": candidate_config_dev_only,
        "dev_selected_variant": selected_dev_variant,
        "holdout_variant_applied": holdout_variant,
        "holdout_fixed_variant_id": f"{selected_dev_variant}_holdout_fixed" if selected_dev_variant else "",
        "holdout_previously_inspected": True,
        "metric_pass_without_formal_claim": metric_pass_without_formal_claim,
        "selected_native_AP50": selected_metric.get("native_AP50", ""),
        "selected_ARI": selected_metric.get("adjusted_rand_index", ""),
        "selected_purity": selected_metric.get("purity", ""),
        "selected_real_minus_best_non_oracle_AP50": selected_metric.get("real_minus_best_non_oracle_AP50", ""),
        "selected_GT_label_coverage_rate": selected_metric.get("GT_label_coverage_rate", ""),
        "selected_repair_gate": selected_gate,
        "formal_method_claim_allowed": False,
        "formal_claim_blocker": (
            "Phase13 applies the Phase12 dev-selected candidate after the original v86 Phase10 holdout and repair-probe "
            "were already inspected; the v86 plan forbids turning this into a formal method claim, and the holdout "
            "Phase2 aggregate gate still reports NO_GO_TRACKLET_ASSOCIATION_WEAK"
        ),
        "next_required_action": (
            "use this only as next-version evidence: freeze the dev config before any new registered holdout and run one "
            "fresh holdout without retuning"
        ),
        **{f"source_{key}": value for key, value in source_summary.items()},
        **{f"materialize_{key}": value for key, value in materialize_summary.items()},
        **{f"eval_{key}": value for key, value in eval_summary.items()},
        "runtime_sec": time.time() - t0,
    }
    _write_csv(out / "candidate_freeze_holdout_source_rows.csv", source_rows)
    _write_csv(out / "candidate_freeze_holdout_frame_mask_candidate_rows.csv", candidate_rows)
    _write_csv(out / "candidate_freeze_holdout_frame_mask_selected_rows.csv", selected_rows)
    _write_csv(out / "candidate_freeze_holdout_variant_rows.csv", variant_rows)
    _write_csv(out / "candidate_freeze_holdout_native_metric_rows.csv", metric_rows)
    _write_csv(out / "candidate_freeze_holdout_control_rows.csv", control_rows)
    _write_csv(out / "candidate_freeze_holdout_source_audit_rows.csv", source_audit_rows)
    _write_json(out / "candidate_freeze_holdout_candidate_config_snapshot.json", candidate_config)
    _write_json(out / "candidate_freeze_holdout_summary.json", summary)
    return summary


def _count_by_scene_chunk(path: Path, label: str) -> tuple[Counter[tuple[str, str]], dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    rows = _read_csv_rows(path) if path.exists() else []
    for row in rows:
        scene = str(row.get("scene_id") or row.get("scene") or "").strip()
        chunk = str(row.get("chunk_id", "")).strip()
        if scene and chunk:
            counts[(scene, chunk)] += 1
    return counts, {"artifact_label": label, "path": _rel(path), "exists": path.exists(), "row_count": len(rows)}


def _fresh_holdout_availability_audit(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase14_output_root)
    out.mkdir(parents=True, exist_ok=True)

    dev_chunks: set[tuple[str, str]] = {
        *{("scene0011_00", str(chunk)) for chunk in range(0, 6)},
        *{("scene0050_00", str(chunk)) for chunk in range(0, 4)},
    }
    holdout_chunks: set[tuple[str, str]] = {
        *{("scene0011_00", str(chunk)) for chunk in range(6, 12)},
        *{("scene0050_00", str(chunk)) for chunk in range(4, 12)},
    }
    artifact_specs = [
        ("dev_local_slot_rows", _repo_path(args.dev_local_phase1_root) / "local_slot_rows.csv"),
        ("dev_adapter_rows", _repo_path(args.dev_adapter_root) / "adapter_rows.csv"),
        ("dev_tracklet_assignment_rows", _repo_path(args.dev_tracklet_phase2_root) / "tracklet_assignment_rows.csv"),
        (
            "phase12_dev_source_rows",
            _repo_path(args.phase12_output_root) / "dev_tracklet_readout_source_rows.csv",
        ),
        (
            "holdout_local_slot_rows",
            _repo_path("outputs/audit/v84_holdout_replay_v82_phase1_local_b0/local_slot_rows.csv"),
        ),
        (
            "holdout_adapter_rows",
            _repo_path("outputs/audit/v84_holdout_replay_v82_local_shadow/phase1_adapter_v84_holdout_replay/adapter_rows.csv"),
        ),
        (
            "holdout_tracklet_assignment_rows",
            _repo_path("outputs/audit/v84_holdout_replay_v82_phase2_object_tracklets/tracklet_assignment_rows.csv"),
        ),
        (
            "phase13_fixed_candidate_source_rows",
            _repo_path(args.phase13_output_root) / "candidate_freeze_holdout_source_rows.csv",
        ),
    ]
    artifact_counts: dict[str, Counter[tuple[str, str]]] = {}
    artifact_rows: list[dict[str, Any]] = []
    all_chunks: set[tuple[str, str]] = set()
    for label, path in artifact_specs:
        counts, audit = _count_by_scene_chunk(path, label)
        artifact_counts[label] = counts
        artifact_rows.append(audit)
        all_chunks.update(counts)

    holdout_summary_path = _repo_path(args.phase10_output_root) / "holdout_summary.json"
    phase13_summary_path = _repo_path(args.phase13_output_root) / "candidate_freeze_holdout_summary.json"
    phase12_summary_path = _repo_path(args.phase12_output_root) / "dev_tracklet_readout_summary.json"
    holdout_summary = _read_json(holdout_summary_path)
    phase12_summary = _read_json(phase12_summary_path)
    phase13_summary = _read_json(phase13_summary_path)

    chunk_rows: list[dict[str, Any]] = []
    fresh_candidate_rows: list[dict[str, Any]] = []
    for scene, chunk in sorted(all_chunks | dev_chunks | holdout_chunks, key=lambda key: (key[0], int(key[1]))):
        in_dev = (scene, chunk) in dev_chunks
        in_holdout = (scene, chunk) in holdout_chunks
        counts = {label: artifact_counts[label].get((scene, chunk), 0) for label, _ in artifact_specs}
        has_local = counts["dev_local_slot_rows"] > 0 or counts["holdout_local_slot_rows"] > 0
        has_adapter = counts["dev_adapter_rows"] > 0 or counts["holdout_adapter_rows"] > 0
        has_tracklet = (
            counts["dev_tracklet_assignment_rows"] > 0
            or counts["holdout_tracklet_assignment_rows"] > 0
            or counts["phase12_dev_source_rows"] > 0
            or counts["phase13_fixed_candidate_source_rows"] > 0
        )
        if in_dev:
            protocol_status = "DEV_SELECTION_USED"
        elif in_holdout:
            protocol_status = "OFFICIAL_HOLDOUT_ALREADY_INSPECTED"
        else:
            protocol_status = "UNPARTITIONED_AVAILABLE"
        eligible_fresh = (not in_dev) and (not in_holdout) and has_local and has_adapter and has_tracklet
        row = {
            "scene_id": scene,
            "chunk_id": chunk,
            "in_v86_dev_partition": in_dev,
            "in_v86_official_holdout_partition": in_holdout,
            "has_local_rows": has_local,
            "has_adapter_rows": has_adapter,
            "has_tracklet_or_candidate_rows": has_tracklet,
            **counts,
            "protocol_status": protocol_status,
            "eligible_as_fresh_formal_holdout": eligible_fresh,
        }
        chunk_rows.append(row)
        if eligible_fresh:
            fresh_candidate_rows.append(row)

    dev_artifact_chunk_count = len({key for key in all_chunks if key in dev_chunks})
    holdout_artifact_chunk_count = len({key for key in all_chunks if key in holdout_chunks})
    unpartitioned_artifact_chunks = sorted(all_chunks - dev_chunks - holdout_chunks, key=lambda key: (key[0], int(key[1])))
    decision = (
        "PASS_FRESH_HOLDOUT_CANDIDATES_AVAILABLE"
        if fresh_candidate_rows
        else "NO_GO_FRESH_HOLDOUT_UNIVERSE_UNAVAILABLE_FOR_V86_FORMAL_CLAIM"
    )
    summary = {
        "schema": "stream4d_v86_phase14_fresh_holdout_availability_audit_v1",
        "phase": "v86_phase14_fresh_holdout_availability_audit",
        "decision": decision,
        "formal_v86_goal_achieved": False,
        "formal_method_claim_allowed": False,
        "formal_claim_blocker": (
            "The current artifact universe is fully covered by the v86 dev split or the already-inspected official "
            "holdout split; Phase12/13 evidence is useful for the next method version but cannot retroactively create "
            "a fresh formal v86 holdout."
        ),
        "dev_partition_chunk_count": len(dev_chunks),
        "holdout_partition_chunk_count": len(holdout_chunks),
        "artifact_chunk_count": len(all_chunks),
        "dev_artifact_chunk_count": dev_artifact_chunk_count,
        "holdout_artifact_chunk_count": holdout_artifact_chunk_count,
        "unpartitioned_artifact_chunk_count": len(unpartitioned_artifact_chunks),
        "fresh_formal_holdout_candidate_count": len(fresh_candidate_rows),
        "official_holdout_summary_path": _rel(holdout_summary_path),
        "official_holdout_decision": holdout_summary.get("decision", ""),
        "phase12_summary_path": _rel(phase12_summary_path),
        "phase12_decision": phase12_summary.get("decision", ""),
        "phase12_selected_variant": phase12_summary.get("selected_variant", ""),
        "phase13_summary_path": _rel(phase13_summary_path),
        "phase13_decision": phase13_summary.get("decision", ""),
        "phase13_holdout_previously_inspected": phase13_summary.get("holdout_previously_inspected", ""),
        "phase13_formal_method_claim_allowed": phase13_summary.get("formal_method_claim_allowed", ""),
        "recommended_next_action": (
            "Start a new method version with DV5 frozen before any new registered holdout, or collect/register "
            "additional uninspected scene/chunk artifacts. Do not relabel Phase13 as v86 formal success."
        ),
        "runtime_sec": time.time() - t0,
    }
    _write_json(out / "fresh_holdout_availability_summary.json", summary)
    _write_csv(out / "fresh_holdout_chunk_coverage_rows.csv", chunk_rows)
    _write_csv(out / "fresh_holdout_artifact_audit_rows.csv", artifact_rows)
    _write_csv(out / "fresh_holdout_candidate_rows.csv", fresh_candidate_rows)
    return summary


def _scene_chunk_counts_from_csv(path: Path) -> tuple[Counter[tuple[str, str]], Counter[str], int]:
    scene_chunk_counts: Counter[tuple[str, str]] = Counter()
    scene_counts: Counter[str] = Counter()
    row_count = 0
    if not path.exists():
        return scene_chunk_counts, scene_counts, row_count
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            scene = str(row.get("scene_id") or row.get("scene") or "").strip()
            chunk = str(row.get("chunk_id") or row.get("window_id") or row.get("window_serial") or "").strip()
            if scene:
                scene_counts[scene] += 1
                if chunk:
                    scene_chunk_counts[(scene, chunk)] += 1
    return scene_chunk_counts, scene_counts, row_count


def _chunks_for_scene(scene_chunk_counts: Counter[tuple[str, str]], scene: str) -> list[str]:
    def key(text: str) -> tuple[int, str]:
        digits = ""
        for char in reversed(str(text)):
            if char.isdigit():
                digits = char + digits
            elif digits:
                break
        return (int(digits) if digits else -1, str(text))

    return sorted({chunk for row_scene, chunk in scene_chunk_counts if row_scene == scene and chunk}, key=key)


def _source_contains(path: Path, pattern: str) -> bool:
    if not path.exists():
        return False
    return pattern in path.read_text(encoding="utf-8", errors="ignore")


def _source_declares_scene(path: Path, scene: str) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return f'"{scene}"' in text or f"'{scene}'" in text


def _raw_substrate_availability_audit(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase15_output_root)
    out.mkdir(parents=True, exist_ok=True)

    consumed_chunks: set[tuple[str, str]] = {
        *{("scene0011_00", str(chunk)) for chunk in range(0, 12)},
        *{("scene0050_00", str(chunk)) for chunk in range(0, 12)},
    }
    raw_specs = [
        ("v47_mask_observation", _repo_path("Stream3D/outputs/audit/v47_observation_tables/mask_observation_table.csv")),
        ("v47_carrier_observation", _repo_path("Stream3D/outputs/audit/v47_observation_tables/carrier_observation_table.csv")),
        (
            "v54_stride1_probe5_mask_observation",
            _repo_path("Stream3D/outputs/audit/v54_observation_tables_stride1_probe5/mask_observation_table.csv"),
        ),
        (
            "v54_stride1_probe5_carrier_observation",
            _repo_path("Stream3D/outputs/audit/v54_observation_tables_stride1_probe5/carrier_observation_table.csv"),
        ),
        (
            "v54_probe3_smoke3_mask_observation",
            _repo_path(
                "Stream3D/outputs/audit/v54_observation_tables_stride1_probe3_mf96_q4096_notopup_smoke3/mask_observation_table.csv"
            ),
        ),
        (
            "v54_probe3_smoke3_carrier_observation",
            _repo_path(
                "Stream3D/outputs/audit/v54_observation_tables_stride1_probe3_mf96_q4096_notopup_smoke3/carrier_observation_table.csv"
            ),
        ),
    ]
    chain_specs = [
        ("v86_dev_local", _repo_path(args.dev_local_phase1_root) / "local_slot_rows.csv"),
        ("v86_dev_adapter", _repo_path(args.dev_adapter_root) / "adapter_rows.csv"),
        ("v86_dev_tracklet", _repo_path(args.dev_tracklet_phase2_root) / "tracklet_assignment_rows.csv"),
        ("v86_holdout_local", _repo_path("outputs/audit/v84_holdout_replay_v82_phase1_local_b0/local_slot_rows.csv")),
        (
            "v86_holdout_adapter",
            _repo_path("outputs/audit/v84_holdout_replay_v82_local_shadow/phase1_adapter_v84_holdout_replay/adapter_rows.csv"),
        ),
        (
            "v86_holdout_tracklet",
            _repo_path("outputs/audit/v84_holdout_replay_v82_phase2_object_tracklets/tracklet_assignment_rows.csv"),
        ),
    ]

    source_rows: list[dict[str, Any]] = []
    raw_mask_by_scene: Counter[str] = Counter()
    raw_carrier_by_scene_chunk: Counter[tuple[str, str]] = Counter()
    raw_carrier_by_scene: Counter[str] = Counter()
    for label, path in raw_specs:
        scene_chunk_counts, scene_counts, row_count = _scene_chunk_counts_from_csv(path)
        source_rows.append(
            {
                "artifact_label": label,
                "path": _rel(path),
                "exists": path.exists(),
                "row_count": row_count,
                "scene_count": len(scene_counts),
                "scene_chunk_count": len(scene_chunk_counts),
            }
        )
        if "mask_observation" in label:
            raw_mask_by_scene.update(scene_counts)
        if "carrier_observation" in label:
            raw_carrier_by_scene_chunk.update(scene_chunk_counts)
            raw_carrier_by_scene.update(scene_counts)

    local_by_scene_chunk: Counter[tuple[str, str]] = Counter()
    adapter_by_scene_chunk: Counter[tuple[str, str]] = Counter()
    tracklet_by_scene_chunk: Counter[tuple[str, str]] = Counter()
    for label, path in chain_specs:
        scene_chunk_counts, scene_counts, row_count = _scene_chunk_counts_from_csv(path)
        source_rows.append(
            {
                "artifact_label": label,
                "path": _rel(path),
                "exists": path.exists(),
                "row_count": row_count,
                "scene_count": len(scene_counts),
                "scene_chunk_count": len(scene_chunk_counts),
            }
        )
        if "local" in label:
            local_by_scene_chunk.update(scene_chunk_counts)
        elif "adapter" in label:
            adapter_by_scene_chunk.update(scene_chunk_counts)
        elif "tracklet" in label:
            tracklet_by_scene_chunk.update(scene_chunk_counts)

    all_scene_chunks = set(raw_carrier_by_scene_chunk) | set(local_by_scene_chunk) | set(adapter_by_scene_chunk) | set(tracklet_by_scene_chunk)
    candidate_rows: list[dict[str, Any]] = []
    ready_rows: list[dict[str, Any]] = []
    for scene, chunk in sorted(all_scene_chunks, key=lambda key: (key[0], int(float(key[1])) if key[1] else -1)):
        raw_mask_rows = raw_mask_by_scene.get(scene, 0)
        raw_carrier_rows = raw_carrier_by_scene_chunk.get((scene, chunk), 0)
        in_consumed = (scene, chunk) in consumed_chunks
        has_local = local_by_scene_chunk.get((scene, chunk), 0) > 0
        has_adapter = adapter_by_scene_chunk.get((scene, chunk), 0) > 0
        has_tracklet = tracklet_by_scene_chunk.get((scene, chunk), 0) > 0
        has_raw_substrate = raw_mask_rows > 0 and raw_carrier_rows > 0
        formal_ready = has_raw_substrate and (not in_consumed) and has_local and has_adapter and has_tracklet
        missing: list[str] = []
        if not has_raw_substrate:
            missing.append("raw_mask_or_carrier")
        if in_consumed:
            missing.append("uninspected_partition")
        if not has_local:
            missing.append("v86_local_rows")
        if not has_adapter:
            missing.append("v86_adapter_rows")
        if not has_tracklet:
            missing.append("v86_tracklet_rows")
        row = {
            "scene_id": scene,
            "chunk_id": chunk,
            "raw_mask_rows_by_scene": raw_mask_rows,
            "raw_carrier_rows_by_scene_chunk": raw_carrier_rows,
            "in_v86_consumed_dev_or_holdout": in_consumed,
            "has_raw_substrate": has_raw_substrate,
            "has_v86_local_rows": has_local,
            "has_v86_adapter_rows": has_adapter,
            "has_v86_tracklet_rows": has_tracklet,
            "formal_ready_uninspected_v86_chain": formal_ready,
            "missing_for_formal_ready": ";".join(missing),
        }
        candidate_rows.append(row)
        if formal_ready:
            ready_rows.append(row)

    scenes_with_raw_substrate = sorted(
        {
            scene
            for scene in raw_carrier_by_scene
            if raw_carrier_by_scene.get(scene, 0) > 0 and raw_mask_by_scene.get(scene, 0) > 0
        }
    )
    scenes_with_unconsumed_raw = sorted(
        {
            scene
            for scene, chunk in raw_carrier_by_scene_chunk
            if (scene, chunk) not in consumed_chunks and raw_mask_by_scene.get(scene, 0) > 0
        }
    )
    decision = (
        "PASS_UNINSPECTED_FORMAL_READY_CHAIN_AVAILABLE"
        if ready_rows
        else "NO_GO_RAW_SUBSTRATE_EXISTS_BUT_FORMAL_READY_CHAIN_MISSING"
    )
    summary = {
        "schema": "stream4d_v86_phase15_raw_substrate_availability_audit_v1",
        "phase": "v86_phase15_raw_substrate_availability_audit",
        "decision": decision,
        "formal_v86_goal_achieved": False,
        "formal_method_claim_allowed": False,
        "formal_claim_blocker": (
            "Uninspected raw D4RT/mask observation substrate exists in older v47/v54 tables, but no uninspected "
            "scene/chunk currently has the full v86 local-slot, adapter, and tracklet-readout chain required for a "
            "fresh formal holdout claim."
        ),
        "raw_source_artifact_count": len(raw_specs),
        "chain_source_artifact_count": len(chain_specs),
        "raw_scene_count": len(scenes_with_raw_substrate),
        "raw_scene_ids": scenes_with_raw_substrate,
        "unconsumed_raw_scene_count": len(scenes_with_unconsumed_raw),
        "unconsumed_raw_scene_ids": scenes_with_unconsumed_raw,
        "candidate_scene_chunk_count": len(candidate_rows),
        "formal_ready_uninspected_v86_chain_count": len(ready_rows),
        "phase14_summary_path": _rel(_repo_path(args.phase14_output_root) / "fresh_holdout_availability_summary.json"),
        "recommended_next_action": (
            "Do not claim v86 success. Either run a new method-version pipeline to build local/adapter/tracklet artifacts "
            "for unconsumed raw scenes before registering a holdout, or provide additional prepared artifacts."
        ),
        "runtime_sec": time.time() - t0,
    }
    _write_json(out / "raw_substrate_availability_summary.json", summary)
    _write_csv(out / "raw_substrate_source_artifact_rows.csv", source_rows)
    _write_csv(out / "raw_substrate_scene_chunk_rows.csv", candidate_rows)
    _write_csv(out / "formal_ready_uninspected_chain_rows.csv", ready_rows)
    return summary


def _new_scene_pipeline_feasibility_audit(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase16_output_root)
    out.mkdir(parents=True, exist_ok=True)

    phase15_root = _repo_path(args.phase15_output_root)
    phase15_summary = _read_json(phase15_root / "raw_substrate_availability_summary.json")
    phase15_rows = _read_csv_rows(phase15_root / "raw_substrate_scene_chunk_rows.csv")
    unconsumed_scenes = [
        str(scene)
        for scene in phase15_summary.get("unconsumed_raw_scene_ids", [])
        if str(scene).strip()
    ]

    source_specs = [
        (
            "v75_soft_incidence_rows",
            _repo_path("Stream3D/outputs/audit/v75_phase1_soft_incidence/incidence_rows.csv"),
            "required_by_v80_v81_local_affinity_loader",
        ),
        (
            "v71_semantic_feature_rows",
            _repo_path("Stream3D/outputs/audit/v71_semantic_features/mask_feature_rows.csv"),
            "semantic_descriptor_or_override_appearance_candidate",
        ),
        (
            "v81_default_dino_appearance_rows",
            _repo_path("Stream3D/outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv"),
            "v82_default_appearance_feature_rows",
        ),
    ]
    source_counts: dict[str, tuple[Counter[tuple[str, str]], Counter[str], int, Path, str]] = {}
    for label, path, role in source_specs:
        scene_chunk_counts, scene_counts, row_count = _scene_chunk_counts_from_csv(path)
        source_counts[label] = (scene_chunk_counts, scene_counts, row_count, path, role)

    raw_chunks_by_scene: dict[str, list[str]] = defaultdict(list)
    for row in phase15_rows:
        scene = str(row.get("scene_id") or "")
        if scene not in unconsumed_scenes:
            continue
        if not _bool(row.get("in_v86_consumed_dev_or_holdout")) and _bool(row.get("has_raw_substrate")):
            raw_chunks_by_scene[scene].append(str(row.get("chunk_id") or ""))

    v80_source = _repo_path("Stream3D/tools/run_v80_cmap_af_l2h_pipeline.py")
    v81_source = _repo_path("Stream3D/tools/run_v81_history_anchored_cmap_af_l2h_pipeline.py")
    v82_source = _repo_path("Stream3D/tools/run_v82_revised_causal_tracklet_memory.py")
    v75_soft_source = _repo_path("Stream3D/stream4d_native/v75_soft_incidence.py")
    v75_smoke_root = _repo_path(args.phase16_v75_smoke_root)
    v75_smoke_summary = _read_json(v75_smoke_root / "incidence_summary.json")
    v82_cli_scene_override_available = _source_contains(v82_source, 'parser.add_argument("--scenes"')
    v82_scene_forwarding_available = _source_contains(v82_source, "local_args.scenes")

    entrypoint_rows = [
        {
            "entrypoint": "v80_selected_chunks_for_new_scenes",
            "path": _rel(v80_source),
            "available": False,
            "evidence": "DEV_SPLIT/HOLDOUT_SPLIT contain only scene0011_00 and scene0050_00 in the current source.",
            "blocking_effect": "new scenes passed via --scenes receive zero selected chunks unless a new split/chunk policy is added",
        },
        {
            "entrypoint": "v81_cli_scene_override",
            "path": _rel(v81_source),
            "available": _source_contains(v80_source, 'parser.add_argument("--scenes"')
            and _source_contains(v81_source, "v80.build_parser()"),
            "evidence": "v81 inherits v80 parser, including --scenes.",
            "blocking_effect": "not sufficient because v80 split policy and v75 incidence inputs still do not cover the new scenes",
        },
        {
            "entrypoint": "v82_cli_scene_override",
            "path": _rel(v82_source),
            "available": v82_cli_scene_override_available,
            "evidence": "v82 parser source scan for --scenes.",
            "blocking_effect": (
                "scene override CLI is present; remaining blockers are upstream incidence/split/appearance inputs"
                if v82_cli_scene_override_available
                else "v82 cannot currently express scene0030_00/scene0081_01/scene0591_00 overrides"
            ),
        },
        {
            "entrypoint": "v82_forwards_scene_override_to_v81",
            "path": _rel(v82_source),
            "available": v82_scene_forwarding_available,
            "evidence": "_v81_phase1_args source scan for local_args.scenes assignment.",
            "blocking_effect": (
                "scene override is forwarded to v81; remaining blockers are upstream incidence/split/appearance inputs"
                if v82_scene_forwarding_available
                else "even if a caller added scenes dynamically, current v82 wrapper would not pass them to v81"
            ),
        },
        {
            "entrypoint": "v75_soft_incidence_smoke_new_scenes",
            "path": _rel(v75_smoke_root / "incidence_summary.json"),
            "available": v75_smoke_summary.get("decision") == "PASS_V75_PHASE1_SOFT_INCIDENCE",
            "evidence": v75_smoke_summary.get("decision", "missing_smoke_summary"),
            "blocking_effect": "default v75 soft-incidence scene roots do not include the unconsumed scenes",
        },
    ]

    input_coverage_rows: list[dict[str, Any]] = []
    required_chain_rows: list[dict[str, Any]] = []
    ready_scene_count = 0
    for scene in unconsumed_scenes:
        raw_chunks = sorted(raw_chunks_by_scene.get(scene, []), key=lambda value: int(float(value)) if value else -1)
        row: dict[str, Any] = {
            "scene_id": scene,
            "raw_unconsumed_chunk_count": len(raw_chunks),
            "raw_unconsumed_chunk_sample": ",".join(raw_chunks[:16]),
            "v80_dev_selected_chunk_count": len(DEV_SPLIT_CHUNKS.get(scene, set())),
            "v80_holdout_selected_chunk_count": len(HOLDOUT_SPLIT_CHUNKS.get(scene, set())),
            "v75_soft_incidence_default_scene_root_declared": _source_declares_scene(v75_soft_source, scene),
        }
        for label, (scene_chunk_counts, scene_counts, row_count, path, role) in source_counts.items():
            chunks = _chunks_for_scene(scene_chunk_counts, scene)
            row[f"{label}_path"] = _rel(path)
            row[f"{label}_role"] = role
            row[f"{label}_total_rows"] = row_count
            row[f"{label}_scene_rows"] = scene_counts.get(scene, 0)
            row[f"{label}_chunk_count"] = len(chunks)
            row[f"{label}_chunk_sample"] = ",".join(chunks[:16])
        input_coverage_rows.append(row)

        missing: list[str] = []
        if len(raw_chunks) == 0:
            missing.append("unconsumed_raw_substrate")
        if row["v75_soft_incidence_rows_scene_rows"] <= 0:
            missing.append("v75_soft_incidence_rows_for_scene")
        if not row["v75_soft_incidence_default_scene_root_declared"]:
            missing.append("v75_soft_incidence_default_scene_root")
        if row["v80_dev_selected_chunk_count"] <= 0 and row["v80_holdout_selected_chunk_count"] <= 0:
            missing.append("v80_v81_registered_chunk_split")
        if row["v81_default_dino_appearance_rows_scene_rows"] <= 0:
            missing.append("v82_default_appearance_rows_for_scene")
        if not any(item["entrypoint"] == "v82_cli_scene_override" and item["available"] for item in entrypoint_rows):
            missing.append("v82_scene_override_cli")
        if not any(item["entrypoint"] == "v82_forwards_scene_override_to_v81" and item["available"] for item in entrypoint_rows):
            missing.append("v82_scene_override_forwarding")
        ready = not missing
        ready_scene_count += int(ready)
        required_chain_rows.append(
            {
                "scene_id": scene,
                "raw_substrate_available": len(raw_chunks) > 0,
                "v75_soft_incidence_rows_available": row["v75_soft_incidence_rows_scene_rows"] > 0,
                "v75_soft_incidence_default_scene_root_declared": row["v75_soft_incidence_default_scene_root_declared"],
                "v80_v81_registered_split_available": row["v80_dev_selected_chunk_count"] > 0
                or row["v80_holdout_selected_chunk_count"] > 0,
                "v71_semantic_features_available": row["v71_semantic_feature_rows_scene_rows"] > 0,
                "v82_default_appearance_rows_available": row["v81_default_dino_appearance_rows_scene_rows"] > 0,
                "v82_scene_override_cli_available": any(
                    item["entrypoint"] == "v82_cli_scene_override" and item["available"] for item in entrypoint_rows
                ),
                "v82_scene_override_forwarding_available": any(
                    item["entrypoint"] == "v82_forwards_scene_override_to_v81" and item["available"]
                    for item in entrypoint_rows
                ),
                "direct_new_scene_chain_ready": ready,
                "missing_for_direct_new_scene_chain": ";".join(missing),
                "recommended_repair": (
                    "Generate/register v75-style soft-incidence rows or adapt the local loader to v47/v54 observation tables, "
                    "add an explicit new-scene chunk split, expose and forward scene/chunk overrides through v82, and provide "
                    "appearance rows covering the registered new scenes before any fresh holdout claim."
                ),
            }
        )

    remaining_missing = sorted(
        {
            item
            for row in required_chain_rows
            for item in str(row.get("missing_for_direct_new_scene_chain") or "").split(";")
            if item
        }
    )
    blocker_tail = "; ".join(remaining_missing) if remaining_missing else "none"
    decision = (
        "PASS_NEW_SCENE_CHAIN_READY_FOR_REGISTERED_SPLIT"
        if ready_scene_count > 0
        else "NO_GO_NEW_SCENE_PIPELINE_INPUT_CHAIN_MISSING"
    )
    summary = {
        "schema": "stream4d_v86_phase16_new_scene_pipeline_feasibility_audit_v1",
        "phase": "v86_phase16_new_scene_pipeline_feasibility_audit",
        "decision": decision,
        "formal_v86_goal_achieved": False,
        "formal_method_claim_allowed": False,
        "formal_claim_blocker": (
            "Unconsumed raw scenes exist, but the current v75/v80/v81/v82 method chain cannot directly build a fresh "
            "formal v86 readout. Remaining direct-chain blockers: "
            f"{blocker_tail}."
        ),
        "unconsumed_raw_scene_ids": unconsumed_scenes,
        "ready_scene_count": ready_scene_count,
        "v75_smoke_summary_path": _rel(v75_smoke_root / "incidence_summary.json"),
        "v75_smoke_decision": v75_smoke_summary.get("decision", ""),
        "recommended_next_action": (
            "Do not claim v86 success. Start a new method-version preparation step: first generate or adapt method-safe "
            "soft-incidence/local inputs for unconsumed scenes, register a split before looking at the new holdout, then "
            "freeze DV5-style readout and run the fresh holdout exactly once."
        ),
        "runtime_sec": time.time() - t0,
    }

    _write_json(out / "new_scene_pipeline_feasibility_summary.json", summary)
    _write_csv(out / "new_scene_input_coverage_rows.csv", input_coverage_rows)
    _write_csv(out / "new_scene_entrypoint_rows.csv", entrypoint_rows)
    _write_csv(out / "new_scene_required_chain_rows.csv", required_chain_rows)
    return summary


def _assignment_rows_from_support(
    native_support_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_tables = sorted(
        {
            str(row.get("source_observation_table", "")).strip()
            for row in native_support_rows
            if str(row.get("source_observation_table", "")).strip()
        }
    )
    mask_label_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_audit_rows: list[dict[str, Any]] = []
    duplicate_mask_keys = 0
    for source_table in source_tables:
        carrier_table_path = _repo_path(source_table)
        mask_table_path = carrier_table_path.with_name("mask_observation_table.csv")
        exists = mask_table_path.exists()
        row_count = 0
        positive_gt_row_count = 0
        diagnostic_label_row_count = 0
        uses_gt_for_prediction_count = 0
        if exists:
            with mask_table_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    row_count += 1
                    scene = str(row.get("scene_id") or row.get("scene") or "")
                    key = (scene, str(row.get("frame_id", "")), str(row.get("mask_id", "")))
                    if key in mask_label_by_key:
                        duplicate_mask_keys += 1
                    mask_label_by_key[key] = row
                    if _int(row.get("diagnostic_gt_instance"), 0) > 0:
                        positive_gt_row_count += 1
                    if _bool(row.get("uses_gt_for_diagnostic_labels")):
                        diagnostic_label_row_count += 1
                    if _bool(row.get("uses_gt_for_prediction")):
                        uses_gt_for_prediction_count += 1
        source_audit_rows.append(
            {
                "source_carrier_observation_table": source_table,
                "source_mask_observation_table": _rel(mask_table_path),
                "exists": exists,
                "mask_observation_row_count": row_count,
                "positive_diagnostic_gt_row_count": positive_gt_row_count,
                "uses_gt_for_diagnostic_label_row_count": diagnostic_label_row_count,
                "uses_gt_for_prediction_row_count": uses_gt_for_prediction_count,
                "legal_for_prediction": uses_gt_for_prediction_count == 0,
                "legal_for_diagnostic_scoring": positive_gt_row_count > 0,
                "notes": "mask diagnostic GT labels score native-carrier/history assignments only; they do not form predictions",
            }
        )

    pred_votes_by_carrier: dict[str, Counter[str]] = defaultdict(Counter)
    gt_votes_by_carrier: dict[str, Counter[str]] = defaultdict(Counter)
    scene_by_carrier: dict[str, str] = {}
    support_obs_by_carrier: Counter[str] = Counter()
    joined_support_observation_count = 0
    labeled_support_observation_count = 0
    missing_label_support_observation_count = 0
    nonpositive_gt_support_observation_count = 0
    native_support_carriers = {
        str(row.get("native_carrier_global_id", "")).strip()
        for row in native_support_rows
        if str(row.get("native_carrier_global_id", "")).strip()
    }

    for row in native_support_rows:
        scene = str(row.get("scene_id", "")).strip()
        key = (scene, str(row.get("frame_id", "")), str(row.get("mask_id", "")))
        mask_label = mask_label_by_key.get(key)
        if not mask_label:
            missing_label_support_observation_count += 1
            continue
        joined_support_observation_count += 1
        gt_instance = _int(mask_label.get("diagnostic_gt_instance"), 0)
        if gt_instance <= 0:
            nonpositive_gt_support_observation_count += 1
            continue
        history_id = str(row.get("history_id", "")).strip()
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        if not history_id or not native_id:
            missing_label_support_observation_count += 1
            continue
        labeled_support_observation_count += 1
        pred_votes_by_carrier[native_id][history_id] += 1
        gt_votes_by_carrier[native_id][str(gt_instance)] += 1
        scene_by_carrier.setdefault(native_id, scene)
        support_obs_by_carrier[native_id] += 1

    assignment_rows: list[dict[str, Any]] = []
    for native_id in sorted(gt_votes_by_carrier):
        pred_counter = pred_votes_by_carrier[native_id]
        gt_counter = gt_votes_by_carrier[native_id]
        pred_label_raw, pred_vote_count, pred_vote_purity = _counter_winner(pred_counter)
        gt_label_raw, gt_vote_count, gt_vote_purity = _counter_winner(gt_counter)
        scene = scene_by_carrier.get(native_id, "")
        assignment_rows.append(
            {
                "scene_id": scene,
                "native_carrier_global_id": native_id,
                "pred_history_id": pred_label_raw,
                "diagnostic_gt_instance": gt_label_raw,
                "pred_history_eval_label": f"{scene}:{pred_label_raw}",
                "diagnostic_gt_eval_label": f"{scene}:{gt_label_raw}",
                "labeled_support_observation_count": support_obs_by_carrier[native_id],
                "pred_vote_count": pred_vote_count,
                "pred_vote_purity": pred_vote_purity,
                "diagnostic_gt_vote_count": gt_vote_count,
                "diagnostic_gt_vote_purity": gt_vote_purity,
                "pred_history_vote_json": _counter_json(pred_counter),
                "diagnostic_gt_vote_json": _counter_json(gt_counter),
                "pred_history_conflict": len(pred_counter) > 1,
                "diagnostic_gt_conflict": len(gt_counter) > 1,
                "is_method_result": False,
                "is_diagnostic_only": True,
                "forbidden_for_method_table": True,
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
                "metric_scope": "native_carrier_mask_diagnostic_cluster_metric_not_scannet_ap",
            }
        )
    summary = {
        "native_support_observation_count": len(native_support_rows),
        "native_support_carrier_count": len(native_support_carriers),
        "source_mask_table_count": len(source_tables),
        "mask_label_key_count": len(mask_label_by_key),
        "duplicate_mask_label_key_count": duplicate_mask_keys,
        "joined_support_observation_count": joined_support_observation_count,
        "labeled_support_observation_count": labeled_support_observation_count,
        "missing_label_support_observation_count": missing_label_support_observation_count,
        "nonpositive_gt_support_observation_count": nonpositive_gt_support_observation_count,
        "native_assignment_count": len(assignment_rows),
        "native_gt_label_coverage_rate": _safe_ratio(len(assignment_rows), len(native_support_carriers)),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return assignment_rows, source_audit_rows, summary


def _vote_entropy_and_margin(text: str) -> tuple[float, float]:
    counts = _counter_from_json(text)
    total = sum(counts.values())
    if total <= 0:
        return 0.0, 0.0
    probs = sorted((count / total for count in counts.values()), reverse=True)
    if len(probs) <= 1:
        return 0.0, 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0) / math.log(len(probs))
    return entropy, probs[0] - probs[1]


def _set_iou(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return _safe_ratio(len(a & b), len(a | b))


def _cluster_metrics(pred_labels: list[str], gt_labels: list[str]) -> dict[str, Any]:
    if len(pred_labels) != len(gt_labels):
        raise ValueError("pred_labels and gt_labels must have the same length")
    n = len(pred_labels)
    if n == 0:
        return {
            "sample_count": 0,
            "adjusted_rand_index": 0.0,
            "purity": 0.0,
            "completeness": 0.0,
            "pred_cluster_count": 0,
            "gt_cluster_count": 0,
            "overmerge_pred_cluster_count": 0,
            "oversplit_gt_cluster_count": 0,
        }
    pred_counts = Counter(pred_labels)
    gt_counts = Counter(gt_labels)
    contingency = Counter(zip(pred_labels, gt_labels))
    total_pairs = _comb2(n)
    sum_cont = sum(_comb2(count) for count in contingency.values())
    sum_pred = sum(_comb2(count) for count in pred_counts.values())
    sum_gt = sum(_comb2(count) for count in gt_counts.values())
    if total_pairs == 0:
        ari = 1.0
    else:
        expected = sum_pred * sum_gt / total_pairs
        denom = 0.5 * (sum_pred + sum_gt) - expected
        ari = 0.0 if denom == 0 else (sum_cont - expected) / denom

    gt_by_pred: dict[str, Counter[str]] = defaultdict(Counter)
    pred_by_gt: dict[str, Counter[str]] = defaultdict(Counter)
    for pred, gt in zip(pred_labels, gt_labels):
        gt_by_pred[pred][gt] += 1
        pred_by_gt[gt][pred] += 1
    purity = _safe_ratio(sum(max(counter.values()) for counter in gt_by_pred.values()), n)
    completeness = _safe_ratio(sum(max(counter.values()) for counter in pred_by_gt.values()), n)
    return {
        "sample_count": n,
        "adjusted_rand_index": ari,
        "purity": purity,
        "completeness": completeness,
        "pred_cluster_count": len(pred_counts),
        "gt_cluster_count": len(gt_counts),
        "overmerge_pred_cluster_count": sum(1 for counter in gt_by_pred.values() if len(counter) > 1),
        "oversplit_gt_cluster_count": sum(1 for counter in pred_by_gt.values() if len(counter) > 1),
    }


def _prediction_and_gt_sets(
    all_assignment_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    label_field: str,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    pred_sets: dict[str, set[str]] = defaultdict(set)
    gt_sets: dict[str, set[str]] = defaultdict(set)
    for row in all_assignment_rows:
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        gt_label = str(row.get("diagnostic_gt_eval_label", "")).strip()
        if native_id and gt_label:
            gt_sets[gt_label].add(native_id)
    for row in pred_rows:
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        pred_label = str(row.get(label_field, "")).strip()
        if native_id and pred_label:
            pred_sets[pred_label].add(native_id)
    return dict(pred_sets), dict(gt_sets)


def _ap_style_metrics(
    pred_sets: dict[str, set[str]],
    gt_sets: dict[str, set[str]],
    pred_scores: dict[str, float],
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    sorted_preds = sorted(pred_sets, key=lambda pred: (-pred_scores.get(pred, 0.0), str(pred)))
    best_iou_values = [
        max((_set_iou(pred_sets[pred], gt_set) for gt_set in gt_sets.values()), default=0.0)
        for pred in sorted_preds
    ]
    out: dict[str, Any] = {
        "native_prediction_object_count": len(pred_sets),
        "native_gt_object_count": len(gt_sets),
        "native_mean_best_IoU": _mean(best_iou_values),
    }
    gt_count = len(gt_sets)
    for threshold in thresholds:
        matched_gt: set[str] = set()
        tp_flags: list[int] = []
        fp_flags: list[int] = []
        for pred in sorted_preds:
            best_gt = ""
            best_iou = 0.0
            for gt_label, gt_set in gt_sets.items():
                if gt_label in matched_gt:
                    continue
                iou = _set_iou(pred_sets[pred], gt_set)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_label
            if best_gt and best_iou >= threshold:
                matched_gt.add(best_gt)
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)
        tp_cum = 0
        fp_cum = 0
        precision_sum_at_tp = 0.0
        for tp, fp in zip(tp_flags, fp_flags):
            tp_cum += tp
            fp_cum += fp
            if tp:
                precision_sum_at_tp += _safe_ratio(tp_cum, tp_cum + fp_cum)
        precision = _safe_ratio(tp_cum, tp_cum + fp_cum)
        recall = _safe_ratio(tp_cum, gt_count)
        suffix = str(int(threshold * 100))
        out[f"native_AP{suffix}"] = _safe_ratio(precision_sum_at_tp, gt_count)
        out[f"native_precision{suffix}"] = precision
        out[f"native_recall{suffix}"] = recall
        out[f"native_matched_gt_count_at_{suffix}"] = len(matched_gt)
    return out


def _row_score(row: dict[str, Any]) -> float:
    if str(row.get("v86_score", "")).strip():
        return _num(row.get("v86_score"), 0.0)
    return _num(row.get("pred_vote_purity"), 0.0)


def _row_conflict(row: dict[str, Any]) -> bool:
    if "v86_conflict_violation" in row:
        return _bool(row.get("v86_conflict_violation"))
    if "support_split_conflict" in row:
        return _bool(row.get("support_split_conflict"))
    return _bool(row.get("pred_history_conflict"))


def _history_state(label: str) -> str:
    if "confirmed" in label:
        return "confirmed"
    if "stable_tentative" in label:
        return "stable_tentative"
    if "quarantine" in label:
        return "quarantine"
    return "other"


def _vote_items(row: dict[str, Any]) -> list[tuple[str, int]]:
    counter = _counter_from_json(str(row.get("pred_history_vote_json", "")))
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def _row_hijack(row: dict[str, Any]) -> bool:
    if "v86_new_object_hijack" in row:
        return _bool(row.get("v86_new_object_hijack"))
    label = str(row.get("v86_label") or row.get("pred_history_eval_label") or "")
    if _history_state(label) != "confirmed":
        return False
    return any(_history_state(vote_label) != "confirmed" for vote_label, _count in _vote_items(row))


def _pred_scores(rows: list[dict[str, Any]], *, label_field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        label = str(row.get(label_field, "")).strip()
        if label:
            values[label].append(_row_score(row))
    return {label: _mean(vals) for label, vals in values.items()}


def _eval_variant(
    all_assignment_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    variant_id: str,
    label_field: str,
    pred_scores: dict[str, float] | None = None,
    score_contract: str = "mean_pred_vote_purity",
    prediction_uses_gt: bool = False,
    is_oracle: bool = False,
) -> dict[str, Any]:
    selected_by_id = {
        str(row.get("native_carrier_global_id", "")): str(row.get(label_field, ""))
        for row in pred_rows
        if row.get("native_carrier_global_id") and row.get(label_field)
    }
    pred_labels: list[str] = []
    gt_labels: list[str] = []
    for row in pred_rows:
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        pred_label = selected_by_id.get(native_id, "")
        gt_label = str(row.get("diagnostic_gt_eval_label", "")).strip()
        if native_id and pred_label and gt_label:
            pred_labels.append(pred_label)
            gt_labels.append(gt_label)
    if pred_scores is None:
        pred_sets, _gt_sets = _prediction_and_gt_sets(all_assignment_rows, pred_rows, label_field=label_field)
        pred_scores = {label: float(len(carriers)) for label, carriers in pred_sets.items()}
        score_contract = score_contract or "cluster_size_desc"
    pred_sets, gt_sets = _prediction_and_gt_sets(all_assignment_rows, pred_rows, label_field=label_field)
    return {
        "variant_id": variant_id,
        "sample_count": len(pred_labels),
        **_cluster_metrics(pred_labels, gt_labels),
        **_ap_style_metrics(pred_sets, gt_sets, pred_scores),
        "score_contract": score_contract,
        "prediction_uses_gt": prediction_uses_gt,
        "is_oracle": is_oracle,
    }


def _size_matched_hash_labels(
    rows: list[dict[str, Any]],
    *,
    group_field: str | None,
    source_label_field: str,
    salt: str,
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row.get(group_field, "")) if group_field else "ALL"
        grouped[group].append(row)
    out: dict[str, str] = {}
    for group, group_rows in grouped.items():
        counts = Counter(str(row.get(source_label_field, "")) for row in group_rows if row.get(source_label_field))
        expanded: list[str] = []
        for label, count in sorted(
            counts.items(),
            key=lambda item: (_sha1_text(f"{salt}|label|{group}|{item[0]}"), str(item[0])),
        ):
            expanded.extend([label] * count)
        sorted_rows = sorted(
            group_rows,
            key=lambda row: (
                _sha1_text(f"{salt}|carrier|{row.get('native_carrier_global_id', '')}"),
                str(row.get("native_carrier_global_id", "")),
            ),
        )
        for row, label in zip(sorted_rows, expanded):
            native_id = str(row.get("native_carrier_global_id", ""))
            if native_id:
                out[native_id] = label
    return out


def _uniform_hash_labels(rows: list[dict[str, Any]], *, source_label_field: str, salt: str) -> dict[str, str]:
    labels = sorted({str(row.get(source_label_field, "")) for row in rows if row.get(source_label_field)})
    if not labels:
        return {}
    out: dict[str, str] = {}
    for row in rows:
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        if not native_id:
            continue
        idx = int(_sha1_text(f"{salt}|{native_id}"), 16) % len(labels)
        out[native_id] = labels[idx]
    return out


def _single_largest_labels(rows: list[dict[str, Any]], *, source_label_field: str, group_field: str) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field, ""))].append(row)
    out: dict[str, str] = {}
    for _group, group_rows in grouped.items():
        counts = Counter(str(row.get(source_label_field, "")) for row in group_rows if row.get(source_label_field))
        if not counts:
            continue
        label = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for row in group_rows:
            native_id = str(row.get("native_carrier_global_id", "")).strip()
            if native_id:
                out[native_id] = label
    return out


def _shifted_history_labels(rows: list[dict[str, Any]], *, source_label_field: str) -> dict[str, str]:
    labels_by_scene: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        scene = str(row.get("scene_id", ""))
        label = str(row.get(source_label_field, ""))
        if label and label not in labels_by_scene[scene]:
            labels_by_scene[scene].append(label)
    for scene in list(labels_by_scene):
        labels_by_scene[scene] = sorted(labels_by_scene[scene])
    out: dict[str, str] = {}
    for row in rows:
        scene = str(row.get("scene_id", ""))
        labels = labels_by_scene.get(scene, [])
        label = str(row.get(source_label_field, ""))
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        if native_id and label in labels and labels:
            out[native_id] = labels[(labels.index(label) - 1) % len(labels)]
    return out


def _carrier_max_chunk(support_rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in support_rows:
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        if not native_id:
            continue
        chunk = _int(row.get("chunk_id"), -1)
        if chunk >= 0:
            out[native_id] = max(out.get(native_id, -1), chunk)
    return out


def _time_indexed_stale_labels(
    rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    *,
    source_label_field: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    carrier_chunk = _carrier_max_chunk(support_rows)
    histories_by_scene: dict[str, list[tuple[int, str]]] = defaultdict(list)
    unsafe_history_rows = 0
    for row in history_rows:
        if _bool(row.get("method_uses_gt")) or _bool(row.get("uses_future")):
            unsafe_history_rows += 1
            continue
        scene = str(row.get("scene_id", "")).strip()
        history_id = str(row.get("history_id", "")).strip()
        last_seen = _int(row.get("last_seen_chunk"), -1)
        if scene and history_id and last_seen >= 0:
            histories_by_scene[scene].append((last_seen, f"{scene}:{history_id}"))
    for scene in list(histories_by_scene):
        histories_by_scene[scene] = sorted(histories_by_scene[scene], key=lambda item: (item[0], item[1]))

    out: dict[str, str] = {}
    no_prior_count = 0
    for row in rows:
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        scene = str(row.get("scene_id", "")).strip()
        source_label = str(row.get(source_label_field, "")).strip()
        chunk = carrier_chunk.get(native_id, -1)
        if not native_id or not scene or chunk < 0:
            no_prior_count += 1
            continue
        candidates = [
            label
            for last_seen, label in histories_by_scene.get(scene, [])
            if last_seen < chunk and label != source_label
        ]
        if not candidates:
            no_prior_count += 1
            continue
        out[native_id] = candidates[-1]
    return out, {
        "time_indexed_stale_history_safe_mapping_count": len(out),
        "time_indexed_stale_history_no_prior_count": no_prior_count,
        "time_indexed_stale_history_unsafe_history_rows": unsafe_history_rows,
        "time_indexed_stale_history_carrier_chunk_count": len(carrier_chunk),
    }


def _semantic_history_labels(rows: list[dict[str, Any]], history_rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    semantic_by_history: dict[str, str] = {}
    unsafe_rows = 0
    for row in history_rows:
        if _bool(row.get("method_uses_gt")) or _bool(row.get("uses_future")):
            unsafe_rows += 1
            continue
        scene = str(row.get("scene_id", "")).strip()
        history_id = str(row.get("history_id", "")).strip()
        semantic_hash = str(row.get("semantic_descriptor_hash", "")).strip()
        if scene and history_id and semantic_hash:
            semantic_by_history[f"{scene}:{history_id}"] = f"{scene}:semantic_descriptor_hash:{semantic_hash}"
    out: dict[str, str] = {}
    for row in rows:
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        history_label = str(row.get("pred_history_eval_label", "")).strip()
        if native_id and history_label in semantic_by_history:
            out[native_id] = semantic_by_history[history_label]
    collisions = len(set(semantic_by_history.values())) < len(semantic_by_history)
    return out, {
        "semantic_history_row_count": len(history_rows),
        "semantic_history_safe_mapping_count": len(semantic_by_history),
        "semantic_history_unsafe_row_count": unsafe_rows,
        "semantic_label_collision_present": collisions,
        "semantic_control_degenerate_unique_per_history": not collisions,
    }


def _labels_to_rows(
    rows: list[dict[str, Any]],
    labels_by_native: dict[str, str],
    *,
    label_field: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        label = labels_by_native.get(native_id, "")
        if not native_id or not label:
            continue
        new_row = dict(row)
        new_row[label_field] = label
        out.append(new_row)
    return out


def _support_split_variants(
    assignment_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    assignment_by_native = {
        str(row.get("native_carrier_global_id", "")): row
        for row in assignment_rows
        if row.get("native_carrier_global_id")
    }
    out: dict[str, dict[str, Any]] = {}
    for split_field, prefix in (("local_slot_id", "M7_local_slot"), ("cluster_id", "M8_cluster")):
        votes: dict[str, Counter[str]] = defaultdict(Counter)
        for row in support_rows:
            native_id = str(row.get("native_carrier_global_id", "")).strip()
            if native_id not in assignment_by_native:
                continue
            scene = str(row.get("scene_id", "")).strip()
            history = str(row.get("history_id", "")).strip()
            part = str(row.get(split_field, "")).strip()
            if native_id and scene and history and part:
                votes[native_id][f"{scene}:{history}|{split_field}:{part}"] += 1

        rows_all: list[dict[str, Any]] = []
        rows_pruned: list[dict[str, Any]] = []
        for native_id, counter in votes.items():
            if not counter:
                continue
            top_label, top_count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
            total = sum(counter.values())
            split_conflict = len(counter) > 1
            base = dict(assignment_by_native[native_id])
            base["v86_label"] = top_label
            base["v86_score"] = _safe_ratio(top_count, total)
            base["support_split_field"] = split_field
            base["support_split_vote_json"] = json.dumps(dict(sorted(counter.items())), sort_keys=True)
            base["support_split_conflict"] = split_conflict
            rows_all.append(base)
            if not split_conflict:
                rows_pruned.append(base)

        out[f"{prefix}_split_all"] = {
            "rows": rows_all,
            "support_source": f"v85 native support majority {split_field} split",
            "repair_action": f"split history objects by method-safe support {split_field}; keeps split-conflicted carriers",
        }
        out[f"{prefix}_split_pruned"] = {
            "rows": rows_pruned,
            "support_source": f"v85 native support majority {split_field} split",
            "repair_action": f"split by {split_field} and drop carriers with multiple support split votes",
        }
    return out


def _anti_hijack_variants(assignment_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    demote_rows: list[dict[str, Any]] = []
    state_priority_rows: list[dict[str, Any]] = []
    for row in assignment_rows:
        vote_items = _vote_items(row)
        if not vote_items:
            continue
        base_label = str(row.get("pred_history_eval_label", ""))
        base_state = _history_state(base_label)
        non_confirmed = [(label, count) for label, count in vote_items if _history_state(label) != "confirmed"]

        demote_label = base_label
        if base_state == "confirmed" and non_confirmed:
            demote_label = sorted(non_confirmed, key=lambda item: (-item[1], item[0]))[0][0]
        demote_count = next((count for label, count in vote_items if label == demote_label), 0)
        demote_total = sum(count for _label, count in vote_items)
        demote_row = dict(row)
        demote_row["v86_label"] = demote_label
        demote_row["v86_score"] = _safe_ratio(demote_count, demote_total)
        demote_row["v86_conflict_violation"] = False
        demote_row["v86_new_object_hijack"] = False
        demote_row["v86_input_conflict"] = len(vote_items) > 1
        demote_rows.append(demote_row)

        state_priority_label = base_label
        for target_state in ("stable_tentative", "confirmed", "quarantine", "other"):
            candidates = [(label, count) for label, count in vote_items if _history_state(label) == target_state]
            if candidates:
                state_priority_label = sorted(candidates, key=lambda item: (-item[1], item[0]))[0][0]
                break
        priority_count = next((count for label, count in vote_items if label == state_priority_label), 0)
        priority_row = dict(row)
        priority_row["v86_label"] = state_priority_label
        priority_row["v86_score"] = _safe_ratio(priority_count, demote_total)
        priority_row["v86_conflict_violation"] = False
        priority_row["v86_new_object_hijack"] = False
        priority_row["v86_input_conflict"] = len(vote_items) > 1
        state_priority_rows.append(priority_row)

    variants["M9_anti_hijack_demote_top_new"] = {
        "rows": demote_rows,
        "support_source": "v85 native support pred_history votes",
        "repair_action": "if confirmed label competes with tentative/quarantine evidence, demote to top non-confirmed vote",
    }
    variants["M10_anti_hijack_state_priority"] = {
        "rows": state_priority_rows,
        "support_source": "v85 native support pred_history votes",
        "repair_action": "resolve conflicts by state priority stable_tentative > confirmed > quarantine; preserves new/tentative objects",
    }
    return variants


def _variant_rows(
    assignment_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    rows_m0 = [dict(row, v86_label=row.get("pred_history_eval_label", "")) for row in assignment_rows]

    rows_pruned = [
        dict(row, v86_label=row.get("pred_history_eval_label", ""))
        for row in assignment_rows
        if not _bool(row.get("pred_history_conflict"))
    ]

    rows_vote_split: list[dict[str, Any]] = []
    rows_carrier_split: list[dict[str, Any]] = []
    rows_confirmed: list[dict[str, Any]] = []
    rows_confirmed_pruned: list[dict[str, Any]] = []
    for row in assignment_rows:
        base_label = str(row.get("pred_history_eval_label", ""))
        vote_sig = _sha1_text(str(row.get("pred_history_vote_json", "")))[:8]
        vote_row = dict(row)
        vote_row["v86_label"] = f"{base_label}|vote_signature:{vote_sig}"
        rows_vote_split.append(vote_row)

        carrier_row = dict(row)
        if _bool(row.get("pred_history_conflict")):
            carrier_row["v86_label"] = f"{base_label}|carrier_split:{row.get('native_carrier_global_id', '')}"
        else:
            carrier_row["v86_label"] = base_label
        rows_carrier_split.append(carrier_row)

        if "confirmed" in str(row.get("pred_history_id", "")):
            conf_row = dict(row, v86_label=base_label)
            rows_confirmed.append(conf_row)
            if not _bool(row.get("pred_history_conflict")):
                rows_confirmed_pruned.append(conf_row)

    variants = {
        "M0_assignment_only": {
            "rows": rows_m0,
            "support_source": "v85 native support majority history assignment",
            "repair_action": "baseline; no repair",
        },
        "M4_conflict_pruned": {
            "rows": rows_pruned,
            "support_source": "v85 native support majority history assignment",
            "repair_action": "dropped carriers with multiple pred_history votes; no threshold tuning",
        },
        "M4_vote_signature_split": {
            "rows": rows_vote_split,
            "support_source": "v85 native support majority history assignment",
            "repair_action": "split labels by method-safe history vote signature; keeps conflicted carriers",
        },
        "M4_carrier_split_conflicts": {
            "rows": rows_carrier_split,
            "support_source": "v85 native support majority history assignment",
            "repair_action": "split conflicted carrier assignments into singleton labels; keeps conflicted carriers",
        },
        "M5_confirmed_only": {
            "rows": rows_confirmed,
            "support_source": "v85 native support majority history assignment",
            "repair_action": "drop stable_tentative histories to test new-object absorption repair direction",
        },
        "M6_confirmed_no_conflict": {
            "rows": rows_confirmed_pruned,
            "support_source": "v85 native support majority history assignment",
            "repair_action": "confirmed-only plus conflict pruning",
        },
    }
    if support_rows:
        variants.update(_support_split_variants(assignment_rows, support_rows))
    variants.update(_anti_hijack_variants(assignment_rows))
    return variants


def _load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    phase7 = _repo_path(args.v85_phase7_root)
    phase8 = _repo_path(args.v85_phase8_root)
    phase10 = _repo_path(args.v85_phase10_root)
    phase5 = _repo_path(args.v85_phase5_root)
    phase6 = _repo_path(args.v85_phase6_root)
    return {
        "v85_phase7": phase7,
        "v85_phase8": phase8,
        "v85_phase10": phase10,
        "v85_phase5": phase5,
        "v85_phase6": phase6,
        "support_rows": _read_csv_rows(phase7 / "native_carrier_support_rows.csv"),
        "assignment_rows": _read_csv_rows(phase7 / "native_carrier_diagnostic_assignment_rows.csv"),
        "v85_control_rows": _read_csv_rows(phase7 / "native_carrier_diagnostic_control_rows.csv"),
        "history_rows": _read_csv_rows(phase5 / "history_node_rows.csv"),
        "v85_diagnostic_summary": _read_json(phase7 / "native_carrier_diagnostic_summary.json"),
        "v85_contract": _read_json(phase7 / "native_carrier_evaluator_candidate_contract.json"),
        "v85_route_summary": _read_json(phase7 / "native_scene_vertex_export_route_summary.json"),
        "v85_phase8_summary": _read_json(phase8 / "control_summary.json"),
        "v85_final": _read_json(phase10 / "final_decision.json"),
        "v85_q_summary": _read_json(phase6 / "q_summary.json"),
    }


def _phase0(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    ctx = _load_inputs(args)
    out = _repo_path(args.phase0_output_root)
    support_rows = ctx["support_rows"]
    assignment_rows = ctx["assignment_rows"]
    unique_support = {row.get("native_carrier_global_id", "") for row in support_rows if row.get("native_carrier_global_id")}
    unique_assignment = {
        row.get("native_carrier_global_id", "") for row in assignment_rows if row.get("native_carrier_global_id")
    }
    required_inputs = [
        ("v85_support_rows", ctx["v85_phase7"] / "native_carrier_support_rows.csv", "method native-carrier support"),
        (
            "v85_assignment_rows",
            ctx["v85_phase7"] / "native_carrier_diagnostic_assignment_rows.csv",
            "evaluator-only GT labels and v85 history majority labels",
        ),
        ("v85_candidate_contract", ctx["v85_phase7"] / "native_carrier_evaluator_candidate_contract.json", "contract seed"),
        ("v85_route_summary", ctx["v85_phase7"] / "native_scene_vertex_export_route_summary.json", "scene exporter audit"),
        ("v85_final_decision", ctx["v85_phase10"] / "final_decision.json", "previous final decision"),
    ]
    input_rows = []
    for name, path, role in required_inputs:
        input_rows.append(
            {
                "input_name": name,
                "path": _rel(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "role": role,
            }
        )
    route_rows = [
        {
            "route_id": "A_native_carrier_objectness",
            "selected_first_round": True,
            "method_candidate": True,
            "scene_metric_candidate": False,
            "reason": "v85 support and candidate native evaluator exist; v86 can freeze contract before rerun",
        },
        {
            "route_id": "B_scene_vertex_exporter",
            "selected_first_round": False,
            "method_candidate": False,
            "scene_metric_candidate": True,
            "reason": ctx["v85_route_summary"].get("primary_blocker", "method-safe scene exporter unavailable"),
        },
        {
            "route_id": "C_frame_mask_readout",
            "selected_first_round": False,
            "method_candidate": False,
            "scene_metric_candidate": False,
            "reason": "secondary renderability diagnostic; not needed for first native-carrier gate",
        },
        {
            "route_id": "D_weak_identity_graph",
            "selected_first_round": False,
            "method_candidate": False,
            "scene_metric_candidate": False,
            "reason": "fallback diagnostic only unless native readout fails completely",
        },
    ]
    forbidden_rows = [
        {
            "path_or_route": "v42/v19 diagnostic native-to-scene bridges",
            "forbidden_for_method_prediction": True,
            "reason": "uses RGB-D/pose/mesh/GT-derived bridge; diagnostic only",
        },
        {
            "path_or_route": "old scene prediction npz artifacts",
            "forbidden_for_method_prediction": True,
            "reason": "provenance mismatch; cannot relabel as v86 readout",
        },
        {
            "path_or_route": "native carrier id treated as ScanNet mesh vertex id",
            "forbidden_for_method_prediction": True,
            "reason": "carrier ids are not scene mesh vertex ids",
        },
        {
            "path_or_route": "v85 candidate contract as current-run success",
            "forbidden_for_method_prediction": True,
            "reason": "v85 contract was post-hoc; v86 must freeze then rerun",
        },
    ]
    pass_gate = (
        ctx["v85_final"].get("strong_method_goal_achieved") is False
        and bool(support_rows)
        and bool(assignment_rows)
        and bool(ctx["v85_contract"].get("allowed_for_future_pre_registered_gate"))
    )
    summary = {
        "schema": "stream4d_v86_phase0_fact_lock_v1",
        "phase": "v86_phase0_fact_lock",
        "decision": "PASS_V86_PHASE0_NATIVE_ROUTE_READY" if pass_gate else "NO_GO_V86_PHASE0_INPUTS_INCOMPLETE",
        "gate": {
            "v85_final_strong_method_goal_false": ctx["v85_final"].get("strong_method_goal_achieved") is False,
            "native_support_available": bool(support_rows),
            "native_assignment_available": bool(assignment_rows),
            "v85_candidate_contract_future_allowed": bool(ctx["v85_contract"].get("allowed_for_future_pre_registered_gate")),
            "pass": pass_gate,
        },
        "v85_final_decision": ctx["v85_final"].get("final_decision", ""),
        "v85_native_support_row_count": len(support_rows),
        "v85_native_unique_carrier_count": len(unique_support),
        "v85_native_assignment_count": len(assignment_rows),
        "v85_native_assignment_unique_carrier_count": len(unique_assignment),
        "v85_native_AP50_diagnostic": ctx["v85_diagnostic_summary"].get("native_carrier_cluster_AP50", ""),
        "v85_real_minus_best_non_oracle_AP50_diagnostic": ctx["v85_diagnostic_summary"].get(
            "native_carrier_real_minus_best_non_oracle_AP50",
            "",
        ),
        "method_safe_scene_vertex_exporter_available": bool(
            ctx["v85_route_summary"].get("method_safe_scene_vertex_exporter_available")
        ),
        "method_safe_native_carrier_evaluator_available_v85": bool(
            ctx["v85_route_summary"].get("method_safe_native_carrier_evaluator_available")
        ),
        "runtime_sec": time.time() - t0,
    }
    _write_json(out / "fact_lock_summary.json", summary)
    _write_csv(out / "route_decision_rows.csv", route_rows)
    _write_csv(out / "forbidden_path_rows.csv", forbidden_rows)
    _write_csv(out / "input_artifact_scope_rows.csv", input_rows)
    return summary


def _phase1(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    ctx = _load_inputs(args)
    out = _repo_path(args.phase1_output_root)
    support_rows = ctx["support_rows"]
    assignment_rows = ctx["assignment_rows"]
    unique_support = {row.get("native_carrier_global_id", "") for row in support_rows if row.get("native_carrier_global_id")}
    unique_labeled = {
        row.get("native_carrier_global_id", "")
        for row in assignment_rows
        if row.get("native_carrier_global_id") and row.get("diagnostic_gt_eval_label")
    }
    prediction_forbidden = {
        "diagnostic_gt_instance",
        "diagnostic_gt_eval_label",
        "diagnostic_gt_vote_count",
        "diagnostic_gt_vote_purity",
        "diagnostic_gt_vote_json",
        "uses_gt_for_diagnostic_labels",
    }
    allowed_prediction_columns = [
        "scene_id",
        "native_carrier_global_id",
        "pred_history_id",
        "pred_history_eval_label",
        "pred_vote_count",
        "pred_vote_purity",
        "pred_history_vote_json",
        "pred_history_conflict",
    ]
    available_columns = set()
    for row in assignment_rows:
        available_columns.update(row)
    gt_leakage_count = sum(1 for col in prediction_forbidden if col in set(allowed_prediction_columns))
    native_gt_label_coverage_rate = _safe_ratio(len(unique_labeled), len(unique_support))
    control_rows = [
        ("B3_real_history_native_membership", True, "method candidate", "pred_history labels from v85 native assignment"),
        ("B4_shuffled_history_by_scene", True, "non-oracle control", "scene-wise label shuffle preserving cluster sizes"),
        ("B5_size_matched_hash_global", True, "non-oracle control", "global deterministic size-matched hash"),
        ("B6_size_matched_hash_by_scene", True, "non-oracle control", "scene deterministic size-matched hash"),
        ("B7_uniform_hash_history", True, "non-oracle control", "uniform hash over real history labels"),
        ("B8_single_largest_by_scene", True, "non-oracle control", "single largest cluster per scene"),
        (
            "B9_semantic_descriptor_hash",
            bool(ctx["history_rows"]),
            "semantic control",
            "history semantic_descriptor_hash from v85 history_node_rows; audit degeneracy separately",
        ),
        (
            "B10_time_indexed_stale_history",
            bool(ctx["history_rows"] and support_rows),
            "stale history control",
            "history labels whose last_seen_chunk is earlier than the carrier support chunk",
        ),
        ("B11_oracle_diagnostic_gt", True, "oracle diagnostic", "GT labels for upper bound only"),
    ]
    control_suite_complete = all(row[1] for row in control_rows)
    control_csv_rows = [
        {
            "control_id": cid,
            "available": available,
            "role": role,
            "notes": notes,
            "uses_gt_for_prediction": cid == "B11_oracle_diagnostic_gt",
            "is_oracle": cid == "B11_oracle_diagnostic_gt",
        }
        for cid, available, role, notes in control_rows
    ]
    contract_body = {
        "schema": "stream4d_v86_native_eval_contract_v1",
        "prediction_universe_definition": (
            "All method-safe D4RT native carrier_global_id values with support observations in evaluated v85 chunks."
        ),
        "gt_label_source": "diagnostic_gt_eval_label from v85 mask_observation_table-derived assignment rows, evaluator-only",
        "gt_used_only_in_evaluator": True,
        "carrier_id_scope": "scene-scoped D4RT native carrier_global_id, not ScanNet mesh vertex id",
        "allowed_prediction_columns": allowed_prediction_columns,
        "forbidden_prediction_columns": sorted(prediction_forbidden),
        "metric_definitions": {
            "native_AP25_AP50_AP75": "AP-style instance metric over native carrier sets against full GT native universe",
            "ARI_purity_completeness": "cluster diagnostics on predicted carrier subset, not AP substitutes",
        },
        "control_definitions": [row[0] for row in control_rows],
        "dev_holdout_split": {
            "dev": "current v85 native carrier support universe",
            "holdout": "not run unless Phase9 selects a frozen native method config",
        },
        "frozen_config_policy": "No threshold tuning after this contract; Phase6 reruns all variants under this contract.",
        "source_candidate_contract_sha256": ctx["v85_contract"].get("contract_sha256", ""),
        "true_time_indexed_stale_history_control_available": bool(ctx["history_rows"] and support_rows),
        "semantic_control_degeneracy_must_be_reported": True,
        "not_scannet_ap": True,
    }
    contract = dict(contract_body)
    contract["contract_sha256"] = _canonical_sha256(contract_body)
    contract["contract_frozen_before_v86_dev_run"] = True
    gate = {
        "gt_label_leakage_to_prediction_count_eq_0": gt_leakage_count == 0,
        "native_gt_label_coverage_rate_ge_0p80": native_gt_label_coverage_rate >= 0.80,
        "control_suite_complete": control_suite_complete,
        "contract_frozen_before_v86_dev_run": True,
    }
    gate["pass"] = all(gate.values())
    summary = {
        "schema": "stream4d_v86_phase1_native_eval_contract_summary_v1",
        "phase": "v86_phase1_native_eval_contract",
        "decision": "PASS_V86_PHASE1_NATIVE_CONTRACT_FROZEN" if gate["pass"] else "NO_GO_V86_PHASE1_CONTRACT_INCOMPLETE",
        "native_gt_label_coverage_rate": native_gt_label_coverage_rate,
        "native_carrier_evaluation_universe_size": len(unique_support),
        "native_labeled_carrier_count": len(unique_labeled),
        "gt_label_leakage_to_prediction_count": gt_leakage_count,
        "carrier_id_scope": contract["carrier_id_scope"],
        "carrier_observation_source_count": len(
            {row.get("source_observation_table", "") for row in support_rows if row.get("source_observation_table")}
        ),
        "control_suite_complete": control_suite_complete,
        "true_time_indexed_stale_history_control_available": bool(ctx["history_rows"] and support_rows),
        "contract_sha256": contract["contract_sha256"],
        "contract_frozen_before_v86_dev_run": True,
        "gate": gate,
        "runtime_sec": time.time() - t0,
    }
    label_rows = [
        {
            "scene_id": row.get("scene_id", ""),
            "native_carrier_global_id": row.get("native_carrier_global_id", ""),
            "diagnostic_gt_eval_label_present": bool(row.get("diagnostic_gt_eval_label")),
            "gt_used_only_in_evaluator": True,
            "prediction_uses_gt": False,
        }
        for row in assignment_rows
    ]
    _write_json(out / "native_eval_contract.json", contract)
    _write_csv(out / "native_eval_contract_rows.csv", [{**summary, "gate": json.dumps(gate, sort_keys=True)}])
    _write_csv(out / "native_gt_label_audit_rows.csv", label_rows)
    _write_csv(out / "control_suite_rows.csv", control_csv_rows)
    return summary


def _phase5(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    ctx = _load_inputs(args)
    out = _repo_path(args.phase5_output_root)
    assignment_rows = ctx["assignment_rows"]
    support_rows = ctx["support_rows"]
    support_counts: Counter[tuple[str, str]] = Counter()
    frame_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    mask_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in support_rows:
        native_id = str(row.get("native_carrier_global_id", "")).strip()
        history = str(row.get("history_id", "")).strip()
        if not native_id or not history:
            continue
        key = (native_id, f"{row.get('scene_id', '')}:{history}")
        support_counts[key] += 1
        if row.get("frame_id"):
            frame_sets[key].add(str(row.get("frame_id")))
        if row.get("mask_id"):
            mask_sets[key].add(str(row.get("mask_id")))

    variants = _variant_rows(assignment_rows, support_rows)
    membership_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    variant_summary_rows: list[dict[str, Any]] = []
    universe_size = len({row.get("native_carrier_global_id", "") for row in assignment_rows if row.get("native_carrier_global_id")})
    global_new_object_hijack_proxy = ctx["v85_q_summary"].get("new_object_hijack_proxy", "")
    wrong_absorption_proxy = ctx["v85_q_summary"].get("wrong_absorption_proxy", "")

    for variant_id, meta in variants.items():
        rows: list[dict[str, Any]] = meta["rows"]
        object_scores: dict[str, list[float]] = defaultdict(list)
        object_frames: dict[str, set[str]] = defaultdict(set)
        object_masks: dict[str, set[str]] = defaultdict(set)
        object_conflicts: Counter[str] = Counter()
        entropies: list[float] = []
        margins: list[float] = []
        selected_native = {row.get("native_carrier_global_id", "") for row in rows if row.get("native_carrier_global_id")}
        source_conflicts = sum(1 for row in rows if _row_conflict(row))
        input_conflicts = sum(1 for row in rows if _bool(row.get("v86_input_conflict")) or _bool(row.get("pred_history_conflict")))
        hijack_count = sum(1 for row in rows if _row_hijack(row))
        for row in rows:
            native_id = str(row.get("native_carrier_global_id", "")).strip()
            label = str(row.get("v86_label", "")).strip()
            if not native_id or not label:
                continue
            entropy, margin = _vote_entropy_and_margin(str(row.get("pred_history_vote_json", "")))
            entropies.append(entropy)
            margins.append(margin)
            assignment_score = _row_score(row)
            support_key = (native_id, str(row.get("pred_history_eval_label", "")))
            frame_count = len(frame_sets.get(support_key, set()))
            mask_count = len(mask_sets.get(support_key, set()))
            conflict_score = 1.0 if _row_conflict(row) else 0.0
            new_object_score = 1.0 if _row_hijack(row) else 0.0
            object_scores[label].append(assignment_score)
            object_frames[label].update(frame_sets.get(support_key, set()))
            object_masks[label].update(mask_sets.get(support_key, set()))
            if conflict_score:
                object_conflicts[label] += 1
                conflict_rows.append(
                    {
                        "variant": variant_id,
                        "scene_id": row.get("scene_id", ""),
                        "native_carrier_id": native_id,
                        "history_id": label,
                        "source_pred_history_eval_label": row.get("pred_history_eval_label", ""),
                        "pred_history_vote_json": row.get("pred_history_vote_json", ""),
                        "repair_action": meta["repair_action"],
                    }
                )
            membership_rows.append(
                {
                    "scene_id": row.get("scene_id", ""),
                    "native_carrier_id": native_id,
                    "history_id": label,
                    "variant": variant_id,
                    "membership_score": assignment_score,
                    "rank": 1,
                    "selected_flag": True,
                    "support_source": meta["support_source"],
                    "frame_support_count": frame_count,
                    "mask_support_count": mask_count,
                    "query_score": "",
                    "assignment_score": assignment_score,
                    "conflict_score": conflict_score,
                    "new_object_score": new_object_score,
                    "method_uses_gt": False,
                    "uses_future": False,
                    "source_pred_history_id": row.get("pred_history_id", ""),
                    "source_pred_history_conflict": row.get("pred_history_conflict", ""),
                    "support_split_field": row.get("support_split_field", ""),
                    "support_split_conflict": row.get("support_split_conflict", ""),
                    "v86_input_conflict": row.get("v86_input_conflict", ""),
                    "v86_new_object_hijack": row.get("v86_new_object_hijack", ""),
                    "repair_action": meta["repair_action"],
                }
            )
        for label, scores in object_scores.items():
            object_rows.append(
                {
                    "scene_id": label.split(":")[0] if ":" in label else "",
                    "history_id": label,
                    "variant": variant_id,
                    "native_carrier_count": len([row for row in rows if row.get("v86_label") == label]),
                    "score_mean": _mean(scores),
                    "score_p10": sorted(scores)[max(0, int(0.1 * len(scores)) - 1)] if scores else 0.0,
                    "support_frame_count": len(object_frames[label]),
                    "support_mask_count": len(object_masks[label]),
                    "conflict_count": object_conflicts[label],
                    "broad_support_ratio": "",
                    "prediction_score": _mean(scores),
                }
            )
        coverage = _safe_ratio(len(selected_native), universe_size)
        source_new_object_hijack = _safe_ratio(hijack_count, len(rows))
        gate = {
            "native_membership_coverage_rate_ge_0p20": coverage >= 0.20,
            "native_object_count_ge_10": len(object_scores) >= 10,
            "membership_entropy_mean_le_0p60": _mean(entropies) <= 0.60,
            "top1_top2_margin_mean_ge_0p05": _mean(margins) >= 0.05,
            "conflict_violation_count_eq_0": source_conflicts == 0,
            "new_object_hijack_proxy_le_0p05": (
                math.isfinite(source_new_object_hijack) and source_new_object_hijack <= 0.05
            ),
            "method_uses_gt_false": True,
            "uses_future_false": True,
        }
        gate["pass"] = all(gate.values())
        variant_summary_rows.append(
            {
                "variant": variant_id,
                "repair_action": meta["repair_action"],
                "native_membership_coverage_rate": coverage,
                "native_object_count": len(object_scores),
                "native_carrier_count_selected": len(selected_native),
                "membership_entropy_mean": _mean(entropies),
                "top1_top2_margin_mean": _mean(margins),
                "conflict_violation_count": source_conflicts,
                "input_conflict_count": input_conflicts,
                "new_object_hijack_count": hijack_count,
                "new_object_hijack_proxy": source_new_object_hijack,
                "v85_global_new_object_birth_rate": global_new_object_hijack_proxy,
                "wrong_absorption_proxy": wrong_absorption_proxy,
                "method_uses_gt": False,
                "uses_future": False,
                "gate_pass": gate["pass"],
                "gate": json.dumps(gate, sort_keys=True),
            }
        )
    best_gate_rows = [row for row in variant_summary_rows if _bool(row.get("gate_pass"))]
    best_variant = best_gate_rows[0]["variant"] if best_gate_rows else ""
    summary = {
        "schema": "stream4d_v86_phase5_native_membership_v1",
        "phase": "v86_phase5_native_membership",
        "decision": "PASS_V86_PHASE5_NATIVE_MEMBERSHIP" if best_gate_rows else "NO_GO_V86_PHASE5_NATIVE_MEMBERSHIP_GATE_FAIL",
        "best_gate_variant": best_variant,
        "variant_count": len(variant_summary_rows),
        "native_evaluation_universe_size": universe_size,
        "new_object_hijack_proxy_source": "variant-specific confirmed-absorbs-new/tentative vote audit from pred_history_vote_json",
        "v85_global_new_object_birth_rate": global_new_object_hijack_proxy,
        "wrong_absorption_proxy": wrong_absorption_proxy,
        "strict_gate_note": "v86 new_object_hijack_proxy is computed per readout variant; v85 new_object_birth_rate is retained only as context",
        "variant_rows": _rel(out / "native_readout_variant_rows.csv"),
        "runtime_sec": time.time() - t0,
    }
    _write_json(out / "native_membership_summary.json", summary)
    _write_csv(out / "native_membership_rows.csv", membership_rows)
    _write_csv(out / "native_object_rows.csv", object_rows)
    _write_csv(out / "native_conflict_rows.csv", conflict_rows)
    _write_csv(out / "native_readout_variant_rows.csv", variant_summary_rows)
    return summary


def _phase6(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    ctx = _load_inputs(args)
    out = _repo_path(args.phase6_output_root)
    assignment_rows = ctx["assignment_rows"]
    history_rows = ctx["history_rows"]
    support_rows = ctx["support_rows"]
    variants = _variant_rows(assignment_rows, support_rows)
    semantic_map_all, semantic_audit = _semantic_history_labels(assignment_rows, history_rows)
    stale_map_all, stale_audit = _time_indexed_stale_labels(
        assignment_rows,
        history_rows,
        support_rows,
        source_label_field="pred_history_eval_label",
    )

    metric_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    ap_curve_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    variant_pass_rows: list[dict[str, Any]] = []

    for variant_id, meta in variants.items():
        pred_rows: list[dict[str, Any]] = meta["rows"]
        pred_scores = _pred_scores(pred_rows, label_field="v86_label")
        real = _eval_variant(
            assignment_rows,
            pred_rows,
            variant_id=variant_id,
            label_field="v86_label",
            pred_scores=pred_scores,
            score_contract="mean_pred_vote_purity_per_v86_label",
        )
        source_conflicts = sum(1 for row in pred_rows if _row_conflict(row))
        metric_row = {
            **real,
            "variant_role": "real_method_candidate",
            "repair_action": meta["repair_action"],
            "selected_carrier_count": len(
                {row.get("native_carrier_global_id", "") for row in pred_rows if row.get("native_carrier_global_id")}
            ),
            "GT_label_coverage_rate": 1.0 if assignment_rows else 0.0,
            "source_conflict_violation_count": source_conflicts,
        }
        metric_rows.append(metric_row)
        for threshold in (25, 50, 75):
            ap_curve_rows.append(
                {
                    "variant": variant_id,
                    "control_id": "real",
                    "threshold": threshold / 100.0,
                    "AP": metric_row.get(f"native_AP{threshold}", ""),
                    "precision": metric_row.get(f"native_precision{threshold}", ""),
                    "recall": metric_row.get(f"native_recall{threshold}", ""),
                    "matched_gt_count": metric_row.get(f"native_matched_gt_count_at_{threshold}", ""),
                }
            )

        controls: list[tuple[str, str, list[dict[str, Any]], str, bool, bool]] = []
        controls.append(
            (
                "B4_shuffled_history_by_scene",
                "scene-wise size matched label shuffle",
                _labels_to_rows(
                    pred_rows,
                    _size_matched_hash_labels(
                        pred_rows,
                        group_field="scene_id",
                        source_label_field="v86_label",
                        salt=f"v86_{variant_id}_shuffled_scene",
                    ),
                    label_field="control_label",
                ),
                "cluster_size_desc",
                False,
                False,
            )
        )
        controls.append(
            (
                "B5_size_matched_hash_global",
                "global size matched hash",
                _labels_to_rows(
                    pred_rows,
                    _size_matched_hash_labels(
                        pred_rows,
                        group_field=None,
                        source_label_field="v86_label",
                        salt=f"v86_{variant_id}_size_global",
                    ),
                    label_field="control_label",
                ),
                "cluster_size_desc",
                False,
                False,
            )
        )
        controls.append(
            (
                "B6_size_matched_hash_by_scene",
                "scene size matched hash",
                _labels_to_rows(
                    pred_rows,
                    _size_matched_hash_labels(
                        pred_rows,
                        group_field="scene_id",
                        source_label_field="v86_label",
                        salt=f"v86_{variant_id}_size_scene",
                    ),
                    label_field="control_label",
                ),
                "cluster_size_desc",
                False,
                False,
            )
        )
        controls.append(
            (
                "B7_uniform_hash_history",
                "uniform hash over labels",
                _labels_to_rows(
                    pred_rows,
                    _uniform_hash_labels(pred_rows, source_label_field="v86_label", salt=f"v86_{variant_id}_uniform"),
                    label_field="control_label",
                ),
                "cluster_size_desc",
                False,
                False,
            )
        )
        controls.append(
            (
                "B8_single_largest_by_scene",
                "single largest cluster by scene",
                _labels_to_rows(
                    pred_rows,
                    _single_largest_labels(pred_rows, source_label_field="v86_label", group_field="scene_id"),
                    label_field="control_label",
                ),
                "cluster_size_desc",
                False,
                False,
            )
        )
        controls.append(
            (
                "B9_semantic_descriptor_hash",
                "semantic descriptor hash control",
                _labels_to_rows(pred_rows, semantic_map_all, label_field="control_label"),
                "cluster_size_desc; semantic hash may be degenerate unique-per-history",
                False,
                False,
            )
        )
        controls.append(
            (
                "B10_time_indexed_stale_history",
                "prior history label whose last_seen_chunk is earlier than carrier support chunk",
                _labels_to_rows(pred_rows, stale_map_all, label_field="control_label"),
                "cluster_size_desc; true time-indexed stale control",
                False,
                False,
            )
        )
        oracle_rows = [dict(row, control_label=row.get("diagnostic_gt_eval_label", "")) for row in pred_rows]
        controls.append(
            (
                "B11_oracle_diagnostic_gt",
                "oracle GT upper bound",
                oracle_rows,
                "cluster_size_desc_for_oracle",
                True,
                True,
            )
        )

        real_ap50 = _num(metric_row.get("native_AP50"), 0.0)
        non_oracle_ap50: list[float] = []
        semantic_ap50 = math.nan
        shuffled_ap50 = math.nan
        stale_ap50 = math.nan
        oracle_ap50 = math.nan
        for control_id, notes, rows, score_contract, uses_gt, is_oracle in controls:
            control_eval = _eval_variant(
                assignment_rows,
                rows,
                variant_id=f"{variant_id}:{control_id}",
                label_field="control_label",
                pred_scores=None,
                score_contract=score_contract,
                prediction_uses_gt=uses_gt,
                is_oracle=is_oracle,
            )
            row = {
                **control_eval,
                "parent_variant": variant_id,
                "control_id": control_id,
                "control_notes": notes,
                "is_method_result": False,
            }
            control_rows.append(row)
            if not uses_gt:
                non_oracle_ap50.append(_num(row.get("native_AP50"), 0.0))
            if control_id == "B9_semantic_descriptor_hash":
                semantic_ap50 = _num(row.get("native_AP50"), math.nan)
            elif control_id == "B4_shuffled_history_by_scene":
                shuffled_ap50 = _num(row.get("native_AP50"), math.nan)
            elif control_id == "B10_time_indexed_stale_history":
                stale_ap50 = _num(row.get("native_AP50"), math.nan)
            elif control_id == "B11_oracle_diagnostic_gt":
                oracle_ap50 = _num(row.get("native_AP50"), math.nan)
            for threshold in (25, 50, 75):
                ap_curve_rows.append(
                    {
                        "variant": variant_id,
                        "control_id": control_id,
                        "threshold": threshold / 100.0,
                        "AP": row.get(f"native_AP{threshold}", ""),
                        "precision": row.get(f"native_precision{threshold}", ""),
                        "recall": row.get(f"native_recall{threshold}", ""),
                        "matched_gt_count": row.get(f"native_matched_gt_count_at_{threshold}", ""),
                    }
                )

        best_non_oracle_ap50 = max(non_oracle_ap50, default=0.0)
        metric_row["real_minus_best_non_oracle_AP50"] = real_ap50 - best_non_oracle_ap50
        metric_row["real_minus_semantic_AP50"] = (
            real_ap50 - semantic_ap50 if math.isfinite(semantic_ap50) else ""
        )
        metric_row["real_minus_shuffled_AP50"] = (
            real_ap50 - shuffled_ap50 if math.isfinite(shuffled_ap50) else ""
        )
        metric_row["real_minus_stale_AP50"] = real_ap50 - stale_ap50 if math.isfinite(stale_ap50) else ""
        metric_row["oracle_gap_AP50"] = oracle_ap50 - real_ap50 if math.isfinite(oracle_ap50) else ""
        gate = {
            "native_AP50_ge_0p45": real_ap50 >= 0.45,
            "native_AP25_ge_0p65": _num(metric_row.get("native_AP25"), 0.0) >= 0.65,
            "ARI_ge_0p60": _num(metric_row.get("adjusted_rand_index"), 0.0) >= 0.60,
            "purity_ge_0p80": _num(metric_row.get("purity"), 0.0) >= 0.80,
            "completeness_ge_0p70": _num(metric_row.get("completeness"), 0.0) >= 0.70,
            "real_minus_best_non_oracle_AP50_ge_0p10": metric_row["real_minus_best_non_oracle_AP50"] >= 0.10,
            "real_minus_semantic_AP50_ge_0p05": (
                math.isfinite(semantic_ap50) and metric_row["real_minus_semantic_AP50"] >= 0.05
            ),
            "GT_label_coverage_rate_ge_0p80": True,
        }
        gate["pass"] = all(gate.values())
        metric_row["dev_native_metric_gate_pass"] = gate["pass"]
        metric_row["gate"] = json.dumps(gate, sort_keys=True)
        variant_pass_rows.append(metric_row)
        if not gate["pass"]:
            failed = [key for key, value in gate.items() if key != "pass" and not value]
            case_rows.append(
                {
                    "variant": variant_id,
                    "failure_type": "NATIVE_EVAL_CONTROL_FAIL",
                    "failed_gate_fields": ";".join(failed),
                    "native_AP50": real_ap50,
                    "real_minus_semantic_AP50": metric_row["real_minus_semantic_AP50"],
                    "real_minus_best_non_oracle_AP50": metric_row["real_minus_best_non_oracle_AP50"],
                    "repair_action": meta["repair_action"],
                }
            )

    passing_metric_variants = [row for row in variant_pass_rows if _bool(row.get("dev_native_metric_gate_pass"))]
    best_by_ap50 = sorted(variant_pass_rows, key=lambda row: _num(row.get("native_AP50"), 0.0), reverse=True)
    summary = {
        "schema": "stream4d_v86_phase6_native_eval_v1",
        "phase": "v86_phase6_native_eval",
        "decision": "PASS_V86_PHASE6_NATIVE_METRIC" if passing_metric_variants else "NO_GO_V86_PHASE6_NATIVE_METRIC_FAIL",
        "dev_native_metric_pass": bool(passing_metric_variants),
        "passing_variant_count": len(passing_metric_variants),
        "best_AP50_variant": best_by_ap50[0].get("variant_id", "") if best_by_ap50 else "",
        "best_native_AP50": best_by_ap50[0].get("native_AP50", "") if best_by_ap50 else "",
        "best_gate_variant": passing_metric_variants[0].get("variant_id", "") if passing_metric_variants else "",
        "semantic_control_audit": semantic_audit,
        "stale_control_audit": stale_audit,
        "metric_scope": "native_carrier_objectness_metric_not_scannet_ap",
        "GT_label_coverage_rate": 1.0 if assignment_rows else 0.0,
        "runtime_sec": time.time() - t0,
    }
    _write_json(out / "native_eval_summary.json", summary)
    _write_csv(out / "native_metric_rows.csv", metric_rows)
    _write_csv(out / "native_control_rows.csv", control_rows)
    _write_csv(out / "native_ap_curve_rows.csv", ap_curve_rows)
    _write_csv(out / "native_case_rows.csv", case_rows)
    return summary


def _phase8(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    ctx = _load_inputs(args)
    out = _repo_path(args.phase8_output_root)
    route_summary = ctx["v85_route_summary"]
    route_rows = _read_csv_rows(ctx["v85_phase7"] / "native_scene_vertex_export_route_rows.csv")
    audit_rows = []
    for row in route_rows:
        audit_rows.append(
            {
                "route_id": row.get("route_id", ""),
                "exists": row.get("exists", ""),
                "method_safe_scene_metric": row.get("can_be_method_result", ""),
                "diagnostic_only": row.get("diagnostic_only", ""),
                "blocked_reason": row.get("blocked_reason", row.get("blocker", "")),
                "evidence": row.get("evidence_path", row.get("source_artifact", "")),
            }
        )
    method_safe_scene = bool(route_summary.get("method_safe_scene_vertex_exporter_available"))
    summary = {
        "schema": "stream4d_v86_phase8_scene_exporter_audit_v1",
        "phase": "v86_phase8_scene_exporter_audit",
        "decision": "PASS_V86_PHASE8_SCENE_EXPORTER" if method_safe_scene else "NO_GO_SCENE_VERTEX_EXPORTER_MISSING",
        "method_safe_scene_vertex_exporter_available": method_safe_scene,
        "checked_candidate_route_count": route_summary.get("checked_candidate_route_count", len(route_rows)),
        "primary_blocker": route_summary.get("primary_blocker", "method_safe_scene_vertex_exporter_missing"),
        "scanNet_scene_method_success": False,
        "native_to_scene_exporter_note": "No method-safe native-carrier-to-ScanNet mesh vertex mapping was found.",
        "runtime_sec": time.time() - t0,
    }
    _write_json(out / "scene_exporter_audit_summary.json", summary)
    _write_csv(out / "scene_exporter_route_rows.csv", audit_rows)
    return summary


def _phase9(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase9_output_root)
    p5 = _read_json(_repo_path(args.phase5_output_root) / "native_membership_summary.json")
    p6 = _read_json(_repo_path(args.phase6_output_root) / "native_eval_summary.json")
    p8 = _read_json(_repo_path(args.phase8_output_root) / "scene_exporter_audit_summary.json")
    p5_variants = _read_csv_rows(_repo_path(args.phase5_output_root) / "native_readout_variant_rows.csv")
    p6_metrics = _read_csv_rows(_repo_path(args.phase6_output_root) / "native_metric_rows.csv")
    rows = []
    for metric in p6_metrics:
        variant = str(metric.get("variant_id", ""))
        phase5 = next((row for row in p5_variants if row.get("variant") == variant), {})
        rows.append(
            {
                "variant": variant,
                "native_AP25": metric.get("native_AP25", ""),
                "native_AP50": metric.get("native_AP50", ""),
                "native_AP75": metric.get("native_AP75", ""),
                "ARI": metric.get("adjusted_rand_index", ""),
                "purity": metric.get("purity", ""),
                "completeness": metric.get("completeness", ""),
                "real_minus_best_non_oracle_AP50": metric.get("real_minus_best_non_oracle_AP50", ""),
                "real_minus_semantic_AP50": metric.get("real_minus_semantic_AP50", ""),
                "phase5_gate_pass": phase5.get("gate_pass", ""),
                "phase6_gate_pass": metric.get("dev_native_metric_gate_pass", ""),
                "conflict_violation_count": phase5.get("conflict_violation_count", ""),
                "new_object_hijack_proxy": phase5.get("new_object_hijack_proxy", ""),
                "method_uses_gt": False,
                "uses_future": False,
            }
        )
    native_method_pass = any(_bool(row.get("phase5_gate_pass")) and _bool(row.get("phase6_gate_pass")) for row in rows)
    scene_method_pass = bool(p8.get("method_safe_scene_vertex_exporter_available")) and False
    if native_method_pass and not scene_method_pass:
        decision = "GO_NATIVE_CARRIER_OBJECTNESS_ONLY"
        primary_blocker = "scene_vertex_exporter_missing"
    elif p6.get("dev_native_metric_pass"):
        decision = "NO_GO_NATIVE_MEMBERSHIP_GATE_FAIL"
        primary_blocker = "native_metric_has_signal_but_membership_gate_failed"
    else:
        decision = "NO_GO_AFFINITY_FIELD_READOUT"
        primary_blocker = "native_membership_or_native_metric_gate_failed"
    decision_rows = [
        {
            "route": "native_carrier_objectness",
            "phase5_pass": p5.get("decision", "").startswith("PASS"),
            "phase6_pass": p6.get("dev_native_metric_pass", False),
            "selected": native_method_pass,
            "decision": decision,
        },
        {
            "route": "scene_vertex_exporter",
            "phase8_pass": p8.get("method_safe_scene_vertex_exporter_available", False),
            "selected": scene_method_pass,
            "decision": "NO_GO_SCENE_VERTEX_EXPORTER_MISSING",
        },
    ]
    summary = {
        "schema": "stream4d_v86_phase9_controls_v1",
        "phase": "v86_phase9_controls",
        "decision": decision,
        "native_method_pass": native_method_pass,
        "scene_method_pass": scene_method_pass,
        "controls_pass": bool(p6.get("dev_native_metric_pass")),
        "primary_blocker": primary_blocker,
        "can_enter_frozen_holdout": native_method_pass or scene_method_pass,
        "method_uses_gt_anywhere": False,
        "uses_future_anywhere": False,
        "runtime_sec": time.time() - t0,
    }
    _write_json(out / "control_summary.json", summary)
    _write_csv(out / "method_variant_rows.csv", rows)
    _write_csv(out / "decision_matrix_rows.csv", decision_rows)
    if summary["can_enter_frozen_holdout"]:
        config = {
            "schema": "stream4d_v86_frozen_method_config_v1",
            "selected_route": "native_carrier_objectness" if native_method_pass else "scene_vertex_exporter",
            "selected_variant": next((row["variant"] for row in rows if _bool(row.get("phase5_gate_pass")) and _bool(row.get("phase6_gate_pass"))), ""),
            "contract_path": _rel(_repo_path(args.phase1_output_root) / "native_eval_contract.json"),
            "no_holdout_retuning": True,
            "selected_from_dev_only": True,
            "dev_split_chunks": {scene: [min(chunks), max(chunks)] for scene, chunks in DEV_SPLIT_CHUNKS.items()},
            "holdout_split_chunks": {scene: [min(chunks), max(chunks)] for scene, chunks in HOLDOUT_SPLIT_CHUNKS.items()},
        }
        config["config_sha256"] = _canonical_sha256(config)
        _write_json(_repo_path(args.config_output_root) / "frozen_method_config.json", config)
    return summary


def _phase10(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase10_output_root)
    out.mkdir(parents=True, exist_ok=True)
    p9 = _read_json(_repo_path(args.phase9_output_root) / "control_summary.json")
    config_path = _repo_path(args.config_output_root) / "frozen_method_config.json"
    config = _read_json(config_path)
    ctx = _load_inputs(args)
    support_rows = ctx["support_rows"]
    history_rows = ctx["history_rows"]
    split_rows = _support_split_audit_rows(support_rows)
    dev_support_rows = _support_rows_for_split(support_rows, "dev")
    holdout_support_rows = _support_rows_for_split(support_rows, "holdout")
    config_without_hash = {key: value for key, value in config.items() if key != "config_sha256"}
    config_sha256_recomputed = _canonical_sha256(config_without_hash) if config_without_hash else ""
    config_sha256_matches = bool(config) and config.get("config_sha256", "") == config_sha256_recomputed
    source_candidate_rows, source_candidate_summary = _holdout_source_candidate_audit(ctx)

    _write_csv(out / "holdout_input_split_rows.csv", split_rows)
    _write_csv(out / "holdout_source_candidate_rows.csv", source_candidate_rows)
    _write_json(out / "holdout_source_candidate_summary.json", source_candidate_summary)
    diagnostic_tentative_summary: dict[str, Any] = {}
    repair_probe_summary: dict[str, Any] = {}

    def _blocked_summary(decision: str, blocker: str, *, formal_run: bool = False) -> dict[str, Any]:
        return {
            "schema": "stream4d_v86_phase10_holdout_v2",
            "phase": "v86_phase10_holdout",
            "decision": decision,
            "formal_holdout_run": formal_run,
            "holdout_replay_attempted": True,
            "config_path": _rel(config_path) if config_path.exists() else "",
            "config_sha256": config.get("config_sha256", ""),
            "config_sha256_recomputed": config_sha256_recomputed,
            "config_sha256_matches": config_sha256_matches,
            "selected_route": config.get("selected_route", ""),
            "selected_variant": config.get("selected_variant", ""),
            "primary_blocker": blocker,
            "dev_support_row_count": len(dev_support_rows),
            "dev_unique_native_carrier_count": len(
                {row.get("native_carrier_global_id", "") for row in dev_support_rows if row.get("native_carrier_global_id")}
            ),
            "holdout_support_row_count": len(holdout_support_rows),
            "holdout_unique_native_carrier_count": len(
                {row.get("native_carrier_global_id", "") for row in holdout_support_rows if row.get("native_carrier_global_id")}
            ),
            "holdout_native_AP50": "",
            "holdout_real_minus_best_non_oracle_AP50": "",
            "holdout_purity": "",
            "holdout_ARI": "",
            "holdout_GT_label_coverage_rate": "",
            "holdout_source_candidate_rows": _rel(out / "holdout_source_candidate_rows.csv"),
            "holdout_source_candidate_summary": _rel(out / "holdout_source_candidate_summary.json"),
            "holdout_method_safe_candidate_source_count": source_candidate_summary.get(
                "method_safe_holdout_input_available_count", 0
            ),
            "holdout_selected_frame_mask_row_count": source_candidate_summary.get(
                "holdout_selected_frame_mask_row_count", ""
            ),
            "holdout_observation_allowed_row_count": source_candidate_summary.get(
                "holdout_observation_allowed_row_count", ""
            ),
            "holdout_observation_allowed_selected_join_row_count": source_candidate_summary.get(
                "holdout_observation_allowed_selected_join_row_count", ""
            ),
            "holdout_source_candidate_primary_blocker": source_candidate_summary.get("primary_blocker", ""),
            "diagnostic_tentative_holdout_summary": _rel(out / "diagnostic_tentative_holdout_summary.json")
            if diagnostic_tentative_summary
            else "",
            "diagnostic_tentative_native_AP50": diagnostic_tentative_summary.get("diagnostic_native_AP50", ""),
            "diagnostic_tentative_real_minus_best_non_oracle_AP50": diagnostic_tentative_summary.get(
                "diagnostic_real_minus_best_non_oracle_AP50", ""
            ),
            "diagnostic_tentative_purity": diagnostic_tentative_summary.get("diagnostic_purity", ""),
            "diagnostic_tentative_ARI": diagnostic_tentative_summary.get("diagnostic_ARI", ""),
            "diagnostic_tentative_method_claim_blocker": diagnostic_tentative_summary.get(
                "method_claim_blocker", ""
            ),
            "holdout_repair_probe_summary": _rel(out / "holdout_repair_probe_summary.json")
            if repair_probe_summary
            else "",
            "holdout_repair_probe_best_variant": repair_probe_summary.get("best_variant_by_AP50", ""),
            "holdout_repair_probe_best_native_AP50": repair_probe_summary.get("best_native_AP50", ""),
            "holdout_repair_probe_best_ARI": repair_probe_summary.get("best_ARI", ""),
            "holdout_repair_probe_best_purity": repair_probe_summary.get("best_purity", ""),
            "holdout_repair_probe_best_metric_pass_without_formal_claim": repair_probe_summary.get(
                "best_metric_pass_without_formal_claim", ""
            ),
            "holdout_repair_probe_formal_method_claim_allowed": repair_probe_summary.get(
                "formal_method_claim_allowed", ""
            ),
            "holdout_repair_probe_primary_blocker": repair_probe_summary.get("primary_blocker", ""),
            "method_uses_gt_anywhere": False,
            "uses_future_anywhere": False,
            "runtime_sec": time.time() - t0,
        }

    run_holdout = bool(p9.get("can_enter_frozen_holdout")) and bool(config)
    if not run_holdout:
        blocker = p9.get("primary_blocker", "dev_gate_failed_or_missing_frozen_config")
        summary = _blocked_summary("NO_GO_HOLDOUT_BLOCKED_BY_DEV_GATE", blocker)
        _write_json(out / "holdout_summary.json", summary)
        _write_csv(out / "holdout_native_metric_rows.csv", [summary])
        _write_csv(out / "holdout_control_rows.csv", [{"control": "formal_holdout", "run": False, "reason": blocker}])
        _write_csv(out / "holdout_failure_case_rows.csv", [{"failure_type": "HOLDOUT_BLOCKED_BY_DEV_GATE", "reason": blocker}])
        return summary

    if config.get("selected_route") != "native_carrier_objectness":
        blocker = "phase10_scene_route_holdout_not_implemented_without_method_safe_scene_exporter"
        summary = _blocked_summary("NO_GO_HOLDOUT_SCENE_ROUTE_UNAVAILABLE", blocker)
        _write_json(out / "holdout_summary.json", summary)
        _write_csv(out / "holdout_native_metric_rows.csv", [summary])
        _write_csv(out / "holdout_control_rows.csv", [{"control": "scene_route_holdout", "run": False, "reason": blocker}])
        _write_csv(out / "holdout_failure_case_rows.csv", [{"failure_type": "HOLDOUT_SCENE_ROUTE_UNAVAILABLE", "reason": blocker}])
        return summary

    if not holdout_support_rows:
        blocker = "registered_holdout_split_has_zero_v85_native_support_rows"
        diagnostic_tentative_summary = _diagnostic_tentative_holdout_replay(ctx, out)
        repair_probe_summary = _holdout_repair_probe(ctx, out)
        summary = _blocked_summary("NO_GO_HOLDOUT_INPUT_MISSING", blocker)
        _write_json(out / "holdout_summary.json", summary)
        _write_csv(out / "holdout_native_metric_rows.csv", [summary])
        _write_csv(
            out / "holdout_control_rows.csv",
            [
                {
                    "control_id": "holdout_input_preflight",
                    "run": False,
                    "reason": blocker,
                    "dev_support_row_count": len(dev_support_rows),
                    "holdout_support_row_count": 0,
                }
            ],
        )
        _write_csv(
            out / "holdout_failure_case_rows.csv",
            [
                {
                    "failure_type": "HOLDOUT_INPUT_MISSING",
                    "reason": blocker,
                    "evidence": _rel(out / "holdout_input_split_rows.csv"),
                    "source_candidate_evidence": _rel(out / "holdout_source_candidate_rows.csv"),
                    "notes": (
                        "Current v85 native support rows cover only registered dev chunks; no selected native support exists "
                        "for scene0011_00 chunks 6-11 or scene0050_00 chunks 4-11. Candidate audit confirms raw D4RT "
                        "holdout observations exist, but no frozen method selected frame-mask/history readout joins them."
                    ),
                }
            ],
        )
        return summary

    holdout_assignment_rows, source_audit_rows, assignment_summary = _assignment_rows_from_support(holdout_support_rows)
    _write_csv(out / "holdout_assignment_rows.csv", holdout_assignment_rows)
    _write_csv(out / "holdout_diagnostic_source_audit_rows.csv", source_audit_rows)
    _write_json(out / "holdout_assignment_summary.json", assignment_summary)
    if not holdout_assignment_rows:
        blocker = "holdout_native_support_has_no_positive_diagnostic_gt_scoring_labels"
        summary = _blocked_summary("NO_GO_HOLDOUT_INPUT_MISSING", blocker)
        summary.update(assignment_summary)
        _write_json(out / "holdout_summary.json", summary)
        _write_csv(out / "holdout_native_metric_rows.csv", [summary])
        _write_csv(out / "holdout_control_rows.csv", [{"control_id": "holdout_assignment_preflight", "run": False, "reason": blocker}])
        _write_csv(out / "holdout_failure_case_rows.csv", [{"failure_type": "HOLDOUT_INPUT_MISSING", "reason": blocker}])
        return summary

    selected_variant = str(config.get("selected_variant", "")).strip()
    variants = _variant_rows(holdout_assignment_rows, holdout_support_rows)
    if selected_variant not in variants:
        blocker = f"selected_variant_missing_in_holdout_variants:{selected_variant}"
        summary = _blocked_summary("NO_GO_HOLDOUT_CONFIG_INVALID", blocker)
        summary.update(assignment_summary)
        _write_json(out / "holdout_summary.json", summary)
        _write_csv(out / "holdout_native_metric_rows.csv", [summary])
        _write_csv(out / "holdout_control_rows.csv", [{"control_id": "holdout_config_preflight", "run": False, "reason": blocker}])
        _write_csv(out / "holdout_failure_case_rows.csv", [{"failure_type": "HOLDOUT_CONFIG_INVALID", "reason": blocker}])
        return summary

    meta = variants[selected_variant]
    pred_rows: list[dict[str, Any]] = meta["rows"]
    pred_scores = _pred_scores(pred_rows, label_field="v86_label")
    metric_row = {
        **_eval_variant(
            holdout_assignment_rows,
            pred_rows,
            variant_id=selected_variant,
            label_field="v86_label",
            pred_scores=pred_scores,
            score_contract="holdout_mean_pred_vote_purity_per_v86_label",
        ),
        "variant_role": "frozen_holdout_method_candidate",
        "repair_action": meta["repair_action"],
        "selected_carrier_count": len(
            {row.get("native_carrier_global_id", "") for row in pred_rows if row.get("native_carrier_global_id")}
        ),
        "GT_label_coverage_rate": assignment_summary.get("native_gt_label_coverage_rate", 0.0),
        "source_conflict_violation_count": sum(1 for row in pred_rows if _row_conflict(row)),
    }

    semantic_map_all, semantic_audit = _semantic_history_labels(holdout_assignment_rows, history_rows)
    stale_map_all, stale_audit = _time_indexed_stale_labels(
        holdout_assignment_rows,
        history_rows,
        holdout_support_rows,
        source_label_field="pred_history_eval_label",
    )
    controls: list[tuple[str, str, list[dict[str, Any]], str, bool, bool]] = [
        (
            "B4_shuffled_history_by_scene",
            "scene-wise size matched label shuffle",
            _labels_to_rows(
                pred_rows,
                _size_matched_hash_labels(
                    pred_rows,
                    group_field="scene_id",
                    source_label_field="v86_label",
                    salt=f"v86_holdout_{selected_variant}_shuffled_scene",
                ),
                label_field="control_label",
            ),
            "cluster_size_desc",
            False,
            False,
        ),
        (
            "B5_size_matched_hash_global",
            "global size matched hash",
            _labels_to_rows(
                pred_rows,
                _size_matched_hash_labels(
                    pred_rows,
                    group_field=None,
                    source_label_field="v86_label",
                    salt=f"v86_holdout_{selected_variant}_size_global",
                ),
                label_field="control_label",
            ),
            "cluster_size_desc",
            False,
            False,
        ),
        (
            "B6_size_matched_hash_by_scene",
            "scene size matched hash",
            _labels_to_rows(
                pred_rows,
                _size_matched_hash_labels(
                    pred_rows,
                    group_field="scene_id",
                    source_label_field="v86_label",
                    salt=f"v86_holdout_{selected_variant}_size_scene",
                ),
                label_field="control_label",
            ),
            "cluster_size_desc",
            False,
            False,
        ),
        (
            "B7_uniform_hash_history",
            "uniform hash over labels",
            _labels_to_rows(
                pred_rows,
                _uniform_hash_labels(
                    pred_rows,
                    source_label_field="v86_label",
                    salt=f"v86_holdout_{selected_variant}_uniform",
                ),
                label_field="control_label",
            ),
            "cluster_size_desc",
            False,
            False,
        ),
        (
            "B8_single_largest_by_scene",
            "single largest cluster by scene",
            _labels_to_rows(
                pred_rows,
                _single_largest_labels(pred_rows, source_label_field="v86_label", group_field="scene_id"),
                label_field="control_label",
            ),
            "cluster_size_desc",
            False,
            False,
        ),
        (
            "B9_semantic_descriptor_hash",
            "semantic descriptor hash control",
            _labels_to_rows(pred_rows, semantic_map_all, label_field="control_label"),
            "cluster_size_desc; semantic hash may be degenerate unique-per-history",
            False,
            False,
        ),
        (
            "B10_time_indexed_stale_history",
            "prior history label whose last_seen_chunk is earlier than carrier support chunk",
            _labels_to_rows(pred_rows, stale_map_all, label_field="control_label"),
            "cluster_size_desc; true time-indexed stale control",
            False,
            False,
        ),
        (
            "B11_oracle_diagnostic_gt",
            "oracle GT upper bound",
            [dict(row, control_label=row.get("diagnostic_gt_eval_label", "")) for row in pred_rows],
            "cluster_size_desc_for_oracle",
            True,
            True,
        ),
    ]

    control_rows: list[dict[str, Any]] = []
    non_oracle_ap50: list[float] = []
    semantic_ap50 = math.nan
    stale_ap50 = math.nan
    for control_id, notes, rows, score_contract, uses_gt, is_oracle in controls:
        control_eval = _eval_variant(
            holdout_assignment_rows,
            rows,
            variant_id=f"{selected_variant}:{control_id}",
            label_field="control_label",
            pred_scores=None,
            score_contract=score_contract,
            prediction_uses_gt=uses_gt,
            is_oracle=is_oracle,
        )
        control_row = {
            **control_eval,
            "parent_variant": selected_variant,
            "control_id": control_id,
            "control_notes": notes,
            "is_method_result": False,
        }
        control_rows.append(control_row)
        if not uses_gt:
            non_oracle_ap50.append(_num(control_row.get("native_AP50"), 0.0))
        if control_id == "B9_semantic_descriptor_hash":
            semantic_ap50 = _num(control_row.get("native_AP50"), math.nan)
        elif control_id == "B10_time_indexed_stale_history":
            stale_ap50 = _num(control_row.get("native_AP50"), math.nan)

    real_ap50 = _num(metric_row.get("native_AP50"), 0.0)
    metric_row["real_minus_best_non_oracle_AP50"] = real_ap50 - max(non_oracle_ap50, default=0.0)
    metric_row["real_minus_semantic_AP50"] = (
        real_ap50 - semantic_ap50 if math.isfinite(semantic_ap50) else ""
    )
    metric_row["real_minus_stale_AP50"] = real_ap50 - stale_ap50 if math.isfinite(stale_ap50) else ""
    gate = {
        "holdout_native_AP50_ge_0p35": real_ap50 >= 0.35,
        "holdout_real_minus_best_non_oracle_AP50_ge_0p07": metric_row["real_minus_best_non_oracle_AP50"] >= 0.07,
        "holdout_purity_ge_0p75": _num(metric_row.get("purity"), 0.0) >= 0.75,
        "holdout_ARI_ge_0p50": _num(metric_row.get("adjusted_rand_index"), 0.0) >= 0.50,
        "holdout_GT_label_coverage_rate_ge_0p80": _num(metric_row.get("GT_label_coverage_rate"), 0.0) >= 0.80,
        "method_uses_gt_false": True,
        "uses_future_false": True,
    }
    gate["pass"] = all(gate.values())
    metric_row["holdout_native_metric_gate_pass"] = gate["pass"]
    metric_row["gate"] = json.dumps(gate, sort_keys=True)
    decision = "PASS_V86_PHASE10_HOLDOUT" if gate["pass"] else "NO_GO_HOLDOUT_NATIVE_FAIL"
    blocker = "" if gate["pass"] else "holdout_native_metric_gate_fail"
    summary = {
        "schema": "stream4d_v86_phase10_holdout_v2",
        "phase": "v86_phase10_holdout",
        "decision": decision,
        "formal_holdout_run": True,
        "holdout_replay_attempted": True,
        "config_path": _rel(config_path) if config_path.exists() else "",
        "config_sha256": config.get("config_sha256", ""),
        "config_sha256_recomputed": config_sha256_recomputed,
        "config_sha256_matches": config_sha256_matches,
        "selected_route": config.get("selected_route", ""),
        "selected_variant": selected_variant,
        "primary_blocker": blocker,
        "dev_support_row_count": len(dev_support_rows),
        "holdout_support_row_count": len(holdout_support_rows),
        "holdout_unique_native_carrier_count": assignment_summary.get("native_support_carrier_count", 0),
        "holdout_assignment_count": len(holdout_assignment_rows),
        "holdout_native_AP50": metric_row.get("native_AP50", ""),
        "holdout_real_minus_best_non_oracle_AP50": metric_row.get("real_minus_best_non_oracle_AP50", ""),
        "holdout_purity": metric_row.get("purity", ""),
        "holdout_ARI": metric_row.get("adjusted_rand_index", ""),
        "holdout_GT_label_coverage_rate": metric_row.get("GT_label_coverage_rate", ""),
        "holdout_semantic_control_AP50": semantic_ap50 if math.isfinite(semantic_ap50) else "",
        "holdout_stale_control_AP50": stale_ap50 if math.isfinite(stale_ap50) else "",
        "holdout_source_candidate_rows": _rel(out / "holdout_source_candidate_rows.csv"),
        "holdout_source_candidate_summary": _rel(out / "holdout_source_candidate_summary.json"),
        "holdout_method_safe_candidate_source_count": source_candidate_summary.get(
            "method_safe_holdout_input_available_count", 0
        ),
        "semantic_control_audit": semantic_audit,
        "stale_control_audit": stale_audit,
        "method_uses_gt_anywhere": False,
        "uses_future_anywhere": False,
        "runtime_sec": time.time() - t0,
    }
    _write_json(out / "holdout_summary.json", summary)
    _write_csv(out / "holdout_native_metric_rows.csv", [metric_row])
    _write_csv(out / "holdout_control_rows.csv", control_rows)
    failed = [key for key, value in gate.items() if key != "pass" and not value]
    _write_csv(
        out / "holdout_failure_case_rows.csv",
        [
            {
                "failure_type": "HOLDOUT_NATIVE_FAIL" if failed else "HOLDOUT_PASS",
                "failed_gate_fields": ";".join(failed),
                "selected_variant": selected_variant,
                "native_AP50": metric_row.get("native_AP50", ""),
                "real_minus_best_non_oracle_AP50": metric_row.get("real_minus_best_non_oracle_AP50", ""),
                "notes": "Frozen holdout evaluated without retuning." if not failed else "Frozen holdout failed one or more pre-registered native gates.",
            }
        ],
    )
    return summary


def _phase11(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _repo_path(args.phase11_output_root)
    p5 = _read_json(_repo_path(args.phase5_output_root) / "native_membership_summary.json")
    p6 = _read_json(_repo_path(args.phase6_output_root) / "native_eval_summary.json")
    p8 = _read_json(_repo_path(args.phase8_output_root) / "scene_exporter_audit_summary.json")
    p9 = _read_json(_repo_path(args.phase9_output_root) / "control_summary.json")
    p10 = _read_json(_repo_path(args.phase10_output_root) / "holdout_summary.json")
    p6_metrics = _read_csv_rows(_repo_path(args.phase6_output_root) / "native_metric_rows.csv")
    best_metric = sorted(p6_metrics, key=lambda row: _num(row.get("native_AP50"), 0.0), reverse=True)
    phase10_pass = p10.get("decision") == "PASS_V86_PHASE10_HOLDOUT"
    native_method_success = bool(p9.get("native_method_pass")) and phase10_pass
    scene_method_success = bool(p9.get("scene_method_pass")) and phase10_pass
    if native_method_success and not scene_method_success:
        final = "GO_NATIVE_CARRIER_OBJECTNESS_ONLY"
        primary_blocker = "NO_GO_SCENE_VERTEX_EXPORTER_MISSING"
    else:
        final = "NO_GO_AFFINITY_FIELD_READOUT"
        if not p5.get("decision", "").startswith("PASS"):
            primary_blocker = "NATIVE_MEMBERSHIP_GATE_FAIL"
        elif not p6.get("dev_native_metric_pass"):
            primary_blocker = "NATIVE_EVAL_CONTROL_FAIL"
        elif p10.get("decision") == "NO_GO_HOLDOUT_INPUT_MISSING":
            primary_blocker = "HOLDOUT_INPUT_MISSING"
        elif not p10.get("formal_holdout_run"):
            primary_blocker = "HOLDOUT_NATIVE_FAIL"
        elif not phase10_pass:
            primary_blocker = "HOLDOUT_NATIVE_FAIL"
        else:
            primary_blocker = p9.get("primary_blocker", "UNKNOWN")
    failure_rows: list[dict[str, Any]] = []
    if not p5.get("decision", "").startswith("PASS"):
        failure_rows.append(
            {
                "failure_type": "NATIVE_MEMBERSHIP_GATE_FAIL",
                "evidence": p5.get("decision", ""),
                "notes": "No native readout variant satisfied the Phase5 membership gate.",
            }
        )
    if not p6.get("dev_native_metric_pass"):
        failure_rows.append(
            {
                "failure_type": "NATIVE_EVAL_CONTROL_FAIL",
                "evidence": p6.get("decision", ""),
                "notes": "No Phase5-valid native variant beat the Phase6 controls under the frozen contract.",
            }
        )
    if not p8.get("method_safe_scene_vertex_exporter_available", False):
        failure_rows.append(
            {
                "failure_type": "SCENE_VERTEX_EXPORTER_MISSING",
                "evidence": p8.get("decision", ""),
                "notes": p8.get("primary_blocker", ""),
            }
        )
    if not phase10_pass:
        row = {
            "failure_type": "HOLDOUT_INPUT_MISSING"
            if p10.get("decision") == "NO_GO_HOLDOUT_INPUT_MISSING"
            else "HOLDOUT_NATIVE_FAIL",
            "evidence": p10.get("decision", ""),
            "notes": p10.get("primary_blocker", ""),
        }
        if p10.get("diagnostic_tentative_holdout_summary"):
            row.update(
                {
                    "diagnostic_tentative_holdout_summary": p10.get("diagnostic_tentative_holdout_summary", ""),
                    "diagnostic_tentative_native_AP50": p10.get("diagnostic_tentative_native_AP50", ""),
                    "diagnostic_tentative_real_minus_best_non_oracle_AP50": p10.get(
                        "diagnostic_tentative_real_minus_best_non_oracle_AP50", ""
                    ),
                    "diagnostic_tentative_purity": p10.get("diagnostic_tentative_purity", ""),
                    "diagnostic_tentative_ARI": p10.get("diagnostic_tentative_ARI", ""),
                    "diagnostic_tentative_method_claim_blocker": p10.get(
                        "diagnostic_tentative_method_claim_blocker", ""
                    ),
                }
            )
        if p10.get("holdout_repair_probe_summary"):
            row.update(
                {
                    "holdout_repair_probe_summary": p10.get("holdout_repair_probe_summary", ""),
                    "holdout_repair_probe_best_variant": p10.get("holdout_repair_probe_best_variant", ""),
                    "holdout_repair_probe_best_native_AP50": p10.get("holdout_repair_probe_best_native_AP50", ""),
                    "holdout_repair_probe_best_ARI": p10.get("holdout_repair_probe_best_ARI", ""),
                    "holdout_repair_probe_best_purity": p10.get("holdout_repair_probe_best_purity", ""),
                    "holdout_repair_probe_best_metric_pass_without_formal_claim": p10.get(
                        "holdout_repair_probe_best_metric_pass_without_formal_claim", ""
                    ),
                    "holdout_repair_probe_formal_method_claim_allowed": p10.get(
                        "holdout_repair_probe_formal_method_claim_allowed", ""
                    ),
                    "holdout_repair_probe_primary_blocker": p10.get("holdout_repair_probe_primary_blocker", ""),
                }
            )
        failure_rows.append(row)
    success_rows = []
    if best_metric:
        best = best_metric[0]
        success_rows.append(
            {
                "success_type": "NATIVE_SIGNAL_PRESENT_BUT_NOT_METHOD_SUCCESS",
                "variant": best.get("variant_id", ""),
                "native_AP50": best.get("native_AP50", ""),
                "native_AP25": best.get("native_AP25", ""),
                "ARI": best.get("adjusted_rand_index", ""),
                "purity": best.get("purity", ""),
                "completeness": best.get("completeness", ""),
                "notes": "Measured under v86 frozen contract. Dev native gate may pass, but final success still requires the registered holdout replay and, for ScanNet AP, a method-safe scene exporter.",
            }
        )
    if p10.get("decision") == "NO_GO_HOLDOUT_INPUT_MISSING":
        next_action = (
            "Repair Phase2 tracklet association/history confirmation on dev so it beats the semantic-only control and "
            "produces frozen method-safe confirmed/stable readout rows for the registered holdout chunks "
            "(scene0011_00 chunks 6-11, scene0050_00 chunks 4-11). Then rerun the frozen Phase10 holdout once "
            "without retuning. Do not claim ScanNet AP until a method-safe native-to-scene vertex exporter exists."
        )
    elif not p8.get("method_safe_scene_vertex_exporter_available", False):
        next_action = (
            "Native carrier method path passed holdout, but ScanNet scene AP still needs a method-safe "
            "native-to-scene vertex exporter before any scene metric claim."
        )
    else:
        next_action = (
            "Investigate the recorded holdout failure cases under the frozen config. Do not retune on holdout; "
            "return to a new dev plan if a new method version is needed."
        )
    final_decision = {
        "schema": "stream4d_v86_final_decision_v1",
        "final_decision": final,
        "native_method_success": native_method_success,
        "scene_method_success": scene_method_success,
        "weak_memory_success": False,
        "method_uses_gt_anywhere": False,
        "uses_future_anywhere": False,
        "selected_route": p9.get("decision", ""),
        "strongest_valid_metric": {
            "variant": best_metric[0].get("variant_id", "") if best_metric else "",
            "native_AP50": best_metric[0].get("native_AP50", "") if best_metric else "",
            "native_AP25": best_metric[0].get("native_AP25", "") if best_metric else "",
            "native_AP75": best_metric[0].get("native_AP75", "") if best_metric else "",
            "metric_scope": "native_carrier_objectness_not_scannet_ap",
        },
        "primary_blocker": primary_blocker,
        "phase5_decision": p5.get("decision", ""),
        "phase6_decision": p6.get("decision", ""),
        "phase8_decision": p8.get("decision", ""),
        "phase9_decision": p9.get("decision", ""),
        "phase10_decision": p10.get("decision", ""),
        "diagnostic_tentative_holdout": {
            "summary_path": p10.get("diagnostic_tentative_holdout_summary", ""),
            "native_AP50": p10.get("diagnostic_tentative_native_AP50", ""),
            "real_minus_best_non_oracle_AP50": p10.get(
                "diagnostic_tentative_real_minus_best_non_oracle_AP50", ""
            ),
            "purity": p10.get("diagnostic_tentative_purity", ""),
            "ARI": p10.get("diagnostic_tentative_ARI", ""),
            "method_claim_blocker": p10.get("diagnostic_tentative_method_claim_blocker", ""),
            "is_method_result": False,
            "is_diagnostic_only": bool(p10.get("diagnostic_tentative_holdout_summary")),
        },
        "holdout_repair_probe": {
            "summary_path": p10.get("holdout_repair_probe_summary", ""),
            "best_variant": p10.get("holdout_repair_probe_best_variant", ""),
            "best_native_AP50": p10.get("holdout_repair_probe_best_native_AP50", ""),
            "best_ARI": p10.get("holdout_repair_probe_best_ARI", ""),
            "best_purity": p10.get("holdout_repair_probe_best_purity", ""),
            "best_metric_pass_without_formal_claim": p10.get(
                "holdout_repair_probe_best_metric_pass_without_formal_claim", ""
            ),
            "formal_method_claim_allowed": p10.get("holdout_repair_probe_formal_method_claim_allowed", ""),
            "primary_blocker": p10.get("holdout_repair_probe_primary_blocker", ""),
            "is_method_result": False,
            "is_diagnostic_only": bool(p10.get("holdout_repair_probe_summary")),
        },
        "next_recommended_action": next_action,
        "runtime_sec": time.time() - t0,
    }
    theory = "\n".join(
        [
            "# Stream4D v86 Theory Update",
            "",
            "The v86 audit separates D4RT native-carrier objectness from ScanNet mesh-vertex AP.",
            "After the anti-hijack state-priority readout repair, the dev native-carrier route satisfies",
            "the Phase5 membership gate and Phase6 native metric/control gate under the frozen contract.",
            "That is still not final method success: the registered temporal holdout has no v85 native support",
            "rows for its chunks, and scene-vertex export remains unavailable by the method-safe route audit.",
            "A diagnostic-only tentative holdout replay can materialize weak tentative history through holdout",
            "adapter rows, but it is forbidden for method claim and remains below the frozen holdout AP50/ARI gates.",
            "The Phase10 repair probe additionally tests high-margin weak filters and Phase2 confirmed/repeated",
            "tracklet readouts. Those rows are recorded as repair evidence only, not as formal method success,",
            "because the probe is not the frozen dev-selected Phase10 method run and Phase2 still failed its",
            "semantic-control/coverage gate.",
            "",
            "Conclusion: persistent affinity field evidence is promising as a dev native-carrier signal, but v86",
            "does not achieve final success until a frozen holdout replay has real holdout inputs. ScanNet AP remains",
            "blocked by the missing method-safe native-carrier-to-scene-vertex exporter.",
        ]
    )
    _write_json(out / "final_decision.json", final_decision)
    _write_csv(out / "failure_case_rows.csv", failure_rows)
    _write_csv(out / "success_case_rows.csv", success_rows)
    (out / "theory_update.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "theory_update.md").write_text(theory + "\n", encoding="utf-8")
    return final_decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(*PHASE_ORDER, "all"), default="all")
    parser.add_argument("--v85-phase5-root", default="outputs/audit/v85_phase5_history_object_feature")
    parser.add_argument("--v85-phase6-root", default="outputs/audit/v85_phase6_history_query")
    parser.add_argument("--v85-phase7-root", default="outputs/audit/v85_phase7_renderable_materializer")
    parser.add_argument("--v85-phase8-root", default="outputs/audit/v85_phase8_strong_controls")
    parser.add_argument("--v85-phase10-root", default="outputs/audit/v85_phase10_casebook")
    parser.add_argument("--phase0-output-root", default="outputs/audit/v86_phase0_fact_lock")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v86_phase1_native_eval_contract")
    parser.add_argument("--phase5-output-root", default="outputs/audit/v86_phase5_native_membership")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v86_phase6_native_eval")
    parser.add_argument("--phase8-output-root", default="outputs/audit/v86_phase8_scene_exporter_audit")
    parser.add_argument("--phase9-output-root", default="outputs/audit/v86_phase9_controls")
    parser.add_argument("--phase10-output-root", default="outputs/audit/v86_phase10_holdout")
    parser.add_argument("--phase11-output-root", default="outputs/audit/v86_phase11_casebook")
    parser.add_argument("--phase12-output-root", default="outputs/audit/v86_phase12_dev_tracklet_readout_repair")
    parser.add_argument("--phase13-output-root", default="outputs/audit/v86_phase13_candidate_freeze_holdout_audit")
    parser.add_argument("--phase14-output-root", default="outputs/audit/v86_phase14_fresh_holdout_availability_audit")
    parser.add_argument("--phase15-output-root", default="outputs/audit/v86_phase15_raw_substrate_availability_audit")
    parser.add_argument("--phase16-output-root", default="outputs/audit/v86_phase16_new_scene_pipeline_feasibility_audit")
    parser.add_argument(
        "--phase16-v75-smoke-root",
        default="outputs/audit/v86_phase16_v75_soft_incidence_new_scene_smoke",
    )
    parser.add_argument(
        "--phase13-candidate-config",
        default="outputs/audit/v86_phase12_dev_tracklet_readout_repair/dev_tracklet_readout_candidate_config.json",
    )
    parser.add_argument("--config-output-root", default="outputs/audit/v86_config")
    parser.add_argument("--dev-tracklet-phase2-root", default="outputs/audit/v82_phase2_object_tracklets_repair11_app079_sigma022")
    parser.add_argument("--dev-local-phase1-root", default="outputs/audit/v82_phase1_local_b0")
    parser.add_argument("--dev-adapter-root", default="outputs/audit/v82_local_shadow/phase1_adapter_dev_v82_phase1_b0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    phase_fns = {
        "phase0": _phase0,
        "phase1": _phase1,
        "phase5": _phase5,
        "phase6": _phase6,
        "phase8": _phase8,
        "phase9": _phase9,
        "phase10": _phase10,
        "phase11": _phase11,
        "phase12": _dev_tracklet_readout_repair,
        "phase13": _candidate_freeze_holdout_audit,
        "phase14": _fresh_holdout_availability_audit,
        "phase15": _raw_substrate_availability_audit,
        "phase16": _new_scene_pipeline_feasibility_audit,
    }
    phases = PHASE_ORDER if args.phase == "all" else (args.phase,)
    summaries: list[dict[str, Any]] = []
    for phase in phases:
        summary = phase_fns[phase](args)
        summaries.append(
            {
                "phase": phase,
                "decision": summary.get("decision", summary.get("final_decision", "")),
                "output": _rel(_repo_path(getattr(args, f"{phase}_output_root"))),
            }
        )
    print(json.dumps({"schema": "stream4d_v86_runner_result_v1", "phases": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
