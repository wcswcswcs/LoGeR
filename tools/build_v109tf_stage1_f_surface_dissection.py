#!/usr/bin/env python3
"""Dissect v108 F-surface action frames for ACL2 v109TF."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
OUT = RESULT_ROOT / "stage1_f_surface_dissection"

V108 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
STAGE2 = V108 / "stage2_semantic_cue_bank"
STAGE5 = V108 / "stage5_full_kitti_00_01_02_05_validation"

SURFACES = ("F", "E")
MAIN_POLICY = "semantic_plus_internal"
ROLE_COLS = (
    "stable_structure_mass",
    "dynamic_mass",
    "boundary_mass",
    "weak_context_mass",
    "road_ground_mass",
    "sky_lowobs_mass",
)
SEMANTIC_COLS = (
    *ROLE_COLS,
    "semantic_trust_mean",
    "semantic_patch_purity_mean",
    "semantic_continuity_score",
    "Q_ref_sem_balanced",
    "Q_ref_sem_risk_strict",
    "Q_ref_sem_stable_strict",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def parse_indices(value: str) -> set[int]:
    out: set[int] = set()
    for part in str(value or "").split(";"):
        if not part.strip():
            continue
        out.add(int(float(part)))
    return out


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def rel_improvement(base: float, action: float) -> float:
    if not math.isfinite(base) or abs(base) < 1e-12 or not math.isfinite(action):
        return float("nan")
    return (base - action) / base


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_repr(counter: Counter[str]) -> str:
    return ";".join(f"{key}:{value}" for key, value in counter.most_common() if key)


def source_frames_for(
    snap_map: dict[tuple[str, str, str, int], list[dict[str, str]]],
    surface: str,
    policy_id: str,
    seq: str,
    frame_id: int,
) -> list[int]:
    rows = snap_map.get((surface, policy_id, seq, frame_id), [])
    return sorted({int(float(row["source_frame"])) for row in rows if boolish(row.get("accepted", ""))})


def op_aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
    op_counter: Counter[str] = Counter()
    context_counter: Counter[str] = Counter()
    token_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    keyframe_count = 0
    scale_frame_count = 0
    trajectory_count = 0
    cache_append_count = 0
    special_count = 0
    local_reference_count = 0
    source_ages: list[float] = []
    for row in rows:
        op = row.get("operation_type", "")
        context = row.get("context_path", "")
        token = row.get("token_type", "")
        op_counter[op] += 1
        context_counter[context] += 1
        token_counter[token] += 1
        if row.get("target_kind"):
            target_counter[row["target_kind"]] += 1
        keyframe_count += int(boolish(row.get("keyframe_flag", "")))
        scale_frame_count += int(boolish(row.get("scale_frame_flag", "")))
        trajectory_count += int(op == "trajectory_write" or boolish(row.get("trajectory_memory_flag", "")))
        cache_append_count += int(op == "cache_append")
        special_count += int(op == "special_token_update" or token != "image_patch")
        local_reference_count += int(context == "local_pose_reference_window")
        source_ages.append(fnum(row.get("source_frame_age", "")))
    return {
        "operation_row_count": len(rows),
        "operation_types_touched": count_repr(op_counter),
        "context_paths_touched": count_repr(context_counter),
        "token_types_touched": count_repr(token_counter),
        "target_kinds_touched": count_repr(target_counter),
        "is_base_keyframe": keyframe_count > 0,
        "is_scale_frame": scale_frame_count > 0,
        "is_cache_append_frame": cache_append_count > 0,
        "is_trajectory_write_frame": trajectory_count > 0,
        "is_special_token_update_frame": special_count > 0,
        "cache_append_count": cache_append_count,
        "special_token_update_count": special_count,
        "trajectory_write_count": trajectory_count,
        "local_reference_count": local_reference_count,
        "source_age_mean": mean(source_ages),
    }


def role_tags(row: dict[str, Any]) -> dict[str, Any]:
    stable = fnum(row.get("stable_structure_mass"))
    dynamic = fnum(row.get("dynamic_mass"))
    boundary = fnum(row.get("boundary_mass"))
    weak = fnum(row.get("weak_context_mass"))
    road = fnum(row.get("road_ground_mass"))
    sky = fnum(row.get("sky_lowobs_mass"))
    dynamic_boundary = sum(v for v in [dynamic, boundary] if math.isfinite(v))
    weak_road = sum(v for v in [weak, road] if math.isfinite(v))
    role_values = {
        "stable_structure": stable,
        "dynamic_boundary": dynamic_boundary,
        "weak_context_road_ground": weak_road,
        "sky_lowobs": sky,
    }
    dominant = max(role_values, key=lambda k: role_values[k] if math.isfinite(role_values[k]) else float("-inf"))
    return {
        "stable_structure_heavy": dominant == "stable_structure",
        "dynamic_boundary_heavy": dominant == "dynamic_boundary",
        "weak_context_road_ground_heavy": dominant == "weak_context_road_ground",
        "sky_lowobs_heavy": dominant == "sky_lowobs",
        "dominant_semantic_role_group": dominant,
        "dynamic_boundary_mass": dynamic_boundary,
        "weak_context_road_ground_mass": weak_road,
    }


def build_selected_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frame_map = {
        (row["seq_id"], int(float(row["frame_id"]))): row
        for row in read_csv(STAGE2 / "frame_semantic_summary.csv")
    }
    op_map: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(STAGE2 / "operation_semantic_summary.csv"):
        try:
            frame_id = int(float(row.get("current_frame") or -1))
        except ValueError:
            continue
        if frame_id < 0:
            continue
        for surface in [sid for sid in row.get("candidate_surface_ids", "").split(";") if sid]:
            op_map[(surface, row.get("seq_id", ""), frame_id)].append(row)

    snap_map: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(STAGE5 / "keyframe_snap_rows.csv"):
        if row.get("surface_id") not in SURFACES or not boolish(row.get("accepted", "")):
            continue
        snapped = int(float(row["snapped_base_keyframe"]))
        snap_map[(row["surface_id"], row["policy_id"], row["seq"], snapped)].append(row)

    fidelity_map = {
        (row["surface_id"], row["policy_id"], row["seq"]): row
        for row in read_csv(STAGE5 / "action_fidelity_rows.csv")
        if row.get("surface_id") in SURFACES
    }
    config_rows = [
        row for row in read_csv(STAGE5 / "action_config_rows.csv")
        if row.get("surface_id") in SURFACES
    ]
    raw_cache: dict[Path, dict[int, dict[str, Any]]] = {}

    selected_rows: list[dict[str, Any]] = []
    op_rows_out: list[dict[str, Any]] = []
    join = {
        "selected_total": 0,
        "frame_semantic_join": 0,
        "exact_operation_join": 0,
        "attribution_operation_join": 0,
        "by_surface": defaultdict(lambda: Counter()),
    }

    for cfg in config_rows:
        surface = cfg["surface_id"]
        policy_id = cfg["policy_id"]
        policy_family = cfg["policy_family"]
        seq = cfg["seq"]
        selected = sorted(parse_indices(cfg.get("selected_global_frame_indices", "")))
        source_selected = sorted(parse_indices(cfg.get("source_selected_global_frame_indices", "")))
        fid = fidelity_map.get((surface, policy_id, seq), {})
        effective = parse_indices(fid.get("effective_action_indices", ""))
        observed = parse_indices(fid.get("observed_action_indices", ""))
        action_file = Path(cfg.get("action_file", ""))
        if action_file and action_file.exists() and action_file not in raw_cache:
            keep = set(selected) | effective | observed
            raw_cache[action_file] = {
                int(float(row.get("sample_position", -1))): row
                for row in load_jsonl(action_file)
                if int(float(row.get("sample_position", -1))) in keep
            }
        raw_map = raw_cache.get(action_file, {})

        for frame_id in selected:
            frame = frame_map.get((seq, frame_id), {})
            exact_ops = op_map.get((surface, seq, frame_id), [])
            source_frames = source_frames_for(snap_map, surface, policy_id, seq, frame_id)
            source_ops: list[dict[str, str]] = []
            for source_frame in source_frames:
                source_ops.extend(op_map.get((surface, seq, source_frame), []))
            attribution_ops = exact_ops if exact_ops else source_ops
            attribution_source = "exact_selected_frame" if exact_ops else ("snap_source_frame" if source_ops else "missing")
            agg = op_aggregate(attribution_ops)
            raw = raw_map.get(frame_id, {})
            semantic_vals = {col: frame.get(col, "") for col in SEMANTIC_COLS}
            selected_source_dist = min((abs(frame_id - src) for src in source_frames), default="")
            nearest_source = min(source_frames, key=lambda src: abs(frame_id - src)) if source_frames else ""
            row: dict[str, Any] = {
                "schema": "acl2_v109tf_stage1_selected_frame_row_v1",
                "surface_id": surface,
                "policy_id": policy_id,
                "policy_family": policy_family,
                "seq": seq,
                "frame_id": frame_id,
                "source_selected_frame_count_for_snap": len(source_frames),
                "source_selected_frames_for_snap": ";".join(str(x) for x in source_frames),
                "nearest_source_selected_frame": nearest_source,
                "nearest_source_selected_distance": selected_source_dist,
                "source_selected_total": len(source_selected),
                "selected_frame_count": len(selected),
                "action_type": cfg.get("stage5_action_mode", ""),
                "expected_action_field": cfg.get("expected_action_field", ""),
                "action_observed": frame_id in observed,
                "action_effective": frame_id in effective,
                "base_is_keyframe_log": raw.get("base_is_keyframe", ""),
                "final_is_keyframe_log": raw.get("final_is_keyframe", ""),
                "forced_anchor_only_log": raw.get("forced_anchor_only", ""),
                "forced_context_only_log": raw.get("forced_context_only", ""),
                "context_only_special_mode_log": raw.get("context_only_special_mode", ""),
                "frame_semantic_join": bool(frame),
                "exact_operation_join": bool(exact_ops),
                "attribution_operation_join": bool(attribution_ops),
                "operation_join_source": attribution_source,
                "internal_score_used_by_policy": agg["special_token_update_count"] if surface == "F" else agg["local_reference_count"],
                "operation_type_touched": agg["operation_types_touched"],
                **semantic_vals,
                **agg,
            }
            row.update(role_tags(row))
            selected_rows.append(row)
            join["selected_total"] += 1
            join["frame_semantic_join"] += int(bool(frame))
            join["exact_operation_join"] += int(bool(exact_ops))
            join["attribution_operation_join"] += int(bool(attribution_ops))
            bys = join["by_surface"][surface]
            bys["selected_total"] += 1
            bys["frame_semantic_join"] += int(bool(frame))
            bys["exact_operation_join"] += int(bool(exact_ops))
            bys["attribution_operation_join"] += int(bool(attribution_ops))

            for op in attribution_ops:
                op_rows_out.append(
                    {
                        "schema": "acl2_v109tf_stage1_action_effective_operation_row_v1",
                        "surface_id": surface,
                        "policy_id": policy_id,
                        "policy_family": policy_family,
                        "seq": seq,
                        "selected_frame_id": frame_id,
                        "action_effective": frame_id in effective,
                        "operation_join_source": attribution_source,
                        "operation_id": op.get("operation_id", ""),
                        "operation_frame": op.get("current_frame", ""),
                        "operation_type": op.get("operation_type", ""),
                        "plan_operation_type": op.get("plan_operation_type", ""),
                        "context_path": op.get("context_path", ""),
                        "token_type": op.get("token_type", ""),
                        "target_id": op.get("target_id", ""),
                        "target_kind": op.get("target_kind", ""),
                        "source_frame": op.get("source_frame", ""),
                        "source_frame_age": op.get("source_frame_age", ""),
                        "keyframe_flag": op.get("keyframe_flag", ""),
                        "scale_frame_flag": op.get("scale_frame_flag", ""),
                        "trajectory_memory_flag": op.get("trajectory_memory_flag", ""),
                        "cache_keep_drop_status": op.get("cache_keep_drop_status", ""),
                    }
                )

    join["by_surface"] = {
        surface: {
            **dict(counter),
            "frame_semantic_join_ratio": counter["frame_semantic_join"] / counter["selected_total"] if counter["selected_total"] else 0.0,
            "exact_operation_join_ratio": counter["exact_operation_join"] / counter["selected_total"] if counter["selected_total"] else 0.0,
            "attribution_operation_join_ratio": counter["attribution_operation_join"] / counter["selected_total"] if counter["selected_total"] else 0.0,
        }
        for surface, counter in join["by_surface"].items()
    }
    join["frame_semantic_join_ratio"] = join["frame_semantic_join"] / join["selected_total"] if join["selected_total"] else 0.0
    join["exact_operation_join_ratio"] = join["exact_operation_join"] / join["selected_total"] if join["selected_total"] else 0.0
    join["attribution_operation_join_ratio"] = join["attribution_operation_join"] / join["selected_total"] if join["selected_total"] else 0.0
    return selected_rows, op_rows_out, join


def role_distribution_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        if row["surface_id"] == "F":
            groups[(row["surface_id"], row["policy_id"], row["policy_family"], row["seq"])].append(row)
    out: list[dict[str, Any]] = []
    for (surface, policy_id, family, seq), rows in sorted(groups.items()):
        effective_rows = [r for r in rows if boolish(r.get("action_effective"))]
        out.append(
            {
                "schema": "acl2_v109tf_stage1_f_semantic_role_distribution_row_v1",
                "surface_id": surface,
                "policy_id": policy_id,
                "policy_family": family,
                "seq": seq,
                "selected_count": len(rows),
                "effective_count": len(effective_rows),
                "exact_operation_join_ratio": sum(boolish(r["exact_operation_join"]) for r in rows) / len(rows) if rows else 0.0,
                "attribution_operation_join_ratio": sum(boolish(r["attribution_operation_join"]) for r in rows) / len(rows) if rows else 0.0,
                **{f"{col}_mean": mean([fnum(r.get(col)) for r in rows]) for col in ROLE_COLS},
                "semantic_trust_mean": mean([fnum(r.get("semantic_trust_mean")) for r in rows]),
                "semantic_patch_purity_mean": mean([fnum(r.get("semantic_patch_purity_mean")) for r in rows]),
                "semantic_continuity_score_mean": mean([fnum(r.get("semantic_continuity_score")) for r in rows]),
                "dominant_semantic_role_group_counts": count_repr(Counter(str(r.get("dominant_semantic_role_group", "")) for r in rows)),
            }
        )
    return out


def keyframe_cache_overlap_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        if row["surface_id"] == "F":
            groups[(row["surface_id"], row["policy_id"], row["policy_family"], row["seq"])].append(row)
    for (surface, policy_id, family, seq), rows in sorted(groups.items()):
        n = len(rows)
        out.append(
            {
                "schema": "acl2_v109tf_stage1_f_keyframe_cache_overlap_row_v1",
                "surface_id": surface,
                "policy_id": policy_id,
                "policy_family": family,
                "seq": seq,
                "selected_count": n,
                "action_effective_count": sum(boolish(r["action_effective"]) for r in rows),
                "base_keyframe_log_count": sum(boolish(r["base_is_keyframe_log"]) for r in rows),
                "operation_keyframe_count": sum(boolish(r["is_base_keyframe"]) for r in rows),
                "cache_append_frame_count": sum(boolish(r["is_cache_append_frame"]) for r in rows),
                "special_token_update_frame_count": sum(boolish(r["is_special_token_update_frame"]) for r in rows),
                "trajectory_write_frame_count": sum(boolish(r["is_trajectory_write_frame"]) for r in rows),
                "scale_frame_count": sum(boolish(r["is_scale_frame"]) for r in rows),
                "exact_operation_join_count": sum(boolish(r["exact_operation_join"]) for r in rows),
                "snap_source_fallback_count": sum(str(r["operation_join_source"]) == "snap_source_frame" for r in rows),
                "missing_operation_join_count": sum(str(r["operation_join_source"]) == "missing" for r in rows),
                "action_effective_ratio": sum(boolish(r["action_effective"]) for r in rows) / n if n else 0.0,
            }
        )
    return out


def metric_maps() -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[tuple[str, str, str], dict[str, str]], dict[tuple[str, str, str], dict[str, str]]]:
    full = {
        (row["surface_id"], row["policy_family"], row["seq"]): row
        for row in read_csv(STAGE5 / "full_sequence_metric_rows.csv")
    }
    rolling = {
        (row["surface_id"], row["policy_family"], row["seq"]): row
        for row in read_csv(STAGE5 / "rolling_metric_rows.csv")
    }
    fidelity = {
        (row["surface_id"], row["policy_family"], row["seq"]): row
        for row in read_csv(STAGE5 / "action_fidelity_rows.csv")
    }
    return full, rolling, fidelity


def control_overlap_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_policy: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    effective_by_policy: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in selected_rows:
        if row["surface_id"] != "F":
            continue
        key = (row["surface_id"], row["policy_family"], row["seq"])
        rows_by_policy[key].add(int(row["frame_id"]))
        if boolish(row["action_effective"]):
            effective_by_policy[key].add(int(row["frame_id"]))
    full, rolling, _fid = metric_maps()
    out: list[dict[str, Any]] = []
    for seq in sorted({key[2] for key in rows_by_policy}):
        base_selected = rows_by_policy.get(("F", MAIN_POLICY, seq), set())
        base_effective = effective_by_policy.get(("F", MAIN_POLICY, seq), set())
        base_metric = full.get(("F", MAIN_POLICY, seq), {})
        for family in sorted({key[1] for key in rows_by_policy if key[2] == seq and key[1] != MAIN_POLICY}):
            selected = rows_by_policy.get(("F", family, seq), set())
            effective = effective_by_policy.get(("F", family, seq), set())
            union = base_selected | selected
            eff_union = base_effective | effective
            metric = full.get(("F", family, seq), {})
            roll = rolling.get(("F", family, seq), {})
            out.append(
                {
                    "schema": "acl2_v109tf_stage1_f_control_overlap_row_v1",
                    "surface_id": "F",
                    "seq": seq,
                    "reference_policy_family": MAIN_POLICY,
                    "control_policy_family": family,
                    "reference_selected_count": len(base_selected),
                    "control_selected_count": len(selected),
                    "selected_overlap_count": len(base_selected & selected),
                    "selected_jaccard": len(base_selected & selected) / len(union) if union else 0.0,
                    "reference_effective_count": len(base_effective),
                    "control_effective_count": len(effective),
                    "effective_overlap_count": len(base_effective & effective),
                    "effective_jaccard": len(base_effective & effective) / len(eff_union) if eff_union else 0.0,
                    "reference_full_rel_improvement": base_metric.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                    "control_full_rel_improvement": metric.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                    "reference_full_ATE_sim3": base_metric.get("full_ATE_sim3", ""),
                    "control_full_ATE_sim3": metric.get("full_ATE_sim3", ""),
                    "control_rolling_p90_rel_improvement": roll.get("rolling_ATE_p90_relative_improvement_vs_baseline", ""),
                }
            )
    return out


def seq05_harm_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full, rolling, _fid = metric_maps()
    metric = full.get(("F", MAIN_POLICY, "05"), {})
    roll = rolling.get(("F", MAIN_POLICY, "05"), {})
    out: list[dict[str, Any]] = []
    for row in selected_rows:
        if row["surface_id"] != "F" or row["policy_family"] != MAIN_POLICY or row["seq"] != "05":
            continue
        if not boolish(row.get("action_effective")):
            continue
        local_context_heavy = int(row.get("local_reference_count", 0)) > 0
        tags = role_tags(row)
        out.append(
            {
                "schema": "acl2_v109tf_stage1_f_seq05_harm_row_v1",
                "surface_id": "F",
                "policy_id": row["policy_id"],
                "seq": "05",
                "frame_id": row["frame_id"],
                "true_keyframe_or_cache_write_point": boolish(row.get("base_is_keyframe_log")) or boolish(row.get("is_cache_append_frame")) or boolish(row.get("is_special_token_update_frame")),
                "non_keyframe_noop": (not boolish(row.get("base_is_keyframe_log"))) or (not boolish(row.get("action_effective"))),
                "local_context_heavy": local_context_heavy,
                "stable_structure_heavy": tags["stable_structure_heavy"],
                "dynamic_boundary_heavy": tags["dynamic_boundary_heavy"],
                "weak_context_road_ground_heavy": tags["weak_context_road_ground_heavy"],
                "dominant_semantic_role_group": tags["dominant_semantic_role_group"],
                "operation_join_source": row.get("operation_join_source", ""),
                "operation_type_touched": row.get("operation_type_touched", ""),
                "context_paths_touched": row.get("context_paths_touched", ""),
                "stable_structure_mass": row.get("stable_structure_mass", ""),
                "dynamic_mass": row.get("dynamic_mass", ""),
                "boundary_mass": row.get("boundary_mass", ""),
                "weak_context_mass": row.get("weak_context_mass", ""),
                "road_ground_mass": row.get("road_ground_mass", ""),
                "semantic_trust_mean": row.get("semantic_trust_mean", ""),
                "semantic_patch_purity_mean": row.get("semantic_patch_purity_mean", ""),
                "semantic_continuity_score": row.get("semantic_continuity_score", ""),
                "seq05_full_ATE_sim3": metric.get("full_ATE_sim3", ""),
                "seq05_baseline_full_ATE_sim3": metric.get("baseline_full_ATE_sim3", ""),
                "seq05_full_rel_improvement": metric.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                "seq05_final_error_rel_improvement": metric.get("final_error_relative_improvement_vs_baseline", ""),
                "seq05_rolling_p90_rel_improvement": roll.get("rolling_ATE_p90_relative_improvement_vs_baseline", ""),
            }
        )
    return out


def e_vs_f_scope_diff_rows(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full, rolling, fidelity = metric_maps()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        if row["policy_family"] == MAIN_POLICY:
            groups[(row["surface_id"], row["seq"])].append(row)
    out: list[dict[str, Any]] = []
    for seq in sorted({seq for _surface, seq in groups}):
        e = groups.get(("E", seq), [])
        f = groups.get(("F", seq), [])
        e_set = {int(r["frame_id"]) for r in e}
        f_set = {int(r["frame_id"]) for r in f}
        e_metric = full.get(("E", MAIN_POLICY, seq), {})
        f_metric = full.get(("F", MAIN_POLICY, seq), {})
        e_fid = fidelity.get(("E", MAIN_POLICY, seq), {})
        f_fid = fidelity.get(("F", MAIN_POLICY, seq), {})
        e_roll = rolling.get(("E", MAIN_POLICY, seq), {})
        f_roll = rolling.get(("F", MAIN_POLICY, seq), {})
        out.append(
            {
                "schema": "acl2_v109tf_stage1_e_vs_f_scope_diff_row_v1",
                "seq": seq,
                "e_selected_count": len(e),
                "f_selected_count": len(f),
                "selected_overlap_count": len(e_set & f_set),
                "selected_overlap_jaccard": len(e_set & f_set) / len(e_set | f_set) if e_set or f_set else 0.0,
                "e_effective_count": sum(boolish(r["action_effective"]) for r in e),
                "f_effective_count": sum(boolish(r["action_effective"]) for r in f),
                "e_expected_action_frame_count": e_fid.get("expected_action_frame_count", ""),
                "f_expected_action_frame_count": f_fid.get("expected_action_frame_count", ""),
                "e_full_rel_improvement": e_metric.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                "f_full_rel_improvement": f_metric.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                "e_full_ATE_sim3": e_metric.get("full_ATE_sim3", ""),
                "f_full_ATE_sim3": f_metric.get("full_ATE_sim3", ""),
                "e_final_rel_improvement": e_metric.get("final_error_relative_improvement_vs_baseline", ""),
                "f_final_rel_improvement": f_metric.get("final_error_relative_improvement_vs_baseline", ""),
                "e_local_window_median_rel": e_metric.get("local_window_ATE_rel_improvement_vs_baseline_median", ""),
                "f_local_window_median_rel": f_metric.get("local_window_ATE_rel_improvement_vs_baseline_median", ""),
                "e_rolling_p90_rel_improvement": e_roll.get("rolling_ATE_p90_relative_improvement_vs_baseline", ""),
                "f_rolling_p90_rel_improvement": f_roll.get("rolling_ATE_p90_relative_improvement_vs_baseline", ""),
                "e_stable_structure_mass_mean": mean([fnum(r.get("stable_structure_mass")) for r in e]),
                "f_stable_structure_mass_mean": mean([fnum(r.get("stable_structure_mass")) for r in f]),
                "e_dynamic_boundary_mass_mean": mean([fnum(r.get("dynamic_boundary_mass")) for r in e]),
                "f_dynamic_boundary_mass_mean": mean([fnum(r.get("dynamic_boundary_mass")) for r in f]),
                "e_special_token_update_frame_count": sum(boolish(r["is_special_token_update_frame"]) for r in e),
                "f_special_token_update_frame_count": sum(boolish(r["is_special_token_update_frame"]) for r in f),
                "e_local_reference_frame_count": sum(int(r.get("local_reference_count", 0)) > 0 for r in e),
                "f_local_reference_frame_count": sum(int(r.get("local_reference_count", 0)) > 0 for r in f),
            }
        )
    return out


def write_report(summary: dict[str, Any], seq05_rows: list[dict[str, Any]], scope_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ACL2 v109TF Stage1 F-Surface Dissection Report",
        "",
        f"stage1_pass: {summary['stage1_pass']}",
        f"blocker: {summary['blocker']}",
        f"selected_total: {summary['selected_total']}",
        f"frame_semantic_join_ratio: {summary['frame_semantic_join_ratio']}",
        f"exact_operation_join_ratio: {summary['exact_operation_join_ratio']}",
        f"attribution_operation_join_ratio: {summary['attribution_operation_join_ratio']}",
        "",
        "## Join interpretation",
        "",
        "Exact operation join uses the snapped action frame itself. Attribution operation join first uses exact rows and then falls back to source frames that snapped into the action keyframe. This preserves the keyframe-aware action schedule while documenting where snap-source fallback was needed.",
        "",
        "## KITTI05 F semantic+ effective frames",
        "",
    ]
    for row in seq05_rows:
        lines.append(
            "- frame {frame}: role={role}, key_or_cache={key}, nonkey_noop={noop}, op_source={source}, ops={ops}".format(
                frame=row["frame_id"],
                role=row["dominant_semantic_role_group"],
                key=row["true_keyframe_or_cache_write_point"],
                noop=row["non_keyframe_noop"],
                source=row["operation_join_source"],
                ops=row["operation_type_touched"],
            )
        )
    lines.extend(["", "## E vs F semantic+ scope", ""])
    for row in scope_rows:
        lines.append(
            "- seq {seq}: E selected={esel} F selected={fsel} overlap={ov} E_rel={erel} F_rel={frel}".format(
                seq=row["seq"],
                esel=row["e_selected_count"],
                fsel=row["f_selected_count"],
                ov=row["selected_overlap_count"],
                erel=row["e_full_rel_improvement"],
                frel=row["f_full_rel_improvement"],
            )
        )
    write_text(OUT / "stage1_report.md", "\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected_rows, op_rows, join = build_selected_rows()
    f_selected_rows = [row for row in selected_rows if row["surface_id"] == "F"]
    role_rows = role_distribution_rows(selected_rows)
    key_rows = keyframe_cache_overlap_rows(selected_rows)
    control_rows = control_overlap_rows(selected_rows)
    seq05_rows = seq05_harm_rows(selected_rows)
    scope_rows = e_vs_f_scope_diff_rows(selected_rows)

    write_csv(OUT / "f_selected_frame_rows.csv", f_selected_rows)
    write_csv(OUT / "f_action_effective_operation_rows.csv", [row for row in op_rows if row["surface_id"] == "F"])
    write_csv(OUT / "f_semantic_role_distribution_by_seq.csv", role_rows)
    write_csv(OUT / "f_keyframe_cache_overlap_rows.csv", key_rows)
    write_csv(OUT / "f_control_overlap_rows.csv", control_rows)
    write_csv(OUT / "f_seq05_harm_rows.csv", seq05_rows)
    write_csv(OUT / "e_vs_f_scope_diff_rows.csv", scope_rows)

    selected_total = int(join["selected_total"])
    frame_join = float(join["frame_semantic_join_ratio"])
    exact_op_join = float(join["exact_operation_join_ratio"])
    attribution_op_join = float(join["attribution_operation_join_ratio"])
    f_attribution = join["by_surface"].get("F", {}).get("attribution_operation_join_ratio", 0.0)
    e_attribution = join["by_surface"].get("E", {}).get("attribution_operation_join_ratio", 0.0)
    seq05_located = len(seq05_rows) > 0 and all(boolish(row["true_keyframe_or_cache_write_point"]) for row in seq05_rows)
    stage1_pass = frame_join >= 0.95 and attribution_op_join >= 0.95 and f_attribution >= 0.95 and e_attribution >= 0.95 and seq05_located
    blocker = "" if stage1_pass else "ACTION_ATTRIBUTION_JOIN_BLOCKED"
    summary = {
        "schema": "acl2_v109tf_stage1_summary_v1",
        "stage1_pass": stage1_pass,
        "blocker": blocker,
        "selected_total": selected_total,
        "f_selected_frame_row_count": len(f_selected_rows),
        "f_action_effective_operation_row_count": sum(1 for row in op_rows if row["surface_id"] == "F"),
        "f_seq05_harm_row_count": len(seq05_rows),
        "frame_semantic_join_ratio": frame_join,
        "exact_operation_join_ratio": exact_op_join,
        "attribution_operation_join_ratio": attribution_op_join,
        "join_by_surface": join["by_surface"],
        "seq05_effective_frames_located": seq05_located,
        "exact_operation_join_note": "Exact join can be lower because v108 Stage5 snapped source-selected frames to base keyframes. attribution_operation_join_ratio includes documented snap-source fallback.",
        "outputs": {
            "f_selected_frame_rows": rel(OUT / "f_selected_frame_rows.csv"),
            "f_action_effective_operation_rows": rel(OUT / "f_action_effective_operation_rows.csv"),
            "f_semantic_role_distribution_by_seq": rel(OUT / "f_semantic_role_distribution_by_seq.csv"),
            "f_keyframe_cache_overlap_rows": rel(OUT / "f_keyframe_cache_overlap_rows.csv"),
            "f_control_overlap_rows": rel(OUT / "f_control_overlap_rows.csv"),
            "f_seq05_harm_rows": rel(OUT / "f_seq05_harm_rows.csv"),
            "e_vs_f_scope_diff_rows": rel(OUT / "e_vs_f_scope_diff_rows.csv"),
            "stage1_report": rel(OUT / "stage1_report.md"),
            "stage1_summary": rel(OUT / "stage1_summary.json"),
        },
    }
    write_json(OUT / "stage1_summary.json", summary)
    write_report(summary, seq05_rows, scope_rows)
    if not stage1_pass:
        write_text(
            OUT / "ACTION_ATTRIBUTION_JOIN_BLOCKED.md",
            "# ACTION_ATTRIBUTION_JOIN_BLOCKED\n\n"
            f"- frame_semantic_join_ratio: `{frame_join}`\n"
            f"- exact_operation_join_ratio: `{exact_op_join}`\n"
            f"- attribution_operation_join_ratio: `{attribution_op_join}`\n"
            f"- join_by_surface: `{clean_json(join['by_surface'])}`\n"
            "- Checked frame id, global/local id, keyframe snap rows, and source-frame fallback in this script.\n"
            "- Do not run new action until attribution coverage is repaired above threshold.\n",
        )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
