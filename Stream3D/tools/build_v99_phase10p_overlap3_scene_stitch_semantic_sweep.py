#!/usr/bin/env python3
"""Postprocess Phase10O with broader overlap3 semantic scene-stitch sweeps."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase10k_holdout_chunk_object_birth_sweep as p10k  # noqa: E402
from tools import build_v99_phase10l_frozen_p2d2_regenerated_birth_holdout as p10l  # noqa: E402
from tools import build_v99_phase10o_overlap3_scene_stitch_repair as p10o  # noqa: E402
from tools.build_v99_phase9_scene_local2history import DSU  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10p_overlap3_scene_stitch_semantic_sweep"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE10O_DIR = AUDIT_ROOT / "v99_phase10o_overlap3_scene_stitch_repair"
BASE_VARIANT = "O0_overlap3_chunk_birth_primary_emit"
CHUNK_SIZE = 32
OVERLAP = 3


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _base_rows() -> list[dict[str, Any]]:
    rows = [dict(row) for row in _read_csv(PHASE10O_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == BASE_VARIANT]
    if not rows:
        raise RuntimeError("missing Phase10O O0 base primary rows")
    return rows


def _object_infos(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    features = p10l._load_holdout_residual_features()
    infos: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": [], "frames": set(), "chunks": set(), "features": []})
    for row in rows:
        oid = str(row["mv_object_id"])
        scene = str(row["scene_id"])
        chunk = str(row["chunk_id"])
        frame = int(float(row["frame_id"]))
        mask = int(float(row["selected_mask_id"]))
        infos[oid]["rows"].append(row)
        infos[oid]["scene_id"] = scene
        infos[oid]["chunk_id"] = chunk
        infos[oid]["frames"].add(frame)
        infos[oid]["chunks"].add(chunk)
        feat = features.get((scene, frame, mask))
        if feat is not None:
            infos[oid]["features"].append(feat)
    for oid, info in infos.items():
        frames = sorted(info["frames"])
        info["first_frame"] = frames[0] if frames else -1
        info["last_frame"] = frames[-1] if frames else -1
        info["frame_count"] = len(frames)
        feats = info["features"]
        if feats:
            info["feature"] = p1._normalize_rows(np.mean(np.stack(feats).astype(np.float32), axis=0, keepdims=True))[0]
        else:
            info["feature"] = None
    return dict(infos)


def _chunk_index(chunk_id: str) -> int:
    if chunk_id.startswith("c"):
        return int(chunk_id[1:])
    return int(chunk_id)


def _exact_candidates(ids: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _read_csv(PHASE10O_DIR / "local2history_candidate_rows.csv"):
        if row.get("candidate_family") != "exact_overlap_frame_mask":
            continue
        a = str(row.get("mv_object_id_a"))
        b = str(row.get("mv_object_id_b"))
        if a not in ids or b not in ids:
            continue
        new = dict(row)
        new["phase_id"] = "v99_phase10p_overlap3_scene_stitch_semantic_sweep"
        out.append(new)
    return out


def _semantic_candidates(infos: dict[str, dict[str, Any]], *, tau: float, max_chunk_gap: int) -> list[dict[str, Any]]:
    by_scene: dict[str, list[str]] = defaultdict(list)
    for oid, info in infos.items():
        by_scene[str(info["scene_id"])].append(oid)
    out: list[dict[str, Any]] = []
    for scene, ids in sorted(by_scene.items()):
        for a in ids:
            ca = _chunk_index(str(infos[a]["chunk_id"]))
            fa = infos[a].get("feature")
            if fa is None:
                continue
            for b in ids:
                cb = _chunk_index(str(infos[b]["chunk_id"]))
                if cb <= ca or cb - ca > max_chunk_gap:
                    continue
                fb = infos[b].get("feature")
                if fb is None:
                    continue
                sim = float(np.dot(fa, fb))
                if sim < tau:
                    continue
                out.append(
                    {
                        "schema_version": "stream4d_v99_phase10p_local2history_candidate_v1",
                        "phase_id": "v99_phase10p_overlap3_scene_stitch_semantic_sweep",
                        "scene_id": scene,
                        "left_chunk_id": str(infos[a]["chunk_id"]),
                        "right_chunk_id": str(infos[b]["chunk_id"]),
                        "mv_object_id_a": a,
                        "mv_object_id_b": b,
                        "candidate_family": f"semantic_residual_tau{tau:.2f}_gap{max_chunk_gap}",
                        "shared_frame_mask_count": 0,
                        "shared_frame_count": 0,
                        "semantic_cosine": sim,
                        "chunk_gap": cb - ca,
                        "affinity": sim,
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    return out


def _one_to_one_mapping(ids: list[str], candidates: list[dict[str, Any]], *, variant_id: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    dsu = DSU(ids)
    used_left: set[tuple[str, str, str]] = set()
    used_right: set[tuple[str, str, str]] = set()
    accepted: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: (_num(item.get("affinity")), _num(item.get("shared_frame_mask_count")), str(item.get("mv_object_id_a")), str(item.get("mv_object_id_b"))), reverse=True):
        a = str(row.get("mv_object_id_a"))
        b = str(row.get("mv_object_id_b"))
        scene = str(row.get("scene_id"))
        left_chunk = str(row.get("left_chunk_id"))
        right_chunk = str(row.get("right_chunk_id"))
        left_key = (scene, left_chunk, a)
        right_key = (scene, right_chunk, b)
        if a not in dsu.parent or b not in dsu.parent or left_key in used_left or right_key in used_right:
            continue
        if dsu.union(a, b):
            new = dict(row)
            new["schema_version"] = "stream4d_v99_phase10p_local2history_merge_v1"
            new["variant_id"] = variant_id
            new["merge_policy"] = "overlap3_semantic_sweep_one_to_one_greedy"
            accepted.append(new)
            used_left.add(left_key)
            used_right.add(right_key)
    mapping = {oid: f"{variant_id}:{dsu.find(oid)}" for oid in ids}
    return mapping, accepted


def _apply_mapping(rows: list[dict[str, Any]], *, variant_id: str, mapping: dict[str, str], policy: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["phase10p_parent_mv_object_id"] = oid
        new["mv_object_id"] = mapping.get(oid, f"{variant_id}:{oid}")
        new["object_id"] = new["mv_object_id"]
        new["object_id_policy"] = policy
        new["score_scope"] = "current_chunk_score_scene_stitched_identity"
        new["score_policy"] = str(row.get("score_policy", "")) + "__phase10p_overlap3_semantic_sweep"
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def _component_stats(mapping: dict[str, str]) -> dict[str, Any]:
    comps: dict[str, int] = defaultdict(int)
    for root in mapping.values():
        comps[root] += 1
    original = len(mapping)
    scene_objects = len(comps)
    merged = original - scene_objects
    large = sum(1 for size in comps.values() if size > 3)
    return {
        "scene_object_count": scene_objects,
        "history_merge_count": merged,
        "history_split_count": 0,
        "duplicate_scene_object_count": merged,
        "fragmentation_rate_proxy": float(scene_objects / original) if original else 0.0,
        "overmerge_rate_proxy_large_component_gt3": float(large / scene_objects) if scene_objects else 0.0,
        "max_component_size": max(comps.values()) if comps else 0,
    }


def _aggregate_decoupled(variant_id: str, local_rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]], stats: dict[str, Any], candidate_count: int, accepted_count: int) -> dict[str, Any]:
    local_agg = p1._aggregate_metrics(local_rows)[0]
    scene_agg = p1._aggregate_metrics(scene_rows)[0]
    row = dict(local_agg)
    for key, value in scene_agg.items():
        if key.endswith("_scene"):
            row[key] = value
    row["variant_id"] = variant_id
    row["metric_composition"] = "local_from_phase10o_primary_chunk_ids_scene_from_phase10p_stitched_ids"
    row["history_candidate_count"] = int(candidate_count)
    row["accepted_history_merge_edge_count"] = int(accepted_count)
    row.update(stats)
    return row


def _eval(variant_id: str, rows: list[dict[str, Any]], eval_scope: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return p1._evaluate_variant(variant_id, rows, eval_scope)


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p10k._patch_phase1_inputs()
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    phase10o = json.loads((PHASE10O_DIR / "summary.json").read_text(encoding="utf-8"))

    scope = p10o._build_overlap3_scope()
    eval_scope = p10o._eval_scope_from_overlap(scope)
    base_rows = _base_rows()
    infos = _object_infos(base_rows)
    ids = sorted(infos)
    exact = _exact_candidates(set(ids))
    base_metric_rows, base_frame_rows = _eval(BASE_VARIANT, base_rows, eval_scope)
    base_agg = p1._aggregate_metrics(base_metric_rows)[0]
    base_agg["metric_composition"] = "phase10o_primary_chunk_scoped_ids"
    base_agg.update(_component_stats({oid: f"{BASE_VARIANT}:{oid}" for oid in ids}))
    base_agg["history_candidate_count"] = 0
    base_agg["accepted_history_merge_edge_count"] = 0

    # The v65 evaluator is CPU-heavy because each variant rereads full-resolution
    # masks and GT. Keep this sweep targeted; Phase10O already tested high
    # thresholds and adjacent-only exact overlap.
    variant_specs: list[dict[str, Any]] = []
    targeted_specs = [
        (0.85, 99),
        (0.80, 4),
        (0.80, 99),
        (0.75, 99),
        (0.70, 99),
    ]
    for tau, gap in targeted_specs:
        label = f"{tau:.2f}".replace(".", "p")
        variant_specs.append(
            {
                "variant_id": f"P_sem_tau{label}_gap{gap}",
                "family": "overlap_exact_plus_semantic_memory",
                "policy": f"exact_overlap_or_semantic_tau{tau:.2f}_max_chunk_gap{gap}",
                "candidates": exact + _semantic_candidates(infos, tau=tau, max_chunk_gap=gap),
                "tau": tau,
                "max_chunk_gap": gap,
            }
        )

    metric_rows: list[dict[str, Any]] = [base_agg]
    single_identity_metric_rows: list[dict[str, Any]] = [base_agg]
    scene_metric_rows: list[dict[str, Any]] = list(base_metric_rows)
    frame_rows: list[dict[str, Any]] = list(base_frame_rows)
    merge_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = [
        {
            "schema_version": "stream4d_v99_phase10p_variant_config_v1",
            "phase_id": "v99_phase10p_overlap3_scene_stitch_semantic_sweep",
            "variant_id": BASE_VARIANT,
            "family": "baseline",
            "policy": "phase10o_primary_emit_no_scene_stitch",
            "candidate_count": 0,
            "accepted_history_merge_edge_count": 0,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]

    for spec in variant_specs:
        variant_id = str(spec["variant_id"])
        candidates = list(spec["candidates"])
        mapping, accepted = _one_to_one_mapping(ids, candidates, variant_id=variant_id)
        rows = _apply_mapping(base_rows, variant_id=variant_id, mapping=mapping, policy=str(spec["policy"]))
        per_metric, frames = _eval(variant_id, rows, eval_scope)
        single_agg = p1._aggregate_metrics(per_metric)[0]
        stats = _component_stats(mapping)
        single_agg["metric_composition"] = "local_and_scene_from_same_stitched_ids"
        single_agg["history_candidate_count"] = len(candidates)
        single_agg["accepted_history_merge_edge_count"] = len(accepted)
        single_agg.update(stats)
        decoupled = _aggregate_decoupled(variant_id, base_metric_rows, per_metric, stats, len(candidates), len(accepted))
        metric_rows.append(decoupled)
        single_identity_metric_rows.append(single_agg)
        scene_metric_rows.extend(per_metric)
        frame_rows.extend(frames)
        merge_rows.extend(accepted)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase10p_variant_config_v1",
                "phase_id": "v99_phase10p_overlap3_scene_stitch_semantic_sweep",
                "variant_id": variant_id,
                "family": spec["family"],
                "policy": spec["policy"],
                "semantic_tau": spec["tau"],
                "max_chunk_gap": spec["max_chunk_gap"],
                "candidate_count": len(candidates),
                "accepted_history_merge_edge_count": len(accepted),
                "chunk_size": CHUNK_SIZE,
                "overlap": OVERLAP,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    best = max(metric_rows, key=lambda row: (_num(row.get("MV_AP_scene")), _num(row.get("MV_AP50_scene")), _num(row.get("MV_AP_window"))))
    best_single = max(single_identity_metric_rows, key=lambda row: (_num(row.get("MV_AP_scene")), _num(row.get("MV_AP50_scene")), _num(row.get("MV_AP_window"))))

    holdout_scene_gate = float(phase0["F2_base_holdout_MV_AP_scene"]) + 0.010
    holdout_scene_ap50_gate = float(phase0["F2_base_holdout_MV_AP50_scene"]) + 0.015
    holdout_window_floor = float(phase0["F2_base_holdout_MV_AP_window"]) - 0.003
    strict_local_gate_window = float(phase0["F2_base_holdout_MV_AP_window"]) + 0.005
    strict_local_gate_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"]) + 0.010
    local_gate = _num(best.get("MV_AP_window")) >= strict_local_gate_window and _num(best.get("MV_AP50_window")) >= strict_local_gate_ap50
    scene_gate = (
        _num(best.get("MV_AP_scene")) >= holdout_scene_gate
        and _num(best.get("MV_AP50_scene")) >= holdout_scene_ap50_gate
        and _num(best.get("MV_AP_window")) >= holdout_window_floor
    )
    safety_gate = (
        int(_num(best.get("same_frame_collision_count"), 1)) == 0
        and int(_num(best.get("missing_mask_raster_count"), 1)) == 0
        and not bool(scope.get("source_uses_future", False))
        and not bool(scope.get("source_uses_gt_for_prediction", False))
    )
    gate_rows = [
        {
            "gate_id": "strict_local_holdout_gate",
            "pass": local_gate,
            "expected": f"MV_AP_window>={strict_local_gate_window} and MV_AP50_window>={strict_local_gate_ap50}",
            "observed": f"{best['variant_id']} MV_AP_window={best.get('MV_AP_window')} MV_AP50_window={best.get('MV_AP50_window')}",
            "severity": "method_gate",
        },
        {
            "gate_id": "scene_holdout_gate_vs_F2_holdout",
            "pass": scene_gate,
            "expected": f"MV_AP_scene>={holdout_scene_gate} and MV_AP50_scene>={holdout_scene_ap50_gate} and MV_AP_window>={holdout_window_floor}",
            "observed": f"{best['variant_id']} MV_AP_scene={best.get('MV_AP_scene')} MV_AP50_scene={best.get('MV_AP50_scene')} MV_AP_window={best.get('MV_AP_window')}",
            "severity": "scene_method_gate",
        },
        {
            "gate_id": "safety_no_gt_no_future_no_missing_no_collision",
            "pass": safety_gate,
            "expected": "uses_gt_for_prediction=false; uses_future=false; same_frame_collision_count=0; missing_mask_raster_count=0",
            "observed": f"uses_gt={scope.get('source_uses_gt_for_prediction')} uses_future={scope.get('source_uses_future')} same_frame_collision={best.get('same_frame_collision_count')} missing_mask={best.get('missing_mask_raster_count')}",
            "severity": "safety",
        },
        {
            "gate_id": "formal_claim_allowed_after_repair",
            "pass": False,
            "expected": "fresh pre-registered run and surfel identity chunk-causal proof",
            "observed": "post-final semantic sweep; surfel_dependency_proven_chunk_causal=false remains a formal proof blocker",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If semantic sweep still fails scene gate, run GT-only oracle/IoU decomposition to distinguish under-merge from wrong local object birth; do not tune AP thresholds.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10p_overlap3_scene_stitch_semantic_sweep_summary_v1",
        "phase_id": "v99_phase10p_overlap3_scene_stitch_semantic_sweep",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "GO_OVERLAP3_SEMANTIC_SCENE_STITCH_METRIC_REPAIRED_FORMAL_REVIEW_REQUIRED" if (local_gate and scene_gate and safety_gate) else "NO_GO_OVERLAP3_SEMANTIC_SCENE_STITCH_SWEEP",
        "metric_gate_pass": bool(local_gate and scene_gate and safety_gate),
        "local_gate_pass": bool(local_gate),
        "scene_gate_pass": bool(scene_gate),
        "safety_gate_pass": bool(safety_gate),
        "formal_claim_allowed": False,
        "source_phase10o_summary": _rel(PHASE10O_DIR / "summary.json"),
        "source_phase10o_best_MV_AP_scene": phase10o.get("best_MV_AP_scene"),
        "best_variant_id": best["variant_id"],
        "best_MV_AP_window": float(_num(best.get("MV_AP_window"))),
        "best_MV_AP50_window": float(_num(best.get("MV_AP50_window"))),
        "best_MV_AP_scene": float(_num(best.get("MV_AP_scene"))),
        "best_MV_AP50_scene": float(_num(best.get("MV_AP50_scene"))),
        "best_history_merge_count": int(_num(best.get("history_merge_count"))),
        "best_scene_object_count": int(_num(best.get("scene_object_count"))),
        "best_max_component_size": int(_num(best.get("max_component_size"))),
        "best_overmerge_rate_proxy_large_component_gt3": float(_num(best.get("overmerge_rate_proxy_large_component_gt3"))),
        "best_single_identity_variant_id": best_single["variant_id"],
        "best_single_identity_MV_AP_window": float(_num(best_single.get("MV_AP_window"))),
        "best_single_identity_MV_AP_scene": float(_num(best_single.get("MV_AP_scene"))),
        "base_MV_AP_window": float(_num(base_agg.get("MV_AP_window"))),
        "base_MV_AP_scene": float(_num(base_agg.get("MV_AP_scene"))),
        "F2_base_holdout_MV_AP_window": float(phase0["F2_base_holdout_MV_AP_window"]),
        "F2_base_holdout_MV_AP50_window": float(phase0["F2_base_holdout_MV_AP50_window"]),
        "F2_base_holdout_MV_AP_scene": float(phase0["F2_base_holdout_MV_AP_scene"]),
        "F2_base_holdout_MV_AP50_scene": float(phase0["F2_base_holdout_MV_AP50_scene"]),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "single_identity_metric_rows": _rel(OUT_DIR / "single_identity_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "local2history_merge_rows": _rel(OUT_DIR / "local2history_merge_rows.csv"),
            "frame_eval_rows": _rel(OUT_DIR / "frame_eval_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "variant_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "single_identity_metric_rows.csv", single_identity_metric_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", scene_metric_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "local2history_merge_rows.csv", merge_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["metric_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
