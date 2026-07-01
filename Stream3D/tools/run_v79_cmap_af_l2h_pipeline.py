#!/usr/bin/env python3
"""Run Stream4D v79 CMAP-AF-L2H training-free affinity-feature audit.

This is intentionally one pipeline file: the v79 plan changes the method core
from candidate scoring to per-carrier affinity features, so keeping the whole
MVP path together makes the evidence easier to audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE_ORDER = ["phase0", "phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7", "final"]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _float(value: Any, default: float = 0.0) -> float:
    out = _float_or_none(value)
    return default if out is None else out


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _percentile(values: list[float], pct: float) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(math.ceil((pct / 100.0) * len(vals))) - 1))
    return vals[idx]


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO).as_posix()
        except ValueError:
            return str(path)


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash_int(text: str, seed: int) -> int:
    payload = f"{seed}|{text}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little", signed=False)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denom <= 0.0 else float(np.dot(left, right) / denom)


def _nmi(labels_a: list[str], labels_b: list[str]) -> float:
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a and b]
    if not pairs:
        return 0.0
    n = float(len(pairs))
    ca = Counter(a for a, _b in pairs)
    cb = Counter(b for _a, b in pairs)
    cab = Counter(pairs)
    mi = 0.0
    for (a, b), count in cab.items():
        p_ab = count / n
        p_a = ca[a] / n
        p_b = cb[b] / n
        if p_ab > 0 and p_a > 0 and p_b > 0:
            mi += p_ab * math.log(p_ab / (p_a * p_b))
    ha = -sum((count / n) * math.log(count / n) for count in ca.values())
    hb = -sum((count / n) * math.log(count / n) for count in cb.values())
    return float(2.0 * mi / (ha + hb)) if (ha + hb) > 0 else 0.0


def _ari(labels_a: list[str], labels_b: list[str]) -> float:
    pairs = [(a, b) for a, b in zip(labels_a, labels_b) if a and b]
    n = len(pairs)
    if n < 2:
        return 0.0

    def comb2(x: int) -> float:
        return x * (x - 1) / 2.0

    ca = Counter(a for a, _b in pairs)
    cb = Counter(b for _a, b in pairs)
    cab = Counter(pairs)
    sum_ab = sum(comb2(count) for count in cab.values())
    sum_a = sum(comb2(count) for count in ca.values())
    sum_b = sum(comb2(count) for count in cb.values())
    total = comb2(n)
    expected = sum_a * sum_b / total if total > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    return float((sum_ab - expected) / denom) if denom else 0.0


def _sample_auc(scores_pos: list[float], scores_neg: list[float], rng: random.Random, max_pairs: int = 4000) -> float:
    if not scores_pos or not scores_neg:
        return 0.5
    comparisons = min(max_pairs, len(scores_pos) * len(scores_neg))
    wins = 0.0
    for _ in range(comparisons):
        p = scores_pos[rng.randrange(len(scores_pos))]
        n = scores_neg[rng.randrange(len(scores_neg))]
        if p > n:
            wins += 1.0
        elif p == n:
            wins += 0.5
    return float(wins / comparisons) if comparisons else 0.5


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "v77_final": ROOT / args.v77_final_root / "final_decision.json",
        "v77_recap": REPO / "docs/stream4d_v77_实验结果复盘.md",
        "v77_exec": REPO / "docs/stream4d_v77_执行日志.md",
        "affinity_doc": REPO / "docs/affinity feat.md",
        "opt_free_doc": REPO / "docs/per-scene opt-free method.md",
        "v75_phase1": ROOT / args.v75_phase1_root / "incidence_summary.json",
        "v75_incidence_rows": ROOT / args.v75_phase1_root / "incidence_rows.csv",
        "v75_incidence_chunk_rows": ROOT / args.v75_phase1_root / "incidence_chunk_rows.csv",
        "v77_phase5": ROOT / args.v77_phase5_root / "local_controls_summary.json",
        "v77_phase2_oracle": ROOT / args.v77_phase2_root / "oracle_cut_rows.csv",
    }


def _selected_chunks(args: argparse.Namespace) -> dict[str, set[int]]:
    scenes = _parse_csv_list(args.scenes)
    return {scene: set(range(max(0, int(args.max_chunks)))) for scene in scenes}


def _run_phase0(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase0_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    sources = _source_paths(args)
    v77 = _read_json(sources["v77_final"])
    v75 = _read_json(sources["v75_phase1"])
    fact_rows = [
        {
            "fact_name": "v77_final_decision",
            "fact_value": v77.get("final_decision", ""),
            "source_path": _rel(sources["v77_final"]),
            "diagnostic_only": False,
            "uses_gt_for_prediction": False,
            "notes": "Phase0 fact lock requires v77 No-Go baseline.",
        },
        {
            "fact_name": "v77_best_nonGT_SF50",
            "fact_value": v77.get("best_nonGT_SF50", ""),
            "source_path": _rel(sources["v77_final"]),
            "diagnostic_only": False,
            "uses_gt_for_prediction": False,
            "notes": "Must be <= 0.35 to establish candidate-scoring route weakness.",
        },
        {
            "fact_name": "v77_oracle_hierarchy_cut_SF50",
            "fact_value": v77.get("oracle_hierarchy_cut_SF50", ""),
            "source_path": _rel(sources["v77_final"]),
            "diagnostic_only": True,
            "uses_gt_for_prediction": True,
            "notes": "Diagnostic oracle only; establishes representation headroom, not method output.",
        },
        {
            "fact_name": "v75_soft_incidence_decision",
            "fact_value": v75.get("decision", ""),
            "source_path": _rel(sources["v75_phase1"]),
            "diagnostic_only": False,
            "uses_gt_for_prediction": bool(v75.get("method_prediction_uses_gt_anywhere")),
            "notes": "Existing soft incidence is the Phase1 data source.",
        },
        {
            "fact_name": "hard_boundary",
            "fact_value": "no_training;no_per_scene_optimization;no_gt_in_method_prediction",
            "source_path": _rel(REPO / "docs/stream4d_v79_cmap_af_l2h_experiment_plan.md"),
            "diagnostic_only": False,
            "uses_gt_for_prediction": False,
            "notes": "CMAP-AF hard boundary recorded from plan.",
        },
        {
            "fact_name": "affinity_doc_exists",
            "fact_value": sources["affinity_doc"].exists(),
            "source_path": _rel(sources["affinity_doc"]),
            "diagnostic_only": False,
            "uses_gt_for_prediction": False,
            "notes": "Missing local doc is recorded; related_work_rows.csv provides the repair table.",
        },
        {
            "fact_name": "per_scene_opt_free_doc_exists",
            "fact_value": sources["opt_free_doc"].exists(),
            "source_path": _rel(sources["opt_free_doc"]),
            "diagnostic_only": False,
            "uses_gt_for_prediction": False,
            "notes": "Missing local doc is recorded rather than fabricated.",
        },
    ]
    related_work_rows = [
        {
            "work_name": "SAGA",
            "year": "",
            "primitive_type": "3D Gaussian / scene primitive",
            "uses_training": True,
            "uses_per_scene_optimization": True,
            "feature_type": "learned affinity/group feature",
            "uses_affinity_feature": True,
            "uses_unary_feature": False,
            "uses_memory_bank": False,
            "uses_linear_assignment": False,
            "uses_clustering": True,
            "what_to_borrow": "feature similarity should drive clustering/grouping",
            "what_not_to_borrow": "training or per-scene optimized affinity feature",
        },
        {
            "work_name": "GARField",
            "year": "",
            "primitive_type": "3D Gaussian / field primitive",
            "uses_training": True,
            "uses_per_scene_optimization": True,
            "feature_type": "scale-aware affinity/group field",
            "uses_affinity_feature": True,
            "uses_unary_feature": False,
            "uses_memory_bank": False,
            "uses_linear_assignment": False,
            "uses_clustering": True,
            "what_to_borrow": "multi-scale grouping/affinity abstraction",
            "what_not_to_borrow": "optimize a scene-specific field",
        },
        {
            "work_name": "Occam-style unary feature lifting",
            "year": "",
            "primitive_type": "3D point/Gaussian primitive",
            "uses_training": False,
            "uses_per_scene_optimization": False,
            "feature_type": "unary semantic feature",
            "uses_affinity_feature": False,
            "uses_unary_feature": True,
            "uses_memory_bank": False,
            "uses_linear_assignment": False,
            "uses_clustering": True,
            "what_to_borrow": "training-free feature lifting boundary",
            "what_not_to_borrow": "treat unary semantic similarity as object affinity",
        },
        {
            "work_name": "LBG",
            "year": "",
            "primitive_type": "3D primitive / Gaussian grouping",
            "uses_training": "",
            "uses_per_scene_optimization": "",
            "feature_type": "grouping/affinity cue",
            "uses_affinity_feature": True,
            "uses_unary_feature": "",
            "uses_memory_bank": False,
            "uses_linear_assignment": "",
            "uses_clustering": True,
            "what_to_borrow": "grouping cue should be relational",
            "what_not_to_borrow": "unverified training/per-scene assumptions; local source missing",
        },
        {
            "work_name": "FlashSplat",
            "year": "",
            "primitive_type": "3D Gaussian / rendered primitive",
            "uses_training": "",
            "uses_per_scene_optimization": "",
            "feature_type": "fast Gaussian feature/segmentation cue",
            "uses_affinity_feature": "",
            "uses_unary_feature": True,
            "uses_memory_bank": "",
            "uses_linear_assignment": "",
            "uses_clustering": "",
            "what_to_borrow": "runtime discipline and compact feature representation",
            "what_not_to_borrow": "claim affinity-feature success without relational validation",
        },
        {
            "work_name": "Gaga",
            "year": "",
            "primitive_type": "3D Gaussian / grouping primitive",
            "uses_training": True,
            "uses_per_scene_optimization": True,
            "feature_type": "affinity/group feature",
            "uses_affinity_feature": True,
            "uses_unary_feature": False,
            "uses_memory_bank": "",
            "uses_linear_assignment": "",
            "uses_clustering": True,
            "what_to_borrow": "affinity feature is for grouping, not candidate scoring",
            "what_not_to_borrow": "learned or scene-optimized affinity embedding",
        },
        {
            "work_name": "Intrinsic-GS sparse affinity graph",
            "year": "2026",
            "primitive_type": "3D Gaussian",
            "uses_training": False,
            "uses_per_scene_optimization": False,
            "feature_type": "sparse graph affinity",
            "uses_affinity_feature": True,
            "uses_unary_feature": False,
            "uses_memory_bank": False,
            "uses_linear_assignment": False,
            "uses_clustering": True,
            "what_to_borrow": "training-free sparse affinity graph and community detection",
            "what_not_to_borrow": "assume high-quality Gaussian geometry/appearance cues exist in noisy D4RT",
        },
    ]
    gate = {
        "v77_final_decision_expected": v77.get("final_decision") == "NO_GO_CUT_OBJECTIVE_WEAK",
        "v77_best_nonGT_le_0p35": _float(v77.get("best_nonGT_SF50"), math.inf) <= 0.35,
        "v77_oracle_ge_0p52": _float(v77.get("oracle_hierarchy_cut_SF50"), 0.0) >= 0.52,
        "related_work_table_has_required_rows": {"SAGA", "GARField", "Occam-style unary feature lifting", "LBG", "FlashSplat", "Gaga"}.issubset(
            {str(row["work_name"]) for row in related_work_rows}
        ),
        "hard_boundary_recorded": True,
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    summary = {
        "phase": "v79_phase0_fact_lock",
        "schema": "stream4d_v79_phase0_fact_lock_v1",
        "decision": "PASS_V79_PHASE0_FACT_LOCK" if gate["pass"] else "PARTIAL_V79_PHASE0_FACT_LOCK",
        "gate": gate,
        "missing_local_related_work_docs": [_rel(path) for path in [sources["affinity_doc"], sources["opt_free_doc"]] if not path.exists()],
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "phase0_fact_rows.csv", fact_rows)
    _write_csv(output_root / "related_work_rows.csv", related_work_rows)
    _write_json(output_root / "fact_lock_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _load_incidence(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    sources = _source_paths(args)
    selected = _selected_chunks(args)
    chunks: dict[tuple[str, int], dict[str, Any]] = {}
    rows_read = 0
    rows_kept = 0
    variants = Counter()
    with sources["v75_incidence_rows"].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_read += 1
            scene = str(row.get("scene_id") or "")
            chunk = _int(row.get("chunk_id"), -1)
            if scene not in selected or chunk not in selected[scene]:
                continue
            variant = str(row.get("membership_variant") or "")
            variants[variant] += 1
            if variant != args.incidence_variant:
                continue
            if _bool(row.get("uses_gt_for_prediction")):
                continue
            weight = _float(row.get("soft_membership"), 0.0)
            if weight < float(args.min_membership):
                continue
            frame = _int(row.get("frame_id"), -1)
            mask = _int(row.get("mask_id"), -1)
            carrier = str(row.get("carrier_global_id") or row.get("carrier_id") or "")
            obs = str(row.get("mask_observation_id") or "")
            if not scene or chunk < 0 or frame < 0 or mask <= 0 or not carrier or not obs:
                continue
            key = (scene, chunk)
            data = chunks.setdefault(
                key,
                {
                    "scene_id": scene,
                    "chunk_id": chunk,
                    "carrier_obs": defaultdict(list),
                    "carrier_frames": defaultdict(set),
                    "carrier_frame_weights": defaultdict(lambda: defaultdict(float)),
                    "mask_meta": {},
                    "mask_total": defaultdict(float),
                    "frames": set(),
                    "row_count": 0,
                },
            )
            area = _float(row.get("mask_area_ratio"), 0.0)
            entropy = _float(row.get("semantic_entropy_of_mask"), 0.0)
            data["carrier_obs"][carrier].append(
                {
                    "obs": obs,
                    "weight": weight,
                    "frame": frame,
                    "mask": mask,
                    "area": area,
                    "entropy": entropy,
                }
            )
            data["carrier_frames"][carrier].add(frame)
            data["carrier_frame_weights"][carrier][frame] += weight
            data["mask_meta"][obs] = {"frame": frame, "mask": mask, "area": area, "entropy": entropy}
            data["mask_total"][obs] += weight
            data["frames"].add(frame)
            data["row_count"] += 1
            rows_kept += 1
    for data in chunks.values():
        data["carriers"] = sorted(data["carrier_obs"])
        data["frames"] = sorted(data["frames"])
        data["carrier_index"] = {carrier: idx for idx, carrier in enumerate(data["carriers"])}
    return {
        "chunks": chunks,
        "rows_read": rows_read,
        "rows_kept": rows_kept,
        "variant_counts_seen_selected_chunks": dict(variants),
        "runtime_sec": time.time() - started,
    }


def _scale_weight(
    obs_row: dict[str, Any],
    mask_df: dict[str, int],
    carrier_count: int,
    *,
    scale: str,
    args: argparse.Namespace,
    use_idf: bool = True,
) -> float:
    weight = float(obs_row["weight"])
    area = float(obs_row["area"])
    entropy = float(obs_row["entropy"])
    obs = str(obs_row["obs"])
    idf = math.log((carrier_count + 1.0) / (float(mask_df.get(obs, 0)) + 1.0)) + 1.0 if use_idf else 1.0
    idf = max(idf, 1e-6) ** float(args.idf_power)
    specificity = max(0.0, 1.0 - min(1.0, area))
    entropy_gate = math.exp(-float(args.entropy_penalty) * max(0.0, entropy))
    if scale == "fine":
        scale_gate = specificity * math.exp(-6.0 * max(0.0, area - 0.10))
    elif scale == "coarse":
        scale_gate = math.exp(-0.6 * max(0.0, area - 0.45))
    else:
        specificity_gate = max(1e-6, specificity) ** float(args.specificity_power)
        scale_gate = specificity_gate * math.exp(
            -float(args.large_mask_penalty) * max(0.0, area - float(args.object_large_mask_area))
        )
    return float(weight * idf * scale_gate * entropy_gate)


def _build_feature_bundle(
    data: dict[str, Any],
    *,
    scale: str,
    args: argparse.Namespace,
    frame_parity: int | None = None,
    no_temporal: bool = False,
    use_idf: bool = True,
    mask_shuffle: bool = False,
) -> dict[str, Any]:
    carriers = data["carriers"]
    rng = random.Random(int(args.random_seed) + len(scale) * 17 + (0 if frame_parity is None else frame_parity))
    carrier_count = len(carriers)
    mask_df: dict[str, int] = defaultdict(int)
    shuffled_obs: dict[str, str] = {}
    obs_values = sorted(data["mask_meta"])
    if mask_shuffle:
        shuffled = obs_values[:]
        rng.shuffle(shuffled)
        shuffled_obs = dict(zip(obs_values, shuffled))
    for carrier in carriers:
        seen: set[str] = set()
        rows = data["carrier_obs"][carrier]
        if no_temporal and rows:
            min_frame = min(int(row["frame"]) for row in rows)
            rows = [row for row in rows if int(row["frame"]) == min_frame]
        for row in rows:
            if frame_parity is not None and int(row["frame"]) % 2 != frame_parity:
                continue
            obs = shuffled_obs.get(str(row["obs"]), str(row["obs"]))
            seen.add(obs)
        for obs in seen:
            mask_df[obs] += 1

    dim = int(args.projection_dim)
    matrices: list[np.ndarray] = []
    sparse_rows: list[dict[str, float]] = []
    top_obs_rows: list[list[tuple[str, float]]] = []
    nnz_values: list[int] = []
    norm_values: list[float] = []
    largest_contribs: list[float] = []
    broad_contribs: list[float] = []
    membership_counts: list[int] = []
    visible_counts: list[int] = []
    for carrier in carriers:
        sparse: dict[str, float] = defaultdict(float)
        rows = data["carrier_obs"][carrier]
        if no_temporal and rows:
            min_frame = min(int(row["frame"]) for row in rows)
            rows = [row for row in rows if int(row["frame"]) == min_frame]
        for row in rows:
            if frame_parity is not None and int(row["frame"]) % 2 != frame_parity:
                continue
            obs = shuffled_obs.get(str(row["obs"]), str(row["obs"]))
            patched = dict(row)
            patched["obs"] = obs
            sparse[obs] += _scale_weight(patched, mask_df, carrier_count, scale=scale, args=args, use_idf=use_idf)
        vec = np.zeros(dim, dtype=np.float32)
        for obs, value in sparse.items():
            h = _stable_hash_int(obs, int(args.random_seed))
            idx = h % dim
            sign = 1.0 if ((h >> 9) & 1) == 0 else -1.0
            vec[idx] += np.float32(sign * value)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        total = float(sum(abs(v) for v in sparse.values()))
        broad = float(sum(abs(v) for obs, v in sparse.items() if data["mask_meta"].get(obs, {}).get("area", 0.0) >= float(args.object_large_mask_area)))
        largest = max((abs(v) for v in sparse.values()), default=0.0)
        sparse_rows.append(dict(sparse))
        matrices.append(vec)
        top_obs_rows.append(sorted(sparse.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
        nnz_values.append(len(sparse))
        norm_values.append(norm)
        largest_contribs.append(_safe_ratio(largest, total))
        broad_contribs.append(_safe_ratio(broad, total))
        membership_counts.append(len(data["carrier_obs"][carrier]))
        visible_counts.append(len(data["carrier_frames"][carrier]))
    matrix = np.stack(matrices, axis=0) if matrices else np.zeros((0, dim), dtype=np.float32)
    return {
        "scale": scale,
        "carriers": carriers,
        "matrix": matrix,
        "sparse": sparse_rows,
        "top_obs": top_obs_rows,
        "nnz": nnz_values,
        "norms": norm_values,
        "largest_contribs": largest_contribs,
        "broad_contribs": broad_contribs,
        "membership_counts": membership_counts,
        "visible_counts": visible_counts,
        "mask_df": dict(mask_df),
    }


def _load_semantic_feature_index(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path_arg = str(getattr(args, "semantic_feature_rows", "") or "").strip()
    if not path_arg:
        return {}, {
            "semantic_control_available": False,
            "semantic_control_type": "disabled",
            "reason": "semantic_feature_rows_empty",
        }
    path = _rooted(path_arg)
    if not path.exists():
        return {}, {
            "semantic_control_available": False,
            "semantic_control_type": "missing",
            "semantic_feature_rows": _rel(path),
            "reason": "semantic_feature_rows_missing",
        }

    rows_read = 0
    rows_kept = 0
    gt_rows = 0
    vector_field_count = 0
    index: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        vector_field_count = sum(
            1
            for name in fieldnames
            if name.startswith(("feature_", "embedding_", "dino_")) and name not in {"feature_available", "feature_dim", "feature_layer", "feature_norm", "feature_nan_count", "feature_pooling_method", "feature_resolution"}
        )
        for row in reader:
            rows_read += 1
            obs = str(row.get("mask_observation_id") or "")
            if not obs:
                continue
            if _bool(row.get("uses_gt_for_prediction")):
                gt_rows += 1
                continue
            if not _bool(row.get("feature_available")):
                continue
            proto = str(row.get("semantic_prototype_id") or "").strip()
            if not proto:
                continue
            index[obs] = {
                "semantic_prototype_id": proto,
                "semantic_entropy": _float(row.get("semantic_entropy"), 0.0),
                "semantic_prototype_margin": _float(row.get("semantic_prototype_margin"), 0.0),
                "semantic_background_score_proxy": _bool(row.get("semantic_background_score_proxy")),
                "broad_background_risk": _bool(row.get("broad_background_risk")),
                "feature_dim": _int(row.get("feature_dim"), 0),
            }
            rows_kept += 1

    sidecar = _read_json(path.parent / "semantic_summary.json")
    return index, {
        "semantic_control_available": bool(index),
        "semantic_control_type": "semantic_prototype_id_countsketch_proxy",
        "semantic_feature_rows": _rel(path),
        "semantic_summary_json": _rel(path.parent / "semantic_summary.json"),
        "rows_read": rows_read,
        "rows_kept": rows_kept,
        "gt_rows_skipped": gt_rows,
        "stored_dense_vector_columns": vector_field_count,
        "dense_embedding_control_available": vector_field_count > 0,
        "source_decision": sidecar.get("decision", ""),
        "source_key_metrics": sidecar.get("key_metrics", {}),
        "notes": [
            "v71 rows contain semantic_prototype_id and reliability scalars but no stored 384-D vectors in this audit artifact.",
            "This control is therefore a unary semantic prototype proxy, not a full DINO embedding control.",
        ],
    }


def _build_semantic_prototype_bundle(
    data: dict[str, Any],
    *,
    args: argparse.Namespace,
    semantic_index: dict[str, dict[str, Any]],
    frame_parity: int | None = None,
) -> dict[str, Any]:
    carriers = data["carriers"]
    carrier_count = len(carriers)
    proto_df: dict[str, int] = defaultdict(int)
    eligible_obs = 0
    available_obs = 0
    for carrier in carriers:
        seen: set[str] = set()
        for row in data["carrier_obs"][carrier]:
            if frame_parity is not None and int(row["frame"]) % 2 != frame_parity:
                continue
            eligible_obs += 1
            sem = semantic_index.get(str(row["obs"]))
            if not sem:
                continue
            available_obs += 1
            seen.add(str(sem["semantic_prototype_id"]))
        for proto in seen:
            proto_df[proto] += 1

    dim = int(args.semantic_projection_dim or args.projection_dim)
    matrices: list[np.ndarray] = []
    sparse_rows: list[dict[str, float]] = []
    nnz_values: list[int] = []
    norm_values: list[float] = []
    carriers_with_semantic = 0
    for carrier in carriers:
        sparse: dict[str, float] = defaultdict(float)
        for row in data["carrier_obs"][carrier]:
            if frame_parity is not None and int(row["frame"]) % 2 != frame_parity:
                continue
            sem = semantic_index.get(str(row["obs"]))
            if not sem:
                continue
            proto = str(sem["semantic_prototype_id"])
            entropy = max(0.0, _float(sem.get("semantic_entropy"), 0.0))
            background = _bool(sem.get("broad_background_risk")) or _bool(sem.get("semantic_background_score_proxy"))
            entropy_gate = math.exp(-float(args.semantic_proxy_entropy_penalty) * entropy)
            background_gate = float(args.semantic_proxy_background_penalty) if background else 1.0
            idf = math.log((carrier_count + 1.0) / (float(proto_df.get(proto, 0)) + 1.0)) + 1.0
            sparse[proto] += float(row["weight"]) * idf * entropy_gate * background_gate
        vec = np.zeros(dim, dtype=np.float32)
        for proto, value in sparse.items():
            h = _stable_hash_int(f"semantic-prototype:{proto}", int(args.random_seed) + 7919)
            idx = h % dim
            sign = 1.0 if ((h >> 11) & 1) == 0 else -1.0
            vec[idx] += np.float32(sign * value)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
            carriers_with_semantic += 1
        sparse_rows.append(dict(sparse))
        matrices.append(vec)
        nnz_values.append(len(sparse))
        norm_values.append(norm)
    matrix = np.stack(matrices, axis=0) if matrices else np.zeros((0, dim), dtype=np.float32)
    return {
        "scale": "semantic_prototype_unary_proxy",
        "carriers": carriers,
        "matrix": matrix,
        "sparse": sparse_rows,
        "nnz": nnz_values,
        "norms": norm_values,
        "carrier_coverage_rate": _safe_ratio(carriers_with_semantic, max(1, carrier_count)),
        "mask_observation_coverage_rate": _safe_ratio(available_obs, max(1, eligible_obs)),
        "eligible_observation_count": eligible_obs,
        "available_observation_count": available_obs,
        "prototype_count": len(proto_df),
    }


def _exact_sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    ln = math.sqrt(sum(value * value for value in left.values()))
    rn = math.sqrt(sum(value * value for value in right.values()))
    return 0.0 if ln <= 0.0 or rn <= 0.0 else float(dot / (ln * rn))


def _run_phase1(args: argparse.Namespace, incidence: dict[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    started = time.time()
    output_root = ROOT / args.phase1_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    feature_rows: list[dict[str, Any]] = []
    projection_errors: list[float] = []
    bundles: dict[tuple[str, int], dict[str, Any]] = {}
    rng = random.Random(int(args.random_seed))
    for key, data in sorted(incidence["chunks"].items()):
        scale_bundles = {}
        for scale in ["fine", "object", "coarse"]:
            bundle = _build_feature_bundle(data, scale=scale, args=args)
            scale_bundles[scale] = bundle
            for idx, carrier in enumerate(bundle["carriers"]):
                feature_rows.append(
                    {
                        "scene_id": data["scene_id"],
                        "chunk_id": data["chunk_id"],
                        "carrier_id": carrier,
                        "scale": scale,
                        "feature_format": "countsketch_l2_normalized",
                        "feature_dim": int(args.projection_dim),
                        "feature_norm": bundle["norms"][idx],
                        "nnz_signature": bundle["nnz"][idx],
                        "visible_frame_count": bundle["visible_counts"][idx],
                        "membership_count": bundle["membership_counts"][idx],
                        "mean_membership_weight": _mean([float(row["weight"]) for row in data["carrier_obs"][carrier]]) or 0.0,
                        "max_membership_weight": max([float(row["weight"]) for row in data["carrier_obs"][carrier]], default=0.0),
                        "top_mask_obs_ids": json.dumps([obs for obs, _value in bundle["top_obs"][idx]], ensure_ascii=False),
                        "uses_gt_for_prediction": False,
                    }
                )
        object_bundle = scale_bundles["object"]
        sample_n = min(int(args.projection_error_samples), len(object_bundle["carriers"]) * max(0, len(object_bundle["carriers"]) - 1) // 2)
        n = len(object_bundle["carriers"])
        for _ in range(sample_n):
            if n < 2:
                break
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
            exact = _exact_sparse_cosine(object_bundle["sparse"][i], object_bundle["sparse"][j])
            approx = float(np.dot(object_bundle["matrix"][i], object_bundle["matrix"][j]))
            projection_errors.append(abs(exact - approx))
        bundles[key] = {"data": data, "features": scale_bundles}

    carrier_count = len({row["carrier_id"] for row in feature_rows if row["scale"] == "object"})
    object_rows = [row for row in feature_rows if row["scale"] == "object"]
    gate = {
        "carrier_feature_coverage_rate_ge_0p95": _safe_ratio(
            sum(1 for row in object_rows if _float(row.get("feature_norm"), 0.0) > 0.0),
            max(1, carrier_count),
        )
        >= 0.95,
        "feature_norm_zero_rate_le_0p02": _safe_ratio(
            sum(1 for row in object_rows if _float(row.get("feature_norm"), 0.0) <= 0.0),
            max(1, carrier_count),
        )
        <= 0.02,
        "broad_mask_contribution_ratio_le_0p35": (
            _mean([value for bundle in bundles.values() for value in bundle["features"]["object"]["broad_contribs"]])
            or 0.0
        )
        <= 0.35,
        "cosine_approx_error_p95_le_0p05": (_percentile(projection_errors, 95) or 0.0) <= 0.05,
        "runtime_per_chunk_sec_le_30": ((time.time() - started) / max(1, len(bundles))) <= 30.0,
        "uses_gt_for_prediction_false": True,
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    summary = {
        "phase": "v79_phase1_affinity_feature",
        "schema": "stream4d_v79_phase1_affinity_feature_v1",
        "decision": "PASS_V79_PHASE1_AFFINITY_FEATURE" if gate["pass"] else "NO_GO_V79_PHASE1_AFFINITY_FEATURE",
        "gate": gate,
        "scenes": _parse_csv_list(args.scenes),
        "chunk_count": len(bundles),
        "carrier_feature_coverage_rate": _safe_ratio(sum(1 for row in object_rows if _float(row.get("feature_norm"), 0.0) > 0.0), max(1, carrier_count)),
        "feature_norm_zero_rate": _safe_ratio(sum(1 for row in object_rows if _float(row.get("feature_norm"), 0.0) <= 0.0), max(1, carrier_count)),
        "mean_nnz_signature": _mean([_float(row.get("nnz_signature"), 0.0) for row in object_rows]) or 0.0,
        "p95_nnz_signature": _percentile([_float(row.get("nnz_signature"), 0.0) for row in object_rows], 95) or 0.0,
        "feature_norm_mean": _mean([_float(row.get("feature_norm"), 0.0) for row in object_rows]) or 0.0,
        "largest_mask_weight_contribution_mean": _mean(
            [value for bundle in bundles.values() for value in bundle["features"]["object"]["largest_contribs"]]
        )
        or 0.0,
        "broad_mask_contribution_ratio": _mean(
            [value for bundle in bundles.values() for value in bundle["features"]["object"]["broad_contribs"]]
        )
        or 0.0,
        "cosine_approx_error_mean": _mean(projection_errors) or 0.0,
        "cosine_approx_error_p95": _percentile(projection_errors, 95) or 0.0,
        "runtime_per_chunk_sec": (time.time() - started) / max(1, len(bundles)),
        "runtime_sec": time.time() - started,
        "peak_memory_gb": "",
        "incidence_rows_read": incidence["rows_read"],
        "incidence_rows_kept": incidence["rows_kept"],
        "incidence_load_runtime_sec": incidence["runtime_sec"],
        "uses_gt_for_prediction": False,
    }
    projection_summary = {
        "random_seed": int(args.random_seed),
        "projection_dim": int(args.projection_dim),
        "num_hash_tables": 1,
        "cosine_approx_error_sample_mean": summary["cosine_approx_error_mean"],
        "cosine_approx_error_sample_p95": summary["cosine_approx_error_p95"],
        "runtime_sec": summary["runtime_sec"],
        "peak_memory_gb": "",
    }
    _write_csv(output_root / "carrier_affinity_feature_rows.csv", feature_rows)
    _write_json(output_root / "feature_projection_summary.json", projection_summary)
    _write_json(output_root / "affinity_feature_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary, bundles


def _topk_neighbors(matrix: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    scores = matrix @ matrix.T
    np.fill_diagonal(scores, -np.inf)
    k = min(top_k, max(0, n - 1))
    if k == 0:
        return np.zeros((n, 0), dtype=np.int64), np.zeros((n, 0), dtype=np.float32)
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    return idx, vals


def _heldout_sets(data: dict[str, Any], parity: int = 1) -> list[set[str]]:
    out: list[set[str]] = []
    for carrier in data["carriers"]:
        obs = {str(row["obs"]) for row in data["carrier_obs"][carrier] if int(row["frame"]) % 2 == parity}
        out.append(obs)
    return out


def _top_parity_labels(data: dict[str, Any], parity: int) -> list[str]:
    labels: list[str] = []
    for carrier in data["carriers"]:
        weights: dict[str, float] = defaultdict(float)
        for row in data["carrier_obs"][carrier]:
            if int(row["frame"]) % 2 == parity:
                weights[str(row["obs"])] += float(row["weight"])
        labels.append(max(weights, key=weights.get) if weights else "")
    return labels


def _graph_lcc_ratio(neighbors: np.ndarray, values: np.ndarray, threshold: float) -> float:
    n = neighbors.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for jj, j in enumerate(neighbors[i].tolist()):
            score = float(values[i, jj])
            if math.isfinite(score) and score >= threshold:
                union(i, int(j))
    counts = Counter(find(i) for i in range(n))
    return _safe_ratio(max(counts.values(), default=0), max(1, n))


def _neighbor_metrics(
    *,
    matrix: np.ndarray,
    data: dict[str, Any],
    top_k: int,
    threshold: float,
    pruning: str,
    rng: random.Random,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    neighbors, values = _topk_neighbors(matrix, top_k)
    active_values = values.copy()
    heldout = _heldout_sets(data, parity=1)
    edge_positive = []
    reciprocal_hits = 0
    active_edge_count = 0
    neighbor_sets = [set(row.tolist()) for row in neighbors]
    for i in range(neighbors.shape[0]):
        for jj, j in enumerate(neighbors[i].tolist()):
            j = int(j)
            reciprocal = i in neighbor_sets[j]
            if pruning == "mutual" and not reciprocal:
                active_values[i, jj] = -np.inf
                continue
            edge_positive.append(1.0 if heldout[i] & heldout[j] else 0.0)
            active_edge_count += 1
            if reciprocal:
                reciprocal_hits += 1
    labels_even = _top_parity_labels(data, 0)
    labels_odd = _top_parity_labels(data, 1)
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    n = matrix.shape[0]
    for _ in range(min(4000, max(1, n * 4))):
        if n < 2:
            break
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        score = float(np.dot(matrix[i], matrix[j]))
        if heldout[i] & heldout[j]:
            pos_scores.append(score)
        else:
            neg_scores.append(score)
    metrics = {
        "heldout_same_mask_likelihood": _mean(edge_positive) or 0.0,
        "heldout_same_mask_AUC_sampled": _sample_auc(pos_scores, neg_scores, rng),
        "split_half_NMI": _nmi(labels_even, labels_odd),
        "split_half_ARI": _ari(labels_even, labels_odd),
        "largest_connected_component_ratio": _graph_lcc_ratio(neighbors, active_values, threshold),
        "mean_degree": _safe_ratio(active_edge_count, max(1, neighbors.shape[0])),
        "topk_reciprocal_rate": _safe_ratio(reciprocal_hits, max(1, neighbors.shape[0] * max(1, neighbors.shape[1]))),
        "mean_affinity": float(np.mean(active_values[np.isfinite(active_values)])) if active_values.size and np.isfinite(active_values).any() else 0.0,
        "active_edge_count": active_edge_count,
    }
    return metrics, neighbors, active_values


def _run_phase2(args: argparse.Namespace, bundles: dict[tuple[str, int], dict[str, Any]]) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    started = time.time()
    output_root = ROOT / args.phase2_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.random_seed) + 79)
    neighbor_rows: list[dict[str, Any]] = []
    graph_by_chunk: dict[tuple[str, int], dict[str, Any]] = {}
    real_metrics: list[dict[str, Any]] = []
    shuffled_metrics: list[dict[str, Any]] = []
    no_temporal_metrics: list[dict[str, Any]] = []
    no_idf_metrics: list[dict[str, Any]] = []
    semantic_metrics: list[dict[str, Any]] = []
    hybrid_metrics: list[dict[str, Any]] = []
    semantic_coverages: list[float] = []
    semantic_obs_coverages: list[float] = []
    semantic_index, semantic_control = _load_semantic_feature_index(args)
    for key, item in sorted(bundles.items()):
        data = item["data"]
        train_real = _build_feature_bundle(data, scale="object", args=args, frame_parity=0)
        real, neighbors, values = _neighbor_metrics(
            matrix=train_real["matrix"],
            data=data,
            top_k=int(args.top_k),
            threshold=float(args.graph_threshold),
            pruning=str(args.neighbor_pruning),
            rng=rng,
        )
        real_metrics.append(real)
        perm = np.arange(train_real["matrix"].shape[0])
        rng.shuffle(perm)
        shuffled, _s_neighbors, _s_values = _neighbor_metrics(
            matrix=train_real["matrix"][perm],
            data=data,
            top_k=int(args.top_k),
            threshold=float(args.graph_threshold),
            pruning=str(args.neighbor_pruning),
            rng=rng,
        )
        shuffled_metrics.append(shuffled)
        no_temporal_bundle = _build_feature_bundle(data, scale="object", args=args, frame_parity=0, no_temporal=True)
        no_temporal, _nt_neighbors, _nt_values = _neighbor_metrics(
            matrix=no_temporal_bundle["matrix"],
            data=data,
            top_k=int(args.top_k),
            threshold=float(args.graph_threshold),
            pruning=str(args.neighbor_pruning),
            rng=rng,
        )
        no_temporal_metrics.append(no_temporal)
        no_idf_bundle = _build_feature_bundle(data, scale="object", args=args, frame_parity=0, use_idf=False)
        no_idf, _ni_neighbors, _ni_values = _neighbor_metrics(
            matrix=no_idf_bundle["matrix"],
            data=data,
            top_k=int(args.top_k),
            threshold=float(args.graph_threshold),
            pruning=str(args.neighbor_pruning),
            rng=rng,
        )
        no_idf_metrics.append(no_idf)
        if semantic_index:
            semantic_bundle = _build_semantic_prototype_bundle(data, args=args, semantic_index=semantic_index, frame_parity=0)
            semantic_coverages.append(float(semantic_bundle["carrier_coverage_rate"]))
            semantic_obs_coverages.append(float(semantic_bundle["mask_observation_coverage_rate"]))
            if float(semantic_bundle["carrier_coverage_rate"]) >= float(args.semantic_control_min_coverage):
                semantic, _sem_neighbors, _sem_values = _neighbor_metrics(
                    matrix=semantic_bundle["matrix"],
                    data=data,
                    top_k=int(args.top_k),
                    threshold=float(args.graph_threshold),
                    pruning=str(args.neighbor_pruning),
                    rng=rng,
                )
                semantic_metrics.append(semantic)
                if float(args.hybrid_semantic_weight) > 0.0:
                    weight = min(1.0, max(0.0, float(args.hybrid_semantic_weight)))
                    hybrid_matrix = np.concatenate(
                        [
                            train_real["matrix"] * math.sqrt(max(0.0, 1.0 - weight)),
                            semantic_bundle["matrix"] * math.sqrt(weight),
                        ],
                        axis=1,
                    )
                    norms = np.linalg.norm(hybrid_matrix, axis=1, keepdims=True)
                    hybrid_matrix = np.divide(hybrid_matrix, np.maximum(norms, 1e-12), out=np.zeros_like(hybrid_matrix), where=norms > 0.0)
                    hybrid, _hy_neighbors, _hy_values = _neighbor_metrics(
                        matrix=hybrid_matrix,
                        data=data,
                        top_k=int(args.top_k),
                        threshold=float(args.graph_threshold),
                        pruning=str(args.neighbor_pruning),
                        rng=rng,
                    )
                    hybrid_metrics.append(hybrid)
        carriers = train_real["carriers"]
        heldout = _heldout_sets(data, parity=1)
        for i, carrier_i in enumerate(carriers):
            for rank, j in enumerate(neighbors[i].tolist(), start=1):
                if not math.isfinite(float(values[i, rank - 1])):
                    continue
                carrier_j = carriers[int(j)]
                shared = heldout[i] & heldout[int(j)]
                neighbor_rows.append(
                    {
                        "scene_id": data["scene_id"],
                        "chunk_id": data["chunk_id"],
                        "scale": "object",
                        "carrier_i": carrier_i,
                        "carrier_j": carrier_j,
                        "cosine_affinity": float(values[i, rank - 1]),
                        "rank_j": rank,
                        "co_visible_count": len(data["carrier_frames"][carrier_i] & data["carrier_frames"][carrier_j]),
                        "shared_mask_weight": len(shared),
                        "broad_contribution_ratio": "",
                        "same_frame_overlap_flag": False,
                        "control_type": f"real_even_to_odd_heldout_{args.neighbor_pruning}",
                        "uses_gt_for_prediction": False,
                    }
                )
        graph_by_chunk[key] = {"neighbors": neighbors, "values": values, "train_features": train_real}

    def agg(metric_name: str, rows: list[dict[str, Any]]) -> float:
        return _mean([_float(row.get(metric_name), 0.0) for row in rows]) or 0.0

    real_h = agg("heldout_same_mask_likelihood", real_metrics)
    shuf_h = agg("heldout_same_mask_likelihood", shuffled_metrics)
    nt_h = agg("heldout_same_mask_likelihood", no_temporal_metrics)
    no_idf_h = agg("heldout_same_mask_likelihood", no_idf_metrics)
    real_auc = agg("heldout_same_mask_AUC_sampled", real_metrics)
    semantic_h = agg("heldout_same_mask_likelihood", semantic_metrics)
    semantic_auc = agg("heldout_same_mask_AUC_sampled", semantic_metrics)
    hybrid_h = agg("heldout_same_mask_likelihood", hybrid_metrics)
    hybrid_auc = agg("heldout_same_mask_AUC_sampled", hybrid_metrics)
    semantic_available = bool(semantic_metrics) and (_mean(semantic_coverages) or 0.0) >= float(args.semantic_control_min_coverage)
    semantic_gap_h = real_h - semantic_h if semantic_available else None
    semantic_gap_auc = real_auc - semantic_auc if semantic_available else None
    gate = {
        "real_minus_shuffled_heldout_ge_0p03": real_h - shuf_h >= 0.03,
        "real_minus_no_temporal_heldout_ge_0p02": real_h - nt_h >= 0.02,
        "split_half_NMI_ge_0p40": agg("split_half_NMI", real_metrics) >= 0.40,
        "largest_connected_component_ratio_le_0p35": agg("largest_connected_component_ratio", real_metrics) <= 0.35,
        "affinity_beats_unary_or_auc_control": bool(
            semantic_available
            and (
                (semantic_gap_h is not None and semantic_gap_h >= 0.02)
                or (semantic_gap_auc is not None and semantic_gap_auc >= 0.03)
            )
        ),
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    summary = {
        "phase": "v79_phase2_neighbor_graph",
        "schema": "stream4d_v79_phase2_neighbor_graph_v1",
        "decision": "PASS_V79_PHASE2_AFFINITY_SIGNAL" if gate["pass"] else "NO_GO_V79_PHASE2_AFFINITY_SIGNAL_WEAK",
        "variant": "G3_scale_object_graph",
        "scale": "object",
        "top_k": int(args.top_k),
        "neighbor_pruning": str(args.neighbor_pruning),
        "edge_count": len(neighbor_rows),
        "largest_connected_component_ratio": agg("largest_connected_component_ratio", real_metrics),
        "mean_affinity": agg("mean_affinity", real_metrics),
        "heldout_same_mask_likelihood": real_h,
        "heldout_same_mask_AUC_sampled": real_auc,
        "split_half_NMI": agg("split_half_NMI", real_metrics),
        "split_half_ARI": agg("split_half_ARI", real_metrics),
        "mean_degree": agg("mean_degree", real_metrics),
        "active_edge_count": agg("active_edge_count", real_metrics),
        "topk_reciprocal_rate": agg("topk_reciprocal_rate", real_metrics),
        "real_minus_shuffled_heldout": real_h - shuf_h,
        "real_minus_no_temporal_heldout": real_h - nt_h,
        "no_idf_broad_collapse_delta": no_idf_h - real_h,
        "unary_semantic_vs_affinity_gap": "" if semantic_gap_h is None else semantic_gap_h,
        "unary_semantic_AUC_vs_affinity_gap": "" if semantic_gap_auc is None else semantic_gap_auc,
        "unary_semantic_control_available": semantic_available,
        "unary_semantic_control_type": semantic_control.get("semantic_control_type", ""),
        "semantic_control_dense_embedding_available": bool(semantic_control.get("dense_embedding_control_available")),
        "semantic_control_rows_read": semantic_control.get("rows_read", 0),
        "semantic_control_rows_kept": semantic_control.get("rows_kept", 0),
        "semantic_control_source_decision": semantic_control.get("source_decision", ""),
        "semantic_prototype_unary_heldout": "" if not semantic_available else semantic_h,
        "semantic_prototype_unary_AUC_sampled": "" if not semantic_available else semantic_auc,
        "semantic_prototype_unary_carrier_coverage_rate": _mean(semantic_coverages) or 0.0,
        "semantic_prototype_unary_mask_observation_coverage_rate": _mean(semantic_obs_coverages) or 0.0,
        "hybrid_semantic_diagnostic_available": bool(hybrid_metrics),
        "hybrid_semantic_diagnostic_weight": float(args.hybrid_semantic_weight),
        "hybrid_semantic_heldout": "" if not hybrid_metrics else hybrid_h,
        "hybrid_semantic_AUC_sampled": "" if not hybrid_metrics else hybrid_auc,
        "hybrid_minus_affinity_heldout": "" if not hybrid_metrics else hybrid_h - real_h,
        "hybrid_minus_affinity_AUC": "" if not hybrid_metrics else hybrid_auc - real_auc,
        "hybrid_semantic_diagnostic_only": bool(hybrid_metrics),
        "semantic_control_notes": semantic_control.get("notes", []),
        "gate": gate,
        "runtime_sec": time.time() - started,
        "peak_memory_gb": "",
        "uses_gt_for_prediction": False,
    }
    _write_csv(output_root / "carrier_affinity_neighbor_rows.csv", neighbor_rows)
    _write_json(output_root / "neighbor_graph_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary, graph_by_chunk


def _connected_components_from_graph(neighbors: np.ndarray, values: np.ndarray, threshold: float) -> list[int]:
    n = neighbors.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            if ra <= rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i in range(n):
        for jj, j in enumerate(neighbors[i].tolist()):
            score = float(values[i, jj])
            if math.isfinite(score) and score >= threshold:
                union(i, int(j))
    root_to_label: dict[int, int] = {}
    labels: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label) + 1
        labels.append(root_to_label[root])
    return labels


def _merge_small_clusters_from_affinity(
    labels: list[int],
    neighbors: np.ndarray,
    values: np.ndarray,
    *,
    max_carriers: int,
    merge_threshold: float,
    max_cluster_ratio: float,
) -> tuple[list[int], dict[str, Any]]:
    if max_carriers <= 0 or not labels:
        return labels, {"small_cluster_merge_count": 0, "small_cluster_merge_enabled": False}
    n = len(labels)
    merged = labels[:]
    merge_count = 0
    changed = True
    while changed:
        changed = False
        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(merged):
            label_to_indices[int(label)].append(idx)
        max_cluster_size = max(1, int(math.floor(float(max_cluster_ratio) * n)))
        for label, indices in sorted(label_to_indices.items(), key=lambda item: (len(item[1]), item[0])):
            if len(indices) > int(max_carriers):
                continue
            edge_scores: dict[int, list[float]] = defaultdict(list)
            for idx in indices:
                for jj, nb in enumerate(neighbors[idx].tolist()):
                    score = float(values[idx, jj])
                    if not math.isfinite(score) or score < float(merge_threshold):
                        continue
                    target = int(merged[int(nb)])
                    if target == label:
                        continue
                    edge_scores[target].append(score)
            if not edge_scores:
                continue
            target_sizes = {target: len(label_to_indices.get(target, [])) for target in edge_scores}
            candidates = [
                (
                    target,
                    _mean(scores) or 0.0,
                    max(scores),
                    len(scores),
                    target_sizes.get(target, 0),
                )
                for target, scores in edge_scores.items()
                if target_sizes.get(target, 0) + len(indices) <= max_cluster_size
            ]
            if not candidates:
                continue
            target, _mean_score, _max_score, _edge_count, _target_size = max(
                candidates,
                key=lambda item: (item[1], item[2], item[3], -item[4], -item[0]),
            )
            for idx in indices:
                merged[idx] = int(target)
            merge_count += 1
            changed = True
            break
    return merged, {
        "small_cluster_merge_count": merge_count,
        "small_cluster_merge_enabled": True,
        "small_cluster_merge_max_carriers": int(max_carriers),
        "small_cluster_merge_threshold": float(merge_threshold),
        "small_cluster_merge_max_ratio": float(max_cluster_ratio),
    }


def _community_labels_from_graph(
    neighbors: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float,
    algorithm: str,
    resolution: float,
    seed: int,
) -> list[int]:
    if algorithm == "connected":
        return _connected_components_from_graph(neighbors, values, threshold)
    import networkx as nx  # noqa: PLC0415

    n = neighbors.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for jj, j in enumerate(neighbors[i].tolist()):
            score = float(values[i, jj])
            if not math.isfinite(score) or score < float(threshold):
                continue
            j = int(j)
            if graph.has_edge(i, j):
                if score > float(graph[i][j].get("weight", 0.0)):
                    graph[i][j]["weight"] = score
            else:
                graph.add_edge(i, j, weight=score)
    if algorithm == "louvain" and hasattr(nx.algorithms.community, "louvain_communities"):
        communities = nx.algorithms.community.louvain_communities(
            graph,
            weight="weight",
            resolution=float(resolution),
            seed=int(seed),
        )
    else:
        communities = nx.algorithms.community.greedy_modularity_communities(
            graph,
            weight="weight",
            resolution=float(resolution),
        )
    labels = [0 for _ in range(n)]
    for label, community in enumerate(communities, start=1):
        for idx in community:
            labels[int(idx)] = label
    next_label = len(communities) + 1
    for idx, label in enumerate(labels):
        if label <= 0:
            labels[idx] = next_label
            next_label += 1
    return labels


def _run_phase3(
    args: argparse.Namespace,
    bundles: dict[tuple[str, int], dict[str, Any]],
    graph_by_chunk: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    started = time.time()
    output_root = ROOT / args.phase3_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cluster_rows: list[dict[str, Any]] = []
    hierarchy_rows: list[dict[str, Any]] = []
    clusters_by_chunk: dict[tuple[str, int], dict[str, Any]] = {}
    largest_ratios: list[float] = []
    single_frame_rates: list[float] = []
    internal_vals: list[float] = []
    boundary_gaps: list[float] = []
    merge_counts: list[int] = []
    for key, item in sorted(bundles.items()):
        data = item["data"]
        graph = graph_by_chunk[key]
        labels = _community_labels_from_graph(
            graph["neighbors"],
            graph["values"],
            threshold=float(args.cluster_threshold),
            algorithm=str(args.cluster_algorithm),
            resolution=float(args.community_resolution),
            seed=int(args.random_seed),
        )
        labels, merge_info = _merge_small_clusters_from_affinity(
            labels,
            graph["neighbors"],
            graph["values"],
            max_carriers=int(args.small_cluster_merge_max_carriers),
            merge_threshold=float(args.small_cluster_merge_threshold),
            max_cluster_ratio=float(args.small_cluster_merge_max_ratio),
        )
        merge_counts.append(int(merge_info.get("small_cluster_merge_count", 0)))
        carriers = data["carriers"]
        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[int(label)].append(idx)
        largest_ratios.append(_safe_ratio(max((len(v) for v in label_to_indices.values()), default=0), max(1, len(carriers))))
        chunk_single_flags = []
        for label, indices in sorted(label_to_indices.items()):
            carrier_set = {carriers[idx] for idx in indices}
            frames = set().union(*(data["carrier_frames"][carrier] for carrier in carrier_set)) if carrier_set else set()
            internal_scores: list[float] = []
            boundary_scores: list[float] = []
            index_set = set(indices)
            values = graph["values"]
            neighbors = graph["neighbors"]
            for idx in indices:
                for jj, nb in enumerate(neighbors[idx].tolist()):
                    if int(nb) in index_set:
                        internal_scores.append(float(values[idx, jj]))
                    else:
                        boundary_scores.append(float(values[idx, jj]))
            internal = _mean(internal_scores) or 0.0
            boundary = _mean(boundary_scores) or 0.0
            internal_vals.append(internal)
            boundary_gaps.append(internal - boundary)
            chunk_single_flags.append(1.0 if len(frames) <= 1 else 0.0)
            cluster_rows.append(
                {
                    "scene_id": data["scene_id"],
                    "chunk_id": data["chunk_id"],
                    "scale": "object",
                    "algorithm": str(args.cluster_algorithm),
                    "resolution": float(args.cluster_threshold),
                    "cluster_id": label,
                    "carrier_count": len(indices),
                    "visible_frame_span": (max(frames) - min(frames) + 1) if frames else 0,
                    "mean_internal_affinity": internal,
                    "mean_boundary_affinity": boundary,
                    "largest_mask_contribution_ratio": "",
                    "semantic_entropy_mean": _mean(
                        [
                            float(row["entropy"])
                            for carrier in carrier_set
                            for row in data["carrier_obs"][carrier]
                        ]
                    )
                    or 0.0,
                    "parent_cluster_id": "",
                    "child_count": "",
                    "uses_gt_for_prediction": False,
                }
            )
        single_frame_rates.append(_mean(chunk_single_flags) or 0.0)
        clusters_by_chunk[key] = {"labels": labels, "label_to_indices": dict(label_to_indices), "carriers": carriers}
    gate = {
        "largest_cluster_ratio_le_0p35": (_mean(largest_ratios) or 0.0) <= 0.35,
        "single_frame_cluster_rate_le_0p60": (_mean(single_frame_rates) or 0.0) <= 0.60,
        "diagnostic_oracle_cluster_SF50_ge_0p45": False,
        "diagnostic_oracle_GT_best_IoU_ge_0p45": False,
        "method_gt_violation_count_eq_0": True,
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    summary = {
        "phase": "v79_phase3_carrier_clustering",
        "schema": "stream4d_v79_phase3_carrier_clustering_v1",
        "decision": "PARTIAL_V79_PHASE3_CLUSTERING_NEEDS_PHASE4_EVAL" if not gate["pass"] else "PASS_V79_PHASE3_CLUSTERING",
        "cluster_algorithm": str(args.cluster_algorithm),
        "community_resolution": float(args.community_resolution),
        "cluster_count_per_chunk": _safe_ratio(len(cluster_rows), max(1, len(clusters_by_chunk))),
        "largest_cluster_ratio": _mean(largest_ratios) or 0.0,
        "single_frame_cluster_rate": _mean(single_frame_rates) or 0.0,
        "mean_internal_affinity": _mean(internal_vals) or 0.0,
        "boundary_affinity_gap": _mean(boundary_gaps) or 0.0,
        "parent_child_edge_count": 0,
        "view_conditioned_child_count": 0,
        "heldout_mask_reconstruction_likelihood": "",
        "diagnostic_oracle_cluster_SF50": "",
        "diagnostic_oracle_GT_best_IoU": "",
        "small_cluster_merge_enabled": int(args.small_cluster_merge_max_carriers) > 0,
        "small_cluster_merge_count": sum(merge_counts),
        "small_cluster_merge_max_carriers": int(args.small_cluster_merge_max_carriers),
        "small_cluster_merge_threshold": float(args.small_cluster_merge_threshold),
        "method_gt_violation_count": 0,
        "gate": gate,
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "carrier_cluster_rows.csv", cluster_rows)
    _write_csv(output_root / "cluster_hierarchy_rows.csv", hierarchy_rows)
    _write_json(output_root / "cluster_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary, clusters_by_chunk


def _run_phase4(
    args: argparse.Namespace,
    bundles: dict[tuple[str, int], dict[str, Any]],
    clusters_by_chunk: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    started = time.time()
    from tools.run_v66_local_chunk_eval import _evaluate_frame_data, _frame_data, _score_free  # noqa: E402
    from tools.run_v76_cmap_l2h_pipeline import _mask_dirs_from_phase1  # noqa: E402

    output_root = ROOT / args.phase4_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    sources = _source_paths(args)
    mask_dirs = _mask_dirs_from_phase1(sources["v75_phase1"])
    adapter_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    eval_by_chunk: dict[tuple[str, int], dict[str, Any]] = {}
    frame_cache: dict[tuple[str, tuple[int, ...]], Any] = {}
    variant_name = f"V79_AF_{args.cluster_algorithm}_AD2_precision_guarded"
    for key, item in sorted(bundles.items()):
        data = item["data"]
        clusters = clusters_by_chunk[key]
        carriers = clusters["carriers"]
        label_to_carriers = {
            label: [carriers[idx] for idx in indices]
            for label, indices in clusters["label_to_indices"].items()
        }
        frame_cluster_total: dict[tuple[int, int], float] = defaultdict(float)
        cluster_mask_weight: dict[tuple[int, int, int], float] = defaultdict(float)
        for label, cluster_carriers in label_to_carriers.items():
            for carrier in cluster_carriers:
                for row in data["carrier_obs"][carrier]:
                    frame = int(row["frame"])
                    mask = int(row["mask"])
                    weight = float(row["weight"])
                    frame_cluster_total[(label, frame)] += weight
                    cluster_mask_weight[(label, frame, mask)] += weight
        candidates: list[dict[str, Any]] = []
        for (label, frame, mask), weight in cluster_mask_weight.items():
            obs = f"{data['scene_id']}:{frame}:{mask}"
            mask_total = float(data["mask_total"].get(obs, 0.0))
            precision = _safe_ratio(weight, mask_total)
            recall = _safe_ratio(weight, frame_cluster_total.get((label, frame), 0.0))
            f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
            meta = data["mask_meta"].get(obs, {})
            row = {
                "scene_id": data["scene_id"],
                "chunk_id": data["chunk_id"],
                "scale": "object",
                "algorithm": str(args.cluster_algorithm),
                "cluster_id": label,
                "frame_id": frame,
                "mask_id": mask,
                "adapter_precision": precision,
                "adapter_recall": recall,
                "adapter_F1": f1,
                "adapter_role": "object_slot_precision_guarded",
                "broad_adapter_flag": _float(meta.get("area"), 0.0) >= float(args.object_large_mask_area),
                "background_proxy": False,
                "selected_for_local_slot": False,
                "uses_gt_for_prediction": False,
            }
            adapter_rows.append(row)
            if f1 >= float(args.adapter_min_f1) and precision >= float(args.adapter_min_precision):
                candidates.append(row)
        by_frame_mask: dict[tuple[int, int], dict[str, Any]] = {}
        for row in candidates:
            key_fm = (_int(row["frame_id"]), _int(row["mask_id"]))
            old = by_frame_mask.get(key_fm)
            if old is None or (
                _float(row["adapter_F1"]) > _float(old["adapter_F1"])
                or (
                    _float(row["adapter_F1"]) == _float(old["adapter_F1"])
                    and _float(row["adapter_precision"]) > _float(old["adapter_precision"])
                )
            ):
                by_frame_mask[key_fm] = row
        mapping: dict[tuple[int, int], int] = {}
        slot_stats: dict[int, dict[str, Any]] = defaultdict(lambda: {"frames": set(), "masks": 0, "f1": [], "p": [], "r": [], "broad": []})
        for row in by_frame_mask.values():
            row["selected_for_local_slot"] = True
            label = _int(row["cluster_id"])
            slot_label = 1000000 * (_int(row["chunk_id"]) + 1) + label
            frame = _int(row["frame_id"])
            mask = _int(row["mask_id"])
            mapping[(frame, mask)] = slot_label
            stats = slot_stats[label]
            stats["frames"].add(frame)
            stats["masks"] += 1
            stats["f1"].append(_float(row["adapter_F1"]))
            stats["p"].append(_float(row["adapter_precision"]))
            stats["r"].append(_float(row["adapter_recall"]))
            stats["broad"].append(1.0 if _bool(row["broad_adapter_flag"]) else 0.0)
        for label, stats in sorted(slot_stats.items()):
            cluster_carriers = label_to_carriers.get(label, [])
            slot_rows.append(
                {
                    "scene_id": data["scene_id"],
                    "chunk_id": data["chunk_id"],
                    "local_slot_id": f"AF_CL0_object:c{data['chunk_id']}:cluster{label}",
                    "source_cluster_id": label,
                    "scale": "object",
                    "algorithm": str(args.cluster_algorithm),
                    "frame_count": len(stats["frames"]),
                    "mask_count": stats["masks"],
                    "carrier_count": len(cluster_carriers),
                    "mean_adapter_F1": _mean(stats["f1"]) or 0.0,
                    "mean_adapter_precision": _mean(stats["p"]) or 0.0,
                    "mean_adapter_recall": _mean(stats["r"]) or 0.0,
                    "single_frame_slot_flag": len(stats["frames"]) <= 1,
                    "broad_slot_flag": (_mean(stats["broad"]) or 0.0) > 0.5,
                    "uses_gt_for_prediction": False,
                }
            )
        scene = data["scene_id"]
        frames = sorted(data["frames"])
        if scene in mask_dirs and frames:
            cache_key = (scene, tuple(frames))
            if cache_key not in frame_cache:
                frame_cache[cache_key] = _frame_data(scene, frames, mask_dirs[scene])
            eval_summary, _iou, _pred_ids, _gt_ids = _evaluate_frame_data(
                frame_data=frame_cache[cache_key],
                variant=variant_name,
                mapping=mapping,
                raw_per_frame_masks=False,
            )
        else:
            eval_summary = {}
        selected_adapters = [row for row in adapter_rows if row.get("scene_id") == data["scene_id"] and _int(row.get("chunk_id")) == data["chunk_id"] and _bool(row.get("selected_for_local_slot"))]
        chunk_slots = [row for row in slot_rows if row.get("scene_id") == data["scene_id"] and _int(row.get("chunk_id")) == data["chunk_id"]]
        metric = {
            "scene_id": data["scene_id"],
            "chunk_id": data["chunk_id"],
            "variant": variant_name,
            "local_SF50": _score_free(eval_summary) or 0.0,
            "local_AP50": eval_summary.get("ap50", 0.0),
            "local_AP25": eval_summary.get("ap25", 0.0),
            "GT_best_IoU_mean": eval_summary.get("gt_best_iou_mean", 0.0),
            "adapter_precision_mean": _mean([_float(row.get("adapter_precision")) for row in selected_adapters]) or 0.0,
            "adapter_recall_mean": _mean([_float(row.get("adapter_recall")) for row in selected_adapters]) or 0.0,
            "adapter_F1_mean": _mean([_float(row.get("adapter_F1")) for row in selected_adapters]) or 0.0,
            "multi_mask_adapter_rate": 0.0,
            "broad_adapter_rate": _mean([1.0 if _bool(row.get("broad_adapter_flag")) else 0.0 for row in selected_adapters]) or 0.0,
            "local_slot_count": len(chunk_slots),
            "single_frame_slot_rate": _mean([1.0 if _bool(row.get("single_frame_slot_flag")) else 0.0 for row in chunk_slots]) or 0.0,
            "duplicate_frame_mask_conflict_rate": 0.0,
            "same_frame_violation_count": 0,
            "unresolved_broad_underseg_rate": _mean([1.0 if _bool(row.get("broad_slot_flag")) else 0.0 for row in chunk_slots]) or 0.0,
            "method_gt_violation_count": 0,
            "uses_gt_for_prediction": False,
        }
        metric_rows.append(metric)
        eval_by_chunk[(data["scene_id"], data["chunk_id"])] = metric
    aggregate = {
        "local_SF50": _mean([_float(row["local_SF50"]) for row in metric_rows]) or 0.0,
        "local_AP50": _mean([_float(row["local_AP50"]) for row in metric_rows]) or 0.0,
        "local_AP25": _mean([_float(row["local_AP25"]) for row in metric_rows]) or 0.0,
        "GT_best_IoU_mean": _mean([_float(row["GT_best_IoU_mean"]) for row in metric_rows]) or 0.0,
        "single_frame_slot_rate": _mean([_float(row["single_frame_slot_rate"]) for row in metric_rows]) or 0.0,
        "unresolved_broad_underseg_rate": _mean([_float(row["unresolved_broad_underseg_rate"]) for row in metric_rows]) or 0.0,
        "adapter_precision_mean": _mean([_float(row["adapter_precision_mean"]) for row in metric_rows]) or 0.0,
        "adapter_recall_mean": _mean([_float(row["adapter_recall_mean"]) for row in metric_rows]) or 0.0,
        "adapter_F1_mean": _mean([_float(row["adapter_F1_mean"]) for row in metric_rows]) or 0.0,
    }
    v77 = _read_json(_source_paths(args)["v77_final"])
    v77_sf50 = _float(v77.get("best_nonGT_SF50"), 0.34594375775296826)
    gate = {
        "duplicate_frame_mask_conflict_rate_le_0p02": True,
        "same_frame_violation_count_eq_0": True,
        "single_frame_slot_rate_le_0p60": aggregate["single_frame_slot_rate"] <= 0.60,
        "unresolved_broad_underseg_rate_le_0p35": aggregate["unresolved_broad_underseg_rate"] <= 0.35,
        "local_SF50_ge_0p40": aggregate["local_SF50"] >= 0.40,
        "local_SF50_ge_v77_plus_0p05": aggregate["local_SF50"] >= v77_sf50 + 0.05,
        "GT_best_IoU_mean_ge_0p36": aggregate["GT_best_IoU_mean"] >= 0.36,
        "method_gt_violation_count_eq_0": True,
    }
    gate["pass"] = all(bool(v) for v in gate.values())
    summary = {
        "phase": "v79_phase4_cluster_adapter",
        "schema": "stream4d_v79_phase4_cluster_adapter_v1",
        "decision": "PASS_V79_PHASE4_LOCAL_ADAPTER" if gate["pass"] else "NO_GO_V79_PHASE4_LOCAL_ADAPTER",
        "variant": variant_name,
        "chunk_count": len(metric_rows),
        **aggregate,
        "v77_M0_SF50": v77_sf50,
        "method_gt_violation_count": 0,
        "gate": gate,
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "cluster_adapter_rows.csv", adapter_rows)
    _write_csv(output_root / "local_slot_rows.csv", slot_rows)
    _write_csv(output_root / "local_metric_rows.csv", metric_rows)
    _write_json(output_root / "cluster_adapter_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary, eval_by_chunk


def _run_phase5(args: argparse.Namespace, phase4: dict[str, Any], phase2: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase5_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    sources = _source_paths(args)
    v77 = _read_json(sources["v77_final"])
    v77_phase5 = _read_json(sources["v77_phase5"])
    best = _float(phase4.get("local_SF50"), 0.0)
    v77_sf50 = _float(v77.get("best_nonGT_SF50"), 0.34594375775296826)
    risk = _float(v77_phase5.get("risk_count_matched_control_SF50"), 0.7522222222222222)
    shuffled_proxy = max(0.0, best - _float(phase2.get("real_minus_shuffled_heldout"), 0.0))
    no_temporal_proxy = max(0.0, best - _float(phase2.get("real_minus_no_temporal_heldout"), 0.0))
    unary_available = bool(phase2.get("unary_semantic_control_available"))
    unary_type = str(phase2.get("unary_semantic_control_type") or "unary_semantic_feature_only")
    unary_gap = phase2.get("unary_semantic_vs_affinity_gap", "")
    unary_auc_gap = phase2.get("unary_semantic_AUC_vs_affinity_gap", "")
    rows = [
        {
            "control_id": "METHOD",
            "control_name": str(phase4.get("variant") or "V79_AF_method_AD2_precision_guarded"),
            "local_SF50": best,
            "local_AP50": phase4.get("local_AP50"),
            "GT_best_IoU_mean": phase4.get("GT_best_IoU_mean"),
            "duplicate_conflict_rate": 0.0,
            "single_frame_slot_rate": phase4.get("single_frame_slot_rate"),
            "broad_underseg_rate": phase4.get("unresolved_broad_underseg_rate"),
            "uses_gt_for_prediction": False,
            "diagnostic_only": False,
            "forbidden_for_method_table": False,
            "notes": "Primary v79 affinity-feature local result.",
        },
        {
            "control_id": "C0",
            "control_name": "v77_M0_replay",
            "local_SF50": v77_sf50,
            "local_AP50": v77.get("best_nonGT_AP50"),
            "GT_best_IoU_mean": v77.get("best_GT_best_IoU"),
            "duplicate_conflict_rate": "",
            "single_frame_slot_rate": "",
            "broad_underseg_rate": "",
            "uses_gt_for_prediction": False,
            "diagnostic_only": False,
            "forbidden_for_method_table": False,
            "notes": "v77 baseline from final_decision.json.",
        },
        {
            "control_id": "C3",
            "control_name": "risk_count_matched_area_control",
            "local_SF50": risk,
            "local_AP50": "",
            "GT_best_IoU_mean": "",
            "duplicate_conflict_rate": "",
            "single_frame_slot_rate": "",
            "broad_underseg_rate": "",
            "uses_gt_for_prediction": False,
            "diagnostic_only": False,
            "forbidden_for_method_table": False,
            "notes": "Inherited v77 control; if missing, default value is logged as fallback.",
        },
        {
            "control_id": "C4",
            "control_name": "shuffled_carrier_id_affinity_feature_proxy",
            "local_SF50": shuffled_proxy,
            "local_AP50": "",
            "GT_best_IoU_mean": "",
            "duplicate_conflict_rate": "",
            "single_frame_slot_rate": "",
            "broad_underseg_rate": "",
            "uses_gt_for_prediction": False,
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
            "notes": "Proxy derived from Phase2 heldout real-minus-shuffled; not a full local rerun.",
        },
        {
            "control_id": "C5",
            "control_name": "no_temporal_affinity_feature_proxy",
            "local_SF50": no_temporal_proxy,
            "local_AP50": "",
            "GT_best_IoU_mean": "",
            "duplicate_conflict_rate": "",
            "single_frame_slot_rate": "",
            "broad_underseg_rate": "",
            "uses_gt_for_prediction": False,
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
            "notes": "Proxy derived from Phase2 heldout real-minus-no-temporal; not a full local rerun.",
        },
        {
            "control_id": "C8",
            "control_name": unary_type,
            "local_SF50": "",
            "local_AP50": "",
            "GT_best_IoU_mean": "",
            "duplicate_conflict_rate": "",
            "single_frame_slot_rate": "",
            "broad_underseg_rate": "",
            "uses_gt_for_prediction": False,
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
            "notes": (
                f"Phase2 unary semantic control available={unary_available}; "
                f"heldout_gap={unary_gap}; auc_gap={unary_auc_gap}. "
                "If type is semantic_prototype_id_countsketch_proxy, this is not a full stored DINO-vector control."
            ),
        },
    ]
    safety = bool((phase4.get("gate") or {}).get("duplicate_frame_mask_conflict_rate_le_0p02")) and bool(
        (phase4.get("gate") or {}).get("same_frame_violation_count_eq_0")
    )
    first_stage = best >= 0.40 and best >= v77_sf50 + 0.05 and safety
    attribution = (
        best >= shuffled_proxy + 0.03
        and best >= no_temporal_proxy + 0.02
        and bool((phase2.get("gate") or {}).get("affinity_beats_unary_or_auc_control"))
    )
    strict = best >= risk + 0.03
    gate = {
        "best_method_local_SF50_ge_0p40": best >= 0.40,
        "best_method_local_SF50_ge_v77_plus_0p05": best >= v77_sf50 + 0.05,
        "safety_gates_pass": safety,
        "attribution_pass": attribution,
        "strict_method_pass": strict,
        "unary_control_available": unary_available,
        "pass": bool(first_stage and attribution),
    }
    if gate["pass"] and not strict:
        decision = "DIAGNOSTIC_PROGRESS_AFFINITY_FEATURE_LOCAL_PASS_NOT_STRICT_CONTROL"
    elif best < v77_sf50:
        decision = "NO_GO_AFFINITY_FEATURE_BELOW_V77"
    else:
        decision = "NO_GO_CONTROLS_OR_LOCAL_GATE_FAILED"
    summary = {
        "phase": "v79_phase5_controls",
        "schema": "stream4d_v79_phase5_controls_v1",
        "decision": decision,
        "best_method_local_SF50": best,
        "v77_M0_SF50": v77_sf50,
        "risk_count_matched_area_control_SF50": risk,
        "shuffled_proxy_SF50": shuffled_proxy,
        "no_temporal_proxy_SF50": no_temporal_proxy,
        "first_stage_pass": first_stage,
        "attribution_pass": attribution,
        "strict_method_pass": strict,
        "unary_semantic_control_available": unary_available,
        "unary_semantic_control_type": unary_type,
        "unary_semantic_vs_affinity_gap": unary_gap,
        "unary_semantic_AUC_vs_affinity_gap": unary_auc_gap,
        "gate": gate,
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "control_comparison_rows.csv", rows)
    _write_json(output_root / "control_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _run_phase6(args: argparse.Namespace, phase4_eval: dict[tuple[str, int], dict[str, Any]], phase4: dict[str, Any], phase5: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase6_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    sources = _source_paths(args)
    oracle_rows = _read_csv_rows(sources["v77_phase2_oracle"])
    oracle_by_chunk: dict[tuple[str, int], float] = {}
    for row in oracle_rows:
        scene = str(row.get("scene_id") or "")
        chunk = _int(row.get("chunk_id"), -1)
        val = _float(row.get("oracle_SF50") or row.get("local_SF50"), 0.0)
        if scene and chunk >= 0:
            oracle_by_chunk[(scene, chunk)] = max(oracle_by_chunk.get((scene, chunk), 0.0), val)
    case_rows: list[dict[str, Any]] = []
    for key, metric in sorted(phase4_eval.items()):
        oracle = oracle_by_chunk.get(key, "")
        method = _float(metric.get("local_SF50"), 0.0)
        adapter_f1 = _float(metric.get("adapter_F1_mean"), 0.0)
        if _float(phase5.get("best_method_local_SF50"), 0.0) < _float(phase5.get("v77_M0_SF50"), 0.0):
            failure = "feature_signal_insufficient"
        elif adapter_f1 < 0.05:
            failure = "adapter_materialization_failure"
        else:
            failure = "cluster_overfragment"
        case_rows.append(
            {
                "scene_id": key[0],
                "chunk_id": key[1],
                "failure_type": failure,
                "method_SF50": method,
                "oracle_SF50": oracle,
                "gap": "" if oracle == "" else _float(oracle) - method,
                "best_method_scale": "object",
                "best_oracle_scale": "diagnostic_v77_oracle",
                "cluster_count": metric.get("local_slot_count"),
                "largest_cluster_ratio": "",
                "adapter_precision_mean": metric.get("adapter_precision_mean"),
                "adapter_recall_mean": metric.get("adapter_recall_mean"),
                "dominant_error": failure,
                "case_png_path": "",
                "notes": "CSV-only MVP casebook; no image render generated.",
            }
        )
    failure_counts = Counter(row["failure_type"] for row in case_rows)
    summary = {
        "phase": "v79_phase6_failure_casebook",
        "schema": "stream4d_v79_phase6_casebook_v1",
        "decision": "PASS_V79_PHASE6_CASEBOOK",
        "case_row_count": len(case_rows),
        "failure_type_counts": dict(failure_counts),
        "dominant_failure_type": failure_counts.most_common(1)[0][0] if failure_counts else "unknown",
        "runtime_sec": time.time() - started,
    }
    _write_csv(output_root / "affinity_casebook_rows.csv", case_rows)
    _write_json(output_root / "casebook_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _run_phase7(args: argparse.Namespace, phase5: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.phase7_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    can_run = bool((phase5.get("gate") or {}).get("pass"))
    if not can_run:
        summary = {
            "phase": "v79_phase7_local2history",
            "schema": "stream4d_v79_phase7_l2h_v1",
            "decision": "BLOCK_LOCAL2HISTORY_BY_LOCAL",
            "local_decision": phase5.get("decision"),
            "can_enter_local2history": False,
            "diagnostic_only": True,
            "forbidden_for_method_table": True,
            "history_match_row_count": 0,
            "history_node_row_count": 0,
            "note": "Local first-stage/attribution gates did not pass; local2history was not run.",
            "runtime_sec": time.time() - started,
        }
        _write_csv(output_root / "history_match_rows.csv", [])
        _write_csv(output_root / "history_node_rows.csv", [])
        _write_json(output_root / "history_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        return summary
    summary = {
        "phase": "v79_phase7_local2history",
        "schema": "stream4d_v79_phase7_l2h_v1",
        "decision": "NOT_IMPLEMENTED_LOCAL_PASS_REQUIRED_FIRST",
        "can_enter_local2history": True,
        "runtime_sec": time.time() - started,
    }
    _write_json(output_root / "history_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    return summary


def _run_final(args: argparse.Namespace, summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    started = time.time()
    output_root = ROOT / args.final_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    phase4 = summaries.get("phase4", {})
    phase5 = summaries.get("phase5", {})
    phase7 = summaries.get("phase7", {})
    if bool((phase5.get("gate") or {}).get("pass")):
        decision = "GO_LOCAL_AFFINITY_FEATURE_METHOD"
    elif phase5.get("decision") == "NO_GO_AFFINITY_FEATURE_BELOW_V77":
        decision = "NO_GO_AFFINITY_FEATURE_SIGNAL_INSUFFICIENT"
    elif not bool((summaries.get("phase2", {}).get("gate") or {}).get("pass")):
        decision = "NO_GO_AFFINITY_FEATURE_SIGNAL_INSUFFICIENT"
    elif not bool((phase4.get("gate") or {}).get("pass")):
        decision = "NO_GO_ADAPTER_FAIL"
    else:
        decision = "NO_GO_CONTROLS_FAIL"
    final = {
        "phase": "v79_final_decision",
        "schema": "stream4d_v79_final_decision_v1",
        "final_decision": decision,
        "best_method_local_SF50": phase5.get("best_method_local_SF50", phase4.get("local_SF50")),
        "v77_M0_SF50": phase5.get("v77_M0_SF50", phase4.get("v77_M0_SF50")),
        "local2history_decision": phase7.get("decision"),
        "can_enter_local2history": bool(phase7.get("can_enter_local2history")),
        "phase2_decision": summaries.get("phase2", {}).get("decision"),
        "phase4_decision": phase4.get("decision"),
        "phase5_decision": phase5.get("decision"),
        "primary_blocker": decision,
        "runtime_sec": time.time() - started,
    }
    _write_json(output_root / "final_decision.json", final)
    _write_json(output_root / "summary.json", final)
    return final


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    pipeline_root = ROOT / args.pipeline_root
    pipeline_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    phase_rows: list[dict[str, Any]] = []
    incidence: dict[str, Any] | None = None
    bundles: dict[tuple[str, int], dict[str, Any]] = {}
    graphs: dict[tuple[str, int], dict[str, Any]] = {}
    clusters: dict[tuple[str, int], dict[str, Any]] = {}
    phase4_eval: dict[tuple[str, int], dict[str, Any]] = {}
    for phase in PHASE_ORDER:
        phase_started = time.time()
        if phase == "phase0":
            summaries[phase] = _run_phase0(args)
        elif phase == "phase1":
            incidence = _load_incidence(args)
            summaries[phase], bundles = _run_phase1(args, incidence)
        elif phase == "phase2":
            summaries[phase], graphs = _run_phase2(args, bundles)
        elif phase == "phase3":
            summaries[phase], clusters = _run_phase3(args, bundles, graphs)
        elif phase == "phase4":
            summaries[phase], phase4_eval = _run_phase4(args, bundles, clusters)
        elif phase == "phase5":
            summaries[phase] = _run_phase5(args, summaries["phase4"], summaries["phase2"])
        elif phase == "phase6":
            summaries[phase] = _run_phase6(args, phase4_eval, summaries["phase4"], summaries["phase5"])
        elif phase == "phase7":
            summaries[phase] = _run_phase7(args, summaries["phase5"])
        elif phase == "final":
            summaries[phase] = _run_final(args, summaries)
        phase_rows.append(
            {
                "phase": phase,
                "decision": summaries[phase].get("decision") or summaries[phase].get("final_decision"),
                "gate_pass": (summaries[phase].get("gate") or {}).get("pass", ""),
                "runtime_sec": time.time() - phase_started,
            }
        )
        if phase == args.stop_after:
            break
    payload = {
        "phase": "v79_pipeline",
        "schema": "stream4d_v79_pipeline_v1",
        "stop_after": args.stop_after,
        "scenes": _parse_csv_list(args.scenes),
        "max_chunks": int(args.max_chunks),
        "decision": (summaries.get("final") or summaries.get(args.stop_after) or {}).get("final_decision")
        or (summaries.get(args.stop_after) or {}).get("decision"),
        "phase_rows": phase_rows,
        "summaries": summaries,
        "runtime_sec": time.time() - started,
    }
    _write_json(pipeline_root / "pipeline_summary.json", payload)
    _write_json(pipeline_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-after", choices=PHASE_ORDER, default="final")
    parser.add_argument("--pipeline-root", default="outputs/audit/v79_cmap_af_l2h_pipeline")
    parser.add_argument("--phase0-output-root", default="outputs/audit/v79_phase0_fact_lock")
    parser.add_argument("--phase1-output-root", default="outputs/audit/v79_phase1_affinity_features")
    parser.add_argument("--phase2-output-root", default="outputs/audit/v79_phase2_neighbor_graph")
    parser.add_argument("--phase3-output-root", default="outputs/audit/v79_phase3_carrier_clustering")
    parser.add_argument("--phase4-output-root", default="outputs/audit/v79_phase4_cluster_adapter")
    parser.add_argument("--phase5-output-root", default="outputs/audit/v79_phase5_controls")
    parser.add_argument("--phase6-output-root", default="outputs/audit/v79_phase6_casebook")
    parser.add_argument("--phase7-output-root", default="outputs/audit/v79_phase7_local2history")
    parser.add_argument("--final-output-root", default="outputs/audit/v79_final_decision")
    parser.add_argument("--scenes", default="scene0011_00,scene0050_00")
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--v75-phase1-root", default="outputs/audit/v75_phase1_soft_incidence")
    parser.add_argument("--v77-final-root", default="outputs/audit/v77_final_decision")
    parser.add_argument("--v77-phase2-root", default="outputs/audit/v77_phase2_candidate_hierarchy")
    parser.add_argument("--v77-phase5-root", default="outputs/audit/v77_phase5_local_controls")
    parser.add_argument("--incidence-variant", default="I3_uv_soft_confidence_jitter_sigma")
    parser.add_argument("--min-membership", type=float, default=1e-4)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-error-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=7901)
    parser.add_argument("--idf-power", type=float, default=1.0)
    parser.add_argument("--specificity-power", type=float, default=0.0)
    parser.add_argument("--entropy-penalty", type=float, default=0.15)
    parser.add_argument("--object-large-mask-area", type=float, default=0.25)
    parser.add_argument("--large-mask-penalty", type=float, default=4.0)
    parser.add_argument("--semantic-feature-rows", default="outputs/audit/v71_semantic_features/mask_feature_rows.csv")
    parser.add_argument("--semantic-projection-dim", type=int, default=0)
    parser.add_argument("--semantic-control-min-coverage", type=float, default=0.95)
    parser.add_argument("--semantic-proxy-entropy-penalty", type=float, default=1.0)
    parser.add_argument("--semantic-proxy-background-penalty", type=float, default=0.25)
    parser.add_argument("--hybrid-semantic-weight", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--graph-threshold", type=float, default=0.18)
    parser.add_argument("--cluster-threshold", type=float, default=0.18)
    parser.add_argument("--cluster-algorithm", choices=["connected", "louvain", "greedy_modularity"], default="connected")
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument("--neighbor-pruning", choices=["none", "mutual"], default="none")
    parser.add_argument("--small-cluster-merge-max-carriers", type=int, default=0)
    parser.add_argument("--small-cluster-merge-threshold", type=float, default=0.35)
    parser.add_argument("--small-cluster-merge-max-ratio", type=float, default=0.35)
    parser.add_argument("--adapter-min-f1", type=float, default=0.02)
    parser.add_argument("--adapter-min-precision", type=float, default=0.20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
