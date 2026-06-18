from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.run_v26_object_quality_diagnostics import _auc, _json_safe, _read_split
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v36_external_downstream_assignment import (
    _collect_observations,
    _load_gt,
    _load_masks,
    _load_tubes,
)
from tools.run_v37_temporal_curriculum import (
    LOCAL_GATE,
    V36_CHAIN2_ARI,
    _aggregate_stage_rows,
    _components_chain_then_closure,
    _components_from_edges,
    _frame_rank_map,
    _labels_for_components,
    _pair_row,
    _region_diagnostics,
    _sample_all_pairs,
    _safe_div,
    _support_pair_counts,
    _temporal_delta,
    _write_csv,
    _write_json,
)


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _feature(row: dict[str, Any]) -> list[float]:
    shared = _float_value(row.get("shared_d4rt_tube_count"))
    jaccard = _float_value(row.get("shared_d4rt_jaccard"))
    delta = _float_value(row.get("delta_t"))
    rgb = _float_value(row.get("rgb_similarity"))
    return [
        shared,
        math.log1p(max(shared, 0.0)),
        jaccard,
        delta,
        1.0 / (1.0 + max(delta, 0.0)),
        rgb,
    ]


def _labeled_training_rows(rows: list[dict[str, Any]], heldout_scene: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("scene")) != str(heldout_scene)
        and row.get("diagnostic_labeled_pair")
        and int(row.get("delta_t") or 0) >= 1
        and row.get("source") in {"d4rt_support", "all_pair_sample"}
    ]


def _labeled_eval_rows(rows: list[dict[str, Any]], scene: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("scene")) == str(scene)
        and row.get("diagnostic_labeled_pair")
        and int(row.get("delta_t") or 0) >= 1
        and row.get("source") in {"d4rt_support", "all_pair_sample"}
    ]


def _fit_loso_model(train_rows: list[dict[str, Any]], seed: int) -> Any | None:
    if len(train_rows) < 50:
        return None
    y = np.asarray([1 if row.get("diagnostic_same_GT") else 0 for row in train_rows], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        return None
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        return None
    x = np.asarray([_feature(row) for row in train_rows], dtype=np.float32)
    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=9,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=1,
    )
    model.fit(x, y)
    return model


def _predict_proba(model: Any, rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0,), dtype=np.float32)
    x = np.asarray([_feature(row) for row in rows], dtype=np.float32)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)


def _score_eval_rows(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"test_pair_count": 0, "test_AUC": None, "test_F1": None}
    labels = np.asarray([1 if row.get("diagnostic_same_GT") else 0 for row in rows], dtype=np.int64)
    prob = _predict_proba(model, rows)
    pred = (prob >= 0.5).astype(np.int64)
    pred_pos = int(np.sum(pred == 1))
    label_pos = int(np.sum(labels == 1))
    true_pos = int(np.sum((pred == 1) & (labels == 1)))
    return {
        "test_pair_count": int(len(rows)),
        "test_AUC": _auc(labels, prob),
        "test_F1": None if len(set(labels.tolist())) < 2 else float((2.0 * true_pos) / max(float(pred_pos + label_pos), 1.0)),
    }


def _edge_rows_for_scene(
    *,
    scene: str,
    nodes: list[Any],
    support_sets: dict[int, set[int]],
    pair_counts: Counter[tuple[int, int]],
    diagnostics: dict[int, dict[str, Any]],
    frame_rank: dict[int, int],
) -> list[dict[str, Any]]:
    rows = []
    for pair in pair_counts:
        row = _pair_row(
            scene=scene,
            nodes=nodes,
            support_sets=support_sets,
            pair_counts=pair_counts,
            diagnostics=diagnostics,
            pair=pair,
            source="d4rt_support",
            frame_rank=frame_rank,
        )
        if int(row.get("delta_t") or 0) >= 1:
            rows.append(row)
    return rows


def _row_edges(
    rows: list[dict[str, Any]],
    prob: np.ndarray,
    *,
    min_prob: float,
    min_delta: int = 1,
    max_delta: int | None = None,
) -> list[tuple[float, int, float, int, int, int]]:
    edges = []
    for row, score in zip(rows, prob.tolist()):
        delta = int(row.get("delta_t") or 0)
        if delta < int(min_delta):
            continue
        if max_delta is not None and delta > int(max_delta):
            continue
        if float(score) < float(min_prob):
            continue
        left = int(row["region_i"])
        right = int(row["region_j"])
        shared = int(float(row.get("shared_d4rt_tube_count") or 0.0))
        jaccard = float(row.get("shared_d4rt_jaccard") or 0.0)
        edges.append((float(score), shared, jaccard, delta, left, right))
    return sorted(edges, reverse=True)


def _evaluate_components(
    *,
    scene: str,
    stage: str,
    nodes: list[Any],
    components: list[list[int]],
    edge_info: dict[str, Any],
    frame_rank: dict[int, int],
    support_by_tube: dict[int, Counter[int]],
    obs_by_tube: dict[int, int],
    gt_labels: dict[int, int],
    min_support: int,
    min_fraction: float,
    model_info: dict[str, Any],
) -> dict[str, Any]:
    labels_pred, unknown_ratio = _labels_for_components(
        components,
        support_by_tube,
        obs_by_tube,
        gt_labels,
        min_support=min_support,
        min_fraction=min_fraction,
    )
    metrics = _cluster_metrics(labels_pred, gt_labels)
    labeled_ids = [tid for tid in sorted(labels_pred) if int(gt_labels.get(int(tid), 0)) > 0]
    row = {
        "scene": scene,
        "stage": stage,
        "ARI": metrics.get("ari"),
        "purity": metrics.get("purity"),
        "completeness": metrics.get("completeness"),
        "unknown_tube_ratio": float(unknown_ratio),
        "labeled_tube_count": metrics.get("labeled_tube_count"),
        "masklet_count": int(len(components)),
        "same_frame_cannot_link_violations": int(
            sum(len([int(nodes[idx].frame_id) for idx in comp]) - len(set(int(nodes[idx].frame_id) for idx in comp)) for comp in components)
        ),
        "min_support": int(min_support),
        "min_fraction": float(min_fraction),
        **edge_info,
        **model_info,
        "_labels_true": [int(gt_labels[int(tid)]) for tid in labeled_ids],
        "_labels_pred": [int(labels_pred[int(tid)]) for tid in labeled_ids],
    }
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _read_split(Path(args.split))
    root = Path(args.output_root)
    learned_dir = root / "v37_phaseH_learned_pair_solver"
    final_dir = root / "v37_final_decision"
    learned_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    scene_data: dict[str, dict[str, Any]] = {}
    pair_rows: list[dict[str, Any]] = []
    for scene in scenes:
        nodes, labels_by_frame, mask_manifest = _load_masks(Path(args.mask_root), scene, args.source, args.mode, int(args.min_region_area))
        frame_rank = _frame_rank_map(labels_by_frame)
        tubes = _load_tubes(scene, args)
        gt_labels = _load_gt(scene, tubes, args)
        support_by_region, support_by_tube, obs_by_tube = _collect_observations(nodes, labels_by_frame, tubes, args)
        support_sets = {idx: set(counter.keys()) for idx, counter in support_by_region.items()}
        diagnostics, _gt_area = _region_diagnostics(scene, nodes, labels_by_frame, compute_rgb=bool(args.compute_rgb))
        pair_counts = _support_pair_counts(
            nodes,
            support_by_region,
            max_pairs_per_tube=int(args.max_support_pairs_per_tube),
            seed=int(args.seed) + len(pair_rows),
            frame_rank=frame_rank,
        )
        support_rows = _edge_rows_for_scene(
            scene=scene,
            nodes=nodes,
            support_sets=support_sets,
            pair_counts=pair_counts,
            diagnostics=diagnostics,
            frame_rank=frame_rank,
        )
        all_sample_pairs = _sample_all_pairs(nodes, max_pairs=int(args.max_allpair_samples_per_scene), seed=int(args.seed) + 311)
        sample_rows = [
            _pair_row(
                scene=scene,
                nodes=nodes,
                support_sets=support_sets,
                pair_counts=pair_counts,
                diagnostics=diagnostics,
                pair=pair,
                source="all_pair_sample",
                frame_rank=frame_rank,
            )
            for pair in all_sample_pairs
        ]
        pair_rows.extend(support_rows)
        pair_rows.extend(sample_rows)
        scene_data[scene] = {
            "nodes": nodes,
            "frame_rank": frame_rank,
            "support_by_tube": support_by_tube,
            "obs_by_tube": obs_by_tube,
            "gt_labels": gt_labels,
            "support_rows": support_rows,
            "mask_manifest": mask_manifest,
        }

    stage_scene_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    variants = [
        ("H1_loso_rf_dt4_p050_s1f005", "greedy", 0.50, 1, 4, 1, 0.05, None),
        ("H2_loso_rf_dt4_p065_s1f005", "greedy", 0.65, 1, 4, 1, 0.05, None),
        ("H3_loso_rf_dt4_p075_s1f005", "greedy", 0.75, 1, 4, 1, 0.05, None),
        ("H4_loso_rf_dt8_p050_s1f005", "greedy", 0.50, 1, 8, 1, 0.05, None),
        ("H5_loso_rf_chain_short4_p050_closure_p050", "chain", 0.50, 1, 4, 1, 0.05, 0.50),
        ("H6_loso_rf_chain_short4_p065_closure_p055", "chain", 0.65, 1, 4, 1, 0.05, 0.55),
        ("H7_loso_rf_chain_short1_p050_closure_p050", "chain", 0.50, 1, 1, 1, 0.05, 0.50),
        ("H8_loso_rf_dt4_p050_s2f005", "greedy", 0.50, 1, 4, 2, 0.05, None),
        ("H9_loso_rf_dt4_p050_bipartite_s1f005", "bipartite", 0.50, 1, 4, 1, 0.05, None),
        ("H10_loso_rf_dt8_p050_bipartite_s1f005", "bipartite", 0.50, 1, 8, 1, 0.05, None),
        ("H11_loso_rf_dt4_p065_bipartite_s1f005", "bipartite", 0.65, 1, 4, 1, 0.05, None),
        ("H12_loso_rf_dt4_p050_bipartite_s2f005", "bipartite", 0.50, 1, 4, 2, 0.05, None),
        ("H13_loso_rf_dt8_p035_s1f005", "greedy", 0.35, 1, 8, 1, 0.05, None),
        ("H14_loso_rf_chain_short4_p035_closure_p035", "chain", 0.35, 1, 4, 1, 0.05, 0.35),
    ]

    for scene in scenes:
        train_rows = _labeled_training_rows(pair_rows, scene)
        eval_rows = _labeled_eval_rows(pair_rows, scene)
        model = _fit_loso_model(train_rows, seed=int(args.seed) + 910 + len(scene))
        if model is None:
            model_rows.append({"scene": scene, "status": "not_enough_training_data", "train_pair_count": len(train_rows)})
            continue
        model_info = {
            "status": "ok_loso_gt_trained_diagnostic",
            "train_pair_count": int(len(train_rows)),
            **_score_eval_rows(model, eval_rows),
        }
        model_rows.append({"scene": scene, **model_info})
        data = scene_data[scene]
        support_rows = data["support_rows"]
        support_prob = _predict_proba(model, support_rows)
        for stage, mode, prob_thr, min_delta, max_delta, min_support, min_fraction, closure_thr in variants:
            if mode == "chain":
                short_edges = _row_edges(support_rows, support_prob, min_prob=prob_thr, min_delta=1, max_delta=max_delta)
                closure_edges = _row_edges(support_rows, support_prob, min_prob=float(closure_thr), min_delta=9, max_delta=None)
                components, edge_info = _components_chain_then_closure(
                    data["nodes"],
                    short_edges,
                    closure_edges,
                    frame_rank=data["frame_rank"],
                )
                edge_info["learned_short_probability_threshold"] = float(prob_thr)
                edge_info["learned_closure_probability_threshold"] = float(closure_thr)
            else:
                edges = _row_edges(support_rows, support_prob, min_prob=prob_thr, min_delta=min_delta, max_delta=max_delta)
                components, edge_info = _components_from_edges(
                    data["nodes"],
                    edges,
                    mode="bipartite" if mode == "bipartite" else "greedy",
                    frame_rank=data["frame_rank"],
                )
                edge_info["learned_probability_threshold"] = float(prob_thr)
            edge_info["candidate_edges"] = int(edge_info.get("candidate_edges") or 0)
            row = _evaluate_components(
                scene=scene,
                stage=stage,
                nodes=data["nodes"],
                components=components,
                edge_info=edge_info,
                frame_rank=data["frame_rank"],
                support_by_tube=data["support_by_tube"],
                obs_by_tube=data["obs_by_tube"],
                gt_labels=data["gt_labels"],
                min_support=min_support,
                min_fraction=min_fraction,
                model_info=model_info,
            )
            stage_scene_rows.append(row)

    stage_summary = _aggregate_stage_rows(stage_scene_rows)
    best = max(stage_summary, key=lambda row: float(row.get("ARI") or -999.0), default={})
    final = {
        "plan": "docs/stream4d_v37_temporal_curriculum_masklet_plan.md",
        "phase": "Phase H learned pair solver diagnostic",
        "final_status": "NO_GO_PHASEH_LEARNED_SOLVER_FAILED",
        "best_stage": best.get("stage"),
        "best_metrics": best,
        "phaseH_gate_pass": bool(best.get("pass_3D_gate")),
        "is_method_result": False,
        "is_diagnostic_only": True,
        "uses_gt_for_training": True,
        "uses_gt_for_prediction_on_heldout": False,
        "allowed_4d": False,
        "allowed_ap": False,
        "note": "LOSO GT-trained calibrated scorer diagnostic; do not promote as training-free method result.",
        "v36_chain2_ARI": V36_CHAIN2_ARI,
        "local_gate": LOCAL_GATE,
    }
    if final["phaseH_gate_pass"]:
        final["final_status"] = "GO_PHASEH_DIAGNOSTIC_ONLY_CALIBRATED_SOLVER"
    _write_csv(learned_dir / "learned_pair_solver_model_rows.csv", model_rows)
    _write_csv(learned_dir / "learned_pair_solver_scene_rows.csv", [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in stage_scene_rows])
    _write_csv(learned_dir / "learned_pair_solver_summary.csv", stage_summary)
    _write_json(learned_dir / "learned_pair_solver_summary.json", stage_summary)
    _write_json(final_dir / "decision_summary.json", final)
    print(json.dumps(_json_safe(final), indent=2, sort_keys=True))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v37 Phase H LOSO learned pair-solver diagnostic.")
    parser.add_argument("--mask-root", default="outputs/audit/v36_external_mask_source_all")
    parser.add_argument("--source", default="watershed")
    parser.add_argument("--mode", default="all_masks")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v21_3_occupancy_d5_warmstart64_probe5_r1")
    parser.add_argument("--output-root", default="outputs/audit/v37_phaseH_learned_pair_solver")
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-region-area", type=int, default=64)
    parser.add_argument("--max-support-pairs-per-tube", type=int, default=20000)
    parser.add_argument("--max-allpair-samples-per-scene", type=int, default=30000)
    parser.add_argument("--compute-rgb", action="store_true")
    parser.add_argument("--seed", type=int, default=3701)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
