#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
PHASE_ID = "v103_phase7_da3_overlap_history_supplement_diagnostic"
PLAN_DOC = REPO_ROOT / "docs" / "stream4d_v103_training_free_primitive_affinity_field_experiment_plan.md"

DEFAULT_HISTORY_ROOT = AUDIT_ROOT / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_r8_i14_e3_veto_ratio100"
DEFAULT_PHASE9B_ROOT = AUDIT_ROOT / "v103_phase9b_da3_provider_readiness"
DEFAULT_SCENE0011_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384"
DEFAULT_SCENE0050_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase7_da3_overlap_history_supplement_r1"


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return np.asarray(image, dtype=np.int32)


def _obs(scene: str, frame_id: int, mask_id: int) -> str:
    return f"{scene}:{int(frame_id)}:{int(mask_id)}"


def _parse_obs(obs_id: str) -> tuple[str, int, int]:
    scene, frame, mask = str(obs_id).split(":")
    return scene, int(frame), int(mask)


def _history_variant_id(history_root: Path, requested: str) -> str:
    requested = str(requested).strip()
    if requested:
        return requested
    return str(_read_json(history_root / "summary.json")["best_variant_id"])


def _load_history(history_root: Path, variant_id: str, scene: str) -> tuple[dict[str, set[str]], dict[str, str]]:
    rows = pd.read_csv(history_root / "merge_selected_rows.csv")
    rows = rows[(rows["variant_id"].astype(str) == str(variant_id)) & (rows["scene_id"].astype(str) == scene)]
    by_obj: dict[str, set[str]] = defaultdict(set)
    owner: dict[str, str] = {}
    for row in rows.to_dict("records"):
        hist_id = str(row["object_id"])
        obs_id = _obs(scene, int(row["frame_id"]), int(row["selected_mask_id"]))
        by_obj[hist_id].add(obs_id)
        owner[obs_id] = hist_id
    return dict(by_obj), owner


def _current_overlap_obs(phase2_root: Path, scene: str, overlap_frames: int) -> set[str]:
    summary = _read_json(phase2_root / "summary.json")
    frame_ids = [int(v) for v in summary["frame_ids"][: int(overlap_frames)]]
    mask_root = _project(summary["mask_root"])
    out: set[str] = set()
    for frame_id in frame_ids:
        mask = _load_mask(mask_root / f"{frame_id}.png")
        for mask_id in np.unique(mask).astype(np.int64).tolist():
            if int(mask_id) > 0:
                out.add(_obs(scene, int(frame_id), int(mask_id)))
    return out


def _entropy(scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0 or float(np.sum(scores)) <= 0:
        return 1.0
    p = scores / max(float(np.sum(scores)), 1e-12)
    h = -float(np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0)))
    return h / max(math.log(max(scores.size, 2)), 1e-12)


def _process_scene(
    *,
    scene: str,
    history_root: Path,
    history_variant_id: str,
    phase9b_root: Path,
    phase2_root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    history_by_obj, history_owner = _load_history(history_root, history_variant_id, scene)
    current_overlap = _current_overlap_obs(phase2_root, scene, int(args.overlap_frames))
    bridge_path = phase9b_root / scene / "mask_pair_primitive_bridge_rows_with_semantic.parquet"
    bridges = pd.read_parquet(bridge_path)
    support_rows: list[dict[str, Any]] = []
    support_by_current: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    diag_diff: list[bool] = []
    diag_same: list[bool] = []
    for row in bridges.to_dict("records"):
        obs_a = str(row["mask_a_observation_id"])
        obs_b = str(row["mask_b_observation_id"])
        hist_obs = ""
        cur_obs = ""
        if obs_a in history_owner and obs_b in current_overlap:
            hist_obs, cur_obs = obs_a, obs_b
        elif obs_b in history_owner and obs_a in current_overlap:
            hist_obs, cur_obs = obs_b, obs_a
        if not hist_obs or not cur_obs:
            continue
        if bool(args.require_semantic_available) and not bool(row.get("semantic_residual_available", False)):
            continue
        final_score = float(row.get("final_bridge_score", 0.0) or 0.0)
        sem = float(row.get("semantic_residual_cosine", 0.0) or 0.0)
        broad = float(row.get("broad_contamination_score", 1.0) or 1.0)
        if final_score < float(args.min_final_bridge_score):
            continue
        if sem < float(args.min_semantic_residual_cosine):
            continue
        if broad > float(args.max_broad_contamination_score):
            continue
        hist_id = history_owner[hist_obs]
        support = float(final_score * sem * max(0.0, 1.0 - broad))
        old = float(support_by_current[cur_obs].get(hist_id, 0.0))
        support_by_current[cur_obs][hist_id] = max(old, support)
        diff = bool(row.get("diagnostic_different_gt", False))
        same = bool(row.get("diagnostic_same_gt", False))
        diag_diff.append(diff)
        diag_same.append(same)
        support_rows.append(
            {
                "schema_version": "stream4d_v103_phase7_da3_overlap_support_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "history_id": hist_id,
                "history_mask_observation_id": hist_obs,
                "current_overlap_mask_observation_id": cur_obs,
                "final_bridge_score": final_score,
                "semantic_residual_cosine": sem,
                "broad_contamination_score": broad,
                "da3_support_score": support,
                "uses_gt_for_prediction": False,
                "diagnostic_same_gt": same,
                "diagnostic_different_gt": diff,
            }
        )

    assignment_rows: list[dict[str, Any]] = []
    history_ids = sorted(history_by_obj)
    assigned_count = 0
    margins: list[float] = []
    entropies: list[float] = []
    for cur_obs, score_by_hist in sorted(support_by_current.items()):
        scores = np.asarray([float(score_by_hist.get(hist_id, 0.0)) for hist_id in history_ids], dtype=np.float64)
        if scores.size == 0:
            continue
        top_order = np.argsort(scores)
        top1_idx = int(top_order[-1])
        top1 = float(scores[top1_idx])
        top2 = float(scores[top_order[-2]]) if scores.size >= 2 else 0.0
        margin = top1 - top2
        entropy = _entropy(scores)
        assigned = top1 >= float(args.tau_hist) and margin >= float(args.tau_margin) and entropy <= float(args.tau_entropy)
        assigned_count += int(assigned)
        margins.append(margin)
        entropies.append(entropy)
        assignment_rows.append(
            {
                "schema_version": "stream4d_v103_phase7_da3_overlap_assignment_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "current_overlap_mask_observation_id": cur_obs,
                "top1_history_id": history_ids[top1_idx],
                "top1_score": top1,
                "top2_score": top2,
                "margin": margin,
                "entropy": entropy,
                "assigned": bool(assigned),
                "uses_gt_for_prediction": False,
            }
        )

    current_overlap_count = len(current_overlap)
    supported_count = len(support_by_current)
    metric = {
        "schema_version": "stream4d_v103_phase7_da3_overlap_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "history_variant_id": history_variant_id,
        "history_object_count": len(history_ids),
        "current_overlap_mask_observation_count": current_overlap_count,
        "da3_supported_current_overlap_mask_count": supported_count,
        "da3_supported_current_overlap_mask_rate": float(supported_count / max(current_overlap_count, 1)),
        "da3_overlap_assignment_count": assigned_count,
        "da3_overlap_assignment_rate_all_overlap_masks": float(assigned_count / max(current_overlap_count, 1)),
        "da3_overlap_assignment_rate_supported_masks": float(assigned_count / max(supported_count, 1)),
        "margin_mean_supported": float(np.mean(margins)) if margins else 0.0,
        "entropy_mean_supported": float(np.mean(entropies)) if entropies else 1.0,
        "accepted_pair_count": len(support_rows),
        "diagnostic_same_gt_rate": float(np.mean(diag_same)) if diag_same else "",
        "diagnostic_different_gt_rate": float(np.mean(diag_diff)) if diag_diff else "",
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    gate_rows = [
        {
            "schema_version": "stream4d_v103_phase7_da3_overlap_gate_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "gate_name": "da3_overlap_assignment_rate_all_overlap_masks_ge_0p10",
            "pass": metric["da3_overlap_assignment_rate_all_overlap_masks"] >= 0.10,
            "observed": metric["da3_overlap_assignment_rate_all_overlap_masks"],
            "required": ">=0.10",
        },
        {
            "schema_version": "stream4d_v103_phase7_da3_overlap_gate_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "gate_name": "diagnostic_different_gt_rate_le_0p20",
            "pass": (metric["diagnostic_different_gt_rate"] == "") or float(metric["diagnostic_different_gt_rate"]) <= 0.20,
            "observed": metric["diagnostic_different_gt_rate"],
            "required": "<=0.20 diagnostic only",
        },
    ]
    return support_rows, assignment_rows, gate_rows, metric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose DA3 overlap-frame supplement for v103 Phase7 history tokens.")
    parser.add_argument("--scene", choices=["scene0011_00", "scene0050_00"], default="scene0011_00")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--history-root", default=str(DEFAULT_HISTORY_ROOT))
    parser.add_argument("--history-variant-id", default="")
    parser.add_argument("--phase9b-root", default=str(DEFAULT_PHASE9B_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--overlap-frames", type=int, default=3)
    parser.add_argument("--min-final-bridge-score", type=float, default=0.40)
    parser.add_argument("--min-semantic-residual-cosine", type=float, default=0.60)
    parser.add_argument("--max-broad-contamination-score", type=float, default=0.02)
    parser.add_argument("--require-semantic-available", action="store_true", default=True)
    parser.add_argument("--tau-hist", type=float, default=0.55)
    parser.add_argument("--tau-margin", type=float, default=0.10)
    parser.add_argument("--tau-entropy", type=float, default=0.75)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    history_root = _project(args.history_root)
    phase9b_root = _project(args.phase9b_root)
    history_variant_id = _history_variant_id(history_root, str(args.history_variant_id))
    phase2_root = _project(args.scene0011_phase2_root if args.scene == "scene0011_00" else args.scene0050_phase2_root)
    support_rows, assignment_rows, gate_rows, metric = _process_scene(
        scene=str(args.scene),
        history_root=history_root,
        history_variant_id=history_variant_id,
        phase9b_root=phase9b_root,
        phase2_root=phase2_root,
        args=args,
    )
    failure_rows = [
        {
            "schema_version": "stream4d_v103_phase7_da3_overlap_failure_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": row["scene_id"],
            "failure_id": row["gate_name"],
            "severity": "diagnostic_blocking",
            "evidence": f"observed={row['observed']} required={row['required']}",
            "repair_direction": "DA3 overlap supplement is not safe/useful enough; do not inject it into Phase7 history token without stricter provider or broader c0001 DA3 export.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    _write_csv(out / "da3_overlap_support_rows.csv", support_rows)
    _write_csv(out / "da3_overlap_assignment_rows.csv", assignment_rows)
    _write_csv(out / "da3_overlap_metric_rows.csv", [metric])
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    decision = "DIAGNOSTIC_DA3_OVERLAP_SUPPLEMENT_USABLE" if not failure_rows else "NO_GO_DA3_OVERLAP_SUPPLEMENT_DIAGNOSTIC"
    summary = {
        "schema_version": "stream4d_v103_phase7_da3_overlap_supplement_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": decision,
        "failure_count": len(failure_rows),
        "metric": metric,
        "history_root": _rel(history_root),
        "history_variant_id": history_variant_id,
        "phase9b_root": _rel(phase9b_root),
        "phase2_root": _rel(phase2_root),
        "thresholds": {
            "min_final_bridge_score": float(args.min_final_bridge_score),
            "min_semantic_residual_cosine": float(args.min_semantic_residual_cosine),
            "max_broad_contamination_score": float(args.max_broad_contamination_score),
        },
        "truthfulness_note": (
            "This is a GT-free DA3 overlap supplement diagnostic. Diagnostic GT rates are reported after filtering only; "
            "they are not used for support selection. This does not emit object predictions or AP."
        ),
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "da3_overlap_support_rows": _rel(out / "da3_overlap_support_rows.csv"),
            "da3_overlap_assignment_rows": _rel(out / "da3_overlap_assignment_rows.csv"),
            "da3_overlap_metric_rows": _rel(out / "da3_overlap_metric_rows.csv"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
        "plan_doc": _rel(PLAN_DOC),
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
