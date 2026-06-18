from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _bool_from_string(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.stream3d_root).resolve()
    output_root = root / args.output_root
    source_root = root / args.source_root
    output_root.mkdir(parents=True, exist_ok=True)

    phase_c = _read_json(root / "outputs/audit/v39_masklet_primitive/masklet_primitive_summary.json")
    decision = _read_json(source_root / "v37_final_decision/decision_summary.json")
    model_rows = _read_csv(source_root / "v37_phaseH_learned_pair_solver/learned_pair_solver_model_rows.csv")
    scene_rows = _read_csv(source_root / "v37_phaseH_learned_pair_solver/learned_pair_solver_scene_rows.csv")
    summary_rows = _read_csv(source_root / "v37_phaseH_learned_pair_solver/learned_pair_solver_summary.csv")

    best_stage = str(decision.get("best_stage") or "")
    best_scene_rows = [row for row in scene_rows if row.get("stage") == best_stage]
    best_summary = next((row for row in summary_rows if row.get("stage") == best_stage), {})

    model_stats = {
        "fold_count": int(len(model_rows)),
        "mean_test_AUC": _mean([_float(row, "test_AUC") for row in model_rows]),
        "min_test_AUC": min([_float(row, "test_AUC") for row in model_rows if _float(row, "test_AUC") is not None], default=None),
        "mean_test_F1": _mean([_float(row, "test_F1") for row in model_rows]),
        "min_test_F1": min([_float(row, "test_F1") for row in model_rows if _float(row, "test_F1") is not None], default=None),
        "total_test_pair_count": int(sum(int(float(row.get("test_pair_count") or 0)) for row in model_rows)),
        "mean_train_pair_count": _mean([_float(row, "train_pair_count") for row in model_rows]),
    }
    best_stage_stats = {
        "best_stage": best_stage,
        "ARI": _float(best_summary, "ARI"),
        "purity": _float(best_summary, "purity"),
        "completeness": _float(best_summary, "completeness"),
        "unknown_tube_ratio": _float(best_summary, "unknown_tube_ratio"),
        "scene0081_ARI": _float(best_summary, "scene0081_ARI"),
        "phaseH_pair_solver_gate_pass": _bool_from_string(best_summary.get("pass_3D_gate")),
        "ari_pass": _bool_from_string(best_summary.get("ari_pass")),
        "purity_pass": _bool_from_string(best_summary.get("purity_pass")),
        "completeness_pass": _bool_from_string(best_summary.get("completeness_pass")),
        "scene0081_pass": _bool_from_string(best_summary.get("scene0081_pass")),
        "unknown_pass": _bool_from_string(best_summary.get("unknown_pass")),
    }
    target_rows = [
        {
            "target": "masklet_pair_same_object",
            "status": "run_from_existing_loso_v37_v2_artifact",
            "fold_count": model_stats["fold_count"],
            "mean_AUC": model_stats["mean_test_AUC"],
            "mean_F1": model_stats["mean_test_F1"],
            "best_stage": best_stage,
            "ARI": best_stage_stats["ARI"],
            "purity": best_stage_stats["purity"],
            "completeness": best_stage_stats["completeness"],
            "scene0081_ARI": best_stage_stats["scene0081_ARI"],
            "gate_pass": best_stage_stats["phaseH_pair_solver_gate_pass"],
            "note": "GT-trained LOSO diagnostic only; not a training-free method result.",
        },
        {
            "target": "candidate_object_representative",
            "status": "not_run",
            "gate_pass": False,
            "note": "No valid v39 Phase D object candidates after Phase C object_birth_primitive_blocker.",
        },
        {
            "target": "tube_to_object_ownership",
            "status": "not_run",
            "gate_pass": False,
            "note": "No selected v39 object set for Phase E attachment after Phase C blocker.",
        },
        {
            "target": "vertex_support_to_object_ownership",
            "status": "not_run",
            "gate_pass": False,
            "note": "No one-object-one-prediction export after Phase C blocker.",
        },
        {
            "target": "object_confidence_iou_regression",
            "status": "not_run",
            "gate_pass": False,
            "note": "No v39 object export/AP rows are available for IoU regression.",
        },
    ]

    phase_c_gate = phase_c.get("phaseC_gate", {})
    phase_h_gate_pass = bool(
        best_stage_stats["phaseH_pair_solver_gate_pass"]
        and model_stats["mean_test_AUC"] is not None
        and model_stats["mean_test_AUC"] >= float(args.min_auc)
    )
    final_status = (
        "GO_PHASEH_DIAGNOSTIC_ONLY_CALIBRATED_PAIR_SOLVER"
        if phase_h_gate_pass
        else "NO_GO_PHASEH_LEARNED_PAIR_DIAGNOSTIC_FAILED_OBJECT_GATE"
    )
    manifest = {
        "phase": "v39_phaseH_learned_diagnostic",
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_gt_for_training": True,
        "method_type": "supervised_or_calibrated_diagnostic",
        "training_free": False,
        "forbidden_for_training_free_table": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "uses_frozen_visual_backbone": False,
        "visual_backbone_name": None,
        "mask_source": "v37/v36 watershed masklet artifact imported for diagnostic",
        "object_birth_source": "none_v39_phaseC_blocked",
        "d4rt_role": "pair association diagnostic only",
        "geometry_field": "d4rt support overlap/jaccard diagnostic features",
        "coordinate_frame": "2d_region_pair_ids",
        "alignment_source": "existing_v37_phaseH_loso_gt_trained_artifact",
        "source_artifacts": [
            str(source_root / "v37_final_decision/decision_summary.json"),
            str(source_root / "v37_phaseH_learned_pair_solver/learned_pair_solver_model_rows.csv"),
            str(source_root / "v37_phaseH_learned_pair_solver/learned_pair_solver_scene_rows.csv"),
            str(source_root / "v37_phaseH_learned_pair_solver/learned_pair_solver_summary.csv"),
        ],
    }
    summary = {
        **manifest,
        "phaseH_status": final_status,
        "phaseH_gate_pass": phase_h_gate_pass,
        "phaseC_gate_pass": bool(phase_c_gate.get("phaseC_gate_pass")),
        "object_birth_primitive_blocker": bool(phase_c_gate.get("object_birth_primitive_blocker")),
        "model_stats": model_stats,
        "best_stage_stats": best_stage_stats,
        "target_rows": target_rows,
        "source_decision_status": decision.get("final_status"),
        "notes": [
            "The available learned diagnostic is a LOSO GT-trained pair scorer, not a full v39 object-set/AP solver.",
            "It shows learnable pair signal, but the best object gate still fails on purity and scene0081 ARI.",
            "Because Phase C did not produce a valid v39 object-birth primitive, Phase H AP/export metrics cannot be honestly claimed.",
        ],
    }
    _write_json(output_root / "learned_diagnostic_manifest.json", manifest)
    _write_json(output_root / "learned_diagnostic_summary.json", summary)
    _write_csv(output_root / "learned_diagnostic_target_rows.csv", target_rows)
    _write_csv(output_root / "learned_diagnostic_model_rows.csv", model_rows)
    _write_csv(output_root / "learned_diagnostic_best_stage_scene_rows.csv", best_scene_rows)
    md = [
        "# Stream4D v39 Phase H Learned Diagnostic",
        "",
        f"`phaseH_status={final_status}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean_test_AUC | {model_stats['mean_test_AUC']} |",
        f"| mean_test_F1 | {model_stats['mean_test_F1']} |",
        f"| best_stage | {best_stage} |",
        f"| best_ARI | {best_stage_stats['ARI']} |",
        f"| best_purity | {best_stage_stats['purity']} |",
        f"| best_completeness | {best_stage_stats['completeness']} |",
        f"| best_scene0081_ARI | {best_stage_stats['scene0081_ARI']} |",
        f"| phaseH_pair_solver_gate_pass | {best_stage_stats['phaseH_pair_solver_gate_pass']} |",
        "",
        "This is diagnostic-only and GT-trained. It is forbidden for training-free/method tables.",
    ]
    (output_root / "learned_diagnostic_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import v37 LOSO learned pair diagnostic as v39 Phase H evidence.")
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument(
        "--source-root",
        default="outputs/audit/v37_phaseH_learned_pair_solver_v2_probe5",
        help="Root containing v37_phaseH_learned_pair_solver and v37_final_decision.",
    )
    parser.add_argument("--output-root", default="outputs/audit/v39_learned_diagnostic")
    parser.add_argument("--min-auc", type=float, default=0.70)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
