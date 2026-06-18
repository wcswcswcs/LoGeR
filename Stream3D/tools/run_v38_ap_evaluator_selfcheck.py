from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCENE = "scene0000_00"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_manifest(pred_dir: Path, config: str) -> None:
    payload = {
        "output_config": config,
        "is_method_result": False,
        "is_diagnostic_only": True,
        "forbidden_for_method_table": True,
        "uses_gt_for_prediction": False,
        "uses_gt": False,
        "notes": "v38 synthetic AP evaluator self-check artifact; diagnostic only.",
    }
    _write_json(pred_dir / "config_manifest.json", payload)


def _write_case(root: Path, config: str, gt_ids: np.ndarray, masks: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    pred_dir = root / "pred" / f"{config}_class_agnostic"
    gt_dir = root / "scannet" / config / "gt"
    tmp_dir = root / "TMP" / config
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(gt_dir / f"{SCENE}.txt", gt_ids.astype(np.int64), fmt="%d")
    np.savez_compressed(
        pred_dir / f"{SCENE}.npz",
        pred_masks=masks.astype(bool, copy=False),
        pred_score=scores.astype(np.float32, copy=False),
        pred_classes=np.zeros((masks.shape[1],), dtype=np.int32),
    )
    np.save(tmp_dir / f"{SCENE}_pre_points.npy", np.arange(gt_ids.shape[0], dtype=np.int64))
    _write_manifest(pred_dir, config)
    return {
        "config": config,
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "tmp_dir": str(tmp_dir),
        "num_vertices": int(gt_ids.shape[0]),
        "num_predictions": int(masks.shape[1]),
    }


def _parse_metric_file(path: Path) -> dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty metric file: {path}")
    parts = lines[-1].split(",")
    if len(parts) != 3:
        raise ValueError(f"Could not parse final AP row from {path}: {lines[-1]}")
    return {"AP": float(parts[0]), "AP50": float(parts[1]), "AP25": float(parts[2])}


def _run_eval(root: Path, case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = str(case["config"])
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    requested_metric = eval_dir / f"{config}.txt"
    log_path = eval_dir / f"{config}.log"
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        str(case["pred_dir"]),
        "--gt_path",
        str(case["gt_dir"]),
        "--dataset",
        "scannet",
        "--no_class",
        "--tmp_root",
        str(root / "TMP"),
        "--tmp_config",
        config,
        "--output_file",
        str(requested_metric),
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=args.stream3d_root, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT)
    metric_path = requested_metric
    if not metric_path.exists():
        fallback = eval_dir / f"{config}_class_agnostic.txt"
        if fallback.exists():
            metric_path = fallback
    metrics = _parse_metric_file(metric_path) if metric_path.exists() else {}
    return {
        **case,
        "command": " ".join(cmd),
        "exit_code": int(proc.returncode),
        "log_path": str(log_path),
        "requested_metric_file": str(requested_metric),
        "metric_file": str(metric_path),
        "output_file_argument_honored": requested_metric.exists(),
        "metrics": metrics,
    }


def _case_specs(root: Path) -> list[dict[str, Any]]:
    n = 240
    one_gt = np.full(n, 2001, dtype=np.int64)
    perfect = np.ones((n, 1), dtype=bool)
    duplicate = np.repeat(perfect, 3, axis=1)

    correct = np.ones((n,), dtype=bool)
    wrong_low_iou = np.zeros((n,), dtype=bool)
    wrong_low_iou[:100] = True

    tiny = np.zeros((n, 1), dtype=bool)
    tiny[:50, 0] = True

    two_gt = np.concatenate([np.full(120, 2001, dtype=np.int64), np.full(120, 2002, dtype=np.int64)])
    conflict = np.zeros((240, 2), dtype=bool)
    conflict[:180, 0] = True
    conflict[60:, 1] = True
    wta = np.zeros((240, 2), dtype=bool)
    wta[:120, 0] = True
    wta[120:, 1] = True

    return [
        _write_case(root, "T0_perfect", one_gt, perfect, np.asarray([1.0], dtype=np.float32)),
        _write_case(root, "T1_duplicate_same_score", one_gt, duplicate, np.asarray([1.0, 1.0, 1.0], dtype=np.float32)),
        _write_case(
            root,
            "T2_wrong_high_correct_low",
            one_gt,
            np.stack([wrong_low_iou, correct], axis=1),
            np.asarray([0.9, 0.1], dtype=np.float32),
        ),
        _write_case(
            root,
            "T2_correct_high_wrong_low",
            one_gt,
            np.stack([wrong_low_iou, correct], axis=1),
            np.asarray([0.1, 0.9], dtype=np.float32),
        ),
        _write_case(root, "T3_tiny_below_min_region", one_gt, tiny, np.asarray([1.0], dtype=np.float32)),
        _write_case(root, "T4_conflict_no_wta", two_gt, conflict, np.asarray([1.0, 1.0], dtype=np.float32)),
        _write_case(root, "T4_conflict_wta", two_gt, wta, np.asarray([1.0, 1.0], dtype=np.float32)),
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "config",
        "exit_code",
        "AP",
        "AP50",
        "AP25",
        "num_predictions",
        "output_file_argument_honored",
        "metric_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics", {})
            writer.writerow(
                {
                    "config": row.get("config"),
                    "exit_code": row.get("exit_code"),
                    "AP": metrics.get("AP"),
                    "AP50": metrics.get("AP50"),
                    "AP25": metrics.get("AP25"),
                    "num_predictions": row.get("num_predictions"),
                    "output_file_argument_honored": row.get("output_file_argument_honored"),
                    "metric_file": row.get("metric_file"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--output-root", default="outputs/audit/v38_ap_evaluator_selfcheck")
    parser.add_argument("--cuda-visible-devices", default="6")
    args = parser.parse_args()
    args.stream3d_root = str(Path(args.stream3d_root).resolve())
    root = Path(args.stream3d_root) / args.output_root
    root.mkdir(parents=True, exist_ok=True)

    rows = [_run_eval(root, case, args) for case in _case_specs(root)]
    by_config = {str(row["config"]): row for row in rows}
    metrics = {name: row["metrics"] for name, row in by_config.items()}
    checks = {
        "ap_evaluator_perfect_test_pass": bool(
            metrics["T0_perfect"].get("AP", 0.0) >= 0.999
            and metrics["T0_perfect"].get("AP50", 0.0) >= 0.999
            and metrics["T0_perfect"].get("AP25", 0.0) >= 0.999
        ),
        "duplicate_behavior_verified": bool(by_config["T1_duplicate_same_score"]["exit_code"] == 0),
        "score_order_behavior_verified": bool(
            metrics["T2_correct_high_wrong_low"].get("AP", -1.0)
            > metrics["T2_wrong_high_correct_low"].get("AP", -1.0)
        ),
        "tiny_filter_behavior_verified": bool(
            metrics["T3_tiny_below_min_region"].get("AP", 1.0) == 0.0
            and metrics["T3_tiny_below_min_region"].get("AP50", 1.0) == 0.0
            and metrics["T3_tiny_below_min_region"].get("AP25", 1.0) == 0.0
        ),
        "conflict_wta_behavior_verified": bool(
            metrics["T4_conflict_wta"].get("AP", -1.0) > metrics["T4_conflict_no_wta"].get("AP", -1.0)
        ),
        "all_eval_commands_exit_zero": all(int(row["exit_code"]) == 0 for row in rows),
    }
    summary = {
        "phase": "v38_ap_evaluator_selfcheck",
        "selfcheck_pass": all(checks.values()),
        "checks": checks,
        "rows": rows,
        "notes": [
            "GT files must live under a path containing scannet/ for evaluation.evaluate scene-id parsing.",
            "output_file_argument_honored records whether evaluation.evaluate wrote exactly to --output_file or used its own derived name.",
            "T1 records exact duplicate behavior without assuming a particular AP drop.",
        ],
    }
    _write_json(root / "ap_evaluator_selfcheck_summary.json", summary)
    _write_csv(root / "ap_evaluator_selfcheck_cases.csv", rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["selfcheck_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
