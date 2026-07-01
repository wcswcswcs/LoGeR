from __future__ import annotations

import argparse
import csv
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import (  # noqa: E402
    SparseSceneIoU,
    _sha256,
    _summarize_iou,
    _write_csv,
    _write_json,
)


REQUIRED_SUMMARY_FIELDS = [
    "diagnostic_only",
    "score_mode",
    "matching_scope",
    "pixel_grid",
    "frame_count",
    "pred_count",
    "gt_count",
]


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return REPO_ROOT / path_obj
    return ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        try:
            return str(path_obj.relative_to(REPO_ROOT))
        except ValueError:
            return str(path_obj)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_project(path).read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with _project(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _python_compile_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        full = _project(path)
        row = {
            "check": "py_compile",
            "path": _rel(full),
            "exists": full.exists(),
            "pass": False,
            "error": "",
            "sha256": _sha256(full) if full.exists() else "",
        }
        if full.exists():
            try:
                py_compile.compile(str(full), doraise=True)
                row["pass"] = True
            except Exception as exc:  # pragma: no cover - recorded in artifact
                row["error"] = repr(exc)
        rows.append(row)
    return rows


def _source_policy_rows() -> list[dict[str, Any]]:
    scene_ap = _project("tools/run_v65_scene_multiview_ap.py")
    text = scene_ap.read_text(encoding="utf-8")
    soma_start = text.index("def _soma_pred_2d")
    stream3d_start = text.index("def _load_stream3d_vertex_labels")
    soma_block = text[soma_start:stream3d_start]
    return [
        {
            "check": "soma_prediction_no_gt_token",
            "path": _rel(scene_ap),
            "pass": "diagnostic_best_gt" not in soma_block and "_load_gt_2d" not in soma_block,
            "evidence": "_soma_pred_2d block does not reference diagnostic_best_gt or _load_gt_2d",
        },
        {
            "check": "soma_prediction_uses_mask_to_object_idx",
            "path": _rel(scene_ap),
            "pass": "mask_to_object_idx" in soma_block,
            "evidence": "_soma_pred_2d assigns pred labels from mask_to_object_idx[(frame_id, mask_id)]",
        },
        {
            "check": "scene_iou_global_accumulator",
            "path": _rel(scene_ap),
            "pass": "class SparseSceneIoU" in text and "self.frame_count += 1" in text,
            "evidence": "SparseSceneIoU accumulates pred/gt area and intersections across add() calls",
        },
        {
            "check": "one_to_one_matching",
            "path": _rel(scene_ap),
            "pass": "linear_sum_assignment" in text and "_max_cardinality_match_count" in text,
            "evidence": "AP and score-free metrics use bipartite max-cardinality matching",
        },
        {
            "check": "score_free_metrics_present",
            "path": _rel(scene_ap),
            "pass": "score_free_match_at_050" in text and "score_free_match_at_025" in text,
            "evidence": "summary includes score-free matching at 0.25 and 0.50",
        },
    ]


def _verify_sha256_sidecar(sidecar: str) -> dict[str, Any]:
    sidecar_path = _project(sidecar)
    if not sidecar_path.exists():
        return {"sidecar": _rel(sidecar_path), "exists": False, "pass": False, "returncode": None, "stdout": "", "stderr": ""}
    proc = subprocess.run(
        ["sha256sum", "-c", str(sidecar_path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "sidecar": _rel(sidecar_path),
        "exists": True,
        "pass": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout_line_count": len(proc.stdout.splitlines()),
        "stderr": proc.stderr.strip(),
    }


def _add_pair(acc: SparseSceneIoU, pred: np.ndarray, gt: np.ndarray) -> None:
    acc.add(pred.astype(np.int64, copy=False), gt.astype(np.int64, copy=False))


def _synthetic_case(case: str) -> dict[str, Any]:
    acc = SparseSceneIoU()
    z = np.zeros((8, 8), dtype=np.int64)
    if case == "perfect":
        gt = z.copy()
        gt[:4, :4] = 1
        gt[4:, 4:] = 2
        pred = gt.copy()
        _add_pair(acc, pred, gt)
    elif case == "all_background_prediction":
        gt = z.copy()
        gt[:4, :4] = 1
        pred = z.copy()
        _add_pair(acc, pred, gt)
    elif case == "split_one_object":
        gt = z.copy()
        gt[:, :4] = 1
        pred = z.copy()
        pred[:4, :4] = 1
        pred[4:, :4] = 2
        _add_pair(acc, pred, gt)
    elif case == "merge_two_objects":
        gt = z.copy()
        gt[:4, :4] = 1
        gt[4:, 4:] = 2
        pred = z.copy()
        pred[:4, :4] = 1
        pred[4:, 4:] = 1
        _add_pair(acc, pred, gt)
    else:
        raise ValueError(case)
    summary, _iou, _pred_ids, _gt_ids = _summarize_iou(
        accumulator=acc,
        min_pred_pixels=1,
        min_gt_pixels=1,
        score_mode="constant",
        input_scores=None,
    )
    expected_pass = {
        "perfect": summary["ap"] == 1.0 and summary["ap50"] == 1.0,
        "all_background_prediction": summary["ap"] == 0.0 and summary["ap50"] == 0.0,
        "split_one_object": summary["score_free_match_at_050"]["tp"] == 1,
        "merge_two_objects": summary["score_free_match_at_050"]["tp"] == 1 and summary["score_free_match_at_050"]["gt_count"] == 2,
    }[case]
    return {
        "case": case,
        "pass": bool(expected_pass),
        "AP": summary["ap"],
        "AP50": summary["ap50"],
        "AP25": summary["ap25"],
        "score_free_match25_TP": summary["score_free_match_at_025"]["tp"],
        "score_free_match50_TP": summary["score_free_match_at_050"]["tp"],
        "score_free_match50_recall": summary["score_free_match_at_050"]["recall"],
        "pred_count": summary["evaluated_pred_count"],
        "gt_count": summary["evaluated_gt_count"],
        "gt_best_iou_mean": summary["gt_best_iou_mean"],
        "pred_best_iou_mean": summary["pred_best_iou_mean"],
    }


def _summary_contract_rows(summary_paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in summary_paths:
        payload = _read_json(path)
        summary = payload.get("summary", {})
        checks = {
            "diagnostic_only": bool(payload.get("diagnostic_only")) is True,
            "score_mode": bool(summary.get("score_mode")),
            "matching_scope": bool(payload.get("matching_scope")),
            "pixel_grid": isinstance(payload.get("pixel_grid"), dict),
            "frame_count": int(payload.get("frame_count") or 0) > 0,
            "pred_count": summary.get("evaluated_pred_count") is not None,
            "gt_count": summary.get("evaluated_gt_count") is not None,
        }
        rows.append(
            {
                "check": "required_summary_fields",
                "path": _rel(path),
                "pass": bool(all(checks.values())),
                "missing_or_failed": ",".join(key for key, ok in checks.items() if not ok),
                "diagnostic_only": payload.get("diagnostic_only"),
                "score_mode": summary.get("score_mode"),
                "matching_scope": payload.get("matching_scope"),
                "pixel_grid": json.dumps(payload.get("pixel_grid", {}), sort_keys=True),
                "frame_count": payload.get("frame_count"),
                "pred_count": summary.get("evaluated_pred_count"),
                "gt_count": summary.get("evaluated_gt_count"),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    compile_paths = [
        "tools/run_v65_scene_multiview_ap.py",
        "tools/audit_v65_scene_multiview_ap_sanity.py",
        "tools/run_v66_metric_lock.py",
    ]
    code_rows = _python_compile_rows(compile_paths) + _source_policy_rows()
    selfcheck_rows = [_synthetic_case(case) for case in ["perfect", "all_background_prediction", "split_one_object", "merge_two_objects"]]

    full_agg = _read_json(args.v65_constant_aggregate)
    predarea_agg = _read_json(args.v65_predarea_aggregate)
    summary_paths = [row["summary_json"] for row in full_agg.get("rows", [])] + [row["summary_json"] for row in predarea_agg.get("rows", [])]
    contract_rows = _summary_contract_rows(summary_paths)
    sidecar_rows = [
        _verify_sha256_sidecar(args.v65_constant_sha256sums),
        _verify_sha256_sidecar(args.v65_predarea_sha256sums),
    ]

    stream3d_rows = [
        row for row in full_agg.get("rows", [])
        if row.get("method") == "stream3d" and int(row.get("stride", 0)) in {5, 10}
    ]
    stream3d_by_stride = {int(row["stride"]): row for row in stream3d_rows}
    stream3d_stride_delta = None
    if 5 in stream3d_by_stride and 10 in stream3d_by_stride:
        stream3d_stride_delta = abs(float(stream3d_by_stride[5]["AP"]) - float(stream3d_by_stride[10]["AP"]))

    gate = {
        "py_compile_pass": all(bool(row["pass"]) for row in code_rows if row["check"] == "py_compile"),
        "synthetic_metric_selfcheck_pass": all(bool(row["pass"]) for row in selfcheck_rows),
        "artifact_sha256_check_pass": all(bool(row["pass"]) for row in sidecar_rows),
        "no_gt_for_soma_prediction": all(bool(row["pass"]) for row in code_rows if row["check"].startswith("soma_prediction")),
        "summary_contract_pass": all(bool(row["pass"]) for row in contract_rows),
        "stream3d_stride5_10_delta_le_0p02": stream3d_stride_delta is not None and stream3d_stride_delta <= 0.02,
    }
    gate["pass"] = all(bool(value) for value in gate.values())

    payload = {
        "phase": "v66_phase0_metric_lock",
        "metric_name": "scene_level_multi_view_2d_AP",
        "diagnostic_only": True,
        "inputs": {
            "v65_constant_aggregate": _rel(args.v65_constant_aggregate),
            "v65_predarea_aggregate": _rel(args.v65_predarea_aggregate),
            "v65_constant_sha256sums": _rel(args.v65_constant_sha256sums),
            "v65_predarea_sha256sums": _rel(args.v65_predarea_sha256sums),
        },
        "gate": gate,
        "stream3d_stride_delta_abs_AP": stream3d_stride_delta,
        "code_rows_csv": _rel(output_root / "code_audit_rows.csv"),
        "metric_selfcheck_rows_csv": _rel(output_root / "metric_selfcheck_rows.csv"),
        "summary_contract_rows_csv": _rel(output_root / "summary_contract_rows.csv"),
        "sha256_sidecar_rows_csv": _rel(output_root / "sha256_sidecar_rows.csv"),
        "required_summary_fields": REQUIRED_SUMMARY_FIELDS,
    }
    _write_csv(output_root / "code_audit_rows.csv", code_rows)
    _write_csv(output_root / "metric_selfcheck_rows.csv", selfcheck_rows)
    _write_csv(output_root / "summary_contract_rows.csv", contract_rows)
    _write_csv(output_root / "sha256_sidecar_rows.csv", sidecar_rows)
    _write_json(output_root / "metric_lock_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v66 metric lock artifacts from v65 scene MV-AP code and outputs.")
    parser.add_argument("--output-root", default="outputs/audit/v66_phase0_metric_lock")
    parser.add_argument(
        "--v65-constant-aggregate",
        default="outputs/audit/v65_scene_multiview_2d_ap_scene0050_full_v4_scoreaudit/aggregate_summary.json",
    )
    parser.add_argument(
        "--v65-predarea-aggregate",
        default="outputs/audit/v65_scene_multiview_2d_ap_scene0050_predarea_diagnostic_v2_scoreaudit/aggregate_summary.json",
    )
    parser.add_argument(
        "--v65-constant-sha256sums",
        default="outputs/audit/v65_scene_multiview_2d_ap_scene0050_full_v4_scoreaudit/SHA256SUMS.txt",
    )
    parser.add_argument(
        "--v65-predarea-sha256sums",
        default="outputs/audit/v65_scene_multiview_2d_ap_scene0050_predarea_diagnostic_v2_scoreaudit/SHA256SUMS.txt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

