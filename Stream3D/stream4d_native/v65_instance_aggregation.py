from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .v47_common import read_csv, write_json
from .v65_common import float_or_none, parse_eval_metric_file, project, rel, sha256_file, write_standard_outputs


ORACLE_SPECS = [
    {
        "variant": "G11",
        "raw_row_id": "A5",
        "source_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g11",
        "output_config": "v65_I6_oracle_gt_union_g11",
        "support_scope": "PREDICTION_UNION_ISLAND",
    },
    {
        "variant": "G12",
        "raw_row_id": "A7",
        "source_config": "v64r2_d4rt_chunk_scale_first_ap_probe5_g12",
        "output_config": "v65_I6_oracle_gt_union_g12",
        "support_scope": "PREDICTION_UNION_ISLAND",
    },
]


def run_v65_instance_aggregation() -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for spec in ORACLE_SPECS:
        _build_oracle_union_config(spec)
        commands.append(_run_eval(spec))
    return commands


def build_v65_instance_aggregation(*, command_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    contract_rows = _contract_rows()
    frag = _fragment_rows()
    rows: list[dict[str, Any]] = []
    fragment_rows: list[dict[str, Any]] = []
    for spec in ORACLE_SPECS:
        raw = contract_rows.get(spec["raw_row_id"], {})
        raw_metrics = _metrics_from_contract(raw)
        base_frag = [row for row in frag if row.get("variant_row_id") == spec["raw_row_id"]]
        rows.append(_aggregation_row("I0", spec, "raw_predictions", raw_metrics, base_frag, forbidden=True))
        rows.append(
            _aggregation_row(
                "I1",
                spec,
                "drop_tiny_fragments_only",
                raw_metrics,
                base_frag,
                forbidden=True,
                note="Evaluator min_region_size=100 already drops tiny predictions; AP is same current-run baseline.",
            )
        )
        for variant, reason in [
            ("I2", "prediction_trace rows do not expose history_id/material ids for D4RT G11/G12 current predictions"),
            ("I3", "same 2D mask/objectlet grouping is not available in current D4RT prediction config"),
            ("I4", "temporal adjacency graph is not linked to AP prediction ids"),
            ("I5", "semantic consistency guard is unavailable for current D4RT prediction ids"),
        ]:
            rows.append(_blocked_row(variant, spec, reason))
        oracle_metrics = parse_eval_metric_file(f"data/evaluation/scannet/{spec['output_config']}_class_agnostic.txt")
        rows.append(
            _aggregation_row(
                "I6",
                spec,
                "oracle_same_GT_union",
                oracle_metrics,
                _oracle_fragment_rows(spec),
                forbidden=True,
                uses_gt_for_prediction=True,
                note="GT-selected diagnostic upper bound; forbidden for method table.",
            )
        )
        stream3d = _stream3d_parity_for(spec["variant"])
        rows.append(
            {
                "variant": "I7",
                "d4rt_variant": spec["variant"],
                "description": "Stream3D same-support diagnostic",
                "AP": float_or_none(stream3d.get("AP")),
                "AP50": float_or_none(stream3d.get("AP50")),
                "AP25": float_or_none(stream3d.get("AP25")),
                "fragment_count_mean": "",
                "fragment_count_per_history_mean": "",
                "tiny_fragment_ratio": "",
                "raw_pred_count_mean": "",
                "kept_pred_count_mean": "",
                "dropped_pred_lt100_mean": "",
                "pred_best_iou_median": "",
                "gt_best_iou_ge_050_mean": "",
                "history_merge_count": "",
                "same_category_false_union_rate": "",
                "uses_gt_for_prediction": False,
                "forbidden_for_method_table": True,
                "status": "diagnostic_reference",
                "note": "Stream3D same-support parity row; not a SOMA aggregation method.",
            }
        )
        fragment_rows.extend(base_frag)
        fragment_rows.extend(_oracle_fragment_rows(spec))
    failed = [row for row in command_rows or [] if int(row.get("returncode", 0)) != 0]
    raw_by_variant = {row["d4rt_variant"]: row for row in rows if row["variant"] == "I0"}
    oracle_by_variant = {row["d4rt_variant"]: row for row in rows if row["variant"] == "I6"}
    oracle_improves = {
        variant: (float_or_none(oracle_by_variant[variant].get("AP")) or 0.0)
        > (float_or_none(raw_by_variant[variant].get("AP")) or 0.0)
        for variant in raw_by_variant
        if variant in oracle_by_variant
    }
    nongt_available = any(row["variant"] in {"I2", "I3", "I4", "I5"} and row["status"] == "ran" for row in rows)
    summary = {
        "phase": "v65_instance_aggregation",
        "aggregation_row_count": len(rows),
        "fragment_row_count": len(fragment_rows),
        "failed_command_count": len(failed),
        "oracle_union_improves_AP": oracle_improves,
        "non_gt_aggregation_available": nongt_available,
        "blocker": "non_GT_fragment_aggregation_signal_insufficient" if any(oracle_improves.values()) and not nongt_available else "",
        "next_algorithm_direction": "object-level aggregation over ownership histories; AP claim remains blocked until non-GT trace exists",
        "gate": {
            "oracle_commands_pass": len(failed) == 0,
            "raw_rows_available": all(row.get("AP") not in (None, "") for row in raw_by_variant.values()),
            "oracle_rows_available": all(row.get("AP") not in (None, "") for row in oracle_by_variant.values()),
            "blocked_non_gt_rows_explained": all(
                bool(row.get("note")) for row in rows if row["variant"] in {"I2", "I3", "I4", "I5"}
            ),
        },
    }
    summary["gate"]["pass"] = bool(all(summary["gate"].values()))
    return {
        "summary": summary,
        "aggregation_metric_rows": rows,
        "fragment_rows": fragment_rows,
        "instance_aggregation_commands": command_rows or [],
    }


def write_v65_instance_aggregation(output_root: str | Path, payload: dict[str, Any]) -> None:
    write_standard_outputs(
        output_root,
        {
            "aggregation_summary.json": payload["summary"],
            "aggregation_metric_rows.csv": payload["aggregation_metric_rows"],
            "fragment_rows.csv": payload["fragment_rows"],
            "instance_aggregation_commands.csv": payload.get("instance_aggregation_commands", []),
        },
    )


def _build_oracle_union_config(spec: dict[str, Any]) -> None:
    source = spec["source_config"]
    output = spec["output_config"]
    out_pred = project("data/prediction") / f"{output}_class_agnostic"
    out_tmp = project("data/TMP") / output
    out_pred.mkdir(parents=True, exist_ok=True)
    out_tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in _probe_scenes():
        pred_path = project("data/prediction") / f"{source}_class_agnostic" / f"{scene}.npz"
        tmp_path = project("data/TMP") / source / f"{scene}_pre_points.npy"
        gt_path = project("data/scannet/gt") / f"{scene}.txt"
        pre_points = np.load(tmp_path).astype(np.int64)
        shutil.copy2(tmp_path, out_tmp / f"{scene}_pre_points.npy")
        gt_full = np.loadtxt(gt_path, dtype=np.int64)
        with np.load(pred_path) as payload:
            pred_masks = np.asarray(payload["pred_masks"], dtype=bool)
            scores = np.asarray(payload["pred_score"], dtype=np.float64)
        scoped = pred_masks[pre_points, :] if pred_masks.shape[0] == gt_full.shape[0] else pred_masks
        pred_areas = scoped.sum(axis=0).astype(np.float64)
        groups: dict[int, list[int]] = {}
        for pred_idx in range(scoped.shape[1]):
            if pred_areas[pred_idx] < 100.0:
                continue
            best_gt, best_iou = _best_gt_for_pred(scoped[:, pred_idx], gt_full[pre_points])
            if best_gt is None or best_iou <= 0.0:
                continue
            groups.setdefault(best_gt, []).append(pred_idx)
        union_masks = []
        union_scores = []
        for gt_id, pred_ids in sorted(groups.items()):
            mask = np.any(pred_masks[:, pred_ids], axis=1)
            if int(mask.sum()) < 100:
                continue
            union_masks.append(mask)
            union_scores.append(float(max(scores[pred_ids])) if scores.size else float(mask.sum()))
        if union_masks:
            out_masks = np.stack(union_masks, axis=1).astype(bool)
            out_scores = np.asarray(union_scores, dtype=np.float64)
            out_classes = np.ones(out_masks.shape[1], dtype=np.int64)
        else:
            out_masks = np.zeros((gt_full.shape[0], 0), dtype=bool)
            out_scores = np.zeros((0,), dtype=np.float64)
            out_classes = np.zeros((0,), dtype=np.int64)
        np.savez_compressed(out_pred / f"{scene}.npz", pred_masks=out_masks, pred_score=out_scores, pred_classes=out_classes)
        rows.append(
            {
                "scene": scene,
                "raw_pred_count": int(scoped.shape[1]),
                "oracle_union_count": int(out_masks.shape[1]),
                "history_merge_count": int(sum(max(0, len(ids) - 1) for ids in groups.values())),
            }
        )
    manifest = {
        "schema_version": "stream4d_prediction_manifest_v1",
        "output_config": output,
        "prediction_config": output,
        "pre_points_config": output,
        "source_configs": [source],
        "support_policy": f"oracle_same_GT_union_on:{source}",
        "pre_points_policy": f"copied_from:{source}",
        "eval_policy": "v65_oracle_same_GT_union_diagnostic",
        "uses_gt": True,
        "uses_gt_for_prediction": True,
        "uses_gt_for_diagnostic": True,
        "is_diagnostic_only": True,
        "is_method_result": False,
        "forbidden_for_method_table": True,
        "notes": "GT-selected oracle fragment union diagnostic for v65 Phase 5; forbidden for method table.",
        "rows": rows,
    }
    write_json(out_pred / "config_manifest.json", manifest)
    write_json(out_tmp / "config_manifest.json", manifest)


def _run_eval(spec: dict[str, Any]) -> dict[str, Any]:
    output = spec["output_config"]
    cmd = [
        sys.executable,
        "-m",
        "evaluation.evaluate",
        "--pred_path",
        f"data/prediction/{output}_class_agnostic",
        "--gt_path",
        "data/scannet/gt",
        "--dataset",
        "scannet",
        "--output_file",
        f"data/evaluation/scannet/{output}_class_agnostic.txt",
        "--tmp_root",
        "data/TMP",
        "--tmp_config",
        output,
        "--no_class",
        "--require-manifest",
        "--allow-oracle-eval",
    ]
    env = os.environ.copy()
    cwd = project(".")
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    return {
        "variant": spec["variant"],
        "command": " ".join(cmd),
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _best_gt_for_pred(pred_mask: np.ndarray, gt_scoped_original: np.ndarray) -> tuple[int | None, float]:
    gt_scoped = gt_scoped_original % 1000 + 2000
    pred_area = float(np.count_nonzero(pred_mask))
    best_gt = None
    best_iou = 0.0
    for gt_id in [int(value) for value in np.unique(gt_scoped) if int(value) >= 1000]:
        gt_mask = gt_scoped == gt_id
        gt_area = float(np.count_nonzero(gt_mask))
        if gt_area < 100.0:
            continue
        inter = float(np.count_nonzero(pred_mask & gt_mask))
        union = pred_area + gt_area - inter
        iou = inter / max(union, 1.0)
        if iou > best_iou:
            best_gt = gt_id
            best_iou = float(iou)
    return best_gt, best_iou


def _aggregation_row(
    variant: str,
    spec: dict[str, Any],
    description: str,
    metrics: dict[str, Any],
    frag_rows: list[dict[str, str]],
    *,
    forbidden: bool,
    uses_gt_for_prediction: bool = False,
    note: str = "",
) -> dict[str, Any]:
    return {
        "variant": variant,
        "d4rt_variant": spec["variant"],
        "description": description,
        "AP": metrics.get("AP"),
        "AP50": metrics.get("AP50"),
        "AP25": metrics.get("AP25"),
        "fragment_count_mean": _mean(frag_rows, "raw_pred_count"),
        "fragment_count_per_history_mean": _mean(frag_rows, "fragment_count_per_history"),
        "tiny_fragment_ratio": _mean(frag_rows, "tiny_fragment_ratio"),
        "raw_pred_count_mean": _mean(frag_rows, "raw_pred_count"),
        "kept_pred_count_mean": _mean(frag_rows, "kept_pred_count"),
        "dropped_pred_lt100_mean": _mean(frag_rows, "dropped_pred_lt100"),
        "pred_best_iou_median": _mean(frag_rows, "pred_best_iou_median"),
        "gt_best_iou_ge_050_mean": _mean(frag_rows, "gt_best_iou_ge_050_mean"),
        "history_merge_count": _mean(frag_rows, "history_merge_count"),
        "same_category_false_union_rate": "",
        "uses_gt_for_prediction": uses_gt_for_prediction,
        "forbidden_for_method_table": forbidden,
        "status": "ran",
        "note": note,
    }


def _blocked_row(variant: str, spec: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "d4rt_variant": spec["variant"],
        "description": "blocked_non_gt_aggregation",
        "AP": "",
        "AP50": "",
        "AP25": "",
        "fragment_count_mean": "",
        "fragment_count_per_history_mean": "",
        "tiny_fragment_ratio": "",
        "raw_pred_count_mean": "",
        "kept_pred_count_mean": "",
        "dropped_pred_lt100_mean": "",
        "pred_best_iou_median": "",
        "gt_best_iou_ge_050_mean": "",
        "history_merge_count": "",
        "same_category_false_union_rate": "",
        "uses_gt_for_prediction": False,
        "forbidden_for_method_table": True,
        "status": "blocked",
        "note": reason,
    }


def _oracle_fragment_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = project("data/prediction") / f"{spec['output_config']}_class_agnostic" / "config_manifest.json"
    if not manifest.exists():
        return []
    payload = __import__("json").loads(manifest.read_text(encoding="utf-8"))
    return [
        {
            "variant_row_id": "I6",
            "config": spec["output_config"],
            "support_scope": spec["support_scope"],
            "scene_id": row["scene"],
            "raw_pred_count": row["raw_pred_count"],
            "kept_pred_count": row["oracle_union_count"],
            "dropped_pred_lt100": "",
            "tiny_fragment_ratio": "",
            "history_merge_count": row["history_merge_count"],
            "fragment_count_per_history": row["oracle_union_count"],
        }
        for row in payload.get("rows", [])
    ]


def _contract_rows() -> dict[str, dict[str, str]]:
    path = project("outputs/audit/v65_ap_contract/ap_contract_rows.csv")
    return {row["row_id"]: row for row in read_csv(path)} if path.exists() else {}


def _fragment_rows() -> list[dict[str, str]]:
    path = project("outputs/audit/v65_ap_failure_decomp/fragmentation_rows.csv")
    return read_csv(path) if path.exists() else []


def _stream3d_parity_for(variant: str) -> dict[str, str]:
    target = "S3D3" if variant == "G11" else "S3D4"
    path = project("outputs/audit/v65_stream3d_parity/stream3d_ap_rows.csv")
    if not path.exists():
        return {}
    for row in read_csv(path):
        if row.get("baseline_variant") == target:
            return row
    return {}


def _metrics_from_contract(row: dict[str, str]) -> dict[str, Any]:
    return {"AP": float_or_none(row.get("AP")), "AP50": float_or_none(row.get("AP50")), "AP25": float_or_none(row.get("AP25"))}


def _probe_scenes() -> list[str]:
    split = project("splits/scannet_v6_probe5.txt")
    return [line.strip() for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float_or_none(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None
