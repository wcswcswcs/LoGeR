#!/usr/bin/env python3
"""Build Stream4D v99 Phase0 fact-lock artifacts.

Phase0 is intentionally read-only with respect to method outputs: it locks the
metric contract and baseline/reference numbers from existing audited artifacts.
It does not rerun any method variant.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"

F2_BASE = "F2_mask_centered_plus_semantic_residual_proxy__score_frame_count"
EXPECTED_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(9)]


BASELINE_FIELDS = [
    "row_id",
    "reference_kind",
    "split",
    "variant_id",
    "metric_family",
    "MV_AP_window",
    "MV_AP50_window",
    "MV_AP25_window",
    "ScoreFreeMatch50_window",
    "MV_AP_scene",
    "MV_AP50_scene",
    "MV_AP25_scene",
    "ScoreFreeMatch50_scene",
    "scene_count",
    "same_frame_collision_count",
    "pixel_collision_rate",
    "missing_mask_raster_count",
    "mean_gt_object_count",
    "mean_pred_object_count",
    "object_count",
    "metric_scope",
    "local_support_policy",
    "metric_source",
    "score_mode",
    "uses_gt_for_prediction",
    "uses_future",
    "source_file",
    "notes",
]


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _str_float(value: Any) -> str:
    parsed = _float(value)
    return "" if parsed is None else repr(parsed)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str] | None:
    for row in rows:
        ok = True
        for key, value in criteria.items():
            if row.get(key) != value:
                ok = False
                break
        if ok:
            return row
    return None


def _blank_row(row_id: str, reference_kind: str, split: str, source_file: Path, notes: str) -> dict[str, Any]:
    row = {field: "" for field in BASELINE_FIELDS}
    row.update(
        {
            "row_id": row_id,
            "reference_kind": reference_kind,
            "split": split,
            "source_file": _rel(source_file),
            "notes": notes,
        }
    )
    return row


def _baseline_from_window(
    row_id: str,
    reference_kind: str,
    split: str,
    row: dict[str, str],
    source_file: Path,
    *,
    variant_key: str = "variant_id",
    notes: str = "",
) -> dict[str, Any]:
    out = _blank_row(row_id, reference_kind, split, source_file, notes)
    metric_scope = row.get("metric_scope", "")
    local_support_policy = row.get("local_support_policy", "")
    if not local_support_policy and (
        "local_window_gt_projection" in metric_scope or "eval_local_window_support" in _rel(source_file)
    ):
        local_support_policy = "local_window_gt_projection"
    out.update(
        {
            "variant_id": row.get(variant_key) or row.get("variant") or row.get("method") or "",
            "metric_family": "MV_AP_window",
            "MV_AP_window": _str_float(row.get("mean_MV_AP_window") or row.get("MV_AP_window") or row.get("mean_MV_AP")),
            "MV_AP50_window": _str_float(
                row.get("mean_MV_AP50_window") or row.get("MV_AP50_window") or row.get("mean_MV_AP50")
            ),
            "MV_AP25_window": _str_float(
                row.get("mean_MV_AP25_window") or row.get("MV_AP25_window") or row.get("mean_MV_AP25")
            ),
            "ScoreFreeMatch50_window": _str_float(
                row.get("mean_score_free_Match50_window")
                or row.get("ScoreFreeMatch50_window")
                or row.get("mean_SF50_recall")
            ),
            "scene_count": row.get("scene_count", ""),
            "same_frame_collision_count": row.get("same_frame_collision_count", ""),
            "pixel_collision_rate": row.get("pixel_collision_rate", ""),
            "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
            "mean_gt_object_count": row.get("mean_gt_object_count", ""),
            "mean_pred_object_count": row.get("mean_pred_object_count", ""),
            "object_count": row.get("object_count", ""),
            "metric_scope": metric_scope,
            "local_support_policy": local_support_policy,
            "metric_source": row.get("canonical_metric_source") or row.get("metric_source") or "",
            "score_mode": row.get("score_mode", ""),
            "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
            "uses_future": row.get("uses_future", ""),
        }
    )
    return out


def _baseline_from_scene(
    row_id: str,
    reference_kind: str,
    split: str,
    row: dict[str, str],
    source_file: Path,
    *,
    notes: str = "",
) -> dict[str, Any]:
    out = _blank_row(row_id, reference_kind, split, source_file, notes)
    out.update(
        {
            "variant_id": row.get("variant_id") or row.get("variant") or row.get("method") or "",
            "metric_family": "MV_AP_scene",
            "MV_AP_scene": _str_float(row.get("mean_MV_AP_scene") or row.get("MV_AP_scene") or row.get("AP")),
            "MV_AP50_scene": _str_float(row.get("mean_MV_AP50_scene") or row.get("MV_AP50_scene") or row.get("AP50")),
            "MV_AP25_scene": _str_float(row.get("mean_MV_AP25_scene") or row.get("MV_AP25_scene") or row.get("AP25")),
            "ScoreFreeMatch50_scene": _str_float(row.get("mean_score_free_Match50_scene")),
            "scene_count": row.get("scene_count") or ("1" if row.get("scene") else ""),
            "same_frame_collision_count": row.get("same_frame_collision_count", ""),
            "missing_mask_raster_count": row.get("missing_mask_raster_count", ""),
            "metric_scope": row.get("metric_scope", ""),
            "metric_source": row.get("canonical_metric_source") or row.get("metric_source") or "",
            "score_mode": row.get("score_mode", ""),
            "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
            "uses_future": row.get("uses_future", ""),
        }
    )
    return out


def _contract() -> dict[str, Any]:
    v65 = STREAM3D_ROOT / "tools/run_v65_scene_multiview_ap.py"
    v89 = STREAM3D_ROOT / "tools/run_v89_recalc_point_projected_mv_ap.py"
    scene_adapter = STREAM3D_ROOT / "tools/build_v98_1_canonical_scene_metrics.py"
    check = STREAM3D_ROOT / "tools/check_mv_ap_contract.py"

    check_proc = subprocess.run(
        [sys.executable, _rel(check)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    import_error = ""
    thresholds: list[float] = []
    has_sparse = has_summarize = has_ap_from_scores = False
    try:
        module = _load_module(v65)
        thresholds = [round(float(x), 2) for x in getattr(module, "AP_THRESHOLDS", [])]
        has_sparse = hasattr(module, "SparseSceneIoU")
        has_summarize = hasattr(module, "_summarize_iou")
        has_ap_from_scores = hasattr(module, "_ap_from_scores")
    except Exception as exc:  # pragma: no cover - diagnostic path
        import_error = repr(exc)

    v89_text = v89.read_text(encoding="utf-8") if v89.is_file() else ""
    scene_text = scene_adapter.read_text(encoding="utf-8") if scene_adapter.is_file() else ""
    thresholds_pass = thresholds == EXPECTED_THRESHOLDS
    formal_pass = (
        check_proc.returncode == 0
        and thresholds_pass
        and has_sparse
        and has_summarize
        and has_ap_from_scores
        and "local_window_gt_projection" in v89_text
        and "scene_level_raw_gt_no_window_split" in scene_text
    )

    return {
        "schema_version": "stream4d_v99_phase0_mv_ap_contract_v1",
        "formal_metric_source_eq_v65": formal_pass,
        "evaluator_file": _rel(v65),
        "window_adapter_file": _rel(v89),
        "scene_adapter_file": _rel(scene_adapter),
        "check_file": _rel(check),
        "check_command": f"{sys.executable} {_rel(check)}",
        "check_returncode": check_proc.returncode,
        "check_stdout": check_proc.stdout,
        "check_stderr": check_proc.stderr,
        "v65_import_error": import_error,
        "ap_thresholds": thresholds,
        "expected_ap_thresholds": EXPECTED_THRESHOLDS,
        "ap_thresholds_pass": thresholds_pass,
        "has_SparseSceneIoU": has_sparse,
        "has__summarize_iou": has_summarize,
        "has__ap_from_scores": has_ap_from_scores,
        "local_support_policy": "local_window_gt_projection"
        if "local_window_gt_projection" in v89_text
        else "",
        "scene_metric_scope_marker": "scene_level_raw_gt_no_window_split"
        if "scene_level_raw_gt_no_window_split" in scene_text
        else "",
        "pass": formal_pass,
    }


def _stream3d_scene_diagnostic() -> dict[str, Any] | None:
    candidates = [
        AUDIT_ROOT / "v65_scene_multiview_2d_ap_scene0050_predarea_diagnostic_v2_scoreaudit/aggregate_rows.csv",
        AUDIT_ROOT / "v65_scene_multiview_2d_ap_scene0050_predarea_diagnostic_v1/aggregate_rows.csv",
        AUDIT_ROOT / "v65_scene_multiview_2d_ap_scene0050_full_v4_scoreaudit/aggregate_rows.csv",
        AUDIT_ROOT / "v65_scene_multiview_2d_ap_scene0050_full_v3/aggregate_rows.csv",
        AUDIT_ROOT / "v65_scene_multiview_2d_ap_scene0050_full_v2/aggregate_rows.csv",
    ]
    best: tuple[float, dict[str, str], Path] | None = None
    for path in candidates:
        if not path.is_file():
            continue
        for row in _read_csv(path):
            if row.get("method") != "stream3d" or row.get("stride") != "5":
                continue
            score = _float(row.get("AP"))
            if score is None:
                continue
            # Prefer the audited pred_area diagnostic when present; otherwise
            # choose the highest valid AP among the ordered candidates.
            if best is None or score > best[0]:
                best = (score, row, path)
    if best is None:
        return None
    score, row, path = best
    out = _baseline_from_scene(
        "Stream3D_scene_diagnostic_MV_AP_scene",
        "stream3d_scene_diagnostic",
        "scene0050_00",
        row,
        path,
        notes="Stream3D scene diagnostic from v65 scene multiview 2D AP artifact; selected method=stream3d,stride=5 highest audited AP among known v65 candidates.",
    )
    out["variant_id"] = f"stream3d_{row.get('score_mode', '')}_stride{row.get('stride', '')}"
    out["metric_scope"] = "scene_level_raw_gt_no_window_split"
    out["metric_source"] = "run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou"
    return out


def _v91_best() -> dict[str, Any] | None:
    best: tuple[float, Path, dict[str, str]] | None = None
    for path in AUDIT_ROOT.glob("v91*/**/mv_metric_aggregate_rows.csv"):
        rows = _read_csv(path)
        for row in rows:
            if _bool(row.get("uses_gt_for_prediction")) is True:
                continue
            if _bool(row.get("uses_future")) is True:
                continue
            score = _float(row.get("mean_MV_AP_window") or row.get("MV_AP_window") or row.get("mean_MV_AP"))
            if score is None:
                continue
            if best is None or score > best[0]:
                best = (score, path, row)
    if best is None:
        return None
    _, path, row = best
    return _baseline_from_window(
        "v91_best_MV_AP_window",
        "v91_best_non_oracle",
        "dev",
        row,
        path,
        notes="Best v91 non-oracle row found by scanning v91*/**/mv_metric_aggregate_rows.csv with uses_gt_for_prediction!=True and uses_future!=True.",
    )


def _append_gate(
    rows: list[dict[str, Any]],
    gate_id: str,
    passed: bool,
    *,
    expected: str = "",
    observed: Any = "",
    severity: str = "required",
    source_file: str = "",
    notes: str = "",
) -> None:
    rows.append(
        {
            "gate_id": gate_id,
            "pass": bool(passed),
            "expected": expected,
            "observed": observed,
            "severity": severity,
            "source_file": source_file,
            "notes": notes,
        }
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = AUDIT_ROOT / "v98_phase15_final_decision/summary.json"
    dev_window_path = AUDIT_ROOT / "v98_phase9_render_snap/canonical_score_repair_mv_metric_aggregate_rows.csv"
    dev_ablation_path = AUDIT_ROOT / "v98_phase9_render_snap/canonical_mv_metric_aggregate_rows.csv"
    holdout_window_path = AUDIT_ROOT / "v98_phase13_holdout_canonical_metrics/canonical_holdout_mv_metric_aggregate_rows.csv"
    dev_scene_path = AUDIT_ROOT / "v98_phase14_mv_ap_scene/dev_mv_scene_metric_aggregate_rows.csv"
    holdout_scene_path = AUDIT_ROOT / "v98_phase14_mv_ap_scene/holdout_mv_scene_metric_aggregate_rows.csv"
    s3d_local_path = AUDIT_ROOT / "v89_recalc_point_projected_mv_ap/eval_local_window_support/mv_aggregate_rows.csv"
    controls_path = AUDIT_ROOT / "v98_phase10_controls/control_metric_rows.csv"

    contract = _contract()

    summary = _read_json(summary_path)
    dev_window = _read_csv(dev_window_path)
    dev_ablation = _read_csv(dev_ablation_path)
    holdout_window = _read_csv(holdout_window_path)
    dev_scene = _read_csv(dev_scene_path)
    holdout_scene = _read_csv(holdout_scene_path)
    s3d_local = _read_csv(s3d_local_path)
    controls = _read_csv(controls_path)

    f2_dev_window = _find_row(dev_window, variant_id=F2_BASE)
    f2_holdout_window = _find_row(holdout_window, variant_id=F2_BASE)
    f2_dev_scene = _find_row(
        dev_scene,
        variant_id=F2_BASE,
        split="dev",
        scene_object_mode="scene_birth_id_no_extra_stitching",
    )
    f2_holdout_scene = _find_row(
        holdout_scene,
        variant_id=F2_BASE,
        split="holdout",
        scene_object_mode="scene_birth_id_no_extra_stitching",
    )
    s3d_local_row = _find_row(s3d_local, method_family="stream3d_local_point_projected")
    b0_row = _find_row(s3d_local, variant="B0_local_only")
    stream3d_scene_row = _stream3d_scene_diagnostic()
    v91_best_row = _v91_best()

    baseline_rows: list[dict[str, Any]] = []
    if f2_dev_window:
        baseline_rows.append(
            _baseline_from_window(
                "F2_base_full_dev_MV_AP_window",
                "f2_base",
                "full_dev",
                f2_dev_window,
                dev_window_path,
                notes="v98.1 score-repaired F2_base local-window metric.",
            )
        )
    if f2_dev_scene:
        baseline_rows.append(
            _baseline_from_scene(
                "F2_base_full_dev_MV_AP_scene",
                "f2_base",
                "full_dev",
                f2_dev_scene,
                dev_scene_path,
                notes="v98.1 F2_base scene metric using scene_birth_id_no_extra_stitching.",
            )
        )
    if f2_holdout_window:
        baseline_rows.append(
            _baseline_from_window(
                "F2_base_holdout_MV_AP_window",
                "f2_base",
                "same_scene_temporal_holdout",
                f2_holdout_window,
                holdout_window_path,
                notes="v98.1 same-scene temporal holdout F2_base local-window metric.",
            )
        )
    if f2_holdout_scene:
        baseline_rows.append(
            _baseline_from_scene(
                "F2_base_holdout_MV_AP_scene",
                "f2_base",
                "same_scene_temporal_holdout",
                f2_holdout_scene,
                holdout_scene_path,
                notes="v98.1 same-scene temporal holdout F2_base scene metric.",
            )
        )
    if s3d_local_row:
        baseline_rows.append(
            _baseline_from_window(
                "Stream3D_corrected_local_window_MV_AP_window",
                "stream3d_corrected_local_window",
                "dev",
                s3d_local_row,
                s3d_local_path,
                variant_key="variant",
                notes="Corrected Stream3D local-window diagnostic with local_window_gt_projection support.",
            )
        )
    if stream3d_scene_row:
        baseline_rows.append(stream3d_scene_row)
    if b0_row:
        baseline_rows.append(
            _baseline_from_window(
                "B0_MV_AP_window",
                "b0_reference",
                "dev",
                b0_row,
                s3d_local_path,
                variant_key="variant",
                notes="B0 local reference from corrected local-window support artifact.",
            )
        )
    if controls:
        best_control = max(controls, key=lambda r: _float(r.get("MV_AP_window")) or float("-inf"))
        baseline_rows.append(
            _baseline_from_window(
                "best_locked_control_MV_AP_window",
                "best_locked_control",
                "full_dev",
                best_control,
                controls_path,
                notes="Best v98 phase10 control by MV_AP_window.",
            )
        )
    if v91_best_row:
        baseline_rows.append(v91_best_row)

    # Keep the non-score-repaired ablation table in the casebook so reviewers can
    # see why F2_score_frame_count is treated as the v99 anchor.
    casebook_rows: list[dict[str, Any]] = []
    for row in dev_ablation:
        casebook_rows.append(
            {
                "case_id": f"v98_ablation::{row.get('variant_id')}",
                "case_type": "v98_1_dev_ablation",
                "variant_id": row.get("variant_id", ""),
                "MV_AP_window": _str_float(row.get("mean_MV_AP_window")),
                "MV_AP50_window": _str_float(row.get("mean_MV_AP50_window")),
                "MV_AP_scene": _str_float(row.get("MV_AP_scene")),
                "MV_AP50_scene": _str_float(row.get("MV_AP50_scene")),
                "source_file": _rel(dev_ablation_path),
                "notes": "v98.1 non-score-repaired ablation reference.",
            }
        )
    for row in baseline_rows:
        casebook_rows.append(
            {
                "case_id": f"phase0_fact::{row.get('row_id')}",
                "case_type": row.get("reference_kind", ""),
                "variant_id": row.get("variant_id", ""),
                "MV_AP_window": row.get("MV_AP_window", ""),
                "MV_AP50_window": row.get("MV_AP50_window", ""),
                "MV_AP_scene": row.get("MV_AP_scene", ""),
                "MV_AP50_scene": row.get("MV_AP50_scene", ""),
                "source_file": row.get("source_file", ""),
                "notes": row.get("notes", ""),
            }
        )

    gate_rows: list[dict[str, Any]] = []
    _append_gate(
        gate_rows,
        "formal_metric_source_eq_v65",
        bool(contract["formal_metric_source_eq_v65"]),
        expected="true",
        observed=contract["formal_metric_source_eq_v65"],
        source_file=contract["evaluator_file"],
    )
    _append_gate(
        gate_rows,
        "ap_thresholds_0p50_to_0p90",
        bool(contract["ap_thresholds_pass"]),
        expected=str(EXPECTED_THRESHOLDS),
        observed=str(contract["ap_thresholds"]),
        source_file=contract["evaluator_file"],
    )
    _append_gate(
        gate_rows,
        "local_support_policy_local_window_gt_projection",
        contract.get("local_support_policy") == "local_window_gt_projection",
        expected="local_window_gt_projection",
        observed=contract.get("local_support_policy", ""),
        source_file=contract["window_adapter_file"],
    )
    _append_gate(gate_rows, "F2_base_full_dev_window_exists", f2_dev_window is not None, source_file=_rel(dev_window_path))
    _append_gate(gate_rows, "F2_base_holdout_window_exists", f2_holdout_window is not None, source_file=_rel(holdout_window_path))
    _append_gate(gate_rows, "F2_base_full_dev_scene_exists", f2_dev_scene is not None, source_file=_rel(dev_scene_path))
    _append_gate(gate_rows, "F2_base_holdout_scene_exists", f2_holdout_scene is not None, source_file=_rel(holdout_scene_path))
    _append_gate(gate_rows, "Stream3D_corrected_local_window_exists", s3d_local_row is not None, source_file=_rel(s3d_local_path))
    _append_gate(
        gate_rows,
        "Stream3D_scene_diagnostic_exists",
        stream3d_scene_row is not None,
        source_file="" if stream3d_scene_row is None else stream3d_scene_row.get("source_file", ""),
    )
    _append_gate(gate_rows, "B0_reference_exists", b0_row is not None, source_file=_rel(s3d_local_path))
    _append_gate(gate_rows, "best_locked_control_exists", bool(controls), source_file=_rel(controls_path))
    _append_gate(
        gate_rows,
        "v91_best_non_oracle_exists",
        v91_best_row is not None,
        source_file="" if v91_best_row is None else v91_best_row.get("source_file", ""),
    )

    for gate_id, key, expected in [
        ("F2_same_frame_collision_zero", "same_frame_collision_count", "0"),
        ("F2_missing_mask_raster_zero", "missing_mask_raster_count", "0"),
        ("F2_uses_gt_for_prediction_false", "uses_gt_for_prediction", "False"),
        ("F2_uses_future_false", "uses_future", "False"),
    ]:
        values = []
        passed = True
        for src, row in [
            (_rel(dev_window_path), f2_dev_window),
            (_rel(holdout_window_path), f2_holdout_window),
            (_rel(dev_scene_path), f2_dev_scene),
            (_rel(holdout_scene_path), f2_holdout_scene),
        ]:
            value = "" if row is None else row.get(key, "")
            values.append(f"{src}:{value}")
            if key in {"uses_gt_for_prediction", "uses_future"}:
                passed = passed and (_bool(value) is False)
            else:
                passed = passed and (_float(value) == 0.0)
        _append_gate(gate_rows, gate_id, passed, expected=expected, observed="; ".join(values))

    gate_fields = ["gate_id", "pass", "expected", "observed", "severity", "source_file", "notes"]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "observed": row["observed"],
            "source_file": row["source_file"],
            "repair_direction": "repair metric adapter/artifact path before Phase1",
            "notes": row["notes"],
        }
        for row in gate_rows
        if not row["pass"]
    ]

    f2_dev_window_metric = next((r for r in baseline_rows if r["row_id"] == "F2_base_full_dev_MV_AP_window"), {})
    f2_dev_scene_metric = next((r for r in baseline_rows if r["row_id"] == "F2_base_full_dev_MV_AP_scene"), {})
    f2_holdout_window_metric = next((r for r in baseline_rows if r["row_id"] == "F2_base_holdout_MV_AP_window"), {})
    f2_holdout_scene_metric = next((r for r in baseline_rows if r["row_id"] == "F2_base_holdout_MV_AP_scene"), {})
    s3d_local_metric = next((r for r in baseline_rows if r["row_id"] == "Stream3D_corrected_local_window_MV_AP_window"), {})
    s3d_scene_metric = next((r for r in baseline_rows if r["row_id"] == "Stream3D_scene_diagnostic_MV_AP_scene"), {})
    b0_metric = next((r for r in baseline_rows if r["row_id"] == "B0_MV_AP_window"), {})
    control_metric = next((r for r in baseline_rows if r["row_id"] == "best_locked_control_MV_AP_window"), {})
    v91_metric = next((r for r in baseline_rows if r["row_id"] == "v91_best_MV_AP_window"), {})

    phase0_pass = not failure_rows
    summary_out = {
        "schema_version": "stream4d_v99_phase0_fact_lock_summary_v1",
        "phase_id": "v99_phase0_fact_lock",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase0_pass": phase0_pass,
        "decision": "PASS_ENTER_PHASE1" if phase0_pass else "BLOCK_PHASE1_REPAIR_PHASE0_FAILURES",
        "failure_count": len(failure_rows),
        "failure_ids": [row["failure_id"] for row in failure_rows],
        "F2_base_variant_id": F2_BASE,
        "F2_base_full_dev_MV_AP_window": _float(f2_dev_window_metric.get("MV_AP_window")),
        "F2_base_full_dev_MV_AP50_window": _float(f2_dev_window_metric.get("MV_AP50_window")),
        "F2_base_full_dev_MV_AP_scene": _float(f2_dev_scene_metric.get("MV_AP_scene")),
        "F2_base_full_dev_MV_AP50_scene": _float(f2_dev_scene_metric.get("MV_AP50_scene")),
        "F2_base_holdout_MV_AP_window": _float(f2_holdout_window_metric.get("MV_AP_window")),
        "F2_base_holdout_MV_AP50_window": _float(f2_holdout_window_metric.get("MV_AP50_window")),
        "F2_base_holdout_MV_AP_scene": _float(f2_holdout_scene_metric.get("MV_AP_scene")),
        "F2_base_holdout_MV_AP50_scene": _float(f2_holdout_scene_metric.get("MV_AP50_scene")),
        "Stream3D_corrected_local_window_MV_AP_window": _float(s3d_local_metric.get("MV_AP_window")),
        "Stream3D_corrected_local_window_MV_AP50_window": _float(s3d_local_metric.get("MV_AP50_window")),
        "Stream3D_scene_diagnostic_MV_AP_scene": _float(s3d_scene_metric.get("MV_AP_scene")),
        "Stream3D_scene_diagnostic_MV_AP50_scene": _float(s3d_scene_metric.get("MV_AP50_scene")),
        "B0_MV_AP_window": _float(b0_metric.get("MV_AP_window")),
        "best_locked_control_MV_AP_window": _float(control_metric.get("MV_AP_window")),
        "v91_best_MV_AP_window": _float(v91_metric.get("MV_AP_window")),
        "v98_final_decision_summary": {
            key: summary.get(key)
            for key in [
                "decision",
                "best_variant",
                "metric_gate_pass",
                "uses_gt_for_prediction",
                "uses_future",
                "strong_reliable_d4rt_dense_semantic_claim_allowed",
                "metric_contract",
            ]
        },
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "baseline_metric_rows": _rel(OUT_DIR / "baseline_metric_rows.csv"),
            "mv_ap_contract": _rel(OUT_DIR / "mv_ap_contract.json"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "casebook_rows": _rel(OUT_DIR / "casebook_rows.csv"),
        },
    }

    _write_json(OUT_DIR / "mv_ap_contract.json", contract)
    _write_csv(OUT_DIR / "baseline_metric_rows.csv", baseline_rows, BASELINE_FIELDS)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows, gate_fields)
    _write_csv(
        OUT_DIR / "variant_failure_rows.csv",
        failure_rows,
        ["failure_id", "severity", "observed", "source_file", "repair_direction", "notes"],
    )
    _write_csv(
        OUT_DIR / "casebook_rows.csv",
        casebook_rows,
        [
            "case_id",
            "case_type",
            "variant_id",
            "MV_AP_window",
            "MV_AP50_window",
            "MV_AP_scene",
            "MV_AP50_scene",
            "source_file",
            "notes",
        ],
    )
    _write_json(OUT_DIR / "summary.json", summary_out)

    print(json.dumps(summary_out, indent=2, sort_keys=True))
    return 0 if phase0_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
