from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


AUDIT_ROOT = ROOT / "outputs/audit"
COMMON_SCHEMA = "stream4d_v92_common_artifact_closure_v1"
COMMON_FIELDS = [
    "schema_version",
    "phase_id",
    "run_id",
    "variant_id",
    "scene_id",
    "split",
    "window_id",
    "chunk_id",
    "uses_gt_for_prediction",
    "uses_future",
    "uses_rgbd_pose_mesh",
    "source_artifact",
    "source_artifact_sha256",
    "created_at",
]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


CREATED_AT = _now()


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], extra_fields: list[str] | None = None) -> None:
    fields = list(COMMON_FIELDS)
    for field in extra_fields or []:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _common(
    *,
    phase_id: str,
    run_id: str,
    variant_id: str,
    source: Path,
    scene_id: str = "aggregate",
    split: str = "dev",
    window_id: str = "",
    chunk_id: str = "",
    uses_gt_for_prediction: Any = False,
    uses_future: Any = False,
    uses_rgbd_pose_mesh: Any = False,
) -> dict[str, Any]:
    return {
        "schema_version": COMMON_SCHEMA,
        "phase_id": phase_id,
        "run_id": run_id,
        "variant_id": variant_id,
        "scene_id": scene_id or "aggregate",
        "split": split or "dev",
        "window_id": window_id,
        "chunk_id": chunk_id,
        "uses_gt_for_prediction": _as_bool(uses_gt_for_prediction),
        "uses_future": _as_bool(uses_future),
        "uses_rgbd_pose_mesh": _as_bool(uses_rgbd_pose_mesh),
        "source_artifact": _rel(source),
        "source_artifact_sha256": _sha256(source) if source.exists() else "",
        "created_at": CREATED_AT,
    }


def _merge_common(
    row: dict[str, Any],
    *,
    phase_id: str,
    run_id: str,
    source: Path,
    variant_id: str | None = None,
) -> dict[str, Any]:
    common = _common(
        phase_id=phase_id,
        run_id=run_id,
        variant_id=variant_id or str(row.get("variant_id") or row.get("variant") or phase_id),
        source=source,
        scene_id=str(row.get("scene_id") or row.get("scene") or "aggregate"),
        split=str(row.get("split") or "dev"),
        window_id=str(row.get("window_id") or ""),
        chunk_id=str(row.get("chunk_id") or ""),
        uses_gt_for_prediction=row.get("uses_gt_for_prediction", False),
        uses_future=row.get("uses_future", False),
        uses_rgbd_pose_mesh=row.get("uses_rgbd_pose_mesh", False),
    )
    merged = dict(row)
    merged.update(common)
    return merged


def _update_hash_sidecar(phase_dir: Path, touched: list[Path]) -> None:
    sidecar = phase_dir / "SHA256SUMS.json"
    payload: dict[str, str] = {}
    if sidecar.exists():
        try:
            payload.update(json.loads(sidecar.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            payload = {}
    for path in touched:
        if path.exists():
            payload[_rel(path)] = _sha256(path)
    _write_json(sidecar, payload)


def _copy_csv_with_common(
    phase_dir: Path,
    *,
    src_name: str,
    dst_name: str,
    phase_id: str,
    run_id: str,
    fallback_fields: list[str] | None = None,
    row_filter: Callable[[dict[str, str]], bool] | None = None,
) -> Path:
    src = phase_dir / src_name
    rows = _read_csv(src)
    if row_filter is not None:
        rows = [row for row in rows if row_filter(row)]
    out_rows = [_merge_common(row, phase_id=phase_id, run_id=run_id, source=src) for row in rows]
    dst = phase_dir / dst_name
    _write_csv(dst, out_rows, fallback_fields)
    return dst


def _phase3c() -> list[Path]:
    phase_dir = AUDIT_ROOT / "v92_phase3c_hr2_uncertainty_readout"
    summary = _json(phase_dir / "summary.json")
    phase_id = "v92_phase3c_hr2_uncertainty_readout"
    run_id = str(summary.get("run_id") or phase_id)
    return [
        _copy_csv_with_common(
            phase_dir,
            src_name="mv_metric_aggregate_rows.csv",
            dst_name="variant_metric_rows.csv",
            phase_id=phase_id,
            run_id=run_id,
        )
    ]


def _phase4() -> list[Path]:
    phase_dir = AUDIT_ROOT / "v92_phase4_semantic_region_affinity"
    summary_path = phase_dir / "summary.json"
    summary = _json(summary_path)
    phase_id = str(summary.get("phase_id") or "v92_phase4_semantic_region_affinity")
    run_id = str(summary.get("run_id") or phase_id)
    touched: list[Path] = []

    config_rows = [
        {
            **_common(
                phase_id=phase_id,
                run_id=run_id,
                variant_id="V92_PHASE4_RADIO_REGION_AFFINITY",
                source=summary_path,
                uses_gt_for_prediction=summary.get("uses_gt_for_prediction", False),
                uses_future=summary.get("uses_future", False),
            ),
            "variant_kind": "diagnostic",
            "semantic_backend": summary.get("semantic_backend", ""),
            "feature_layer": summary.get("feature_layer", ""),
            "region_node_rows": summary.get("region_node_rows", ""),
            "region_edge_rows": summary.get("region_edge_rows", ""),
            "diagnostic_only_uses_gt": summary.get("diagnostic_only_uses_gt", ""),
            "note": "Derived common wrapper; Phase4 is diagnostic-only and does not claim MV_AP method success.",
        }
    ]
    config_path = phase_dir / "variant_config_rows.csv"
    _write_csv(config_path, config_rows)
    touched.append(config_path)

    metric_path = _copy_csv_with_common(
        phase_dir,
        src_name="semantic_diagnostic_auc_rows.csv",
        dst_name="variant_metric_rows.csv",
        phase_id=phase_id,
        run_id=run_id,
        fallback_fields=[
            "scope",
            "metric_name",
            "auc",
            "sample_count",
            "positive_count",
            "negative_count",
            "diagnostic_only_uses_gt",
        ],
    )
    touched.append(metric_path)

    gate_rows = [
        {
            **_common(
                phase_id=phase_id,
                run_id=run_id,
                variant_id="V92_PHASE4_RADIO_REGION_AFFINITY",
                source=summary_path,
                uses_gt_for_prediction=summary.get("uses_gt_for_prediction", False),
                uses_future=summary.get("uses_future", False),
            ),
            "routing_label": summary.get("routing_label", summary.get("decision", "")),
            "diagnostic_gate_pass": summary.get("routing_label") == "SEMANTIC_REGION_SIGNAL_STRONG",
            "source_internal_same_gt_different_gt_AUC_mean": summary.get("source_internal_same_gt_different_gt_AUC_mean", ""),
            "foreground_background_region_AUC_mean": summary.get("foreground_background_region_AUC_mean", ""),
            "method_success_claim_allowed": False,
        }
    ]
    gate_path = phase_dir / "variant_gate_rows.csv"
    _write_csv(gate_path, gate_rows)
    touched.append(gate_path)

    failure_path = _copy_csv_with_common(
        phase_dir,
        src_name="semantic_failure_rows.csv",
        dst_name="variant_failure_rows.csv",
        phase_id=phase_id,
        run_id=run_id,
        fallback_fields=["failure_type", "repair_direction", "detail"],
    )
    touched.append(failure_path)

    case_rows = [
        {
            **_common(
                phase_id=phase_id,
                run_id=run_id,
                variant_id="V92_PHASE4_RADIO_REGION_AFFINITY",
                source=summary_path,
                uses_gt_for_prediction=summary.get("uses_gt_for_prediction", False),
                uses_future=summary.get("uses_future", False),
            ),
            "case_type": "semantic_region_signal",
            "source_internal_same_gt_different_gt_AUC_mean": summary.get("source_internal_same_gt_different_gt_AUC_mean", ""),
            "foreground_background_region_AUC_mean": summary.get("foreground_background_region_AUC_mean", ""),
            "interpretation": "RADIO source-internal instance-affinity signal is present; Phase5 must test readout under MV_AP_window.",
        }
    ]
    case_path = phase_dir / "casebook_rows.csv"
    _write_csv(case_path, case_rows)
    touched.append(case_path)
    return touched


def _phase4b() -> list[Path]:
    phase_dir = AUDIT_ROOT / "v92_phase4b_region_granularity_coarse2"
    summary_path = phase_dir / "summary.json"
    summary = _json(summary_path)
    phase_id = "v92_phase4b_region_granularity_repair"
    run_id = str(summary.get("run_id") or phase_id)
    touched: list[Path] = []
    config_rows = [
        {
            **_common(
                phase_id=phase_id,
                run_id=run_id,
                variant_id="V92_PHASE4B_COARSE2_REGION_GRAPH",
                source=summary_path,
                uses_gt_for_prediction=summary.get("uses_gt_for_prediction", False),
                uses_future=summary.get("uses_future", False),
            ),
            "variant_kind": "region_granularity_repair",
            "coarsen_factor": summary.get("coarsen_factor", ""),
            "input_region_node_rows": summary.get("input_region_node_rows", ""),
            "output_region_node_rows": summary.get("output_region_node_rows", ""),
            "region_node_reduction_ratio": summary.get("region_node_reduction_ratio", ""),
        }
    ]
    config_path = phase_dir / "variant_config_rows.csv"
    _write_csv(config_path, config_rows)
    touched.append(config_path)
    metric_path = phase_dir / "variant_metric_rows.csv"
    _write_csv(
        metric_path,
        [
            {
                **_common(
                    phase_id=phase_id,
                    run_id=run_id,
                    variant_id="V92_PHASE4B_COARSE2_REGION_GRAPH",
                    source=summary_path,
                    uses_gt_for_prediction=summary.get("uses_gt_for_prediction", False),
                    uses_future=summary.get("uses_future", False),
                ),
                "input_region_node_rows": summary.get("input_region_node_rows", ""),
                "output_region_node_rows": summary.get("output_region_node_rows", ""),
                "input_region_edge_rows": summary.get("input_region_edge_rows", ""),
                "output_region_edge_rows": summary.get("output_region_edge_rows", ""),
                "region_node_reduction_ratio": summary.get("region_node_reduction_ratio", ""),
            }
        ],
    )
    touched.append(metric_path)
    gate_path = phase_dir / "variant_gate_rows.csv"
    _write_csv(
        gate_path,
        [
            {
                **_common(
                    phase_id=phase_id,
                    run_id=run_id,
                    variant_id="V92_PHASE4B_COARSE2_REGION_GRAPH",
                    source=summary_path,
                    uses_gt_for_prediction=summary.get("uses_gt_for_prediction", False),
                    uses_future=summary.get("uses_future", False),
                ),
                "materialized_for_phase5": True,
                "method_success_claim_allowed": False,
            }
        ],
    )
    touched.append(gate_path)
    failure_path = phase_dir / "variant_failure_rows.csv"
    _write_csv(failure_path, [])
    touched.append(failure_path)
    case_path = phase_dir / "casebook_rows.csv"
    _write_csv(
        case_path,
        [
            {
                **_common(
                    phase_id=phase_id,
                    run_id=run_id,
                    variant_id="V92_PHASE4B_COARSE2_REGION_GRAPH",
                    source=summary_path,
                    uses_gt_for_prediction=summary.get("uses_gt_for_prediction", False),
                    uses_future=summary.get("uses_future", False),
                ),
                "case_type": "region_granularity_repair",
                "interpretation": "Coarser derived RADIO region graph is a Phase5 repair input; MV_AP success must be judged by Phase5E.",
            }
        ],
    )
    touched.append(case_path)
    return touched


def _phase5(dirname: str) -> list[Path]:
    phase_dir = AUDIT_ROOT / dirname
    summary = _json(phase_dir / "summary.json")
    phase_id = dirname
    run_id = str(summary.get("run_id") or phase_id)
    touched = [
        _copy_csv_with_common(
            phase_dir,
            src_name="field_variant_config_rows.csv",
            dst_name="variant_config_rows.csv",
            phase_id=phase_id,
            run_id=run_id,
        ),
        _copy_csv_with_common(
            phase_dir,
            src_name="control_metric_rows.csv",
            dst_name="variant_metric_rows.csv",
            phase_id=phase_id,
            run_id=run_id,
        ),
        _copy_csv_with_common(
            phase_dir,
            src_name="control_metric_rows.csv",
            dst_name="variant_gate_rows.csv",
            phase_id=phase_id,
            run_id=run_id,
        ),
        _copy_csv_with_common(
            phase_dir,
            src_name="field_failure_rows.csv",
            dst_name="variant_failure_rows.csv",
            phase_id=phase_id,
            run_id=run_id,
            fallback_fields=[
                "failure_type",
                "repair_direction",
                "MV_AP_window",
                "MV_AP50_window",
                "best_control_MV_AP_window",
                "real_minus_best_control_MV_AP_window",
            ],
        ),
    ]
    return touched


def _phase5d() -> list[Path]:
    phase_dir = AUDIT_ROOT / "v92_phase5d_score_calibration"
    summary = _json(phase_dir / "summary.json")
    phase_id = "v92_phase5d_score_calibration"
    run_id = str(summary.get("run_id") or phase_id)
    touched = [
        _copy_csv_with_common(
            phase_dir,
            src_name="score_variant_config_rows.csv",
            dst_name="variant_config_rows.csv",
            phase_id=phase_id,
            run_id=run_id,
        ),
        _copy_csv_with_common(
            phase_dir,
            src_name="control_metric_rows.csv",
            dst_name="variant_metric_rows.csv",
            phase_id=phase_id,
            run_id=run_id,
        ),
        _copy_csv_with_common(
            phase_dir,
            src_name="control_metric_rows.csv",
            dst_name="variant_gate_rows.csv",
            phase_id=phase_id,
            run_id=run_id,
        ),
        _copy_csv_with_common(
            phase_dir,
            src_name="score_failure_rows.csv",
            dst_name="variant_failure_rows.csv",
            phase_id=phase_id,
            run_id=run_id,
            fallback_fields=[
                "failure_type",
                "repair_direction",
                "MV_AP_window",
                "MV_AP50_window",
                "score_free_Match50_window",
                "best_control_MV_AP_window",
                "real_minus_best_control_MV_AP_window",
            ],
        ),
    ]
    case_path = phase_dir / "casebook_rows.csv"
    if not case_path.exists():
        _write_csv(
            case_path,
            [
                {
                    **_common(
                        phase_id=phase_id,
                        run_id=run_id,
                        variant_id=str(summary.get("best_variant_id", "V92_PHASE5D_SCORE_CALIBRATION")),
                        source=phase_dir / "summary.json",
                        uses_gt_for_prediction=summary.get("uses_gt_for_prediction", False),
                        uses_future=summary.get("uses_future", False),
                    ),
                    "case_type": "score_calibration_no_go",
                    "decision": summary.get("decision", ""),
                    "interpretation": "GT-free score calibration over Phase5B masks did not pass the local-window MV_AP/control gate.",
                }
            ],
        )
    touched.append(case_path)
    return touched


def _phase6() -> list[Path]:
    phase_dir = AUDIT_ROOT / "v92_phase6_attribution"
    summary_path = phase_dir / "summary.json"
    summary = _json(summary_path)
    phase_id = "v92_phase6_attribution"
    run_id = phase_id
    touched: list[Path] = []

    config_path = phase_dir / "variant_config_rows.csv"
    config_rows = [
        {
            **_common(
                phase_id=phase_id,
                run_id=run_id,
                variant_id="V92_PHASE6_ATTRIBUTION_MATRIX",
                source=summary_path,
            ),
            "variant_kind": "attribution",
            "phase5b_root": summary.get("phase5b_root", ""),
            "phase4_root": summary.get("phase4_root", ""),
            "note": "Derived common wrapper around Phase6 D4RT/RADIO/fusion attribution.",
        }
    ]
    _write_csv(config_path, config_rows)
    touched.append(config_path)

    touched.append(
        _copy_csv_with_common(
            phase_dir,
            src_name="attribution_metric_rows.csv",
            dst_name="variant_metric_rows.csv",
            phase_id=phase_id,
            run_id=run_id,
            fallback_fields=["metric_name", "value"],
        )
    )
    gate_rows = [
        {
            **_common(
                phase_id=phase_id,
                run_id=run_id,
                variant_id="V92_PHASE6_ATTRIBUTION_MATRIX",
                source=summary_path,
            ),
            "decision": summary.get("decision", ""),
            "gate_pass": summary.get("decision") == "GEOMETRY_SEMANTIC_COMPLEMENTARITY_SUPPORTED",
            "D4RT_plus_RADIO_MV_AP_window": summary.get("D4RT_plus_RADIO_MV_AP_window", ""),
            "best_control_MV_AP_window": summary.get("best_control_MV_AP_window", ""),
            "whole_source_MV_AP_window": summary.get("whole_source_MV_AP_window", ""),
            "method_success_claim_allowed": False,
        }
    ]
    gate_path = phase_dir / "variant_gate_rows.csv"
    _write_csv(gate_path, gate_rows)
    touched.append(gate_path)

    failure_rows = []
    if summary.get("decision") != "GEOMETRY_SEMANTIC_COMPLEMENTARITY_SUPPORTED":
        failure_rows.append(
            {
                **_common(
                    phase_id=phase_id,
                    run_id=run_id,
                    variant_id="V92_PHASE6_ATTRIBUTION_MATRIX",
                    source=summary_path,
                ),
                "failure_type": summary.get("decision", ""),
                "repair_direction": "Improve source-container object membership readout before holdout/local2history.",
                "D4RT_plus_RADIO_MV_AP_window": summary.get("D4RT_plus_RADIO_MV_AP_window", ""),
                "best_control_MV_AP_window": summary.get("best_control_MV_AP_window", ""),
                "real_minus_best_control_MV_AP_window": _num(summary.get("D4RT_plus_RADIO_MV_AP_window")) - _num(summary.get("best_control_MV_AP_window")),
            }
        )
    failure_path = phase_dir / "variant_failure_rows.csv"
    _write_csv(
        failure_path,
        failure_rows,
        ["failure_type", "repair_direction", "D4RT_plus_RADIO_MV_AP_window", "best_control_MV_AP_window", "real_minus_best_control_MV_AP_window"],
    )
    touched.append(failure_path)

    case_rows = [
        {
            **_common(
                phase_id=phase_id,
                run_id=run_id,
                variant_id="V92_PHASE6_ATTRIBUTION_MATRIX",
                source=summary_path,
            ),
            "case_type": "control_bias_attribution",
            "interpretation": "Fusion beats single-route ablations but loses to whole-source/control baselines, so no complementarity claim is allowed.",
            "decision": summary.get("decision", ""),
        }
    ]
    case_path = phase_dir / "casebook_rows.csv"
    _write_csv(case_path, case_rows)
    touched.append(case_path)
    return touched


def _phase9() -> list[Path]:
    phase_dir = AUDIT_ROOT / "v92_phase9_casebook"
    decision_path = phase_dir / "final_decision.json"
    decision = _json(decision_path)
    phase_id = "v92_phase9_casebook"
    run_id = phase_id
    touched: list[Path] = []

    summary = dict(decision)
    summary["schema"] = "stream4d_v92_phase9_summary_v1"
    summary["summary_source_artifact"] = _rel(decision_path)
    summary["summary_source_artifact_sha256"] = _sha256(decision_path)
    summary_path = phase_dir / "summary.json"
    _write_json(summary_path, summary)
    touched.append(summary_path)

    config_rows = [
        {
            **_common(
                phase_id=phase_id,
                run_id=run_id,
                variant_id="V92_PHASE9_FINAL_DECISION",
                source=decision_path,
            ),
            "variant_kind": "final_decision",
            "holdout_decision": decision.get("holdout_decision", ""),
            "da3_branch_decision": decision.get("da3_branch_decision", ""),
        }
    ]
    config_path = phase_dir / "variant_config_rows.csv"
    _write_csv(config_path, config_rows)
    touched.append(config_path)

    metric_rows = []
    for name, value in (decision.get("key_metrics") or {}).items():
        metric_rows.append(
            {
                **_common(
                    phase_id=phase_id,
                    run_id=run_id,
                    variant_id="V92_PHASE9_FINAL_DECISION",
                    source=decision_path,
                ),
                "metric_name": name,
                "metric_value": value,
            }
        )
    metric_path = phase_dir / "variant_metric_rows.csv"
    _write_csv(metric_path, metric_rows, ["metric_name", "metric_value"])
    touched.append(metric_path)

    gate_rows = [
        {
            **_common(
                phase_id=phase_id,
                run_id=run_id,
                variant_id="V92_PHASE9_FINAL_DECISION",
                source=decision_path,
            ),
            "final_decision": decision.get("final_decision", ""),
            "can_claim_local_method_success": decision.get("can_claim_local_method_success", ""),
            "can_enter_local2history": decision.get("can_enter_local2history", ""),
            "holdout_decision": decision.get("holdout_decision", ""),
            "da3_branch_decision": decision.get("da3_branch_decision", ""),
        }
    ]
    gate_path = phase_dir / "variant_gate_rows.csv"
    _write_csv(gate_path, gate_rows)
    touched.append(gate_path)

    failure_rows = []
    if not decision.get("can_claim_local_method_success", False):
        failure_rows.append(
            {
                **_common(
                    phase_id=phase_id,
                    run_id=run_id,
                    variant_id="V92_PHASE9_FINAL_DECISION",
                    source=decision_path,
                ),
                "failure_type": decision.get("final_decision", ""),
                "primary_blocker": decision.get("primary_blocker", ""),
                "secondary_blocker": decision.get("secondary_blocker", ""),
                "repair_direction": "Future work must improve membership-field readout enough to beat controls under MV_AP_window before holdout/local2history.",
            }
        )
    failure_path = phase_dir / "variant_failure_rows.csv"
    _write_csv(failure_path, failure_rows, ["failure_type", "primary_blocker", "secondary_blocker", "repair_direction"])
    touched.append(failure_path)

    case_rows = []
    answers = decision.get("answers") or {}
    for name, value in answers.items():
        case_rows.append(
            {
                **_common(
                    phase_id=phase_id,
                    run_id=run_id,
                    variant_id="V92_PHASE9_FINAL_DECISION",
                    source=decision_path,
                ),
                "case_type": "final_answer",
                "question_key": name,
                "answer": value,
            }
        )
    case_path = phase_dir / "casebook_rows.csv"
    _write_csv(case_path, case_rows, ["case_type", "question_key", "answer"])
    touched.append(case_path)
    return touched


def _validate_common() -> dict[str, list[str]]:
    required = [
        "summary.json",
        "variant_config_rows.csv",
        "variant_metric_rows.csv",
        "variant_gate_rows.csv",
        "variant_failure_rows.csv",
        "casebook_rows.csv",
    ]
    phase_names = [
        "v92_phase0_mv_ap_contract",
        "v92_phase1_source_container_registry",
        "v92_phase2_d4rt_sufficiency",
        "v92_phase3_d4rt_highres",
        "v92_phase3_d4rt_highres_hr2_grid16",
        "v92_phase3c_hr2_uncertainty_readout",
        "v92_phase4_semantic_region_affinity",
        "v92_phase4b_region_granularity_coarse2",
        "v92_phase5_source_container_field",
        "v92_phase5b_source_container_edge_field",
        "v92_phase5c_tight_field_repair",
        "v92_phase5d_score_calibration",
        "v92_phase5e_coarse2_tight_field",
        "v92_phase6_attribution",
        "v92_phase9_casebook",
    ]
    missing: dict[str, list[str]] = {}
    for name in phase_names:
        phase_dir = AUDIT_ROOT / name
        absent = [item for item in required if not (phase_dir / item).exists()]
        if absent:
            missing[name] = absent
    return missing


def run(_: argparse.Namespace) -> dict[str, Any]:
    phase_touched: dict[str, list[Path]] = {
        "v92_phase3c_hr2_uncertainty_readout": _phase3c(),
        "v92_phase4_semantic_region_affinity": _phase4(),
        "v92_phase4b_region_granularity_coarse2": _phase4b(),
        "v92_phase5_source_container_field": _phase5("v92_phase5_source_container_field"),
        "v92_phase5b_source_container_edge_field": _phase5("v92_phase5b_source_container_edge_field"),
        "v92_phase5c_tight_field_repair": _phase5("v92_phase5c_tight_field_repair"),
        "v92_phase5d_score_calibration": _phase5d(),
        "v92_phase5e_coarse2_tight_field": _phase5("v92_phase5e_coarse2_tight_field"),
        "v92_phase6_attribution": _phase6(),
        "v92_phase9_casebook": _phase9(),
    }
    for phase_name, paths in phase_touched.items():
        _update_hash_sidecar(AUDIT_ROOT / phase_name, paths)
    summary = {
        "schema": "stream4d_v92_common_artifact_closure_summary_v1",
        "created_at": CREATED_AT,
        "phase_touched": {name: [_rel(path) for path in paths] for name, paths in phase_touched.items()},
        "missing_common_artifacts_after": _validate_common(),
        "changes_metric_values": False,
        "changes_final_decision": False,
        "note": "Derived common wrappers from already materialized v92 outputs to satisfy the plan Section 5 artifact naming contract.",
    }
    out_path = AUDIT_ROOT / "v92_phase9_casebook" / "common_artifact_closure_summary.json"
    _write_json(out_path, summary)
    _update_hash_sidecar(AUDIT_ROOT / "v92_phase9_casebook", [out_path])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Build missing v92 common audit artifact wrappers.").parse_args()


if __name__ == "__main__":
    run(parse_args())
