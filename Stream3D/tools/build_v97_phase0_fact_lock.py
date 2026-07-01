#!/usr/bin/env python3
"""Build Stream4D v97 Phase0 fact lock and scope contract artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
OUT = ROOT / "outputs/audit/v97_phase0_fact_lock"
PHASE_ID = "v97_phase0_fact_lock"
RUN_ID = "v97_phase0_fact_lock"

PLAN = REPO_ROOT / "docs/stream4d_v97_d4rt_micro_primitive_semantic_affinity_field_plan.md"
V65_EVALUATOR = ROOT / "tools/run_v65_scene_multiview_ap.py"
V91_CONTRACT = ROOT / "outputs/audit/v91_phase0_mv_ap_contract/summary.json"
V93_PHASE0 = ROOT / "outputs/audit/v93_phase0_contract/summary.json"
V93_BASELINES = ROOT / "outputs/audit/v93_phase0_contract/baseline_metric_rows.csv"
V94_CONTROLS = ROOT / "outputs/audit/v94_phase4_controls/summary.json"
V95_PHASE0 = ROOT / "outputs/audit/v95_phase0_fact_lock/summary.json"
V96_PHASE0 = ROOT / "outputs/audit/v96_phase0_fact_lock/summary.json"
V96_FINAL = ROOT / "outputs/audit/v96_phase10_dev_decision_object_core_k512_s010_h1_R32_all_controls/final_dev_decision.json"
V96_PHASE6 = ROOT / "outputs/audit/v96_phase6_render_snap_object_core_k512_s010_h1_frame_count_x_masklet/summary.json"
V96_PHASE9 = ROOT / "outputs/audit/v96_phase9_error_decomposition_object_core_k512_s010_h1_R32/blocker_summary.json"
D4RT_ADAPTER = ROOT / "stream4d/d4rt_adapter.py"
D4RT_ROOT = REPO_ROOT / "Open-d4rt"
D4RT_CONFIG = D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
D4RT_CKPT = D4RT_ROOT / "checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"

SEMANTIC_SOURCES = [
    {
        "source_id": "v91_radio_mask_features_npz",
        "feature_family": "radio",
        "feature_scope": "mask_npz_dev",
        "root": ROOT / "outputs/audit/v91_radio_mask_features_npz",
        "primary_file": ROOT / "outputs/audit/v91_radio_mask_features_npz/mask_features.npz",
        "dense_or_region": "mask_region_proxy",
        "allowed_for_v97_semantic_aware": True,
        "notes": "RADIO mask-feature NPZ can seed semantic attachment if dense tensors are unavailable; must be labelled region_proxy.",
    },
    {
        "source_id": "v91_radio_mask_features_npz_scene0011",
        "feature_family": "radio",
        "feature_scope": "mask_npz_scene0011",
        "root": ROOT / "outputs/audit/v91_radio_mask_features_npz_scene0011",
        "primary_file": ROOT / "outputs/audit/v91_radio_mask_features_npz_scene0011/mask_features.npz",
        "dense_or_region": "mask_region_proxy",
        "allowed_for_v97_semantic_aware": True,
        "notes": "Scene-specific RADIO mask-feature NPZ.",
    },
    {
        "source_id": "v91_radio_mask_features_npz_scene0050",
        "feature_family": "radio",
        "feature_scope": "mask_npz_scene0050",
        "root": ROOT / "outputs/audit/v91_radio_mask_features_npz_scene0050",
        "primary_file": ROOT / "outputs/audit/v91_radio_mask_features_npz_scene0050/mask_features.npz",
        "dense_or_region": "mask_region_proxy",
        "allowed_for_v97_semantic_aware": True,
        "notes": "Scene-specific RADIO mask-feature NPZ.",
    },
    {
        "source_id": "v92_semantic_region_affinity_scene0011",
        "feature_family": "radio_or_dino_region",
        "feature_scope": "region_scene0011",
        "root": ROOT / "outputs/audit/v92_phase4_semantic_region_affinity_scene0011",
        "primary_file": ROOT / "outputs/audit/v92_phase4_semantic_region_affinity_scene0011/frame_feature_rows.csv",
        "dense_or_region": "region_proxy",
        "allowed_for_v97_semantic_aware": True,
        "notes": "Region/frame feature rows; must not be reported as dense tensor sampling.",
    },
    {
        "source_id": "v92_semantic_region_affinity_scene0050",
        "feature_family": "radio_or_dino_region",
        "feature_scope": "region_scene0050",
        "root": ROOT / "outputs/audit/v92_phase4_semantic_region_affinity_scene0050",
        "primary_file": ROOT / "outputs/audit/v92_phase4_semantic_region_affinity_scene0050/frame_feature_rows.csv",
        "dense_or_region": "region_proxy",
        "allowed_for_v97_semantic_aware": True,
        "notes": "Region/frame feature rows; must not be reported as dense tensor sampling.",
    },
    {
        "source_id": "v81_dino_feature_json_scene0011_scene0050",
        "feature_family": "dino",
        "feature_scope": "mask_json_scene0011_scene0050",
        "root": ROOT / "outputs/audit/v81_dino_feature_json_scene0011_scene0050",
        "primary_file": ROOT / "outputs/audit/v81_dino_feature_json_scene0011_scene0050/mask_feature_rows.csv",
        "dense_or_region": "mask_region_proxy",
        "allowed_for_v97_semantic_aware": True,
        "notes": "Large DINO mask-feature CSV; can be used as source-internal region feature if loaded carefully.",
    },
]

EXPECTED_AP_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
FORMAL_METRIC_SOURCE = "tools.run_v65_scene_multiview_ap.SparseSceneIoU/_summarize_iou"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
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
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _sha256(path: Path, *, max_hash_bytes: int = 1024 * 1024 * 1024) -> tuple[str, str]:
    if not path.exists() or not path.is_file():
        return "", "missing_or_not_file"
    size = path.stat().st_size
    if size > max_hash_bytes:
        return "", f"skipped_large_file_size_bytes_{size}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "computed"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _run_text(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout


def _copy_baseline_rows(created_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(V93_BASELINES):
        out: dict[str, Any] = dict(row)
        out["schema_version"] = "stream4d_v97_phase0_baseline_metric_v1"
        out["phase_id"] = PHASE_ID
        out["run_id"] = RUN_ID
        out["source_phase"] = "v93_phase0_contract"
        out["created_at"] = created_at
        rows.append(out)

    v96_final = _read_json(V96_FINAL)
    best = v96_final.get("phase6_c_best_variant", {}) if isinstance(v96_final.get("phase6_c_best_variant"), dict) else {}
    rows.append(
        {
            "schema_version": "stream4d_v97_phase0_baseline_metric_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "variant_id": best.get("readout_variant", ""),
            "variant_family": "v96_latest_best_no_go_boundary",
            "source_version": "v96_phase10_dev_decision_object_core_k512_s010_h1_R32_all_controls",
            "support_policy": "local_window_gt_projection",
            "metric_source": FORMAL_METRIC_SOURCE,
            "metric_scope": "full_dev",
            "MV_AP_window": best.get("MV_AP_window", ""),
            "MV_AP50_window": best.get("MV_AP50_window", ""),
            "MV_AP25_window": best.get("MV_AP25_window", ""),
            "ScoreFreeMatch50_window": best.get("ScoreFreeMatch50_window", ""),
            "ScoreFreeMatch25_window": best.get("ScoreFreeMatch25_window", ""),
            "same_frame_collision_count": best.get("same_frame_collision_count", 0),
            "missing_mask_raster_count": best.get("missing_mask_raster_count", 0),
            "uses_gt_for_prediction_count": int(_bool(best.get("uses_gt_for_prediction"))),
            "uses_future_count": int(_bool(best.get("uses_future"))),
            "notes": "v96 latest best is a locked No-Go boundary, not v97 progress.",
            "scene_id": "ALL_DEV",
            "split": "dev",
            "window_id": "ALL_DEV_WINDOWS",
            "source_artifact": _rel(V96_FINAL),
            "created_at": created_at,
        }
    )
    return rows


def _semantic_rows(created_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in SEMANTIC_SOURCES:
        root = Path(src["root"])
        primary = Path(src["primary_file"])
        digest, digest_status = _sha256(primary)
        stat = primary.stat() if primary.exists() and primary.is_file() else None
        rows.append(
            {
                "schema_version": "stream4d_v97_phase0_semantic_feature_contract_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "source_id": src["source_id"],
                "feature_family": src["feature_family"],
                "feature_scope": src["feature_scope"],
                "semantic_feature_status": "available_region_proxy" if root.exists() and primary.exists() else "missing",
                "semantic_source": src["dense_or_region"],
                "root_path": _rel(root),
                "root_exists": root.exists(),
                "primary_file": _rel(primary),
                "primary_file_exists": primary.exists(),
                "primary_file_size_bytes": int(stat.st_size) if stat else "",
                "primary_file_sha256": digest,
                "primary_file_sha256_status": digest_status,
                "allowed_for_v97_semantic_aware": bool(src["allowed_for_v97_semantic_aware"]),
                "dense_tensor_loaded": False,
                "region_proxy_allowed": True,
                "notes": src["notes"],
                "created_at": created_at,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _d4rt_rows(created_at: str) -> list[dict[str, Any]]:
    entries = [
        ("opend4rt_root", D4RT_ROOT, "directory", "D4RT repo root"),
        ("opend4rt_config", D4RT_CONFIG, "file", "D4RT model config"),
        ("opend4rt_checkpoint", D4RT_CKPT, "file", "D4RT frozen checkpoint"),
        ("stream4d_d4rt_adapter", D4RT_ADAPTER, "file", "repo adapter"),
    ]
    rows: list[dict[str, Any]] = []
    for name, path, kind, role in entries:
        digest, digest_status = _sha256(path)
        stat = path.stat() if path.exists() and path.is_file() else None
        rows.append(
            {
                "schema_version": "stream4d_v97_phase0_d4rt_contract_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "contract_name": name,
                "artifact_path": _rel(path),
                "artifact_kind": kind,
                "artifact_role": role,
                "artifact_exists": path.exists(),
                "artifact_size_bytes": int(stat.st_size) if stat else "",
                "artifact_sha256": digest,
                "artifact_sha256_status": digest_status,
                "allowed_as_micro_track_source": name in {"opend4rt_config", "opend4rt_checkpoint", "stream4d_d4rt_adapter"},
                "old_carrier_npz_allowed_as_method": False,
                "created_at": created_at,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _triton_rows(created_at: str) -> list[dict[str, Any]]:
    torch_spec = importlib.util.find_spec("torch")
    triton_spec = importlib.util.find_spec("triton")
    code = (
        "import torch, triton; "
        "print(torch.__version__); "
        "print(torch.cuda.is_available()); "
        "print(torch.cuda.device_count()); "
        "print(triton.__version__)"
    )
    rc, out = _run_text([sys.executable, "-c", code])
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    gpu_query_rc, gpu_query = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    return [
        {
            "schema_version": "stream4d_v97_phase0_triton_contract_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "contract_name": "python_import_contract",
            "torch_spec_found": torch_spec is not None,
            "triton_spec_found": triton_spec is not None,
            "python_check_returncode": rc,
            "torch_version": lines[0] if len(lines) > 0 else "",
            "torch_cuda_is_available": lines[1] if len(lines) > 1 else "",
            "torch_cuda_device_count": lines[2] if len(lines) > 2 else "",
            "triton_version": lines[3] if len(lines) > 3 else "",
            "triton_available": rc == 0 and triton_spec is not None,
            "gpu_query_returncode": gpu_query_rc,
            "gpu_query_output": gpu_query.strip(),
            "preferred_gpu_ids": "6,7",
            "created_at": created_at,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]


def _forbidden_rows(created_at: str) -> list[dict[str, Any]]:
    rows = [
        (
            "gt_local_window_projection",
            ROOT / "data/scannet/processed",
            "GT projection is evaluator support only; forbidden for v97 query, feature, clustering, render, readout, ranking, or threshold selection.",
        ),
        (
            "old_carrier_npz_as_method_track",
            ROOT / "outputs/audit",
            "Old carrier/NPZ rows are diagnostic controls only; v97 micro-track method must use D4RTAdapter decode path.",
        ),
        (
            "segment_diagnostic_as_full_dev_success",
            ROOT / "outputs/audit",
            "Segment diagnostic cannot be compared to full-dev method gate.",
        ),
        (
            "holdout_or_local2history_before_phase8_pass",
            ROOT / "outputs/audit",
            "Holdout/local2history are forbidden unless Phase8 full-dev gate passes and Phase10 decision is GO_FREEZE_HOLDOUT.",
        ),
    ]
    out: list[dict[str, Any]] = []
    for name, path, reason in rows:
        out.append(
            {
                "schema_version": "stream4d_v97_phase0_forbidden_artifact_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "artifact_name": name,
                "artifact_path": _rel(path),
                "artifact_exists": path.exists(),
                "may_use_for_prediction": False,
                "may_use_for_query_selection": False,
                "may_use_for_variant_selection": False,
                "may_use_for_diagnostic": True,
                "reason": reason,
                "created_at": created_at,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


def _gate(name: str, observed: Any, required: Any, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v97_phase0_gate_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "gate": name,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()

    v91_contract = _read_json(V91_CONTRACT)
    v93_phase0 = _read_json(V93_PHASE0)
    v94_controls = _read_json(V94_CONTROLS)
    v95_phase0 = _read_json(V95_PHASE0)
    v96_phase0 = _read_json(V96_PHASE0)
    v96_final = _read_json(V96_FINAL)
    v96_best = v96_final.get("phase6_c_best_variant", {}) if isinstance(v96_final.get("phase6_c_best_variant"), dict) else {}
    evaluator_text = V65_EVALUATOR.read_text(encoding="utf-8") if V65_EVALUATOR.exists() else ""

    ap_thresholds = v96_phase0.get("AP_thresholds_actual", v95_phase0.get("AP_thresholds_actual", v93_phase0.get("AP_thresholds_actual", [])))
    local_support_policy = str(v96_phase0.get("local_support_policy", v95_phase0.get("local_support_policy", v93_phase0.get("local_support_policy", ""))))
    formal_metric_source_eq_v65 = bool(
        V65_EVALUATOR.exists()
        and "SparseSceneIoU" in evaluator_text
        and "_summarize_iou" in evaluator_text
        and "AP_THRESHOLDS" in evaluator_text
        and _bool(v96_phase0.get("formal_metric_source_eq_v65", v93_phase0.get("formal_metric_source_eq_v65")))
    )
    b0_ap = _num(v96_phase0.get("B0_MV_AP_window"), _num(v93_phase0.get("B0_MV_AP_window")))
    b0_ap50 = _num(v96_phase0.get("B0_MV_AP50_window"), _num(v93_phase0.get("B0_MV_AP50_window")))
    best_control_ap = _num(v96_phase0.get("best_control_MV_AP_window"), _num(v94_controls.get("best_control_MV_AP_window")))
    best_control_ap50 = _num(v96_phase0.get("best_control_MV_AP50_window"), _num(v94_controls.get("best_control_MV_AP50_window")))
    v91_best_ap = _num(v96_phase0.get("v91_best_MV_AP_window"), _num(v93_phase0.get("v91_best_MV_AP_window")))
    v91_best_ap50 = _num(v96_phase0.get("v91_best_MV_AP50_window"), _num(v93_phase0.get("v91_best_MV_AP50_window")))
    required_ap = max(v91_best_ap + 0.006, best_control_ap + 0.010)
    required_ap50 = max(v91_best_ap50 + 0.012, best_control_ap50 + 0.015)

    baseline_rows = _copy_baseline_rows(created_at)
    semantic_rows = _semantic_rows(created_at)
    d4rt_rows = _d4rt_rows(created_at)
    triton_rows = _triton_rows(created_at)
    forbidden_rows = _forbidden_rows(created_at)

    uses_gt_for_prediction_count = 0
    uses_future_count = 0
    semantic_root_exists = any(_bool(row.get("root_exists")) and _bool(row.get("primary_file_exists")) for row in semantic_rows)
    radio_backend_available = any(row.get("feature_family") == "radio" and _bool(row.get("primary_file_exists")) for row in semantic_rows)
    dino_backend_available = any(row.get("feature_family") == "dino" and _bool(row.get("primary_file_exists")) for row in semantic_rows)
    triton_available = any(_bool(row.get("triton_available")) for row in triton_rows)

    gate_rows = [
        _gate("formal_metric_source_eq_v65", formal_metric_source_eq_v65, True, formal_metric_source_eq_v65),
        _gate("local_support_policy", local_support_policy, "local_window_gt_projection", local_support_policy == "local_window_gt_projection"),
        _gate("uses_gt_for_prediction_count", uses_gt_for_prediction_count, 0, uses_gt_for_prediction_count == 0),
        _gate("uses_future_count", uses_future_count, 0, uses_future_count == 0),
        _gate("d4rt_root_exists", D4RT_ROOT.exists(), True, D4RT_ROOT.exists()),
        _gate("d4rt_config_exists", D4RT_CONFIG.exists(), True, D4RT_CONFIG.exists()),
        _gate("d4rt_ckpt_exists", D4RT_CKPT.exists(), True, D4RT_CKPT.exists()),
        _gate("required_MV_AP_window_available", required_ap, ">0", required_ap > 0),
        _gate("required_MV_AP50_window_available", required_ap50, ">0", required_ap50 > 0),
        _gate("triton_available", triton_available, True, triton_available),
        _gate("semantic_feature_root_exists", semantic_root_exists, True, semantic_root_exists),
    ]
    phase0_pass = all(bool(row.get("pass")) for row in gate_rows[:9])

    evaluator_contract = {
        "schema": "stream4d_v97_evaluator_contract_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "formal_metric_source": FORMAL_METRIC_SOURCE,
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "AP_thresholds_actual": ap_thresholds,
        "AP_thresholds_expected": EXPECTED_AP_THRESHOLDS,
        "AP_thresholds_match_expected": [float(x) for x in ap_thresholds] == EXPECTED_AP_THRESHOLDS if ap_thresholds else False,
        "local_support_policy": local_support_policy,
        "metric_scope_rule": "Only full_dev MV_AP_window/MV_AP50_window can support method success; segment/native metrics are diagnostic only.",
        "support_policy_note": "local-window GT projection is evaluator support only and forbidden in prediction path.",
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }

    stream3d_rows = [row for row in baseline_rows if row.get("variant_family") == "stream3d_local_diagnostic"]
    stream3d_row = stream3d_rows[0] if stream3d_rows else {}
    summary = {
        "schema": "stream4d_v97_phase0_fact_lock_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "PASS_V97_PHASE0_FACT_LOCK" if phase0_pass else "NO_GO_V97_PHASE0_FACT_LOCK",
        "formal_metric_source": FORMAL_METRIC_SOURCE,
        "formal_metric_source_eq_v65": formal_metric_source_eq_v65,
        "AP_thresholds_actual": ap_thresholds,
        "local_support_policy": local_support_policy,
        "B0_MV_AP_window": b0_ap,
        "B0_MV_AP50_window": b0_ap50,
        "best_locked_control": v94_controls.get("best_control_variant_id", v96_phase0.get("best_locked_control", "P3_C0_area_semantic_hybrid_score")),
        "best_control_MV_AP_window": best_control_ap,
        "best_control_MV_AP50_window": best_control_ap50,
        "v91_best": "V91_AD4_sr2_adapt_sig8_b05_j075_r12",
        "v91_best_MV_AP_window": v91_best_ap,
        "v91_best_MV_AP50_window": v91_best_ap50,
        "required_MV_AP_window": required_ap,
        "required_MV_AP50_window": required_ap50,
        "stream3d_corrected_local_window_MV_AP_window": _num(v96_phase0.get("stream3d_corrected_local_window_MV_AP_window"), _num(stream3d_row.get("MV_AP_window"))),
        "stream3d_corrected_local_window_MV_AP50_window": _num(v96_phase0.get("stream3d_corrected_local_window_MV_AP50_window"), _num(stream3d_row.get("MV_AP50_window"))),
        "v91_contract_source": _rel(V91_CONTRACT),
        "v91_contract_decision": v91_contract.get("decision", ""),
        "v96_final_decision": v96_final.get("decision", ""),
        "v96_latest_best_MV_AP_window": _num(v96_best.get("MV_AP_window")),
        "v96_latest_best_MV_AP50_window": _num(v96_best.get("MV_AP50_window")),
        "v96_latest_best_variant": v96_best.get("readout_variant", ""),
        "d4rt_root_exists": D4RT_ROOT.exists(),
        "d4rt_config_exists": D4RT_CONFIG.exists(),
        "d4rt_ckpt_exists": D4RT_CKPT.exists(),
        "d4rt_ckpt_size_bytes": D4RT_CKPT.stat().st_size if D4RT_CKPT.exists() else 0,
        "semantic_dense_feature_root_exists": False,
        "semantic_region_or_mask_proxy_root_exists": semantic_root_exists,
        "radio_backend_available": radio_backend_available,
        "dino_backend_available": dino_backend_available,
        "triton_available": triton_available,
        "uses_gt_for_prediction_count": uses_gt_for_prediction_count,
        "uses_future_count": uses_future_count,
        "can_enter_phase1": phase0_pass,
        "can_claim_semantic_aware_dense": False,
        "semantic_feature_note": "Available semantic sources are currently region/mask proxies unless later phases load dense RADIO/DINO tensors.",
        "baseline_metric_rows": _rel(OUT / "baseline_metric_rows.csv"),
        "evaluator_contract": _rel(OUT / "evaluator_contract.json"),
        "forbidden_artifact_rows": _rel(OUT / "forbidden_artifact_rows.csv"),
        "semantic_feature_contract_rows": _rel(OUT / "semantic_feature_contract_rows.csv"),
        "d4rt_contract_rows": _rel(OUT / "d4rt_contract_rows.csv"),
        "triton_contract_rows": _rel(OUT / "triton_contract_rows.csv"),
        "phase0_gate_rows": _rel(OUT / "phase0_gate_rows.csv"),
        "runtime_sec": float(time.time() - started),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }

    _write_csv(OUT / "baseline_metric_rows.csv", baseline_rows)
    _write_json(OUT / "evaluator_contract.json", evaluator_contract)
    _write_csv(OUT / "forbidden_artifact_rows.csv", forbidden_rows)
    _write_csv(OUT / "semantic_feature_contract_rows.csv", semantic_rows)
    _write_csv(OUT / "d4rt_contract_rows.csv", d4rt_rows)
    _write_csv(OUT / "triton_contract_rows.csv", triton_rows)
    _write_csv(OUT / "phase0_gate_rows.csv", gate_rows)
    _write_json(OUT / "summary.json", summary)
    return summary


def main() -> None:
    summary = run()
    print(json.dumps({"decision": summary["decision"], "can_enter_phase1": summary["can_enter_phase1"], "output_root": _rel(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
