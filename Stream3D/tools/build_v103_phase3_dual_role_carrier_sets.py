#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v103_phase3_fast_carrier_reliability_filter import (  # noqa: E402
    _compute_scene_arrays,
    _ensure_mmap_cache,
    _load_cached,
    _project,
    _support_metrics,
)
from diagnose_v103_phase3_reliable_carrier_gt import (  # noqa: E402
    _retained_phase3_semantics,
    _variant_by_id,
)


PHASE_ID = "v103_phase3_dual_role_carrier_sets"
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase3_dual_role_carrier_sets_r1"
DEFAULT_POSITIVE_PHASE3_ROOT = AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_q5c_objlike16384_false_bridge_repair4"
DEFAULT_VETO_PHASE3_ROOT = AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_q5c_objlike16384_source_balanced_repair3"
DEFAULT_SCENE0011_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_SCENE0050_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_first32"


SCENE_POSITIVE_DEFAULTS = {
    "scene0011_00": "D6_interior_only_broad085_jitter006_semhard_top12_floor60b12",
    "scene0050_00": "D4_broad085_jitter006_sem010_top12_floor60b12",
}


def _rel(path: Path | str) -> str:
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


def _selected_by_scene(root: Path) -> dict[str, str]:
    summary = _read_json(root / "summary.json")
    return {str(k): str(v) for k, v in dict(summary.get("selected_variant_by_scene", {})).items()}


def _role_quality(scene: str, role: str, variant_id: str, retained: np.ndarray, arrays: dict[str, np.ndarray], diag: dict[str, Any]) -> dict[str, Any]:
    retained = np.asarray(retained, dtype=bool)
    count = int(np.count_nonzero(retained))
    broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)
    sem_pair_count = np.asarray(arrays["semantic_pair_count"], dtype=np.float64)
    sem_bad_rate = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)
    jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float64)
    unfiltered_broad = float(np.mean(broad))
    filtered_broad = float(np.mean(broad[retained])) if count else 1.0
    unfiltered_sem = float(diag["unfiltered_semantic_contradiction_rate"])
    filtered_sem = (
        float(np.sum(sem_pair_count[retained] * sem_bad_rate[retained]) / max(np.sum(sem_pair_count[retained]), 1.0))
        if count
        else 1.0
    )
    unfiltered_jitter_p90 = float(np.percentile(jitter, 90))
    filtered_jitter_p90 = float(np.percentile(jitter[retained], 90)) if count else 1.0
    support = _support_metrics(diag, retained)
    total = int(retained.shape[0])
    return {
        "schema_version": "stream4d_v103_phase3_dual_role_metric_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "role": role,
        "variant_id": variant_id,
        "carrier_count": count,
        "total_carrier_count": total,
        "retained_carrier_rate": float(count / max(total, 1)),
        "object_like_mask_support_p10": support["object_like_mask_support_p10"],
        "object_like_mask_support_p50": support["object_like_mask_support_p50"],
        "boundary_band_support_p10": support["boundary_band_support_p10"],
        "boundary_band_support_p50": support["boundary_band_support_p50"],
        "mask_support_coverage_after_filter": support["mask_support_coverage_after_filter"],
        "broad_mask_participation_rate": filtered_broad,
        "unfiltered_broad_mask_participation_rate": unfiltered_broad,
        "broad_relative_reduction": float((unfiltered_broad - filtered_broad) / max(unfiltered_broad, 1e-9)),
        "semantic_contradiction_rate": filtered_sem,
        "unfiltered_semantic_contradiction_rate": unfiltered_sem,
        "semantic_relative_reduction": float((unfiltered_sem - filtered_sem) / max(unfiltered_sem, 1e-9)) if unfiltered_sem > 0 else 0.0,
        "normalized_jitter_p90": filtered_jitter_p90,
        "unfiltered_normalized_jitter_p90": unfiltered_jitter_p90,
        "jitter_relative_reduction": float((unfiltered_jitter_p90 - filtered_jitter_p90) / max(unfiltered_jitter_p90, 1e-9)),
        "uses_gt_for_selection": False,
        "uses_gt_for_prediction": False,
        "diagnostic_repair_scaffold": True,
    }


def _gate_rows_for_metric(row: dict[str, Any]) -> list[dict[str, Any]]:
    role = str(row["role"])
    if role == "positive_core":
        specs = [
            ("positive_retained_rate_between_0p03_0p30", 0.03 <= float(row["retained_carrier_rate"]) <= 0.30, row["retained_carrier_rate"], "0.03..0.30"),
            ("positive_broad_reduction_ge_0p20", float(row["broad_relative_reduction"]) >= 0.20, row["broad_relative_reduction"], 0.20),
            ("positive_semantic_reduction_ge_0p20", float(row["semantic_relative_reduction"]) >= 0.20, row["semantic_relative_reduction"], 0.20),
            ("positive_jitter_reduction_ge_0p20", float(row["jitter_relative_reduction"]) >= 0.20, row["jitter_relative_reduction"], 0.20),
        ]
    else:
        specs = [
            ("veto_retained_rate_between_0p05_0p60", 0.05 <= float(row["retained_carrier_rate"]) <= 0.60, row["retained_carrier_rate"], "0.05..0.60"),
            ("veto_object_like_support_p10_ge_50", float(row["object_like_mask_support_p10"]) >= 50.0, row["object_like_mask_support_p10"], 50.0),
            ("veto_boundary_support_p10_ge_10", float(row["boundary_band_support_p10"]) >= 10.0, row["boundary_band_support_p10"], 10.0),
        ]
    return [
        {
            "schema_version": "stream4d_v103_phase3_dual_role_gate_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": row["scene_id"],
            "role": role,
            "variant_id": row["variant_id"],
            "gate_name": name,
            "pass": bool(ok),
            "observed": observed,
            "required": required,
        }
        for name, ok, observed, required in specs
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v103 Phase3 dual-role D4RT carrier sets: sparse positive core plus coverage/veto support.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--positive-phase3-root", default=str(DEFAULT_POSITIVE_PHASE3_ROOT))
    parser.add_argument("--veto-phase3-root", default=str(DEFAULT_VETO_PHASE3_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--scene0011-positive-variant-id", default=SCENE_POSITIVE_DEFAULTS["scene0011_00"])
    parser.add_argument("--scene0050-positive-variant-id", default=SCENE_POSITIVE_DEFAULTS["scene0050_00"])
    parser.add_argument("--scene0011-veto-variant-id", default="")
    parser.add_argument("--scene0050-veto-variant-id", default="")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    positive_root = _project(args.positive_phase3_root)
    veto_root = _project(args.veto_phase3_root)
    veto_selected = _selected_by_scene(veto_root)
    specs = {
        "scene0011_00": {
            "phase2_root": _project(args.scene0011_phase2_root),
            "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_features.npz",
            "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
            "positive_variant_id": str(args.scene0011_positive_variant_id),
            "veto_variant_id": str(args.scene0011_veto_variant_id or veto_selected["scene0011_00"]),
        },
        "scene0050_00": {
            "phase2_root": _project(args.scene0050_phase2_root),
            "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_features.npz",
            "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
            "positive_variant_id": str(args.scene0050_positive_variant_id),
            "veto_variant_id": str(args.scene0050_veto_variant_id or veto_selected["scene0050_00"]),
        },
    }

    metric_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    selected_roles_by_scene: dict[str, dict[str, str]] = {}
    for scene, spec in specs.items():
        scene_out = out / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        diag, _unused_a, _unused_b, arrays = _compute_scene_arrays(scene, spec, scene_out, int(args.cupy_device_id))
        cache_dir, _manifest = _ensure_mmap_cache(spec["phase2_root"])
        batch = _load_cached(cache_dir)
        carrier_ids = np.asarray(batch["carrier_id"], dtype=np.int64)

        positive_variant_id = str(spec["positive_variant_id"])
        veto_variant_id = str(spec["veto_variant_id"])
        positive_retained, _positive_meta = _retained_phase3_semantics(_variant_by_id(positive_variant_id), arrays, diag)
        veto_retained, _veto_meta = _retained_phase3_semantics(_variant_by_id(veto_variant_id), arrays, diag)
        union_retained = np.asarray(positive_retained, dtype=bool) | np.asarray(veto_retained, dtype=bool)
        overlap_retained = np.asarray(positive_retained, dtype=bool) & np.asarray(veto_retained, dtype=bool)

        npz_path = scene_out / "dual_role_carrier_sets.npz"
        np.savez_compressed(
            npz_path,
            positive_carrier_id=carrier_ids[np.flatnonzero(positive_retained)],
            veto_carrier_id=carrier_ids[np.flatnonzero(veto_retained)],
            union_carrier_id=carrier_ids[np.flatnonzero(union_retained)],
            positive_mask=np.asarray(positive_retained, dtype=np.bool_),
            veto_mask=np.asarray(veto_retained, dtype=np.bool_),
        )
        artifact_rows.append(
            {
                "schema_version": "stream4d_v103_phase3_dual_role_artifact_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "artifact_id": "dual_role_carrier_sets",
                "path": _rel(npz_path),
                "positive_count": int(np.count_nonzero(positive_retained)),
                "veto_count": int(np.count_nonzero(veto_retained)),
                "union_count": int(np.count_nonzero(union_retained)),
                "overlap_count": int(np.count_nonzero(overlap_retained)),
            }
        )
        selected_roles_by_scene[scene] = {"positive_core": positive_variant_id, "veto_support": veto_variant_id}

        for role, variant_id, retained in [
            ("positive_core", positive_variant_id, positive_retained),
            ("veto_support", veto_variant_id, veto_retained),
            ("union_diagnostic", f"{positive_variant_id}+{veto_variant_id}", union_retained),
        ]:
            row = _role_quality(scene, role, variant_id, retained, arrays, diag)
            metric_rows.append(row)
            if role != "union_diagnostic":
                role_gates = _gate_rows_for_metric(row)
                gate_rows.extend(role_gates)
                for gate in role_gates:
                    if not bool(gate["pass"]):
                        failure_rows.append(
                            {
                                "schema_version": "stream4d_v103_phase3_dual_role_failure_row_v1",
                                "phase_id": PHASE_ID,
                                "scene_id": scene,
                                "role": role,
                                "variant_id": variant_id,
                                "failure_id": gate["gate_name"],
                                "severity": "blocking_for_dual_role_repair",
                                "evidence": f"observed={gate['observed']} required={gate['required']}",
                                "repair_direction": "Adjust role split or query/support balance; do not treat this as a single reliable-carrier pass.",
                            }
                        )

    metric_path = out / "dual_role_metric_rows.csv"
    gate_path = out / "dual_role_gate_rows.csv"
    failure_path = out / "failure_rows.csv"
    artifact_path = out / "artifact_rows.csv"
    _write_csv(metric_path, metric_rows)
    _write_csv(gate_path, gate_rows)
    _write_csv(failure_path, failure_rows)
    _write_csv(artifact_path, artifact_rows)
    summary = {
        "schema_version": "stream4d_v103_phase3_dual_role_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_DUAL_ROLE_CARRIER_SET_REPAIR_SCAFFOLD" if not failure_rows else "NO_GO_DUAL_ROLE_CARRIER_SET_REPAIR_SCAFFOLD",
        "failure_count": len(failure_rows),
        "positive_phase3_root": _rel(positive_root),
        "veto_phase3_root": _rel(veto_root),
        "selected_roles_by_scene": selected_roles_by_scene,
        "uses_gt_for_selection": False,
        "uses_gt_for_prediction": False,
        "diagnostic_repair_scaffold": True,
        "truthfulness_note": (
            "This artifact separates D4RT positive-core carriers from coverage/veto carriers after Phase3 diagnostics. "
            "It is a repair scaffold, not proof that a single reliable-carrier set passed the original Phase3 gate."
        ),
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "dual_role_metric_rows": _rel(metric_path),
            "dual_role_gate_rows": _rel(gate_path),
            "failure_rows": _rel(failure_path),
            "artifact_rows": _rel(artifact_path),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
