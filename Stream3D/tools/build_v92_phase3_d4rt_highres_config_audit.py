from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE_ID = "v92_phase3_d4rt_highres"
RUN_ID = "v92_phase3a_config_knob_audit"
OUT = ROOT / "outputs/audit/v92_phase3_d4rt_highres"
PHASE0_SUMMARY = ROOT / "outputs/audit/v92_phase0_mv_ap_contract/summary.json"
PHASE2_SUMMARY = ROOT / "outputs/audit/v92_phase2_d4rt_sufficiency/summary.json"
D4RT_RECOMPUTE_SCRIPT = ROOT / "tools/run_v65_d4rt_stride_overlap_geometry.py"
D4RT_PROVIDER = ROOT / "geometry_provider/d4rt_carrier_provider.py"
D4RT_MATERIALIZER = ROOT / "tools/materialize_d4rt_aligned_geometry_for_stream3d.py"
D4RT_CONFIG = REPO_ROOT / "Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml"
D4RT_CKPT = REPO_ROOT / "Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"

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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for key in COMMON_FIELDS:
            if any(key in row for row in rows):
                fieldnames.append(key)
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _common(schema_version: str, variant_id: str, source: Path, created_at: str) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "variant_id": variant_id,
        "scene_id": "ALL_DEV",
        "split": "dev",
        "window_id": "ALL_WINDOWS",
        "chunk_id": "",
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "uses_rgbd_pose_mesh": False,
        "source_artifact": _rel(source),
        "source_artifact_sha256": _sha256(source) if source.exists() else "",
        "created_at": created_at,
    }


def _line_for(path: Path, pattern: str) -> int | str:
    if not path.exists():
        return ""
    regex = re.compile(pattern)
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if regex.search(line):
            return idx
    return ""


def _knob_row(
    *,
    path: Path,
    knob_name: str,
    knob_type: str,
    current_value: Any,
    candidate_values: str,
    controls_sampling_resolution: bool,
    controls_carrier_count: bool,
    requires_training: bool,
    uses_future: bool,
    notes: str,
    created_at: str,
) -> dict[str, Any]:
    row = {
        **_common("stream4d_v92_phase3_config_knob_audit_v1", "PHASE3A_KNOB_AUDIT", path, created_at),
        "script_or_config_path": _rel(path),
        "line_hint": _line_for(path, re.escape(str(knob_name))),
        "knob_name": knob_name,
        "knob_type": knob_type,
        "current_value": current_value,
        "candidate_values": candidate_values,
        "controls_sampling_resolution": controls_sampling_resolution,
        "controls_carrier_count": controls_carrier_count,
        "requires_training": requires_training,
        "uses_future": uses_future,
        "notes": notes,
    }
    return row


def run() -> dict[str, Any]:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()
    phase0 = _read_json(PHASE0_SUMMARY)
    phase2 = _read_json(PHASE2_SUMMARY)
    knob_rows = [
        _knob_row(
            path=D4RT_RECOMPUTE_SCRIPT,
            knob_name="--grid-size",
            knob_type="int",
            current_value=8,
            candidate_values="12,16",
            controls_sampling_resolution=True,
            controls_carrier_count=True,
            requires_training=False,
            uses_future=False,
            notes="True D4RT source grid density knob in fresh recompute; grid16 is approximately 4x grid8 source samples per frame.",
            created_at=created_at,
        ),
        _knob_row(
            path=D4RT_RECOMPUTE_SCRIPT,
            knob_name="--strides",
            knob_type="list[int]",
            current_value="1,2,5,10 default; v92 local support uses stride5 artifacts",
            candidate_values="5 for protocol match; 2/1 only diagnostic if runtime allows",
            controls_sampling_resolution=True,
            controls_carrier_count=True,
            requires_training=False,
            uses_future=False,
            notes="Temporal frame sampling knob. Changing stride changes observation cadence; same-readout MV_AP comparison should keep local support protocol explicit.",
            created_at=created_at,
        ),
        _knob_row(
            path=D4RT_RECOMPUTE_SCRIPT,
            knob_name="--query-chunk-size",
            knob_type="int",
            current_value=4096,
            candidate_values="4096,8192",
            controls_sampling_resolution=False,
            controls_carrier_count=False,
            requires_training=False,
            uses_future=False,
            notes="Runtime batching only; does not increase carrier count by itself.",
            created_at=created_at,
        ),
        _knob_row(
            path=D4RT_RECOMPUTE_SCRIPT,
            knob_name="--min-confidence",
            knob_type="float",
            current_value=0.5,
            candidate_values="0.2,0.5",
            controls_sampling_resolution=False,
            controls_carrier_count=True,
            requires_training=False,
            uses_future=False,
            notes="Post-filter threshold, not a high-res sampling knob. Lowering can increase retained carriers but may reduce quality.",
            created_at=created_at,
        ),
        _knob_row(
            path=D4RT_RECOMPUTE_SCRIPT,
            knob_name="--min-visibility",
            knob_type="float",
            current_value=0.5,
            candidate_values="0.0,0.5",
            controls_sampling_resolution=False,
            controls_carrier_count=True,
            requires_training=False,
            uses_future=False,
            notes="Post-filter threshold, not a high-res sampling knob.",
            created_at=created_at,
        ),
        _knob_row(
            path=D4RT_RECOMPUTE_SCRIPT,
            knob_name="--uv-radius",
            knob_type="float",
            current_value=0.002,
            candidate_values="0.002,0.004",
            controls_sampling_resolution=False,
            controls_carrier_count=False,
            requires_training=False,
            uses_future=False,
            notes="Overlap stitching match radius; affects alignment quality, not source sampling density directly.",
            created_at=created_at,
        ),
        _knob_row(
            path=D4RT_PROVIDER,
            knob_name="nn_radius",
            knob_type="float",
            current_value=0.05,
            candidate_values="0.05,0.08,0.10",
            controls_sampling_resolution=False,
            controls_carrier_count=False,
            requires_training=False,
            uses_future=False,
            notes="Projection/materialization radius after carriers exist; can change support footprint but is not fresh D4RT high-res.",
            created_at=created_at,
        ),
        _knob_row(
            path=D4RT_PROVIDER,
            knob_name="density_alpha",
            knob_type="float",
            current_value=2.0,
            candidate_values="2.0,3.0",
            controls_sampling_resolution=False,
            controls_carrier_count=False,
            requires_training=False,
            uses_future=False,
            notes="Density-mode projection radius scaling; materializer knob, not D4RT recompute sampling.",
            created_at=created_at,
        ),
        _knob_row(
            path=D4RT_MATERIALIZER,
            knob_name="--nn-radius",
            knob_type="float",
            current_value=0.05,
            candidate_values="0.05,0.08,0.10",
            controls_sampling_resolution=False,
            controls_carrier_count=False,
            requires_training=False,
            uses_future=False,
            notes="ScanNet projection/materialization radius; useful control, not high-res D4RT carrier generation.",
            created_at=created_at,
        ),
    ]
    real_sampling_knobs = [row for row in knob_rows if row["controls_sampling_resolution"] and row["controls_carrier_count"]]
    prereq = {
        "d4rt_recompute_script_exists": D4RT_RECOMPUTE_SCRIPT.exists(),
        "d4rt_config_exists": D4RT_CONFIG.exists(),
        "d4rt_ckpt_exists": D4RT_CKPT.exists(),
        "phase2_routes_resolution": phase2.get("routing_label") == "D4RT_RESOLUTION_LIKELY_BLOCKER",
        "real_sampling_knob_count": len(real_sampling_knobs),
    }
    ready = all(
        [
            prereq["d4rt_recompute_script_exists"],
            prereq["d4rt_config_exists"],
            prereq["d4rt_ckpt_exists"],
            prereq["phase2_routes_resolution"],
            prereq["real_sampling_knob_count"] > 0,
        ]
    )
    decision = "PASS_V92_PHASE3A_HIGHRES_KNOBS_FOUND" if ready else "NO_GO_D4RT_HIGHRES_KNOB_MISSING"

    planned = [
        ("LOWRES_AD4_baseline", 1.0, "existing v91 best / Phase2 low-res diagnostic"),
        ("HR1_grid12_same_readout", 2.25, "--grid-size 12, stride 5, same readout protocol"),
        ("HR2_grid16_same_readout", 4.0, "--grid-size 16, stride 5, same readout protocol"),
        ("HR3_grid12_boundary_adaptive_proxy", 2.25, "grid12 plus boundary/source-risk prioritization if downstream support builder allows it"),
        ("HR4_grid16_lowconf_control", 4.0, "grid16 plus min-confidence 0.2 diagnostic control; not pure high-res"),
    ]
    highres_density_rows = []
    highres_mv_metric_rows = []
    for variant, mult, notes in planned:
        common = _common("stream4d_v92_phase3_highres_density_plan_v1", variant, PHASE2_SUMMARY, created_at)
        highres_density_rows.append(
            {
                **common,
                "carrier_count_multiplier": mult,
                "median_carrier_count_inside_source": phase2.get("median_carrier_count_inside_source_unique_key", "") if variant.startswith("LOWRES") else "",
                "median_carrier_support_area_ratio": phase2.get("median_carrier_support_area_ratio_unique_key", "") if variant.startswith("LOWRES") else "",
                "projection_jitter_p90_px": phase2.get("projection_jitter_p90_global", "") if variant.startswith("LOWRES") else "",
                "run_status": "completed_existing_lowres" if variant.startswith("LOWRES") else "not_run_yet",
                "notes": notes,
            }
        )
        metric_common = _common("stream4d_v92_phase3_highres_mv_metric_plan_v1", variant, PHASE0_SUMMARY, created_at)
        highres_mv_metric_rows.append(
            {
                **metric_common,
                "carrier_count_multiplier": mult,
                "MV_AP_window": phase0.get("v91_best_MV_AP_window", "") if variant.startswith("LOWRES") else "",
                "MV_AP50_window": phase0.get("v91_best_MV_AP50_window", "") if variant.startswith("LOWRES") else "",
                "MV_AP25_window": phase0.get("v91_best_MV_AP25_window", "") if variant.startswith("LOWRES") else "",
                "ScoreFreeMatch50_window": "",
                "best_control_MV_AP_window": phase0.get("best_control_MV_AP_window", ""),
                "best_control_MV_AP50_window": phase0.get("best_control_MV_AP50_window", ""),
                "real_minus_best_control_MV_AP_window": "",
                "real_minus_best_control_MV_AP50_window": "",
                "same_frame_collision_count": 0 if variant.startswith("LOWRES") else "",
                "missing_mask_raster_count": 0 if variant.startswith("LOWRES") else "",
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "runtime_gpu_hours": "",
                "run_status": "completed_existing_lowres" if variant.startswith("LOWRES") else "not_run_yet",
            }
        )

    control_rows = [
        {
            **_common("stream4d_v92_phase3_highres_control_v1", "semantic_only_control", PHASE0_SUMMARY, created_at),
            "control_variant": "C0_semantic_only_control",
            "MV_AP_window": phase0.get("C0_MV_AP_window", ""),
            "MV_AP50_window": phase0.get("C0_MV_AP50_window", ""),
            "run_status": "completed_existing_control",
        },
        {
            **_common("stream4d_v92_phase3_highres_control_v1", "area_semantic_control", PHASE0_SUMMARY, created_at),
            "control_variant": phase0.get("best_control_variant", "P3_C0_area_semantic_hybrid_score"),
            "MV_AP_window": phase0.get("best_control_MV_AP_window", ""),
            "MV_AP50_window": phase0.get("best_control_MV_AP50_window", ""),
            "run_status": "completed_existing_control",
        },
        {
            **_common("stream4d_v92_phase3_highres_control_v1", "single_largest_control", PHASE0_SUMMARY, created_at),
            "control_variant": "C4_single_largest_by_scene_control",
            "MV_AP_window": "",
            "MV_AP50_window": "",
            "run_status": "not_relocked_in_phase0",
        },
        {
            **_common("stream4d_v92_phase3_highres_control_v1", "shuffled_carrier_control", PHASE0_SUMMARY, created_at),
            "control_variant": "shuffled_carrier_control",
            "MV_AP_window": "",
            "MV_AP50_window": "",
            "run_status": "not_run_yet",
        },
        {
            **_common("stream4d_v92_phase3_highres_control_v1", "stale_carrier_control", PHASE0_SUMMARY, created_at),
            "control_variant": "stale_carrier_control",
            "MV_AP_window": "",
            "MV_AP50_window": "",
            "run_status": "not_run_yet",
        },
    ]
    failure_rows = []
    if not ready:
        failure_rows.append(
            {
                **_common("stream4d_v92_phase3_highres_failure_v1", "PHASE3A_KNOB_AUDIT", D4RT_RECOMPUTE_SCRIPT, created_at),
                "failure_type": decision,
                "repair_direction": "locate real D4RT sampling controls or install missing Open-d4rt checkpoint/config before any high-res claim",
                "prerequisites": json.dumps(prereq, sort_keys=True),
            }
        )

    empty_common_fields = COMMON_FIELDS + ["status", "notes"]
    _write_csv(OUT / "config_knob_audit_rows.csv", knob_rows)
    _write_csv(OUT / "highres_carrier_observation_rows.csv", [], empty_common_fields)
    _write_csv(OUT / "highres_incidence_rows.csv", [], empty_common_fields)
    _write_csv(OUT / "highres_density_rows.csv", highres_density_rows)
    _write_csv(OUT / "highres_mv_metric_rows.csv", highres_mv_metric_rows)
    _write_csv(OUT / "highres_control_rows.csv", control_rows)
    _write_csv(OUT / "highres_failure_rows.csv", failure_rows)
    _write_csv(OUT / "variant_config_rows.csv", knob_rows)
    _write_csv(OUT / "variant_metric_rows.csv", highres_mv_metric_rows)
    _write_csv(OUT / "variant_gate_rows.csv", [
        {
            **_common("stream4d_v92_phase3_variant_gate_v1", "PHASE3A_KNOB_AUDIT", D4RT_RECOMPUTE_SCRIPT, created_at),
            "gate_name": name,
            "gate_pass": bool(value) if not isinstance(value, int) else value > 0,
            "gate_value": value,
        }
        for name, value in prereq.items()
    ])
    _write_csv(OUT / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT / "casebook_rows.csv", [
        {
            **_common("stream4d_v92_phase3_casebook_v1", "PHASE3A_KNOB_AUDIT", D4RT_RECOMPUTE_SCRIPT, created_at),
            "case_type": "knob_audit",
            "evidence": "grid-size is a real fresh D4RT source sampling knob; nn_radius/density_alpha are materialization controls, not recompute high-res.",
            "decision": decision,
        }
    ])

    summary = {
        "schema": "stream4d_v92_phase3a_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "decision": decision,
        "phase3a_pass": ready,
        "prerequisites": prereq,
        "real_sampling_knobs": [row["knob_name"] for row in real_sampling_knobs],
        "recommended_first_variants": ["HR1_grid12_same_readout", "HR2_grid16_same_readout"],
        "highres_recompute_not_run_yet": True,
        "d4rt_config": _rel(D4RT_CONFIG),
        "d4rt_ckpt": _rel(D4RT_CKPT),
        "phase2_routing_label": phase2.get("routing_label", ""),
        "duration_sec": time.time() - started,
        "created_at": created_at,
    }
    _write_json(OUT / "summary.json", summary)
    sha_rows = {path.name: _sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    _write_json(OUT / "SHA256SUMS.json", sha_rows)
    return summary


if __name__ == "__main__":
    print(json.dumps(_jsonable(run()), indent=2, sort_keys=True))
