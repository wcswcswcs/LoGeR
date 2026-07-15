#!/usr/bin/env python3
"""Run a small v119 HorizonStream D-HS liveness and trace smoke matrix.

This runner is intentionally a liveness/provenance gate, not a terminal
sequence-level claim. It verifies that selected HorizonStream D-HS candidate
and matched-control actions execute through the real pipeline and leave
auditable trace/action rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
RUN_ROOT = OUT / "stage4_hs_dhs_liveness_smoke"
V113_ROOT = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"
HS_ROOT = ROOT / "third_party/HorizonStream"
IMAGE_ROOT = V113_ROOT / "kitti_generalizable"
SEMANTIC_ROOT = V113_ROOT / "semantic_projection"
CONFIG = "configs/horizonstream_infer.yaml"
CHECKPOINT = "checkpoints/HorizonStream.pt"

AUDIT_FILES = (
    "hs_la_action_rows.csv",
    "hs_gq_action_gate_rows.csv",
    "hs_gq_state_action_rows.csv",
    "hs_lq_action_gate_rows.csv",
    "hs_chs_lane_action_rows.csv",
)
TRACE_FILES = (
    "hs_gla_state_probe_rows.csv",
    "hs_gla_direct_kda_probe_rows.csv",
    "hs_local_head_semantic_attention_rows.csv",
    "hs_mrt_readout_probe_rows.csv",
)


@dataclass(frozen=True)
class Case:
    case_id: str
    branch: str
    role: str
    action: str
    control: str
    expected_audit_file: str
    layer_filter_kind: str
    selected_layer: str
    description: str


CASES = (
    Case(
        case_id="baseline_no_action",
        branch="HS-KDA",
        role="baseline",
        action="",
        control="",
        expected_audit_file="",
        layer_filter_kind="",
        selected_layer="",
        description="real HorizonStream baseline with trace enabled",
    ),
    Case(
        case_id="chs2l_semantic_internal_selected_layer",
        branch="C-HS-2L",
        role="candidate",
        action="HS_CHS2L_semantic_internal_lane",
        control="",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="selected-layer transient/persistent explicit lane state cache with semantic+internal routing",
    ),
    Case(
        case_id="chs2l_internal_only_selected_layer",
        branch="C-HS-2L",
        role="matched_control",
        action="HS_CHS2L_internal_only_lane",
        control="",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="internal-only lane assignment control for C-HS-2L",
    ),
    Case(
        case_id="chs2l_semantic_only_selected_layer",
        branch="C-HS-2L",
        role="matched_control",
        action="HS_CHS2L_semantic_only_lane",
        control="",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="semantic-only lane assignment control for C-HS-2L",
    ),
    Case(
        case_id="chs2l_role_reverse_selected_layer",
        branch="C-HS-2L",
        role="matched_control",
        action="HS_CHS2L_semantic_internal_lane",
        control="role_rotation_dynamic_stable",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="dynamic/stable role-reversal control for C-HS-2L",
    ),
    Case(
        case_id="chs2l_random_routing_selected_layer",
        branch="C-HS-2L",
        role="matched_control",
        action="HS_CHS2L_semantic_internal_lane",
        control="random_routing",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="same-capacity random routing control for C-HS-2L",
    ),
    Case(
        case_id="chs2l_equal_split_selected_layer",
        branch="C-HS-2L",
        role="matched_control",
        action="HS_CHS2L_semantic_internal_lane",
        control="equal_split_generic",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="equal-energy generic split control for C-HS-2L",
    ),
    Case(
        case_id="chs2l_duplicated_identical_selected_layer",
        branch="C-HS-2L",
        role="matched_control",
        action="HS_CHS2L_semantic_internal_lane",
        control="duplicated_identical_readout",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="duplicated lane identical-readout capacity control for C-HS-2L",
    ),
    Case(
        case_id="chs2l_metadata_only_selected_layer",
        branch="C-HS-2L",
        role="matched_control",
        action="HS_CHS2L_semantic_internal_lane",
        control="metadata_only_no_routing",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="metadata-only no-routing representation control for C-HS-2L",
    ),
    Case(
        case_id="chs3l_semantic_internal_metric_selected_layer",
        branch="C-HS-3L",
        role="candidate",
        action="HS_CHS3L_semantic_internal_metric_lane",
        control="",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="selected-layer transient/persistent/metric explicit lane state cache",
    ),
    Case(
        case_id="chs3l_internal_only_selected_layer",
        branch="C-HS-3L",
        role="matched_control",
        action="HS_CHS3L_internal_only_metric_lane",
        control="",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="internal-only three-lane assignment control for C-HS-3L",
    ),
    Case(
        case_id="chs3l_semantic_only_selected_layer",
        branch="C-HS-3L",
        role="matched_control",
        action="HS_CHS3L_semantic_only_metric_lane",
        control="",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="semantic-only three-lane assignment control for C-HS-3L",
    ),
    Case(
        case_id="chs3l_role_reverse_selected_layer",
        branch="C-HS-3L",
        role="matched_control",
        action="HS_CHS3L_semantic_internal_metric_lane",
        control="role_rotation_dynamic_stable",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="dynamic/stable role-reversal control for C-HS-3L",
    ),
    Case(
        case_id="chs3l_random_routing_selected_layer",
        branch="C-HS-3L",
        role="matched_control",
        action="HS_CHS3L_semantic_internal_metric_lane",
        control="random_routing",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="same-capacity random routing control for C-HS-3L",
    ),
    Case(
        case_id="chs3l_equal_split_selected_layer",
        branch="C-HS-3L",
        role="matched_control",
        action="HS_CHS3L_semantic_internal_metric_lane",
        control="equal_split_generic",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="equal split representation control for C-HS-3L",
    ),
    Case(
        case_id="chs3l_metadata_only_selected_layer",
        branch="C-HS-3L",
        role="matched_control",
        action="HS_CHS3L_semantic_internal_metric_lane",
        control="metadata_only_no_routing",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="metadata-only no-routing representation control for C-HS-3L",
    ),
    Case(
        case_id="chss_r8_semantic_internal_selected_layer",
        branch="C-HS-S",
        role="candidate",
        action="HS_CHSS_R8_semantic_internal_shadow_lane",
        control="",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="rank-8 selected-channel shadow lane approximation for C-HS-S",
    ),
    Case(
        case_id="chss_r16_semantic_internal_selected_layer",
        branch="C-HS-S",
        role="candidate",
        action="HS_CHSS_R16_semantic_internal_shadow_lane",
        control="",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="rank-16 selected-channel shadow lane approximation for C-HS-S",
    ),
    Case(
        case_id="chss_r32_semantic_internal_selected_layer",
        branch="C-HS-S",
        role="candidate",
        action="HS_CHSS_R32_semantic_internal_shadow_lane",
        control="",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="rank-32 selected-channel shadow lane approximation for C-HS-S",
    ),
    Case(
        case_id="chss_r8_random_routing_selected_layer",
        branch="C-HS-S",
        role="matched_control",
        action="HS_CHSS_R8_semantic_internal_shadow_lane",
        control="random_routing",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="rank-8 shadow lane random-routing control for C-HS-S",
    ),
    Case(
        case_id="chss_r8_role_reverse_selected_layer",
        branch="C-HS-S",
        role="matched_control",
        action="HS_CHSS_R8_semantic_internal_shadow_lane",
        control="role_rotation_dynamic_stable",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="rank-8 shadow lane role-reversal control for C-HS-S",
    ),
    Case(
        case_id="chss_r8_equal_split_selected_layer",
        branch="C-HS-S",
        role="matched_control",
        action="HS_CHSS_R8_semantic_internal_shadow_lane",
        control="equal_split_generic",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="rank-8 shadow lane equal-split control for C-HS-S",
    ),
    Case(
        case_id="chss_r8_metadata_only_selected_layer",
        branch="C-HS-S",
        role="matched_control",
        action="HS_CHSS_R8_semantic_internal_shadow_lane",
        control="metadata_only_no_routing",
        expected_audit_file="hs_chs_lane_action_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="rank-8 shadow lane metadata-only no-routing control for C-HS-S",
    ),
    Case(
        case_id="dhsl_la4_persistent_aligned_tiny",
        branch="D-HS-L",
        role="candidate",
        action="HS_LA4_persistent_aligned_boost_tiny",
        control="",
        expected_audit_file="hs_la_action_rows.csv",
        layer_filter_kind="la",
        selected_layer="3",
        description="local head persistent-aligned semantic/internal attention action",
    ),
    Case(
        case_id="dhsl_la4_same_magnitude_random_logit",
        branch="D-HS-L",
        role="matched_control",
        action="HS_LA4_persistent_aligned_boost_tiny",
        control="same_magnitude_random_logit",
        expected_audit_file="hs_la_action_rows.csv",
        layer_filter_kind="la",
        selected_layer="3",
        description="same-magnitude random-logit control for D-HS-L",
    ),
    Case(
        case_id="dhsl_la4_semantic_shuffle",
        branch="D-HS-L",
        role="matched_control",
        action="HS_LA4_persistent_aligned_boost_tiny",
        control="semantic_shuffle_by_frame",
        expected_audit_file="hs_la_action_rows.csv",
        layer_filter_kind="la",
        selected_layer="3",
        description="semantic-shuffle control for D-HS-L",
    ),
    Case(
        case_id="dhsl_la4_role_rotation_dynamic_stable",
        branch="D-HS-L",
        role="matched_control",
        action="HS_LA4_persistent_aligned_boost_tiny",
        control="role_rotation_dynamic_stable",
        expected_audit_file="hs_la_action_rows.csv",
        layer_filter_kind="la",
        selected_layer="3",
        description="dynamic/stable role-rotation control for D-HS-L",
    ),
    Case(
        case_id="dhsl_la4_low_risk_reverse",
        branch="D-HS-L",
        role="matched_control",
        action="HS_LA4_persistent_aligned_boost_tiny",
        control="low_risk_reverse",
        expected_audit_file="hs_la_action_rows.csv",
        layer_filter_kind="la",
        selected_layer="3",
        description="low-risk reverse control for D-HS-L",
    ),
    Case(
        case_id="dhsl_la4_internal_qk_reverse",
        branch="D-HS-L",
        role="matched_control",
        action="HS_LA4_persistent_aligned_boost_tiny",
        control="internal_qk_reverse",
        expected_audit_file="hs_la_action_rows.csv",
        layer_filter_kind="la",
        selected_layer="3",
        description="internal-QK reverse control for D-HS-L",
    ),
    Case(
        case_id="dhsl_la4_generic_rowmean_logit_shift",
        branch="D-HS-L",
        role="matched_control",
        action="HS_LA4_persistent_aligned_boost_tiny",
        control="generic_rowmean_logit_shift",
        expected_audit_file="hs_la_action_rows.csv",
        layer_filter_kind="la",
        selected_layer="3",
        description="generic row-mean logit shift control for D-HS-L",
    ),
    Case(
        case_id="dhsg_gw3_internal_semantic_lowgain",
        branch="D-HS-G",
        role="candidate",
        action="HS_GW3_internal_plus_semantic_lowgain",
        control="",
        expected_audit_file="hs_gq_action_gate_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="GLA lifetime/value write candidate using internal plus semantic gate",
    ),
    Case(
        case_id="dhsg_gw3_semantic_shuffle",
        branch="D-HS-G",
        role="matched_control",
        action="HS_GW3_internal_plus_semantic_lowgain",
        control="semantic_shuffle_by_frame",
        expected_audit_file="hs_gq_action_gate_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="semantic-shuffle matched control for D-HS-G",
    ),
    Case(
        case_id="dhsm_gq4_mrt_token_lowgain",
        branch="D-HS-M",
        role="candidate",
        action="HS_GQ4_mrt_token_lowgain",
        control="",
        expected_audit_file="hs_gq_action_gate_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="MRT token pre-GLA gain candidate",
    ),
    Case(
        case_id="dhsm_gq4_same_magnitude_random_sign",
        branch="D-HS-M",
        role="matched_control",
        action="HS_GQ4_mrt_token_lowgain",
        control="same_magnitude_random_sign",
        expected_audit_file="hs_gq_action_gate_rows.csv",
        layer_filter_kind="gq",
        selected_layer="4",
        description="same-magnitude random-sign control for D-HS-M",
    ),
    Case(
        case_id="dhsm_lq5_headwise_mrt_lowgain",
        branch="D-HS-M",
        role="candidate_secondary",
        action="HS_LQ5_headwise_mrt_lowgain",
        control="",
        expected_audit_file="hs_lq_action_gate_rows.csv",
        layer_filter_kind="",
        selected_layer="",
        description="headwise local value MRT-ish secondary candidate",
    ),
    Case(
        case_id="dhsm_lq5_same_magnitude_value_scale_random",
        branch="D-HS-M",
        role="matched_control_secondary",
        action="HS_LQ5_headwise_mrt_lowgain",
        control="same_magnitude_value_scale_random",
        expected_audit_file="hs_lq_action_gate_rows.csv",
        layer_filter_kind="",
        selected_layer="",
        description="same-magnitude value-scale random control for secondary D-HS-M",
    ),
    Case(
        case_id="dhsm_lq5_low_risk_reverse",
        branch="D-HS-M",
        role="matched_control_secondary",
        action="HS_LQ5_headwise_mrt_lowgain",
        control="low_risk_reverse",
        expected_audit_file="hs_lq_action_gate_rows.csv",
        layer_filter_kind="",
        selected_layer="",
        description="low-risk reverse control for secondary D-HS-M",
    ),
    Case(
        case_id="dhsm_lq5_internal_confident_only",
        branch="D-HS-M",
        role="matched_control_secondary",
        action="HS_LQ5_headwise_mrt_lowgain_CTRL_internal_confident_only",
        control="",
        expected_audit_file="hs_lq_action_gate_rows.csv",
        layer_filter_kind="",
        selected_layer="",
        description="internal-confident-only control for secondary D-HS-M",
    ),
    Case(
        case_id="dhsm_lq5_semantic_only_rowmean_neutral",
        branch="D-HS-M",
        role="matched_control_secondary",
        action="HS_LQ5_headwise_mrt_lowgain_CTRL_semantic_only_rowmean_neutral",
        control="",
        expected_audit_file="hs_lq_action_gate_rows.csv",
        layer_filter_kind="",
        selected_layer="",
        description="semantic-only rowmean-neutral control for secondary D-HS-M",
    ),
    Case(
        case_id="dhsm_lq5_rowmean_only_generic_scale",
        branch="D-HS-M",
        role="matched_control_secondary",
        action="HS_LQ5_headwise_mrt_lowgain_CTRL_rowmean_only_generic_scale",
        control="",
        expected_audit_file="hs_lq_action_gate_rows.csv",
        layer_filter_kind="",
        selected_layer="",
        description="rowmean-only generic scale control for secondary D-HS-M",
    ),
)

RESULT_FIELDS = [
    "schema",
    "job_id",
    "case_id",
    "branch",
    "role",
    "seq",
    "gpu",
    "action",
    "control",
    "trace_profile",
    "selected_layer",
    "output_root",
    "trace_root",
    "action_audit_root",
    "log_path",
    "command",
    "returncode",
    "duration_sec",
    "ran_eval",
    "ate_rmse",
    "num_pose_pairs",
    "sim3_scale",
    "expected_audit_file",
    "expected_audit_rows",
    "all_audit_rows_json",
    "all_trace_rows_json",
    "trace_total_rows",
    "liveness_pass",
    "truthfulness_boundary",
    "stdout_tail",
]


def artifact_paths(
    max_frames: int,
    trace_profile: str = "full",
    seqs: str = "00",
    run_label: str = "",
) -> dict[str, Path]:
    suffix = f"_max{int(max_frames)}" if int(max_frames) > 0 else "_full"
    if trace_profile != "full":
        suffix = f"{suffix}_{trace_profile}"
    seq_suffix = "_".join(parse_csv_list(seqs))
    if seq_suffix and seq_suffix != "00":
        suffix = f"{suffix}_seq{seq_suffix}"
    label = run_label.strip().replace("/", "_").replace(" ", "_")
    if label:
        suffix = f"{suffix}_{label}"
    return {
        "manifest": RUN_ROOT / f"dhs_liveness_smoke{suffix}_manifest.csv",
        "results_csv": RUN_ROOT / f"dhs_liveness_smoke{suffix}_run_results.csv",
        "results_jsonl": RUN_ROOT / f"dhs_liveness_smoke{suffix}_run_results.jsonl",
        "summary": RUN_ROOT / f"dhs_liveness_smoke{suffix}_summary.json",
    }


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        return max(0, sum(1 for _ in reader) - 1)


def row_counts(root: Path, names: tuple[str, ...]) -> dict[str, int]:
    return {name: count_csv_rows(root / name) for name in names}


def pipeline_summary(output_root: Path, seq: str) -> dict[str, Any]:
    path = output_root / "pipeline_summary.json"
    if not path.exists():
        return {"ran_eval": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = (
        payload.get("infer_eval", {})
        .get("sequences", {})
        .get(f"{seq}/02", {})
        .get("metrics", {})
        .get("main", {})
    )
    return {
        "ran_eval": bool(payload.get("ran_eval")),
        "ate_rmse": metrics.get("ate_rmse"),
        "num_pose_pairs": metrics.get("num_pose_pairs"),
        "sim3_scale": metrics.get("sim3_scale"),
    }


def make_command(job: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--config",
        CONFIG,
        "--img-path",
        str(IMAGE_ROOT.resolve()),
        "--seq-list",
        str(job["seq"]),
        "--camera",
        "02",
        "--checkpoint",
        CHECKPOINT,
        "--output-root",
        str(Path(job["output_root"]).resolve()),
        "--no-camera-preprocess",
        "--offload-outputs-to-cpu",
        "--no-save-videos",
        "--no-save-points",
        "--no-save-images",
        "--no-save-depth",
        "--no-save-depth-conf",
        "--no-mask-sky",
        "--no-point-mask-sky",
        "--no-loop",
        "--eval-pose-variants",
        "main",
    ]
    if int(job["max_frames"]) > 0:
        cmd.extend(["--max-frames", str(int(job["max_frames"]))])
    return cmd


def env_for_job(job: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    trace_profile = str(job.get("trace_profile") or "full")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    env["HS_V113_SEMANTIC_ROOT"] = str(SEMANTIC_ROOT.resolve())
    env["HS_V113_TRACE_ENABLE"] = "1"
    env["HS_V113_TRACE_LOCAL_ENABLE"] = "0" if trace_profile == "global_mrt" else "1"
    env["HS_V113_TRACE_GLA_ENABLE"] = "1"
    env["HS_V113_TRACE_MRT_ENABLE"] = "1"
    env["HS_V113_TRACE_LOCAL_SAMPLE_ROWS"] = "0" if trace_profile == "global_mrt" else "64"
    env["HS_V119_KDA_DIRECT_PROBE_MAX_TOKENS"] = "32" if trace_profile == "global_mrt" else "128"
    env["HS_V113_TRACE_ROOT"] = str(Path(job["trace_root"]).resolve())
    env["HS_V114_ACTION_AUDIT_ROOT"] = str(Path(job["action_audit_root"]).resolve())
    env.pop("HS_V113_ACTION", None)
    env.pop("HS_V113_CONTROL", None)
    env.pop("HS_V116_LA_LAYER_FILTER", None)
    env.pop("HS_V115_GQ_LAYER_FILTER", None)
    if job["action"]:
        env["HS_V113_ACTION"] = str(job["action"])
    if job["control"]:
        env["HS_V113_CONTROL"] = str(job["control"])
    if job["layer_filter_kind"] == "la" and job["selected_layer"]:
        env["HS_V116_LA_LAYER_FILTER"] = str(job["selected_layer"])
    if job["layer_filter_kind"] == "gq" and job["selected_layer"]:
        env["HS_V115_GQ_LAYER_FILTER"] = str(job["selected_layer"])
    return env


def command_for_log(job: dict[str, Any], cmd: list[str]) -> str:
    env = env_for_job(job)
    keys = [
        "PYTORCH_CUDA_ALLOC_CONF",
        "CUDA_VISIBLE_DEVICES",
        "HS_V113_SEMANTIC_ROOT",
        "HS_V113_TRACE_ENABLE",
        "HS_V113_TRACE_LOCAL_ENABLE",
        "HS_V113_TRACE_GLA_ENABLE",
        "HS_V113_TRACE_MRT_ENABLE",
        "HS_V113_TRACE_LOCAL_SAMPLE_ROWS",
        "HS_V119_KDA_DIRECT_PROBE_MAX_TOKENS",
        "HS_V113_TRACE_ROOT",
        "HS_V114_ACTION_AUDIT_ROOT",
        "HS_V113_ACTION",
        "HS_V113_CONTROL",
        "HS_V116_LA_LAYER_FILTER",
        "HS_V115_GQ_LAYER_FILTER",
    ]
    env_parts = [f"{key}={env[key]}" for key in keys if key in env]
    return " ".join(env_parts + cmd)


def clear_case_dirs(job: dict[str, Any]) -> None:
    for key in ("output_root", "trace_root", "action_audit_root"):
        path = Path(job[key])
        if path.exists():
            shutil.rmtree(path)


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    output_root = Path(job["output_root"])
    trace_root = Path(job["trace_root"])
    audit_root = Path(job["action_audit_root"])
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if job["fresh_case"]:
        clear_case_dirs(job)
    output_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    cmd = make_command(job)
    env = env_for_job(job)
    command = command_for_log(job, cmd)
    tail: deque[str] = deque(maxlen=120)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# v119 HorizonStream D-HS liveness smoke command\n")
        log.write(f"cwd={HS_ROOT}\n")
        log.write(command + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(HS_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            tail.append(line)
        returncode = proc.wait()

    summary = pipeline_summary(output_root, str(job["seq"]))
    audit_counts = row_counts(audit_root, AUDIT_FILES)
    trace_counts = row_counts(trace_root, TRACE_FILES)
    expected_file = str(job["expected_audit_file"])
    expected_rows = audit_counts.get(expected_file, 0) if expected_file else 0
    trace_total_rows = sum(trace_counts.values())
    action_rows_ok = bool(expected_rows > 0) if expected_file else True
    liveness_pass = bool(int(returncode) == 0 and summary.get("ran_eval") and action_rows_ok and trace_total_rows > 0)
    truthfulness_boundary = (
        "liveness_trace_smoke_only_max_frames_"
        f"{job['max_frames']}; not terminal, not full-sequence, not cross-sequence"
    )
    return {
        "schema": "acl2_v119tf_hs_dhs_liveness_smoke_result_v1",
        "job_id": job["job_id"],
        "case_id": job["case_id"],
        "branch": job["branch"],
        "role": job["role"],
        "seq": job["seq"],
        "gpu": job["gpu"],
        "action": job["action"],
        "control": job["control"],
        "trace_profile": job.get("trace_profile", "full"),
        "selected_layer": job["selected_layer"],
        "output_root": rel(output_root),
        "trace_root": rel(trace_root),
        "action_audit_root": rel(audit_root),
        "log_path": rel(log_path),
        "command": command,
        "returncode": int(returncode),
        "duration_sec": round(time.time() - started, 3),
        "ran_eval": bool(summary.get("ran_eval")),
        "ate_rmse": summary.get("ate_rmse"),
        "num_pose_pairs": summary.get("num_pose_pairs"),
        "sim3_scale": summary.get("sim3_scale"),
        "expected_audit_file": expected_file,
        "expected_audit_rows": int(expected_rows),
        "all_audit_rows_json": json.dumps(audit_counts, sort_keys=True),
        "all_trace_rows_json": json.dumps(trace_counts, sort_keys=True),
        "trace_total_rows": int(trace_total_rows),
        "liveness_pass": bool(liveness_pass),
        "truthfulness_boundary": truthfulness_boundary,
        "stdout_tail": "".join(tail)[-4000:],
    }


def run_gpu_queue(
    gpu: str,
    rows: list[dict[str, Any]],
    results_csv: Path,
    results_jsonl: Path,
    csv_lock: threading.Lock,
    jsonl_lock: threading.Lock,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        result = run_job(row)
        append_csv(results_csv, result, csv_lock)
        append_jsonl(results_jsonl, result, jsonl_lock)
        print(
            "RETURN "
            f"rc={result['returncode']} pass={result['liveness_pass']} gpu={gpu} "
            f"job={result['job_id']} ate={result.get('ate_rmse')} "
            f"expected_rows={result['expected_audit_rows']} trace_rows={result['trace_total_rows']}",
            flush=True,
        )
        if int(result["returncode"]) != 0:
            print(result["stdout_tail"][-2000:], flush=True)
        out.append(result)
    return out


def build_jobs(
    gpus: list[str],
    seqs: list[str],
    max_frames: int,
    fresh_case: bool,
    trace_profile: str,
    case_ids: list[str],
    run_label: str,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    selected_cases = [case for case in CASES if not case_ids or case.case_id in set(case_ids)]
    if case_ids and len(selected_cases) != len(set(case_ids)):
        known = {case.case_id for case in CASES}
        missing = sorted(set(case_ids) - known)
        raise SystemExit(f"Unknown case ids: {missing}")
    idx = 0
    label_suffix = f"_{run_label.strip().replace('/', '_').replace(' ', '_')}" if run_label.strip() else ""
    for seq in seqs:
        for case in selected_cases:
            profile_suffix = f"_{trace_profile}" if trace_profile != "full" else ""
            case_root = RUN_ROOT / f"seq{seq}_{case.case_id}_max{max_frames}{profile_suffix}{label_suffix}"
            job = {
                "schema": "acl2_v119tf_hs_dhs_liveness_smoke_job_v1",
                "job_id": f"{case.case_id}_seq{seq}",
                "case_id": case.case_id,
                "branch": case.branch,
                "role": case.role,
                "seq": seq,
                "gpu": gpus[idx % len(gpus)],
                "action": case.action,
                "control": case.control,
                "trace_profile": trace_profile,
                "expected_audit_file": case.expected_audit_file,
                "layer_filter_kind": case.layer_filter_kind,
                "selected_layer": case.selected_layer,
                "description": case.description,
                "output_root": str(case_root / "output"),
                "trace_root": str(case_root / "trace"),
                "action_audit_root": str(case_root / "action_audit"),
                "log_path": str(
                    RUN_ROOT / "logs" / f"{case.case_id}_seq{seq}_max{max_frames}{profile_suffix}{label_suffix}.log"
                ),
                "max_frames": int(max_frames),
                "fresh_case": bool(fresh_case),
            }
            jobs.append(job)
            idx += 1
    return jobs


def build_summary(
    results: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    branch_stats: dict[str, dict[str, Any]] = {}
    for row in results:
        branch = str(row["branch"])
        stats = branch_stats.setdefault(
            branch,
            {
                "jobs": 0,
                "liveness_pass_jobs": 0,
                "candidate_jobs": 0,
                "control_jobs": 0,
                "candidate_liveness_pass_jobs": 0,
                "control_liveness_pass_jobs": 0,
                "ate_rmse_by_case": {},
                "audit_rows_by_case": {},
                "trace_rows_by_case": {},
            },
        )
        stats["jobs"] += 1
        if row.get("liveness_pass"):
            stats["liveness_pass_jobs"] += 1
        role = str(row.get("role", ""))
        if "control" in role:
            stats["control_jobs"] += 1
            if row.get("liveness_pass"):
                stats["control_liveness_pass_jobs"] += 1
        elif role != "baseline":
            stats["candidate_jobs"] += 1
            if row.get("liveness_pass"):
                stats["candidate_liveness_pass_jobs"] += 1
        stats["ate_rmse_by_case"][row["case_id"]] = row.get("ate_rmse")
        stats["audit_rows_by_case"][row["case_id"]] = row.get("expected_audit_rows")
        stats["trace_rows_by_case"][row["case_id"]] = row.get("trace_total_rows")

    failures = [row for row in results if int(row.get("returncode", 1)) != 0]
    liveness_failures = [row for row in results if not row.get("liveness_pass")]
    summary = {
        "schema": "acl2_v119tf_hs_dhs_liveness_smoke_summary_v1",
        "generated_at_utc": generated_at,
        "result_root": rel(RUN_ROOT),
        "seqs": args.seqs,
        "max_frames": int(args.max_frames),
        "trace_profile": args.trace_profile,
        "run_label": args.run_label,
        "case_filter": args.case_ids,
        "gpus": args.gpus,
        "max_workers": int(args.max_workers),
        "job_count": len(jobs),
        "result_count": len(results),
        "returncode_failure_count": len(failures),
        "liveness_fail_count": len(liveness_failures),
        "all_jobs_returncode_zero": len(failures) == 0 and len(results) == len(jobs),
        "all_jobs_liveness_pass": len(liveness_failures) == 0 and len(results) == len(jobs),
        "terminal_pass": False,
        "completion_claim": "not_complete_liveness_trace_smoke_only",
        "truthfulness_boundary": (
            f"seqs={args.seqs} small-frame real HorizonStream liveness/provenance smoke only; "
            "does not satisfy full-sequence, cross-sequence, or all-control carrier gates"
        ),
        "branch_stats": branch_stats,
        "manifest": rel(paths["manifest"]),
        "results_csv": rel(paths["results_csv"]),
        "results_jsonl": rel(paths["results_jsonl"]),
    }
    return summary


def run_batch(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(int(args.max_frames), str(args.trace_profile), str(args.seqs), str(args.run_label))
    manifest_path = paths["manifest"]
    manifest_fields = [
        "schema",
        "job_id",
        "case_id",
        "branch",
        "role",
        "seq",
        "gpu",
        "action",
        "control",
        "trace_profile",
        "expected_audit_file",
        "layer_filter_kind",
        "selected_layer",
        "description",
        "output_root",
        "trace_root",
        "action_audit_root",
        "log_path",
        "max_frames",
        "fresh_case",
    ]
    write_csv(manifest_path, jobs, manifest_fields)

    results_csv = paths["results_csv"]
    results_jsonl = paths["results_jsonl"]
    if args.fresh_results:
        results_csv.unlink(missing_ok=True)
        results_jsonl.unlink(missing_ok=True)

    queues: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        queues.setdefault(str(job["gpu"]), []).append(job)
    print(
        f"planned_jobs={len(jobs)} queues={len(queues)} manifest={manifest_path} results_csv={results_csv}",
        flush=True,
    )

    csv_lock = threading.Lock()
    jsonl_lock = threading.Lock()
    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, len(queues)))) as executor:
        futures = {
            executor.submit(run_gpu_queue, gpu, rows, results_csv, results_jsonl, csv_lock, jsonl_lock): gpu
            for gpu, rows in queues.items()
        }
        for future in as_completed(futures):
            all_results.extend(future.result())
    all_results.sort(key=lambda row: row["job_id"])
    summary = build_summary(all_results, jobs, args, paths)
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    return all_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--seqs", default="00")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument(
        "--trace-profile",
        choices=("full", "global_mrt"),
        default="full",
        help="full keeps local+GLA+MRT traces; global_mrt disables local trace for longer windows.",
    )
    parser.add_argument("--case-ids", default="", help="Comma-separated case ids; empty means all cases.")
    parser.add_argument("--run-label", default="", help="Optional suffix for top-level artifact filenames.")
    parser.add_argument("--fresh-case", action="store_true")
    parser.add_argument("--fresh-results", action="store_true")
    args = parser.parse_args()

    gpus = parse_csv_list(args.gpus)
    seqs = parse_csv_list(args.seqs)
    if not gpus:
        raise SystemExit("No GPUs supplied.")
    if not seqs:
        raise SystemExit("No sequences supplied.")
    if not HS_ROOT.is_dir():
        raise SystemExit(f"missing HorizonStream root: {HS_ROOT}")
    if not (HS_ROOT / CONFIG).is_file():
        raise SystemExit(f"missing config: {HS_ROOT / CONFIG}")
    if not (HS_ROOT / CHECKPOINT).is_file():
        raise SystemExit(f"missing checkpoint: {HS_ROOT / CHECKPOINT}")
    if not IMAGE_ROOT.is_dir():
        raise SystemExit(f"missing image root: {IMAGE_ROOT}")
    if not SEMANTIC_ROOT.is_dir():
        raise SystemExit(f"missing semantic root: {SEMANTIC_ROOT}")

    case_ids = parse_csv_list(args.case_ids)
    jobs = build_jobs(
        gpus,
        seqs,
        int(args.max_frames),
        bool(args.fresh_case),
        str(args.trace_profile),
        case_ids,
        str(args.run_label),
    )
    results = run_batch(args, jobs)
    return 0 if all(int(row.get("returncode", 1)) == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
