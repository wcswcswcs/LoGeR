#!/usr/bin/env python3
"""Generate ACL2 v111TF T3 semantic-aware soft token-weighting configs."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_pilot_configs as v108  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
T3 = RESULT_ROOT / "batch_t_t3_soft_token_weighting"
CONFIG_ROOT = T3 / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = T3 / "workspace"
RAW_ACTION = T3 / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
SOURCE = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality/stage4_full_00_01_02_05_validation/candidate_policy_rows.csv"
SEMANTIC = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search/stage2_semantic_cue_bank/frame_semantic_summary.csv"
SEQUENCES = ("00", "01", "02", "05")
SOURCE_POLICY_ID = "B1_semantic_plus_internal"

ALPHA_CAM = 0.0
BETA_CAM = 1.0
ALPHA_REG = 0.0
BETA_REG = 1.0
MU_ANCHOR = 0.5
G_MIN = 0.25
G_MAX = 1.25
EPS = 1e-6

POLICIES = [
    {
        "policy_id": "T3_map_all1_context_parity",
        "policy_family": "map_all1_context_parity",
        "variant": "parity_all1",
        "gamma_cam": 0.0,
        "gamma_reg": 0.0,
        "nu": 0.0,
        "risk_mode": "none",
    },
    {
        "policy_id": "T3_soft_mild_raw",
        "policy_family": "soft_mild_raw",
        "variant": "soft_mild",
        "gamma_cam": 1.0,
        "gamma_reg": 0.7,
        "nu": 0.2,
        "risk_mode": "raw",
    },
    {
        "policy_id": "T3_soft_medium_raw",
        "policy_family": "soft_medium_raw",
        "variant": "soft_medium",
        "gamma_cam": 2.0,
        "gamma_reg": 1.0,
        "nu": 0.3,
        "risk_mode": "raw",
    },
    {
        "policy_id": "T3_soft_strong_raw",
        "policy_family": "soft_strong_raw",
        "variant": "soft_strong",
        "gamma_cam": 4.0,
        "gamma_reg": 2.0,
        "nu": 0.5,
        "risk_mode": "raw",
    },
    {
        "policy_id": "T3_soft_mild_znorm",
        "policy_family": "soft_mild_znorm",
        "variant": "soft_mild",
        "gamma_cam": 1.0,
        "gamma_reg": 0.7,
        "nu": 0.2,
        "risk_mode": "znorm",
    },
    {
        "policy_id": "T3_soft_medium_znorm",
        "policy_family": "soft_medium_znorm",
        "variant": "soft_medium",
        "gamma_cam": 2.0,
        "gamma_reg": 1.0,
        "nu": 0.3,
        "risk_mode": "znorm",
    },
    {
        "policy_id": "T3_soft_strong_znorm",
        "policy_family": "soft_strong_znorm",
        "variant": "soft_strong",
        "gamma_cam": 4.0,
        "gamma_reg": 2.0,
        "nu": 0.5,
        "risk_mode": "znorm",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_indices(raw: str) -> list[int]:
    return [int(float(part)) for part in str(raw).split(";") if str(part).strip()]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def source_rows() -> dict[str, dict[str, str]]:
    return {
        row["seq"]: row
        for row in read_csv(SOURCE)
        if row.get("policy_id") == SOURCE_POLICY_ID and row.get("seq") in SEQUENCES
    }


def semantic_rows() -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_csv(SEMANTIC):
        seq = row.get("seq_id", "")
        if seq not in SEQUENCES:
            continue
        frame = int(float(row.get("frame_id", 0)))
        dynamic = safe_float(row.get("dynamic_mass"))
        boundary = safe_float(row.get("boundary_mass"))
        weak = safe_float(row.get("weak_context_mass"))
        trust = safe_float(row.get("semantic_trust_mean"), 1.0)
        stable = safe_float(row.get("stable_structure_mass"))
        continuity = safe_float(row.get("semantic_continuity_score"))
        risk = dynamic + 0.7 * boundary + 0.3 * weak + 0.5 * (1.0 - trust)
        support = stable + 0.5 * continuity - 0.5 * boundary
        row = dict(row)
        row["v111_risk"] = risk
        row["v111_support"] = support
        out[(seq, frame)] = row
    by_seq: dict[str, list[float]] = {seq: [] for seq in SEQUENCES}
    for (seq, _), row in out.items():
        by_seq[seq].append(float(row["v111_risk"]))
    stats: dict[str, tuple[float, float]] = {}
    for seq, values in by_seq.items():
        med = statistics.median(values)
        mad = statistics.median([abs(value - med) for value in values])
        stats[seq] = (med, mad)
    for (seq, _), row in out.items():
        med, mad = stats[seq]
        row["v111_risk_z"] = (float(row["v111_risk"]) - med) / (mad + EPS)
        row["v111_risk_seq_median"] = med
        row["v111_risk_seq_mad"] = mad
    return out


def weight_mask(policy: dict[str, Any], sem: dict[str, Any]) -> list[float]:
    if policy["risk_mode"] == "none":
        return [1, 1, 1, 1, 1, 1]
    risk = safe_float(sem["v111_risk_z"] if policy["risk_mode"] == "znorm" else sem["v111_risk"])
    support = safe_float(sem["v111_support"])
    g_cam = sigmoid(ALPHA_CAM + BETA_CAM * support - float(policy["gamma_cam"]) * risk)
    g_reg = sigmoid(ALPHA_REG + BETA_REG * support - float(policy["gamma_reg"]) * risk)
    g_anchor = clamp(1.0 + MU_ANCHOR * support - float(policy["nu"]) * risk, G_MIN, G_MAX)
    return [g_cam, g_reg, g_reg, g_reg, g_reg, g_anchor]


def mask_text(mask: list[float]) -> str:
    return ",".join(f"{value:.6g}" for value in mask)


def method_yaml(
    *,
    checkpoint: str,
    env_name: str,
    use_sdpa: bool,
    action_label: str,
    indices: list[int],
    mask_map: dict[int, list[float]],
) -> str:
    lines = [
        "model: lingbot_map",
        f"env: {env_name}",
        f"_checkpoint: {checkpoint}",
        "_device: cuda",
        "_use_amp: true",
        f"_use_sdpa: {str(use_sdpa).lower()}",
        "_image_size: 518",
        "_patch_size: 14",
        "_enable_3d_rope: true",
        "_num_scale_frames: 8",
        "_max_frame_num: 1024",
        "_kv_cache_sliding_window: 64",
        "_kv_cache_scale_frames: 8",
        "_auto_keyframe_threshold: 320",
        "_area_budget: 255000",
        "_align: 14",
        "_mode: streaming",
        "_keyframe_interval: auto",
        f"_force_non_keyframe_indices: {json.dumps(indices)}",
        f"_stage4_action_label: {action_label}",
        "_stage4_action_mode: context_token_mask",
        f"_stage4_context_token_mask_map: {json.dumps({str(k): v for k, v in mask_map.items()}, sort_keys=True)}",
        "",
    ]
    return "\n".join(lines)


def run_config_yaml(dataset: str, method: str) -> str:
    return "\n".join(
        [
            f"workspace: {WORKSPACE.resolve()}",
            "",
            "evaluation:",
            "  traj:",
            "    enable: true",
            "    vis: true",
            "  auc:",
            "    enable: false",
            "  depth:",
            "    enable: false",
            "  points:",
            "    enable: false",
            "",
            "datasets:",
            f"  - {dataset}",
            "",
            "methods:",
            f"  - {method}",
            "",
        ]
    )


def main() -> int:
    env = v108.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    source_by_seq = source_rows()
    sem_by_key = semantic_rows()
    missing_source = [seq for seq in SEQUENCES if seq not in source_by_seq]
    gpu_cycle = ["0", "1", "2", "3", "4", "5"]

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    T3.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v111tf_t3_fullseq_{seq}"
        write_text(
            DATASET_DIR / f"{dataset}.yaml",
            "\n".join(
                [
                    "dataset: kitti",
                    f"raw_data_root: {raw_data_root}",
                    "_target_size: [504, 280]",
                    f"_sequences: [\"{seq}\"]",
                    "",
                ]
            ),
        )

    config_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    prepare_rows_by_seq: dict[str, dict[str, Any]] = {}
    row_index = 0
    missing_semantic: list[dict[str, Any]] = []

    for policy in POLICIES:
        policy_id = str(policy["policy_id"])
        for seq in SEQUENCES:
            source = source_by_seq.get(seq, {})
            selected = parse_indices(source.get("selected_global_frame_indices", ""))
            source_indices = parse_indices(source.get("source_selected_global_frame_indices", ""))
            frames = int(float(source.get("frames", "0"))) if source else 0
            mask_map: dict[int, list[float]] = {}
            for frame in selected:
                sem = sem_by_key.get((seq, frame), {})
                if not sem:
                    missing_semantic.append({"seq": seq, "frame": frame, "policy_id": policy_id})
                    sem = {"v111_risk": 0.0, "v111_risk_z": 0.0, "v111_support": 0.0}
                mask = weight_mask(policy, sem)
                mask_map[frame] = mask
                frame_rows.append(
                    {
                        "schema": "acl2_v111tf_t3_soft_weight_frame_row_v1",
                        "policy_id": policy_id,
                        "policy_family": policy["policy_family"],
                        "seq": seq,
                        "frame": frame,
                        "risk_mode": policy["risk_mode"],
                        "v111_risk": sem.get("v111_risk", ""),
                        "v111_risk_z": sem.get("v111_risk_z", ""),
                        "v111_support": sem.get("v111_support", ""),
                        "camera_weight": mask[0],
                        "register_weight": mask[1],
                        "anchor_weight": mask[5],
                        "token_type_mask": mask_text(mask),
                    }
                )

            dataset = f"kitti_v111tf_t3_fullseq_{seq}"
            method = f"lingbot_map_v111tf_t3_{policy_id}_{seq}"
            action_label = f"v111tf_t3_{policy_id}"
            config = CONFIG_ROOT / f"kitti_lingbot_v111tf_t3_{policy_id}_{seq}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            gpu = gpu_cycle[row_index % len(gpu_cycle)]
            row_index += 1

            write_text(
                method_path,
                method_yaml(
                    checkpoint=checkpoint,
                    env_name=env_name,
                    use_sdpa=use_sdpa,
                    action_label=action_label,
                    indices=selected,
                    mask_map=mask_map,
                ),
            )
            write_text(config, run_config_yaml(dataset, method))
            selected_string = ";".join(str(x) for x in selected)
            source_selected_string = ";".join(str(x) for x in source_indices)
            mask_values = list(mask_map.values())
            cam_mean = sum(mask[0] for mask in mask_values) / len(mask_values) if mask_values else ""
            reg_mean = sum(mask[1] for mask in mask_values) / len(mask_values) if mask_values else ""
            anchor_mean = sum(mask[5] for mask in mask_values) / len(mask_values) if mask_values else ""
            row = {
                "schema": "acl2_v111tf_t3_soft_token_policy_row_v1",
                "surface_id": "T",
                "candidate_id": "T3",
                "policy_id": policy_id,
                "policy_family": policy["policy_family"],
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_label,
                "stage2_action_mode": "context_token_mask",
                "stage4_action_mode": "context_token_mask",
                "source_policy_id": SOURCE_POLICY_ID,
                "source_selected_count": len(source_indices),
                "source_selected_global_frame_indices": source_selected_string,
                "selected_count": len(selected),
                "selected_global_frame_indices": selected_string,
                "expected_action_field": "forced_context_only",
                "risk_mode": policy["risk_mode"],
                "gamma_cam": policy["gamma_cam"],
                "gamma_reg": policy["gamma_reg"],
                "nu": policy["nu"],
                "alpha_cam": ALPHA_CAM,
                "beta_cam": BETA_CAM,
                "alpha_reg": ALPHA_REG,
                "beta_reg": BETA_REG,
                "mu_anchor": MU_ANCHOR,
                "g_min": G_MIN,
                "g_max": G_MAX,
                "camera_token_weight_mean": cam_mean,
                "register_token_weight_mean": reg_mean,
                "anchor_token_weight_mean": anchor_mean,
                "frames": frames,
                "full_sequence_keyframe_interval": v108.keyframe_interval(frames) if frames else "",
                "runtime_boundary": (
                    "Per-frame context_token_mask_map applies soft camera/register/anchor weights only to selected B1 high-risk compact context-token writes."
                ),
                "claim_boundary": (
                    "T3 soft weighting geometry/mechanism study. Semantic causality still requires controls and cannot be claimed from this grid alone."
                ),
                "config": str(config.resolve()),
                "method_config": str(method_path.resolve()),
                "action_file": str(action_file.resolve()),
                "gpu": gpu,
            }
            config_rows.append(row)
            policy_rows.append({k: v for k, v in row.items() if k not in {"config", "method_config", "action_file", "gpu"}})

            prefix = v108.command_prefix(conda_path, pythonpath, gpu)
            prepare_command = (
                f"{prefix} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config.resolve()} --force"
            )
            prepare_rows_by_seq.setdefault(
                seq,
                {
                    "schema": "acl2_v111tf_t3_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v111tf_t3_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"t3_fullseq_{seq}",
                    "target_kind": "full_sequence_dataset_prepare",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": "dataset_prepare_once",
                    "action_family": "prepare",
                    "stage4_action_mode": "dataset_prepare",
                    "selector": "deduplicated_prepare_once_per_dataset_seq_to_avoid_parallel_rmtree_race",
                    "selected_count": 0,
                    "gpu": gpu,
                    "cwd": str(BENCHMARK.resolve()),
                    "config": str(config.resolve()),
                    "trace_file": "",
                    "action_file": "",
                    "command": prepare_command,
                    "status": "planned",
                },
            )
            action_env = (
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
                f"ACL2_V105_STAGE4_ACTION_LABEL={action_label} "
                f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                f"ACL2_V105_GCA_TRACE_METHOD={method} "
                f"ACL2_V108_STAGE4_POLICY_ID={policy_id} "
                f"ACL2_V108_STAGE4_SURFACE_ID=T "
                f"ACL2_V111TF_T3_POLICY_ID={policy_id}"
            )
            commands = {
                "run_worker": (
                    f"{prefix} {action_env} {conda_path} run -n {env_name} "
                    f"python run_worker.py --config {config.resolve()} --method {method} "
                    f"--dataset {dataset} --scene {seq} --force"
                ),
                "evaluate": (
                    f"{prefix} {conda_path} run -n {env_name} "
                    f"python evaluate.py --config {config.resolve()} --force"
                ),
                "report": (
                    f"{prefix} {conda_path} run -n {env_name} "
                    f"python report.py --workspace {WORKSPACE.resolve()} --dataset {dataset}"
                ),
            }
            for phase, command in commands.items():
                manifest_rows.append(
                    {
                        "schema": "acl2_v111tf_t3_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v111tf_t3_{policy_id}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"t3_fullseq_{seq}",
                        "target_kind": "full_sequence",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": policy["policy_family"],
                        "stage4_action_mode": "context_token_mask",
                        "selector": "v110_stage4_B1_semantic_plus_internal_selected_keyframes",
                        "selected_count": len(selected),
                        "force_non_keyframe_indices": selected_string,
                        "gpu": gpu,
                        "cwd": str(BENCHMARK.resolve()),
                        "config": str(config.resolve()),
                        "trace_file": "",
                        "action_file": str(action_file.resolve()),
                        "command": command,
                        "status": "planned",
                    }
                )

    manifest_rows = [prepare_rows_by_seq[seq] for seq in SEQUENCES] + manifest_rows
    summary = {
        "schema": "acl2_v111tf_t3_soft_token_config_summary_v1",
        "t3_config_ready": not missing_source and not missing_semantic,
        "blocker": "missing_source_or_semantic_rows" if (missing_source or missing_semantic) else "",
        "missing_source_sequences": missing_source,
        "missing_semantic_rows": missing_semantic,
        "source_policy_id": SOURCE_POLICY_ID,
        "sequences": list(SEQUENCES),
        "policy_count": len(POLICIES),
        "policy_ids": [str(policy["policy_id"]) for policy in POLICIES],
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "prepare_manifest_rows": len(prepare_rows_by_seq),
        "run_worker_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "run_worker"),
        "evaluate_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "evaluate"),
        "report_manifest_rows": sum(1 for row in manifest_rows if row["phase"] == "report"),
        "weight_defaults": {
            "alpha_cam": ALPHA_CAM,
            "beta_cam": BETA_CAM,
            "alpha_reg": ALPHA_REG,
            "beta_reg": BETA_REG,
            "mu_anchor": MU_ANCHOR,
            "g_min": G_MIN,
            "g_max": G_MAX,
            "note": "Plan specified gamma_cam/gamma_reg/nu ladder; these fixed defaults are implementation assumptions, not GT-fitted.",
        },
        "outputs": {
            "action_config_rows": rel(T3 / "action_config_rows.csv"),
            "candidate_policy_rows": rel(T3 / "candidate_policy_rows.csv"),
            "soft_weight_frame_rows": rel(T3 / "soft_weight_frame_rows.csv"),
            "run_manifest": rel(T3 / "run_manifest.csv"),
            "summary": rel(T3 / "t3_config_generation_summary.json"),
            "workspace": rel(WORKSPACE),
            "raw_action": rel(RAW_ACTION),
        },
    }
    write_csv(T3 / "action_config_rows.csv", config_rows)
    write_csv(T3 / "candidate_policy_rows.csv", policy_rows)
    write_csv(T3 / "soft_weight_frame_rows.csv", frame_rows)
    write_csv(T3 / "run_manifest.csv", manifest_rows)
    write_json(T3 / "t3_config_generation_summary.json", summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True))
    return 0 if summary["t3_config_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
