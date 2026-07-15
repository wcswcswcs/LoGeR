#!/usr/bin/env python3
"""Generate v107R Stage7 full-sequence selected-policy configs.

Stage6 validated a semantic-only threshold policy on selected 96F windows.  This
script embeds the exact selected global frame indices into full KITTI
00/01/02/05 streaming runs, so full-sequence ATE can be measured without
pretending that the 96F-local selector is already a general long-horizon policy.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V105_STAGE0 = V105 / "stage0_repo_env_audit/stage0_summary.json"
V105_FULL_METRICS = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
V107R = ROOT / "results/acl2_v107r_lingbot_semantic_memory_decision_cue_operation_control"
STAGE6_POLICY_ROWS = (
    V107R
    / "stage6_runtime_pilot_or_blocked/semantic_wrapper_policy_pilot/semantic_action_policy_rows.csv"
)
OUT = V107R / "stage7_full_sequence_selected_policy"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"

PRIMARY_ACTION = "semantic_highrisk_early_risk_ge_0p60_force_non_keyframe"
STAGE7_ACTION = "semantic_threshold0p60_selected_fullseq_force_non_keyframe"
SEQUENCES = ["00", "01", "02", "05"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_indices(raw: str | None) -> list[int]:
    if raw is None or raw.strip() == "":
        return []
    return sorted({int(float(x)) for x in raw.replace(",", ";").split(";") if x.strip()})


def selected_indices_by_seq() -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    by_seq = {seq: [] for seq in SEQUENCES}
    source_rows: list[dict[str, Any]] = []
    for row in read_csv(STAGE6_POLICY_ROWS):
        if row["action_name"] != PRIMARY_ACTION:
            continue
        seq = row["seq"]
        indices = parse_indices(row.get("force_global_frame_indices", ""))
        by_seq.setdefault(seq, []).extend(indices)
        source_rows.append(
            {
                "schema": "acl2_v107r_stage7_selected_source_row_v1",
                "seq": seq,
                "source_target_id": row["target_id"],
                "source_target_kind": row["target_kind"],
                "source_action_name": row["action_name"],
                "selected_count": len(indices),
                "selected_global_frame_indices": ";".join(str(x) for x in indices),
                "source_trace_start_idx": row.get("trace_start_idx", ""),
                "source_trace_end_idx_exclusive": row.get("trace_end_idx_exclusive", ""),
                "selector_scope": row.get("selector_scope", ""),
                "risk_threshold": row.get("risk_threshold", ""),
                "selected_risk_mean": row.get("selected_risk_mean", ""),
            }
        )
    return {seq: sorted(set(vals)) for seq, vals in by_seq.items()}, source_rows


def frame_counts_from_v105() -> dict[str, int]:
    rows = read_csv(V105_FULL_METRICS)
    return {row["seq"]: int(float(row["frames"])) for row in rows}


def method_yaml(checkpoint: str, env_name: str, use_sdpa: bool, action_name: str, indices: list[int]) -> str:
    return "\n".join(
        [
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
            f"_stage4_action_label: {action_name}",
            "_stage4_action_mode: force_non_keyframe",
            "",
        ]
    )


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


def command_prefix(conda_path: str, pythonpath: str, gpu: str) -> str:
    return f"PATH={Path(conda_path).parent}:$PATH PYTHONPATH={pythonpath} CUDA_VISIBLE_DEVICES={gpu}"


def build() -> dict[str, Any]:
    stage0 = json.loads(V105_STAGE0.read_text(encoding="utf-8"))
    checkpoint = stage0["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(stage0["kitti"]["resolved_kitti_root"])
    conda_path = stage0["environment"]["conda"]["conda"]
    env_name = stage0["environment"]["conda"]["recommended_env"]
    pythonpath = stage0["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(stage0["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    frame_counts = frame_counts_from_v105()
    selected_by_seq, source_rows = selected_indices_by_seq()

    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)

    policy_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    gpu_cycle = ["0", "1", "2", "3", "4", "5"]

    for seq_index, seq in enumerate(SEQUENCES):
        indices = selected_by_seq.get(seq, [])
        frames = frame_counts.get(seq)
        if frames is None:
            raise ValueError(f"missing v105 full metric frame count for seq={seq}")
        out_of_range = [idx for idx in indices if idx < 0 or idx >= frames]
        if out_of_range:
            raise ValueError(f"selected indices out of range for seq={seq}: {out_of_range}")

        dataset = f"kitti_v107r_stage7_fullseq_{seq}"
        method = f"lingbot_map_v107r_stage7_{STAGE7_ACTION}_{seq}"
        config = CONFIG_ROOT / f"kitti_lingbot_v107r_stage7_{STAGE7_ACTION}_{seq}.yaml"
        method_path = METHOD_DIR / f"{method}.yaml"
        action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
        gpu = gpu_cycle[seq_index % len(gpu_cycle)]

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
        write_text(method_path, method_yaml(checkpoint, env_name, use_sdpa, STAGE7_ACTION, indices))
        write_text(config, run_config_yaml(dataset, method))

        policy_rows.append(
            {
                "schema": "acl2_v107r_stage7_full_sequence_policy_row_v1",
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": STAGE7_ACTION,
                "source_stage6_action": PRIMARY_ACTION,
                "stage4_action_mode": "force_non_keyframe",
                "selected_count": len(indices),
                "selected_global_frame_indices": ";".join(str(x) for x in indices),
                "frames": frames,
                "policy_scope": "full_sequence_embedding_of_stage6_selected_global_indices",
                "policy_boundary": (
                    "This verifies full-sequence ATE for Stage6-selected semantic-threshold interventions; "
                    "it is not yet a general recurrent long-horizon semantic policy."
                ),
            }
        )
        config_rows.append(
            {
                **policy_rows[-1],
                "config": str(config.resolve()),
                "method_config": str(method_path.resolve()),
                "action_file": str(action_file.resolve()),
                "gpu": gpu,
            }
        )

        prefix = command_prefix(conda_path, pythonpath, gpu)
        action_env = (
            f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
            f"ACL2_V105_STAGE4_ACTION_LABEL={STAGE7_ACTION} "
            "ACL2_V107_STAGE7_POLICY_SCOPE=full_sequence_selected_stage6_policy"
        )
        commands = {
            "prepare": (
                f"{prefix} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config.resolve()} --force"
            ),
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
                    "schema": "acl2_v107r_stage7_full_sequence_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v107r_stage7_{STAGE7_ACTION}_{seq}_{phase}",
                    "phase": phase,
                    "target_id": f"fullseq_{seq}",
                    "target_kind": "full_sequence",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": STAGE7_ACTION,
                    "action_family": "semantic_action_threshold_full_sequence_selected",
                    "stage4_action_mode": "force_non_keyframe",
                    "selector": "stage6_semantic_highrisk_early_risk_ge_0p60_selected_global_indices",
                    "selected_count": len(indices),
                    "force_non_keyframe_indices": ";".join(str(x) for x in indices),
                    "trace_start_idx": 0,
                    "trace_end_idx_exclusive": frames,
                    "target_frame_start": 0,
                    "target_frame_end": frames - 1,
                    "gpu": gpu,
                    "cwd": str(BENCHMARK.resolve()),
                    "config": str(config.resolve()),
                    "trace_file": "",
                    "action_file": str(action_file.resolve()),
                    "command": command,
                    "status": "planned",
                }
            )

    write_csv(OUT / "stage6_selected_source_rows.csv", source_rows)
    write_csv(OUT / "full_sequence_policy_rows.csv", policy_rows)
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "run_manifest.csv", manifest_rows)
    summary = {
        "schema": "acl2_v107r_stage7_full_sequence_config_summary_v1",
        "stage6_source_action": PRIMARY_ACTION,
        "stage7_action": STAGE7_ACTION,
        "sequences": SEQUENCES,
        "policy_rows": len(policy_rows),
        "manifest_rows": len(manifest_rows),
        "selected_indices_by_seq": {seq: selected_by_seq.get(seq, []) for seq in SEQUENCES},
        "workspace": rel(WORKSPACE),
        "raw_action": rel(RAW_ACTION),
        "v105_full_metrics": rel(V105_FULL_METRICS),
        "runtime_boundary": "Full-sequence LingBot wrapper force_non_keyframe interventions at Stage6-selected global frames; no output post-processing action.",
    }
    write_json(OUT / "config_generation_summary.json", summary)
    return summary


def main() -> None:
    print(json.dumps(build(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
