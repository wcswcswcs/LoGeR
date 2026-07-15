#!/usr/bin/env python3
"""Generate ACL2 v116-TF Task1 matched-control configs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_pilot_configs as v108  # noqa: E402
import build_v116tf_task1_ab_configs as task1  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence"
OUT = RESULT_ROOT / "task1_ab_controls"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
V108_STAGE3 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search/stage3_operation_cue_screen"

SEQUENCES = ("00", "02")
NUM_ANCHOR = 8
A1_CONTROL_MS = (16, 24, 32)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def hash_key(salt: str, seq: str, frame: int) -> str:
    return hashlib.sha256(f"{salt}|{seq}|{frame}".encode("utf-8")).hexdigest()


def exact_count(preferred: list[int], universe: list[int], n: int, salt: str, seq: str) -> list[int]:
    unique = sorted(set(preferred))
    if len(unique) >= n:
        return sorted(sorted(unique, key=lambda frame: hash_key(salt, seq, frame))[:n])
    used = set(unique)
    ranked_fill = sorted((frame for frame in universe if frame not in used), key=lambda frame: hash_key(salt, seq, frame))
    return sorted(unique + ranked_fill[: max(0, n - len(unique))])


def a1_control_indices(mode: str, seq: str, m: int, sem: dict[tuple[str, int], dict[str, float]]) -> list[int]:
    frames = list(range(m))
    if mode == "random":
        return sorted(sorted(frames, key=lambda frame: hash_key(f"a1_random_{m}_seed0", seq, frame))[:NUM_ANCHOR])
    rows: list[tuple[tuple[Any, ...], int]] = []
    if mode == "semantic_shuffle":
        risks = [sem.get((seq, frame), {}).get("risk", 0.0) for frame in frames]
        offset = 7 % len(frames)
        shifted = risks[offset:] + risks[:offset]
        for frame, shifted_risk in zip(frames, shifted):
            row = sem.get((seq, frame), {})
            rows.append(((shifted_risk, -row.get("stable", 0.0), frame), frame))
    elif mode == "role_rotation":
        for frame in frames:
            row = sem.get((seq, frame), {})
            # Rotate the semantic role contract: prefer low stable / high dynamic frames.
            rows.append(((row.get("stable", 0.0), -row.get("dynamic", 0.0), row.get("weak", 0.0), frame), frame))
    else:
        raise ValueError(f"unknown A1 control mode: {mode}")
    rows.sort(key=lambda item: item[0])
    return sorted(frame for _key, frame in rows[:NUM_ANCHOR])


def v108_b1_source(policy_id: str, seq: str) -> list[int]:
    frames = []
    for row in read_csv(V108_STAGE3 / "surface_policy_frame_rows.csv"):
        if row.get("surface_id") == "B" and row.get("policy_id") == policy_id and row.get("seq_id") == seq:
            frames.append(int(float(row["frame_id"])))
    if not frames:
        raise FileNotFoundError(f"missing v108 B1 source frames for {policy_id} seq={seq}")
    return sorted(set(frames))


def low_risk_base_universe(seq: str, frames: int, sem: dict[tuple[str, int], dict[str, float]]) -> list[int]:
    bases = v108.base_keyframes(frames)
    return sorted(
        bases,
        key=lambda frame: (
            sem.get((seq, frame), {}).get("risk", 0.0),
            -sem.get((seq, frame), {}).get("stable", 0.0),
            frame,
        ),
    )


def b1_control_indices(
    mode: str,
    seq: str,
    frames: int,
    reference_count: int,
    sem: dict[tuple[str, int], dict[str, float]],
) -> tuple[list[int], str]:
    bases = v108.base_keyframes(frames)
    if mode == "semantic_shuffle":
        snapped, _rows = v108.snap_to_nearest_base_keyframe(v108_b1_source("B1_semantic_shuffle", seq), frames)
        return exact_count(snapped, bases, reference_count, "b1_semantic_shuffle_same_count", seq), "v108_B1_semantic_shuffle_snapped_then_exact_count"
    if mode == "same_count_random":
        ranked = sorted(bases, key=lambda frame: hash_key("b1_same_count_random_seed0", seq, frame))
        return sorted(ranked[:reference_count]), "base_keyframe_universe_deterministic_same_count_random_seed0"
    if mode == "low_risk_reverse":
        snapped, _rows = v108.snap_to_nearest_base_keyframe(v108_b1_source("B1_low_risk_reverse", seq), frames)
        return exact_count(snapped, low_risk_base_universe(seq, frames, sem), reference_count, "b1_low_risk_reverse_fill", seq), "v108_B1_low_risk_reverse_snapped_plus_low_risk_fill_exact_count"
    raise ValueError(f"unknown B1 control mode: {mode}")


def control_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "policy_id": "AB_CTRL_B1_semantic_shuffle_same_count",
            "policy_family": "b1_semantic_shuffle_same_count",
            "control_component": "B1",
            "control_mode": "semantic_shuffle",
            "M": 0,
            "with_b1": True,
        },
        {
            "policy_id": "AB_CTRL_B1_same_count_random_seed0",
            "policy_family": "b1_same_count_random",
            "control_component": "B1",
            "control_mode": "same_count_random",
            "M": 0,
            "with_b1": True,
        },
        {
            "policy_id": "AB_CTRL_B1_low_risk_reverse_same_count",
            "policy_family": "b1_low_risk_reverse_same_count",
            "control_component": "B1",
            "control_mode": "low_risk_reverse",
            "M": 0,
            "with_b1": True,
        },
    ]
    for m in A1_CONTROL_MS:
        for mode, family in (
            ("random", "a1_random_same_count_plus_b1"),
            ("semantic_shuffle", "a1_semantic_shuffle_plus_b1"),
            ("role_rotation", "a1_role_rotation_plus_b1"),
        ):
            specs.append(
                {
                    "policy_id": f"AB_CTRL_A1_{mode}_first{m}_plus_B1",
                    "policy_family": family,
                    "control_component": "A1",
                    "control_mode": mode,
                    "M": m,
                    "with_b1": True,
                }
            )
    return specs


def main() -> int:
    env = v108.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    frames_by_seq = task1.frame_counts()
    b1_by_seq = task1.b1_force_indices()
    sem = task1.semantic_by_key()
    gpu_cycle = ["0", "1", "2", "3", "4", "5"]

    for path in (METHOD_DIR, DATASET_DIR, RAW_ACTION, OUT):
        path.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v116tf_task1_ab_control_fullseq_{seq}"
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
    manifest_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prepare_rows_by_seq: dict[str, dict[str, Any]] = {}
    row_index = 0

    for spec in control_specs():
        for seq in SEQUENCES:
            frames = frames_by_seq[seq]
            scale_indices: list[int] = []
            b1_indices: list[int] = []
            selector_note = ""
            if spec["control_component"] == "A1":
                scale_indices = a1_control_indices(str(spec["control_mode"]), seq, int(spec["M"]), sem)
                b1_indices = b1_by_seq[seq]
                selector_note = "A1 control selector with B1 reference force_non_keyframe fixed"
            else:
                b1_indices, selector_note = b1_control_indices(
                    str(spec["control_mode"]),
                    seq,
                    frames,
                    len(b1_by_seq[seq]),
                    sem,
                )
            combined = bool(scale_indices)
            dataset = f"kitti_v116tf_task1_ab_control_fullseq_{seq}"
            action_label = f"v116tf_task1_control_{spec['policy_id']}"
            method = f"lingbot_map_v116tf_task1_control_{spec['policy_id']}_{seq}"
            config = CONFIG_ROOT / f"kitti_lingbot_v116tf_task1_control_{spec['policy_id']}_{seq}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            gpu = gpu_cycle[row_index % len(gpu_cycle)]
            row_index += 1

            write_text(
                method_path,
                task1.method_yaml(
                    checkpoint=checkpoint,
                    env_name=env_name,
                    use_sdpa=use_sdpa,
                    action_label=action_label,
                    b1_indices=b1_indices,
                    scale_indices=scale_indices,
                    combined=combined,
                ),
            )
            write_text(config, task1.run_config_yaml(dataset, method).replace(str(task1.WORKSPACE.resolve()), str(WORKSPACE.resolve())))

            policy = {
                "schema": "acl2_v116tf_task1_control_policy_row_v1",
                "task": "Task1_AB_controls",
                "surface_id": "AB",
                "candidate_id": spec["policy_id"].split("_", 2)[-1],
                "policy_id": spec["policy_id"],
                "policy_family": spec["policy_family"],
                "control_component": spec["control_component"],
                "control_mode": spec["control_mode"],
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_label,
                "stage4_action_mode": "anchor_scale_frame_indices" if combined else "force_non_keyframe",
                "M": spec["M"],
                "num_anchor": len(scale_indices),
                "scale_frame_indices": ";".join(str(x) for x in scale_indices),
                "b1_force_non_keyframe_indices": ";".join(str(x) for x in b1_indices),
                "b1_expected_count": len(b1_indices),
                "b1_reference_count": len(b1_by_seq[seq]),
                "frames": frames,
                "full_sequence_keyframe_interval": v108.keyframe_interval(frames),
                "selector": selector_note,
                "runtime_boundary": "Matched Task1 controls: same runtime action fields as A1+B1 pilot; selector only is changed.",
            }
            policy_rows.append(policy)
            config_rows.append(
                {
                    **policy,
                    "config": str(config.resolve()),
                    "method_config": str(method_path.resolve()),
                    "action_file": str(action_file.resolve()),
                    "gpu": gpu,
                }
            )
            for field, values in (("scale", scale_indices), ("b1_force_non_keyframe", b1_indices)):
                for rank, frame in enumerate(values, start=1):
                    row = sem.get((seq, frame), {})
                    selected_rows.append(
                        {
                            "schema": "acl2_v116tf_task1_control_selected_frame_row_v1",
                            "policy_id": spec["policy_id"],
                            "policy_family": spec["policy_family"],
                            "control_component": spec["control_component"],
                            "control_mode": spec["control_mode"],
                            "seq": seq,
                            "field": field,
                            "rank": rank,
                            "frame": frame,
                            "dynamic_mass": row.get("dynamic", ""),
                            "stable_structure_mass": row.get("stable", ""),
                            "boundary_mass": row.get("boundary", ""),
                            "weak_context_mass": row.get("weak", ""),
                            "risk": row.get("risk", ""),
                        }
                    )

            prefix = v108.command_prefix(conda_path, pythonpath, gpu)
            action_env = (
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
                f"ACL2_V105_STAGE4_ACTION_LABEL={action_label} "
                f"ACL2_V108_STAGE4_POLICY_ID={spec['policy_id']} "
                "ACL2_V108_STAGE4_SURFACE_ID=AB "
                f"ACL2_V116TF_TASK1_POLICY_ID={spec['policy_id']}"
            )
            prepare_command = (
                f"{prefix} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config.resolve()} --force"
            )
            prepare_rows_by_seq.setdefault(
                seq,
                {
                    "schema": "acl2_v116tf_task1_control_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v116tf_task1_control_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"task1_ab_control_fullseq_{seq}",
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
                        "schema": "acl2_v116tf_task1_control_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v116tf_task1_control_{spec['policy_id']}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"task1_ab_control_fullseq_{seq}",
                        "target_kind": "full_sequence",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": spec["policy_family"],
                        "control_component": spec["control_component"],
                        "control_mode": spec["control_mode"],
                        "stage4_action_mode": "anchor_scale_frame_indices" if combined else "force_non_keyframe",
                        "selector": selector_note,
                        "selected_count": len(scale_indices) + len(b1_indices),
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
    write_csv(OUT / "TASK1_CONTROL_RUN_MANIFEST.csv", manifest_rows)
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "control_policy_rows.csv", policy_rows)
    write_csv(OUT / "selected_control_frame_rows.csv", selected_rows)
    summary = {
        "schema": "acl2_v116tf_task1_control_config_summary_v1",
        "sequences": list(SEQUENCES),
        "policy_count": len(control_specs()),
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "prepare_rows": len(prepare_rows_by_seq),
        "run_worker_rows": len([r for r in manifest_rows if r["phase"] == "run_worker"]),
        "workspace": rel(WORKSPACE),
        "raw_action": rel(RAW_ACTION),
        "config_root": rel(CONFIG_ROOT),
        "control_contract": "A1 random/shuffle/role controls keep B1 fixed; B1 shuffle/same-count/reverse controls keep A1 absent and exact-match B1 final count.",
    }
    write_json(OUT / "TASK1_CONTROL_CONFIG_SUMMARY.json", summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
