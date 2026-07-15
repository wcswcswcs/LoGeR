#!/usr/bin/env python3
"""Build ACL2 v119-TF LingBot B1 Stage0 fresh FlashInfer metrics."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_ROOT = RESULT_ROOT / "stage0_lingbot_b1_fresh_baselines"
WORKSPACE = RUN_ROOT / "workspace"
OUT_DIR = RUN_ROOT / "full_sequence_metrics"
DEFAULT_BASELINE_CSV = (
    RESULT_ROOT
    / "stage0_lingbot_fresh_baselines/full_sequence_metrics/stage0_lingbot_flashinfer_baseline_rows.csv"
)
BASE_METRIC_SCRIPT = ROOT / "tools/build_v119tf_lingbot_stage0_fresh_metrics.py"
SEQ_ORDER = ["00", "02"]
FORCED_INDICES = {
    "00": [668, 683, 3113, 3128, 3143, 3158, 3173],
    "02": [2813, 2843, 3818, 3833, 3848, 3863, 3893],
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                rows.append({"_json_error": str(exc), "_line_no": line_no})
    return rows


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def method_for(seq: str) -> str:
    return f"lingbot_map_v119_stage0_B1_semantic_only_flashinfer_{seq}"


def dataset_for(seq: str) -> str:
    return f"kitti_v119_stage0_b1_fullseq_{seq}"


def action_file_for(seq: str) -> Path:
    return RUN_ROOT / "raw_action" / f"{dataset_for(seq)}_{seq}_{method_for(seq)}.jsonl"


def load_metric_module():
    spec = importlib.util.spec_from_file_location("v119_lingbot_stage0_metrics", BASE_METRIC_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_METRIC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_lookup() -> dict[str, float]:
    out: dict[str, float] = {}
    for row in read_csv(DEFAULT_BASELINE_CSV):
        if row.get("reference_source") == "v119_fresh_flashinfer_baseline_rerun":
            out[row["seq"]] = float(row["full_ATE_sim3"])
    return out


def action_fidelity_row(seq: str) -> dict[str, Any]:
    action_file = action_file_for(seq)
    rows = load_jsonl(action_file)
    expected = set(FORCED_INDICES[seq])
    observed: set[int] = set()
    effective: set[int] = set()
    base_keyframes: set[int] = set()
    final_keyframes: set[int] = set()
    trace_error_rows = 0
    for row in rows:
        try:
            sample = int(float(row.get("sample_position", -1)))
        except (TypeError, ValueError):
            trace_error_rows += 1
            continue
        if "_json_error" in row:
            trace_error_rows += 1
            continue
        if boolish(row.get("forced_non_keyframe", False)):
            observed.add(sample)
        if boolish(row.get("base_is_keyframe", False)):
            base_keyframes.add(sample)
        if boolish(row.get("final_is_keyframe", False)):
            final_keyframes.add(sample)
        if (
            boolish(row.get("forced_non_keyframe", False))
            and boolish(row.get("base_is_keyframe", False))
            and not boolish(row.get("final_is_keyframe", True))
        ):
            effective.add(sample)

    missing = expected - observed
    unexpected = observed - expected
    ineffective = expected - effective
    passed = action_file.is_file() and observed == expected and effective == expected and trace_error_rows == 0
    return {
        "schema": "acl2_v119tf_lingbot_b1_stage0_fresh_action_fidelity_row_v1",
        "surface_id": "B",
        "policy_id": "B1_semantic_only",
        "policy_family": "semantic_only",
        "seq": seq,
        "dataset": dataset_for(seq),
        "method": method_for(seq),
        "action_name": "v119_stage0_B1_semantic_only_flashinfer",
        "stage4_action_mode": "force_non_keyframe",
        "expected_action_field": "forced_non_keyframe",
        "expected_action_frame_count": len(expected),
        "observed_action_frame_count": len(observed),
        "action_effective_frame_count": len(effective),
        "action_noop_frame_count": len(ineffective),
        "expected_keyframe_count": len(expected),
        "observed_keyframe_count": len(observed & base_keyframes),
        "expected_cache_append_count": len(expected),
        "observed_cache_append_count": len(effective),
        "trace_error_rows": trace_error_rows,
        "action_fidelity_pass": passed,
        "observed_action_indices": ";".join(str(x) for x in sorted(observed)),
        "effective_action_indices": ";".join(str(x) for x in sorted(effective)),
        "missing_expected_indices": ";".join(str(x) for x in sorted(missing)),
        "unexpected_observed_indices": ";".join(str(x) for x in sorted(unexpected)),
        "ineffective_expected_indices": ";".join(str(x) for x in sorted(ineffective)),
        "base_keyframe_count_observed_log": len(base_keyframes),
        "final_keyframe_count_observed_log": len(final_keyframes),
        "action_log_rows": len(rows),
        "action_file": rel(action_file),
        "action_file_exists": action_file.is_file(),
    }


def main() -> None:
    metric_module = load_metric_module()
    baselines = baseline_lookup()
    full_rows: list[dict[str, Any]] = []
    stage0_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []

    for seq in SEQ_ORDER:
        metric_module.WORKSPACE = WORKSPACE
        metric_module.OUT_DIR = OUT_DIR
        metric_module.DATASET = dataset_for(seq)
        metric_module.METHOD = method_for(seq)
        full_row, stage0_row, local = metric_module.summarize_seq(seq)
        fidelity = action_fidelity_row(seq)
        baseline_ate = baselines.get(seq)
        candidate_ate = float(stage0_row["full_ATE_sim3"])
        rel_improvement = (
            (baseline_ate - candidate_ate) / baseline_ate
            if baseline_ate is not None and baseline_ate != 0.0
            else ""
        )

        full_row.update(
            {
                "schema": "acl2_v119tf_lingbot_b1_stage0_fresh_full_metric_row_v1",
                "setting": "B1_semantic_only_flashinfer_force_non_keyframe",
                "baseline_full_ATE_sim3": baseline_ate if baseline_ate is not None else "",
                "full_ATE_rel_improvement": rel_improvement,
                "action_fidelity_pass": fidelity["action_fidelity_pass"],
                "action_file": fidelity["action_file"],
            }
        )
        stage0_row.update(
            {
                "schema": "acl2_v119tf_stage0_lingbot_b1_fresh_metric_row_v1",
                "reference_source": "v119_fresh_B1_semantic_only_flashinfer_rerun",
                "policy_id": "B1_semantic_only",
                "full_ATE_sim3": candidate_ate,
                "baseline_full_ATE_sim3": baseline_ate if baseline_ate is not None else "",
                "full_ATE_rel_improvement": rel_improvement,
                "action_fidelity_pass": fidelity["action_fidelity_pass"],
                "expected_action_frame_count": fidelity["expected_action_frame_count"],
                "observed_action_frame_count": fidelity["observed_action_frame_count"],
                "action_effective_frame_count": fidelity["action_effective_frame_count"],
                "action_file": fidelity["action_file"],
                "source_path": rel(OUT_DIR / "stage0_lingbot_b1_fresh_rows.csv"),
                "metric_scope_note": "fresh v119 FlashInfer B1 semantic-only carrier replacement; matched to v119 default FlashInfer baseline",
            }
        )

        full_rows.append(full_row)
        stage0_rows.append(stage0_row)
        local_rows.extend(local)
        fidelity_rows.append(fidelity)

    write_csv(OUT_DIR / "lingbot_b1_fresh_full_metrics.csv", full_rows)
    write_csv(OUT_DIR / "stage0_lingbot_b1_fresh_rows.csv", stage0_rows)
    write_csv(OUT_DIR / "action_fidelity_rows.csv", fidelity_rows)
    write_csv(OUT_DIR / "local_window_rows.csv", local_rows)
    summary = {
        "schema": "acl2_v119tf_lingbot_b1_stage0_fresh_metric_summary_v1",
        "sequences": SEQ_ORDER,
        "metric_row_count": len(full_rows),
        "completed_sequences": [
            row["seq"]
            for row in full_rows
            if row.get("pose_depth_available") and row.get("eval_available")
        ],
        "all_action_fidelity": bool(fidelity_rows)
        and len(fidelity_rows) == len(SEQ_ORDER)
        and all(bool(row["action_fidelity_pass"]) for row in fidelity_rows),
        "fresh_lingbot_b1_baseline_complete": len(full_rows) == len(SEQ_ORDER)
        and all(row.get("pose_depth_available") and row.get("eval_available") for row in full_rows)
        and all(bool(row["action_fidelity_pass"]) for row in fidelity_rows),
        "baseline_source": rel(DEFAULT_BASELINE_CSV),
        "stage0_rows": stage0_rows,
        "action_fidelity_rows": fidelity_rows,
        "truthfulness_boundary": "Fresh B1 replacement uses v119 FlashInfer backend; it is not a strict v110 SDPA replay.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stage0_lingbot_b1_fresh_metric_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
