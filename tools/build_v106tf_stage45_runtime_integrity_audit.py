#!/usr/bin/env python3
"""Audit ACL2 v106 Stage4/Stage5 runtime result integrity."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106 = ROOT / "results/acl2_v106tf_lingbot_semantic_aware_memory_role_control"
BASELINE_STAGE = V105 / "stage2_gca_trace"
BASELINE_METHOD = "lingbot_map_stream_default_stage2_notrace"
STAGE4 = V106 / "stage4_local_preserve_reference_block"
STAGE5 = V106 / "stage5_query_head_local_minimization"
OUT = V106 / "stage45_runtime_integrity_audit"


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


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_map(raw: str) -> dict[int, list[int]]:
    if not raw:
        return {}
    data = json.loads(raw)
    return {int(frame): [int(head) for head in heads] for frame, heads in data.items()}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_traj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames: list[int] = []
    mats: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 13:
                raise ValueError(f"bad trajectory row in {path}: {line[:120]}")
            frames.append(int(vals[0]))
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = np.asarray(vals[1:], dtype=np.float64).reshape(3, 4)
            mats.append(mat)
    return np.asarray(frames, dtype=np.int64), np.stack(mats, axis=0)


def traj_max_abs_diff(lhs: Path, rhs: Path) -> tuple[bool, float, str]:
    if not lhs.exists() or not rhs.exists():
        return False, float("nan"), "missing_traj"
    lhs_frames, lhs_mats = load_traj(lhs)
    rhs_frames, rhs_mats = load_traj(rhs)
    if not np.array_equal(lhs_frames, rhs_frames):
        return False, float("nan"), "frame_mismatch"
    return True, float(np.max(np.abs(lhs_mats - rhs_mats))), ""


def manifest_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["seq"], row["action_name"], row["method"], row["phase"])


def observed_pairs(headlocal_rows: list[dict[str, Any]]) -> tuple[set[int], int]:
    frames = {int(row["sample_position"]) for row in headlocal_rows}
    pairs = 0
    for row in headlocal_rows:
        raw = str(row.get("headlocal_action_heads", ""))
        if raw:
            pairs += len([item for item in raw.split(",") if item.strip()])
    return frames, pairs


def audit_stage(stage_name: str, stage_dir: Path, summary_file: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_rows = read_csv(stage_dir / "action_config_rows.csv")
    manifest_rows = read_csv(stage_dir / "run_manifest.csv")
    result_rows = read_csv(stage_dir / "run_results.csv")
    summary = json.loads((stage_dir / summary_file).read_text(encoding="utf-8"))

    manifest_keys = Counter(manifest_key(row) for row in manifest_rows)
    result_keys = Counter(manifest_key(row) for row in result_rows)
    missing_results = sorted(manifest_keys - result_keys)
    extra_results = sorted(result_keys - manifest_keys)
    duplicate_results = sorted(key for key, count in result_keys.items() if count > manifest_keys.get(key, 0))
    nonzero_results = [row for row in result_rows if int(row.get("returncode", 1)) != 0]

    detail_rows: list[dict[str, Any]] = []
    for cfg in config_rows:
        seq = cfg["seq"]
        dataset = cfg["dataset"]
        method = cfg["method"]
        action_name = cfg["action_name"]
        action_map = parse_map(cfg.get("head_action_map_json", ""))
        expected_frames = set(action_map)
        expected_pairs = sum(len(heads) for heads in action_map.values())
        traj = stage_dir / f"workspace/{dataset}/{seq}/{method}/traj.txt"
        trace_file = Path(cfg["trace_file"])
        action_file = Path(cfg["action_file"])
        action_rows = load_jsonl(action_file)
        trace_rows = load_jsonl(trace_file)
        headlocal_rows = [row for row in action_rows if row.get("headlocal_action_enabled")]
        frames, pairs = observed_pairs(headlocal_rows)
        trace_error_rows = [row for row in trace_rows if row.get("row_type") == "trace_error"]
        kv_rows = [row for row in trace_rows if row.get("row_type") == "kv_cache_provenance"]
        if action_name == "no_action":
            action_fidelity_pass = not headlocal_rows and expected_pairs == 0
        else:
            action_fidelity_pass = frames == expected_frames and pairs == expected_pairs
        baseline_traj = BASELINE_STAGE / f"workspace/{dataset}/{seq}/{BASELINE_METHOD}/traj.txt"
        no_action_parity_ok = ""
        no_action_max_abs_pose_diff = ""
        no_action_parity_error = ""
        if action_name == "no_action":
            ok, max_diff, error = traj_max_abs_diff(traj, baseline_traj)
            no_action_parity_ok = bool(ok and max_diff <= 1e-9)
            no_action_max_abs_pose_diff = max_diff
            no_action_parity_error = error
        detail_rows.append(
            {
                "schema": "acl2_v106tf_stage45_runtime_integrity_detail_v1",
                "stage": stage_name,
                "seq": seq,
                "action_name": action_name,
                "method": method,
                "stage_action_mode": cfg.get("stage4_action_mode", cfg.get("stage5_action_mode", "")),
                "head_action_pair_count": expected_pairs,
                "traj_exists": traj.exists(),
                "trace_file_exists": trace_file.exists(),
                "action_file_exists": action_file.exists(),
                "expected_frame_count": len(expected_frames),
                "observed_frame_count": len(frames),
                "observed_pair_count": pairs,
                "action_fidelity_pass": action_fidelity_pass,
                "trace_rows": len(trace_rows),
                "kv_cache_provenance_rows": len(kv_rows),
                "trace_error_rows": len(trace_error_rows),
                "trace_fidelity_basic_pass": traj.exists() and bool(kv_rows) and not trace_error_rows and action_fidelity_pass,
                "no_action_parity_ok": no_action_parity_ok,
                "no_action_max_abs_pose_diff": no_action_max_abs_pose_diff,
                "no_action_parity_error": no_action_parity_error,
            }
        )

    basic_pass = all(parse_bool(row["trace_fidelity_basic_pass"]) for row in detail_rows)
    no_action_rows = [row for row in detail_rows if row["action_name"] == "no_action"]
    no_action_parity_pass = all(row["no_action_parity_ok"] is True for row in no_action_rows)
    stage_summary = {
        "schema": "acl2_v106tf_stage45_runtime_integrity_stage_summary_v1",
        "stage": stage_name,
        "manifest_rows": len(manifest_rows),
        "run_result_rows": len(result_rows),
        "config_rows": len(config_rows),
        "missing_result_count": len(missing_results),
        "extra_result_count": len(extra_results),
        "duplicate_result_count": len(duplicate_results),
        "nonzero_returncode_count": len(nonzero_results),
        "detail_row_count": len(detail_rows),
        "basic_trace_fidelity_all_pass": basic_pass,
        "no_action_parity_pass": no_action_parity_pass,
        "no_action_max_abs_pose_diff_max": max(
            [float(row["no_action_max_abs_pose_diff"]) for row in no_action_rows if row["no_action_max_abs_pose_diff"] != ""],
            default=float("nan"),
        ),
        "summary_status": summary.get("stage4_status", summary.get("stage5_status", "")),
        "summary_pass": summary.get("stage4_action_pass", summary.get("stage5_minimization_pass", "")),
        "summary_blocker": summary.get("blocker", ""),
        "summary_run_result_rows": summary.get("run_result_rows", ""),
        "summary_run_failures": summary.get("run_failures", ""),
        "integrity_pass": (
            len(missing_results) == 0
            and len(extra_results) == 0
            and len(duplicate_results) == 0
            and len(nonzero_results) == 0
            and basic_pass
            and no_action_parity_pass
            and int(summary.get("run_result_rows", -1)) == len(result_rows)
            and int(summary.get("run_failures", -1)) == 0
        ),
        "missing_results": ["|".join(key) for key in missing_results],
        "extra_results": ["|".join(key) for key in extra_results],
        "duplicate_results": ["|".join(key) for key in duplicate_results],
    }
    return stage_summary, detail_rows


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    stage4_summary, stage4_details = audit_stage("stage4", STAGE4, "stage4_summary.json")
    stage5_summary, stage5_details = audit_stage("stage5", STAGE5, "stage5_summary.json")
    detail_rows = stage4_details + stage5_details
    write_csv(OUT / "stage45_runtime_integrity_detail_rows.csv", detail_rows)
    write_csv(OUT / "stage45_runtime_integrity_stage_rows.csv", [stage4_summary, stage5_summary])

    final_summary = {
        "schema": "acl2_v106tf_stage45_runtime_integrity_audit_summary_v1",
        "stage4": stage4_summary,
        "stage5": stage5_summary,
        "integrity_pass": bool(stage4_summary["integrity_pass"] and stage5_summary["integrity_pass"]),
        "no_success_claim": True,
        "conclusion": (
            "Runtime/manifests/traces/no-action parity are internally consistent; "
            "Stage4/Stage5 remain No-Go by measured gates."
        ),
        "outputs": {
            "detail_rows": (OUT / "stage45_runtime_integrity_detail_rows.csv").relative_to(ROOT).as_posix(),
            "stage_rows": (OUT / "stage45_runtime_integrity_stage_rows.csv").relative_to(ROOT).as_posix(),
            "summary": (OUT / "stage45_runtime_integrity_summary.json").relative_to(ROOT).as_posix(),
            "report": (OUT / "stage45_runtime_integrity_report.md").relative_to(ROOT).as_posix(),
        },
    }
    (OUT / "stage45_runtime_integrity_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Stage4/Stage5 Runtime Integrity Audit",
        "",
        "This audit checks manifest coverage, run returncodes, per-config action trace fidelity, "
        "trajectory existence, trace-error absence, KV provenance rows, and no-action trajectory parity.",
        "",
        f"- stage4_integrity_pass: `{stage4_summary['integrity_pass']}`",
        f"- stage4_status: `{stage4_summary['summary_status']}`",
        f"- stage4_blocker: `{stage4_summary['summary_blocker']}`",
        f"- stage4_no_action_max_abs_pose_diff_max: `{stage4_summary['no_action_max_abs_pose_diff_max']}`",
        f"- stage5_integrity_pass: `{stage5_summary['integrity_pass']}`",
        f"- stage5_status: `{stage5_summary['summary_status']}`",
        f"- stage5_blocker: `{stage5_summary['summary_blocker']}`",
        f"- stage5_no_action_max_abs_pose_diff_max: `{stage5_summary['no_action_max_abs_pose_diff_max']}`",
        f"- combined_integrity_pass: `{final_summary['integrity_pass']}`",
        "",
        "Conclusion: integrity checks do not overturn the No-Go result. No Stage7 full/rolling validation is permitted.",
        "",
    ]
    (OUT / "stage45_runtime_integrity_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return final_summary


if __name__ == "__main__":
    build()
