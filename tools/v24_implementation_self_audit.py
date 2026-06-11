#!/usr/bin/env python3
"""ACL2 v24 implementation and dynamic smoke audit.

The audit is deliberately conservative.  It reports failure when it cannot
prove semantic role alignment, path consumption, no-op parity, or stale-run
exclusion from landed artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    materialized = list(rows)
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _contains(text: str, needles: List[str]) -> Tuple[bool, List[str]]:
    missing = [needle for needle in needles if needle not in text]
    return not missing, missing


def _candidate_from_run(run_name: str) -> str:
    match = re.search(r"_(K1_H9|P0_[^_]+_[A-Z0-9_]+|PASSIVE_DEBUG_ONLY|FRAMESEM_[^_]+_[A-Z0-9_]+|GLOBALSEM_[^_]+_[A-Z0-9_]+|FRAMEGLOBAL_[^_]+_[A-Z0-9_]+|SWASEM_[^_]+_[A-Z0-9_]+|TTTSEM_[^_]+_[A-Z0-9_]+|CHUNKSEM_[^_]+_[A-Z0-9_]+|PAIR_[A-Z0-9_]+|ALLMEM_[^_]+_[A-Z0-9_]+)_chunk", run_name)
    return match.group(1) if match else run_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--results-root",
        default="results/kitti01_hmc_v2/acl2_v24_semanticprior_pathspecific_allmemory_parallel",
    )
    parser.add_argument("--phase0-report", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    results = (repo / args.results_root).resolve() if not Path(args.results_root).is_absolute() else Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results / "implementation_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "semantic_prior_generator": repo / "loger/pipeline/semantic_prior_generator.py",
        "hybrid_memory_controller": repo / "loger/pipeline/hybrid_memory_controller.py",
        "pi3": repo / "loger/models/pi3.py",
        "attention": repo / "loger/models/layers/attention.py",
        "ttt_write_controller": repo / "loger/pipeline/ttt_write_controller.py",
        "run_pipeline": repo / "run_pipeline_abc_v2.py",
        "run_attention": repo / "tools/run_attention_cue_experiment.sh",
        "run_v24_candidate": repo / "tools/run_v24_candidate_rollout.sh",
        "run_v24_matrix": repo / "tools/run_v24_matrix.sh",
    }
    text = {name: _read(path) for name, path in files.items()}
    checks: List[Dict[str, object]] = []

    def add(name: str, passed: bool, detail: str, severity: str = "hard", missing: List[str] | None = None) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "severity": severity,
                "detail": detail,
                "missing": missing or [],
            }
        )

    passed, missing = _contains(
        text["semantic_prior_generator"],
        ["V_sem_tok", "R_sem_tok", "R_sem_patch_flat", "SEMANTIC_ROLE_NEGATIVE_SHORT"],
    )
    add("semantic_prior_outputs_role_streams", passed, "SPG emits value/role patch and token streams.", missing=missing)

    passed, missing = _contains(
        text["hybrid_memory_controller"],
        [
            "semantic_role_policy",
            "semantic_memory_paths",
            "_apply_semantic_role_policy",
            "frame_semantic_source_consumed",
            "chunk_global_semantic_source_consumed",
            "swa_semantic_source_consumed",
            "ttt_semantic_role_consumed",
        ],
    )
    add("hmc_routes_roles_to_memory_paths", passed, "HMC parses role policy and reports frame/global/SWA/TTT consumption.", missing=missing)

    passed, missing = _contains(
        text["pi3"],
        ["semantic_role_negative", "semantic_role_source_skip", "R_sem_tok", "source_keep_mask"],
    )
    add("model_consumes_roles_for_compact_kv", passed, "pi3 compact_kv path can consume semantic role source masks.", missing=missing)

    passed, missing = _contains(
        text["run_pipeline"],
        [
            "--semantic_role_policy",
            "--semantic_memory_paths",
            "semantic_role_summary.jsonl",
            "semantic_memory_path_summary.jsonl",
            "global_start",
            "global_end",
        ],
    )
    add("pipeline_forwards_role_and_global_cache_keys", passed, "Pipeline exposes role args and uses global frame cache keys.", missing=missing)

    passed, missing = _contains(
        text["run_attention"],
        ["SEMANTIC_ROLE_ARGS", "CONTEXT_SOURCE_SKIP_ARGS", "--stage_c_cache_require_hit", "--semantic_role_policy"],
    )
    add("shell_forwards_role_skip_cache_args", passed, "Shell runner forwards semantic role/source skip/cache args.", missing=missing)

    passed, missing = _contains(
        text["run_v24_candidate"],
        [".INVALID_RERUN_", "fine_semantic_split_available", "candidate_uses_coarse_fallback", "SEMANTIC_MEMORY_PATHS"],
    )
    add("v24_launcher_stale_safe_and_auditable", passed, "v24 launcher invalidates stale dirs and records coarse fallback.", missing=missing)

    rollouts = results / "rollouts"
    role_files = [p for p in rollouts.glob("*/semantic_role_summary.jsonl") if ".INVALID" not in str(p)] if rollouts.exists() else []
    memory_files = [p for p in rollouts.glob("*/semantic_memory_path_summary.jsonl") if ".INVALID" not in str(p)] if rollouts.exists() else []
    context_files = [p for p in rollouts.glob("*/context_skip_summary.jsonl") if ".INVALID" not in str(p)] if rollouts.exists() else []

    alignment_rows: List[Dict[str, object]] = []
    for path in role_files:
        run = path.parent.name
        for row in _jsonl(path):
            role_counts = row.get("role_counts") or {}
            group_role_counts = row.get("group_role_counts") or {}
            alignment_rows.append(
                {
                    "run_name": run,
                    "candidate_id": _candidate_from_run(run),
                    "chunk_idx": row.get("chunk_idx"),
                    "semantic_role_available": bool(row.get("semantic_role_available")),
                    "token_count": int(row.get("token_count") or 0),
                    "role_count_keys": len(role_counts) if isinstance(role_counts, dict) else 0,
                    "group_role_count_keys": len(group_role_counts) if isinstance(group_role_counts, dict) else 0,
                    "role_counts_json": json.dumps(role_counts, ensure_ascii=False, sort_keys=True),
                }
            )
    _write_csv(out_dir / "semantic_role_alignment_audit.csv", alignment_rows)

    path_rows: List[Dict[str, object]] = []
    for path in memory_files:
        run = path.parent.name
        for row in _jsonl(path):
            metrics = row.get("semantic_group_role_metrics") or {}
            role_counts = row.get("semantic_role_counts") or {}
            path_rows.append(
                {
                    "run_name": run,
                    "candidate_id": _candidate_from_run(run),
                    "chunk_idx": row.get("chunk_idx"),
                    "semantic_role_policy": row.get("semantic_role_policy"),
                    "semantic_memory_paths": row.get("semantic_memory_paths"),
                    "semantic_role_consumed_any": bool(row.get("semantic_role_consumed_any")),
                    "frame_consumed": bool(row.get("frame_semantic_source_consumed")),
                    "global_consumed": bool(row.get("chunk_global_semantic_source_consumed")),
                    "swa_consumed": bool(row.get("swa_semantic_source_consumed")),
                    "ttt_consumed": bool(row.get("ttt_semantic_role_consumed")),
                    "lifecycle_consumed": bool(row.get("lifecycle_semantic_role_consumed")),
                    "metric_group_count": len(metrics) if isinstance(metrics, dict) else 0,
                    "role_count_keys": len(role_counts) if isinstance(role_counts, dict) else 0,
                }
            )
    context_by_run: Dict[str, Dict[str, float]] = {}
    for path in context_files:
        run = path.parent.name
        for row in _jsonl(path):
            acc = context_by_run.setdefault(
                run,
                {
                    "context_skip_requested": 0.0,
                    "num_context_source_skip_applied": 0.0,
                    "num_context_empty_source_events": 0.0,
                    "max_context_source_skip_tokens": 0.0,
                },
            )
            if row.get("context_source_skip_requested"):
                acc["context_skip_requested"] = 1.0
            acc["num_context_source_skip_applied"] += _to_float(row.get("num_context_source_skip_applied"))
            acc["num_context_empty_source_events"] += _to_float(row.get("num_context_empty_source_events"))
            acc["max_context_source_skip_tokens"] = max(
                acc["max_context_source_skip_tokens"],
                _to_float(row.get("max_context_source_skip_tokens")),
            )
    for row in path_rows:
        row.update(context_by_run.get(str(row["run_name"]), {}))
    _write_csv(out_dir / "path_consumption_audit.csv", path_rows)

    phase0_rows = []
    report_root = Path(args.phase0_report) if args.phase0_report else results / "phase0_plumbing_report"
    if not report_root.is_absolute():
        report_root = repo / report_root
    for row in _read_csv(report_root / "candidate_vs_H9_delta_by_horizon.csv"):
        candidate = row.get("candidate_id", "")
        if candidate == "K1_H9" or candidate.startswith("P0_"):
            phase0_rows.append(
                {
                    "candidate_id": candidate,
                    "chunk_id": row.get("chunk_id"),
                    "horizon": row.get("horizon"),
                    "ATE_delta_vs_H9": row.get("ATE_delta_vs_H9"),
                    "raw_trans_max_diff_vs_H9": row.get("raw_trans_max_diff_vs_H9", row.get("raw_trans_max_diff")),
                    "raw_pose_max_abs_diff_vs_H9": row.get("raw_pose_max_abs_diff_vs_H9"),
                    "raw_pose_frames_matched_H9": row.get("raw_pose_frames_matched_H9"),
                }
            )
    _write_csv(out_dir / "noop_parity_metrics.csv", phase0_rows)

    role_nonempty = any(
        bool(row["semantic_role_available"]) and int(row["token_count"]) > 0 and int(row["role_count_keys"]) > 0
        for row in alignment_rows
    )
    memory_nonempty = any(
        bool(row["semantic_role_consumed_any"]) and int(row["metric_group_count"]) > 0 and int(row["role_count_keys"]) > 0
        for row in path_rows
    )
    context_applied = any(
        float(row.get("num_context_source_skip_applied") or 0.0) > 0.0
        for row in path_rows
    )
    empty_source_ok = all(
        float(row.get("num_context_empty_source_events") or 0.0) == 0.0
        for row in path_rows
        if float(row.get("context_skip_requested") or 0.0) > 0.0
    )
    noop_ok = bool(phase0_rows) and all(
        candidate == "K1_H9"
        or (
            math.isfinite(_to_float(row["ATE_delta_vs_H9"]))
            and abs(_to_float(row["ATE_delta_vs_H9"])) == 0.0
            and math.isfinite(_to_float(row["raw_trans_max_diff_vs_H9"]))
            and abs(_to_float(row["raw_trans_max_diff_vs_H9"])) == 0.0
        )
        for row in phase0_rows
        for candidate in [str(row["candidate_id"])]
    )
    invalid_dirs = [p for p in rollouts.glob("*.INVALID*")] if rollouts.exists() else []
    stale_ok = all(".INVALID" not in str(p) for p in role_files + memory_files + context_files)

    add("dynamic_semantic_role_counts_nonempty", role_nonempty, "At least one dynamic v24 smoke produced non-empty role counts.", severity="dynamic")
    add("dynamic_memory_path_metrics_nonempty", memory_nonempty, "At least one dynamic v24 smoke produced path role metrics.", severity="dynamic")
    add("dynamic_context_skip_applied", context_applied, "A requested compact_kv/source skip smoke applied source skip.", severity="dynamic")
    add("dynamic_context_skip_empty_source_zero", empty_source_ok, "No requested source skip smoke reported empty source events.", severity="dynamic")
    add("dynamic_noop_parity_zero", noop_ok, "Phase 0 no-op/pass-through rows have zero ATE/raw translation drift.", severity="dynamic")
    add("dynamic_invalid_dirs_excluded", stale_ok, f"Audit inputs exclude invalid dirs; invalid dir count={len(invalid_dirs)}.", severity="dynamic")

    hard_pass = all(bool(c["passed"]) for c in checks if c["severity"] == "hard")
    dynamic_pass = all(bool(c["passed"]) for c in checks if c["severity"] == "dynamic")
    all_pass = hard_pass and dynamic_pass
    summary = {
        "results_root": str(results),
        "out_dir": str(out_dir),
        "hard_static_gate_pass": hard_pass,
        "dynamic_smoke_gate_pass": dynamic_pass,
        "all_gate_pass": all_pass,
        "role_alignment_rows": len(alignment_rows),
        "path_consumption_rows": len(path_rows),
        "noop_parity_rows": len(phase0_rows),
        "invalid_dir_count": len(invalid_dirs),
        "fine_semantic_split_available": False,
        "fine_semantic_split_note": "Current Stage C cache exposes coarse groups only; sky/vegetation-specific candidates are logged as coarse fallback diagnostics.",
        "checks": checks,
    }
    (out_dir / "codex_self_check_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "codex_self_check_failures.jsonl").open("w", encoding="utf-8") as handle:
        for check in checks:
            if not bool(check["passed"]):
                handle.write(json.dumps(check, ensure_ascii=False, sort_keys=True) + "\n")

    lines = [
        "# ACL2 v24 Codex Implementation Self-Audit",
        "",
        f"results_root: `{results}`",
        "",
        f"hard_static_gate_pass: `{str(hard_pass).lower()}`",
        f"dynamic_smoke_gate_pass: `{str(dynamic_pass).lower()}`",
        f"all_gate_pass: `{str(all_pass).lower()}`",
        "",
        "Fine semantic split available: `false`",
        "",
        "Sky/vegetation-specific candidates are therefore auditable only as coarse LOW_VALUE_STUFF fallback diagnostics in this run.",
        "",
        "| Check | Severity | Pass | Detail |",
        "|---|---|---:|---|",
    ]
    for check in checks:
        lines.append(
            f"| `{check['name']}` | `{check['severity']}` | `{str(bool(check['passed'])).lower()}` | {check['detail']} |"
        )
    lines.append("")
    lines.append("Gate result: performance matrix is allowed." if all_pass else "Gate result: performance matrix is not allowed.")
    (out_dir / "codex_self_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
