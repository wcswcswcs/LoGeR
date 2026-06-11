#!/usr/bin/env python3
"""Implementation self-audit for ACL2 v23 semantic all-memory plumbing.

The script is intentionally conservative: it reports a failed or pending gate
when it cannot prove a path is wired.  It never infers performance success from
static code checks alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _check_contains(name: str, text: str, needles: List[str]) -> Tuple[bool, List[str]]:
    missing = [needle for needle in needles if needle not in text]
    return not missing, missing


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    return value


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _jsonl_rows(path: Path) -> List[Dict[str, object]]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--results-root",
        default="results/kitti01_hmc_v2/acl2_v23_semanticprior_allmemory_durable_target25",
    )
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
        "run_pipeline": repo / "run_pipeline_abc_v2.py",
        "run_attention": repo / "tools/run_attention_cue_experiment.sh",
        "run_v23_candidate": repo / "tools/run_v23_candidate_rollout.sh",
        "run_v23_matrix": repo / "tools/run_v23_matrix.sh",
    }
    text = {name: _read(path) for name, path in files.items()}

    checks: List[Dict[str, object]] = []

    def add(name: str, passed: bool, detail: str, severity: str = "hard", missing: List[str] | None = None) -> None:
        checks.append({
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
            "missing": missing or [],
        })

    passed, missing = _check_contains(
        "PriorOutput semantic role fields",
        text["semantic_prior_generator"],
        ["V_sem_tok", "R_sem_tok", "R_sem_patch_flat", "SEMANTIC_ROLE_NEGATIVE_SHORT"],
    )
    add("semantic_prior_generator_outputs_role", passed, "PriorOutput carries v23 semantic value/role token streams.", missing=missing)

    passed, missing = _check_contains(
        "HMC role control fields",
        text["hybrid_memory_controller"],
        [
            "semantic_role_policy",
            "semantic_memory_paths",
            "_apply_semantic_role_policy",
            "frame_semantic_source_consumed",
            "swa_semantic_source_consumed",
            "ttt_semantic_role_consumed",
        ],
    )
    add("hmc_builds_and_routes_semantic_roles", passed, "HMC has role policy args, role refinement, and path-consumption debug.", missing=missing)

    passed, missing = _check_contains(
        "Model consumes semantic role source skip",
        text["pi3"],
        ["semantic_role_negative", "R_sem_tok", "semantic_role_negative_short_and_highD"],
    )
    add("model_consumes_semantic_role_for_source_skip", passed, "pi3 context-source skip can consume R_sem_tok.", missing=missing)

    passed, missing = _check_contains(
        "Pipeline CLI and summaries",
        text["run_pipeline"],
        [
            "--semantic_role_policy",
            "--semantic_memory_paths",
            "semantic_role_summary.jsonl",
            "semantic_memory_path_summary.jsonl",
            "semantic_role_highd_quantile",
        ],
    )
    add("pipeline_cli_and_debug_outputs", passed, "CLI forwards semantic role controls and writes audit summaries.", missing=missing)

    passed, missing = _check_contains(
        "Shell forwards role controls",
        text["run_attention"],
        ["SEMANTIC_ROLE_POLICY", "SEMANTIC_MEMORY_PATHS", "SEMANTIC_ROLE_ARGS", "--semantic_role_policy"],
    )
    add("shell_forwards_semantic_role_args", passed, "run_attention_cue_experiment forwards semantic role env to Python.", missing=missing)

    passed, missing = _check_contains(
        "v23 launcher stale protection",
        text["run_v23_candidate"],
        [".INVALID_RERUN_", "SEMANTIC_ROLE_POLICY", "SEMANTIC_MEMORY_PATHS", "semantic_role_negative"],
    )
    add("v23_launcher_present_and_stale_safe", passed, "v23 launcher exists, forwards role controls, and invalidates stale dirs.", missing=missing)

    passed, missing = _check_contains(
        "v23 matrix present",
        text["run_v23_matrix"],
        ["phase0", "phase2", "phase3", "run_v23_candidate_rollout.sh"],
    )
    add("v23_matrix_scheduler_present", passed, "v23 matrix scheduler has phase entry points.", missing=missing)

    dynamic_root = results / "rollouts"
    role_files = list(dynamic_root.glob("*/semantic_role_summary.jsonl")) if dynamic_root.exists() else []
    memory_files = list(dynamic_root.glob("*/semantic_memory_path_summary.jsonl")) if dynamic_root.exists() else []
    context_files = list(dynamic_root.glob("*/context_skip_summary.jsonl")) if dynamic_root.exists() else []
    role_rows = [row for path in role_files for row in _jsonl_rows(path)]
    memory_rows = [row for path in memory_files for row in _jsonl_rows(path)]
    role_nonempty = any(
        bool(row.get("semantic_role_available"))
        and int(row.get("token_count") or 0) > 0
        and bool(row.get("role_counts"))
        for row in role_rows
    )
    memory_nonempty = any(
        bool(row.get("semantic_group_role_metrics"))
        and bool(row.get("semantic_role_counts"))
        for row in memory_rows
    )
    add(
        "dynamic_semantic_role_summary_present",
        role_nonempty,
        "At least one v23 rollout wrote non-empty semantic_role_summary.jsonl with role counts.",
        severity="dynamic",
    )
    add(
        "dynamic_semantic_memory_path_summary_present",
        memory_nonempty,
        "At least one v23 rollout wrote non-empty semantic_memory_path_summary.jsonl group/role metrics.",
        severity="dynamic",
    )
    add(
        "dynamic_context_skip_summary_present",
        any(_count_jsonl(path) > 0 for path in context_files),
        "At least one v23 rollout wrote context_skip_summary.jsonl.",
        severity="dynamic",
    )

    hard_pass = all(bool(c["passed"]) for c in checks if c["severity"] == "hard")
    dynamic_pass = all(bool(c["passed"]) for c in checks if c["severity"] == "dynamic")
    all_pass = hard_pass and dynamic_pass
    summary = {
        "results_root": str(results),
        "out_dir": str(out_dir),
        "hard_static_gate_pass": hard_pass,
        "dynamic_smoke_gate_pass": dynamic_pass,
        "all_gate_pass": all_pass,
        "checks": checks,
    }

    (out_dir / "codex_self_check_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_jsonable) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "codex_self_check_failures.jsonl").open("w", encoding="utf-8") as handle:
        for check in checks:
            if not bool(check["passed"]):
                handle.write(json.dumps(check, ensure_ascii=False, sort_keys=True) + "\n")

    lines = [
        "# ACL2 v23 Codex Implementation Self-Audit",
        "",
        f"results_root: `{results}`",
        "",
        f"hard_static_gate_pass: `{str(hard_pass).lower()}`",
        f"dynamic_smoke_gate_pass: `{str(dynamic_pass).lower()}`",
        f"all_gate_pass: `{str(all_pass).lower()}`",
        "",
        "| Check | Severity | Pass | Detail |",
        "|---|---|---:|---|",
    ]
    for check in checks:
        lines.append(
            f"| `{check['name']}` | `{check['severity']}` | `{str(bool(check['passed'])).lower()}` | {check['detail']} |"
        )
    lines.append("")
    if not all_pass:
        lines.append("Gate result: performance matrix is not allowed until failed checks are fixed or completed.")
    else:
        lines.append("Gate result: v23 semantic role plumbing smoke is auditable.")
    (out_dir / "codex_self_check_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
