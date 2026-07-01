#!/usr/bin/env python3
"""Audit and optionally run v81 Phase6 Type-B merge-boundary rescue.

The script is intentionally conservative: v80 merge/qscale/controller families
are treated as prior evidence, while a targeted smoke can be run only on
Type-B seq01 chunks that have real overlap-support tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


V81_ROOT = Path("results/kitti01_hmc_v2/acl2_v81tf_longwindow_semantic_three_memory_control/report_final")
V80_ROOT = Path("results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final")
PHASE1_ROWS = V81_ROOT / "phase1_long_window_cluster_bank/long_window_cluster_rows.csv"
PHASE3_ROWS = V81_ROOT / "phase3_selected_write_risk_rule/selected_write_risk_rows.csv"
PHASE4_ROWS = V81_ROOT / "phase4_read_swa_confirmation/read_swa_confirmation_rows.csv"
PHASE0_FAILED = V81_ROOT / "phase0_v80_evidence_lock/failed_action_family_matrix.csv"
DEFAULT_OUT_DIR = V81_ROOT / "phase6_merge_boundary_typeb_rescue"
DEFAULT_SUPPORT_DIR = V80_ROOT / "phase9_seq01_ref055_v80_error_semantic_support_maps"
DEFAULT_CHUNKS = (7, 8, 9, 10)
DEFAULT_CASES = (
    "native_no_swa",
    "overlap_outlier",
    "geometry_only",
    "overlap_outlier_random",
    "overlap_outlier_shuffled",
)
MERGE_SUMMARY_GLOBS = (
    "phase9_out4_merge_overlap_*/*summary*.json",
    "phase9_seq01_*merge*/*gate_summary.json",
    "phase9_seq01_*qscale*/*gate_summary.json",
    "phase9_seq01_*controller*/*gate_summary.json",
    "phase9_seq01_*proxy*/*gate_summary.json",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    clean[key] = value
            writer.writerow(clean)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        out = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _index(rows: Iterable[Mapping[str, str]], key: str) -> dict[str, Mapping[str, str]]:
    return {str(row.get(key, "")): row for row in rows if str(row.get(key, "")).strip()}


def _phase3_profiles(row: Mapping[str, str]) -> dict[str, bool]:
    return {
        key: _safe_bool(row.get(key))
        for key in (
            "R0_plan_strict",
            "R1_visual_cluster2_ratio_guard",
            "R2_direction_guarded_no_context",
            "R3_seq02_cluster_diagnostic_only",
        )
    }


def _build_typeb_rows(support_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p1_rows = _read_csv(PHASE1_ROWS)
    p3_by_window = _index(_read_csv(PHASE3_ROWS), "window_id")
    p4_by_window = _index(_read_csv(PHASE4_ROWS), "window_id")
    typeb_rows: list[dict[str, Any]] = []
    all_bad_rows: list[dict[str, Any]] = []
    for row in p1_rows:
        if str(row.get("case_type")) != "bad":
            continue
        window_id = str(row.get("window_id", ""))
        p3 = p3_by_window.get(window_id, {})
        p4 = p4_by_window.get(window_id, {})
        profiles = _phase3_profiles(p3)
        triggered = any(profiles.values())
        ratio = _safe_float(row.get("selected_low_support_ratio")) or 0.0
        cluster = _safe_int(row.get("continuous_low_support_cluster_len")) or 0
        future = _safe_float(row.get("downstream_future_consistency")) or 0.0
        scale_cv = _safe_float(row.get("window5_subchunk_scale_cv")) or 0.0
        radio_boundary = _safe_float(row.get("radio_boundary_mean"))
        read_swa_alignment = _safe_float(p4.get("read_swa_alignment"))
        center = _safe_int(row.get("center_chunk"))
        selected_not_dominant = bool(ratio < 0.50 or cluster < 3 or str(row.get("selected_minus_control_downstream_direction")) != "harmful")
        future_or_boundary_high = bool(
            future >= 5.0
            or scale_cv >= 0.10
            or (radio_boundary is not None and radio_boundary >= 0.70)
        )
        merge_residual_high = bool(read_swa_alignment is None or read_swa_alignment < 0.30)
        support_map = support_dir / f"chunk_{int(center or -1):03d}_swa_overlap_source_gate_layer_18.pt"
        reasons: list[str] = []
        if selected_not_dominant:
            reasons.append("selected_low_support_not_dominant")
        if not triggered:
            reasons.append("ttt_selected_write_rule_not_triggered")
        if future_or_boundary_high:
            reasons.append("future_or_boundary_high")
        if merge_residual_high:
            reasons.append("read_swa_alignment_low_or_missing")
        is_typeb = bool(selected_not_dominant and (not triggered) and future_or_boundary_high and merge_residual_high)
        out = {
            "window_id": window_id,
            "seq": row.get("seq"),
            "chunk_start": row.get("chunk_start"),
            "chunk_end": row.get("chunk_end"),
            "center_chunk": center,
            "J_long": _safe_float(row.get("J_long")),
            "downstream_future_consistency": future,
            "window5_subchunk_scale_cv": scale_cv,
            "radio_boundary_mean": radio_boundary,
            "selected_low_support_ratio": ratio,
            "continuous_low_support_cluster_len": cluster,
            "downstream_direction": row.get("selected_minus_control_downstream_direction"),
            "read_swa_alignment": read_swa_alignment,
            "phase3_any_risk_triggered": triggered,
            "phase3_profiles": profiles,
            "typeb_reasons": reasons,
            "is_typeb": is_typeb,
            "support_map": str(support_map),
            "support_map_exists": support_map.is_file(),
            "smoke_eligible": bool(is_typeb and str(row.get("seq")) == "01" and support_map.is_file()),
        }
        all_bad_rows.append(out)
        if is_typeb:
            typeb_rows.append(out)
    return typeb_rows, all_bad_rows


def _classify_family(path: Path, payload: Mapping[str, Any]) -> str:
    text = str(path).lower()
    candidate = str(payload.get("candidate", "")).lower()
    if "out4" in text or "outlier" in text:
        return "OUT4 merge overlap"
    if "qscale" in text or "radio_qscale" in candidate or "thingstuff_radio" in candidate:
        return "RADIO qscale merge"
    if "multiobjective" in text:
        return "multiobjective controller"
    if "safe_positive" in text:
        return "safe-positive controller"
    if "proxy" in text:
        return "proxy controller"
    if "radio_component" in text:
        return "RADIO component merge"
    return "merge/SWA other"


def _load_v80_merge_audit() -> list[dict[str, Any]]:
    failed_by_family = {row.get("family", ""): row for row in _read_csv(PHASE0_FAILED)}
    paths: set[Path] = set()
    for pattern in MERGE_SUMMARY_GLOBS:
        paths.update(V80_ROOT.glob(pattern))
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        family = _classify_family(path, payload)
        failed = failed_by_family.get(family, {})
        rows.append(
            {
                "family": family,
                "summary_path": str(path),
                "candidate": payload.get("candidate"),
                "chunks": payload.get("chunks"),
                "phaseE_gate_pass": payload.get("phaseE_gate_pass"),
                "phaseE_head_tail_pass": payload.get("phaseE_head_tail_pass"),
                "phaseE_overlap_pass": payload.get("phaseE_overlap_pass"),
                "head_tail_pass_count": payload.get("head_tail_pass_count"),
                "overlap_pass_count": payload.get("overlap_pass_count"),
                "head_tail_median_improvement_vs_baseline_ratio": payload.get("head_tail_median_improvement_vs_baseline_ratio"),
                "overlap_median_improvement_vs_baseline_ratio": payload.get("overlap_median_improvement_vs_baseline_ratio"),
                "phase0_v81_rule": failed.get("v81_rule", ""),
                "phase0_reason": failed.get("reason", ""),
            }
        )
    return rows


def _run(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(list(cmd), cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, check=False)
    return {"cmd": list(cmd), "returncode": int(proc.returncode), "log_path": str(log_path)}


def _build_run_command(args: argparse.Namespace, chunks: Sequence[int], smoke_root: Path) -> list[str]:
    return [
        str(args.python),
        "tools/run_v68_phaseE_merge_multichunk.py",
        "--output-root",
        str(smoke_root),
        "--chunks",
        ",".join(str(x) for x in chunks),
        "--cases",
        ",".join(DEFAULT_CASES),
        "--gpus",
        args.gpus,
        "--semantic-merge-overlap-support-dir",
        str(args.support_dir),
        "--semantic-merge-overlap-support-kind",
        "source_gate",
        "--semantic-merge-overlap-support-floor",
        str(args.support_floor),
        "--reject-worse-than-native-overlap",
        "--semantic-conf-min",
        str(args.semantic_conf_min),
        "--blend-alpha",
        str(args.blend_alpha),
        "--max-blend-log-scale-delta",
        str(args.max_blend_log_scale_delta),
        "--skip-existing",
    ]


def _build_eval_command(args: argparse.Namespace, chunks: Sequence[int], smoke_root: Path) -> list[str]:
    cmd = [
        str(args.python),
        "tools/evaluate_v68_phaseE_multichunk.py",
        "--root",
        str(smoke_root),
        "--chunks",
        ",".join(str(x) for x in chunks),
        "--candidate",
        "overlap_outlier",
        "--baseline",
        "native_no_swa",
    ]
    for run in DEFAULT_CASES:
        cmd.extend(["--run", run])
    for control in ("geometry_only", "overlap_outlier_random", "overlap_outlier_shuffled"):
        cmd.extend(["--control", control])
    cmd.extend(
        [
            "--out-json",
            str(smoke_root / "v81_typeb_overlap_outlier_gate_summary.json"),
            "--out-csv",
            str(smoke_root / "v81_typeb_overlap_outlier_decisions.csv"),
            "--out-rows-csv",
            str(smoke_root / "v81_typeb_overlap_outlier_run_metrics.csv"),
        ]
    )
    return cmd


def _expected_trajectories(smoke_root: Path, chunks: Sequence[int]) -> list[Path]:
    return [smoke_root / f"chunk{chunk:02d}" / case / "01.txt" for chunk in chunks for case in DEFAULT_CASES]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_report(out_dir: Path, summary: Mapping[str, Any], typeb_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# ACL2 v81 Phase6 Type-B Merge/SWA Rescue",
        "",
        f"Gate pass: `{summary.get('gate_pass')}`",
        f"Decision: `{summary.get('decision')}`",
        "",
        "## Counts",
        "",
        f"- bad rows audited: {summary.get('bad_rows_audited')}",
        f"- type-B rows: {summary.get('typeb_rows')}",
        f"- smoke eligible rows: {summary.get('smoke_eligible_rows')}",
        f"- targeted smoke executed: {summary.get('targeted_smoke_executed')}",
        "",
        "## Type-B Rows",
        "",
    ]
    for row in typeb_rows:
        lines.append(
            "- {window_id}: center={center_chunk}, J_long={J_long}, future={downstream_future_consistency}, "
            "scale_cv={window5_subchunk_scale_cv}, radio_boundary={radio_boundary_mean}, "
            "support_map_exists={support_map_exists}".format(**row)
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Phase3/Phase4 blocked TTT selected-write action; Phase6 is evaluated as merge/SWA Type-B rescue only.",
            "- Existing v80 merge/qscale/controller families are audited separately to avoid silent forbidden repeats.",
            "- A full method pass requires good-pair protection and controls. If the targeted smoke lacks good-pair coverage, the method gate remains false.",
        ]
    )
    (out_dir / "typeb_rescue_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--support-dir", type=Path, default=DEFAULT_SUPPORT_DIR)
    parser.add_argument("--python", type=Path, default=Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--execute-smoke", action="store_true")
    parser.add_argument("--chunks", default=",".join(str(x) for x in DEFAULT_CHUNKS))
    parser.add_argument("--support-floor", type=float, default=0.25)
    parser.add_argument("--semantic-conf-min", type=float, default=0.05)
    parser.add_argument("--blend-alpha", type=float, default=0.60)
    parser.add_argument("--max-blend-log-scale-delta", type=float, default=0.07)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    typeb_rows, all_bad_rows = _build_typeb_rows(args.support_dir)
    merge_audit_rows = _load_v80_merge_audit()
    _write_csv(args.out_dir / "all_bad_typeb_audit_rows.csv", all_bad_rows)
    _write_csv(args.out_dir / "typeb_case_rows.csv", typeb_rows)
    _write_csv(args.out_dir / "existing_merge_family_audit.csv", merge_audit_rows)

    requested_chunks = [int(part.strip()) for part in str(args.chunks).split(",") if part.strip()]
    eligible_chunks = sorted({int(row["center_chunk"]) for row in typeb_rows if row.get("smoke_eligible")})
    chunks = [chunk for chunk in requested_chunks if chunk in eligible_chunks]
    smoke_root = args.out_dir / "targeted_overlap_outlier_smoke"
    run_result: dict[str, Any] | None = None
    eval_result: dict[str, Any] | None = None
    smoke_summary: dict[str, Any] = {}

    if args.execute_smoke and chunks:
        expected = _expected_trajectories(smoke_root, chunks)
        if expected and all(path.is_file() for path in expected):
            run_result = {
                "cmd": "reuse_existing_outputs",
                "returncode": 0,
                "reused_existing_outputs": True,
                "existing_trajectory_count": len(expected),
                "log_path": str(args.out_dir / "targeted_overlap_outlier_run.log"),
            }
        else:
            run_cmd = _build_run_command(args, chunks, smoke_root)
            run_result = _run(run_cmd, cwd=Path.cwd(), log_path=args.out_dir / "targeted_overlap_outlier_run.log")
        if int(run_result.get("returncode", 1)) == 0:
            eval_cmd = _build_eval_command(args, chunks, smoke_root)
            eval_result = _run(eval_cmd, cwd=Path.cwd(), log_path=args.out_dir / "targeted_overlap_outlier_eval.log")
            smoke_summary = _read_json(smoke_root / "v81_typeb_overlap_outlier_gate_summary.json")
    elif args.execute_smoke:
        run_result = {
            "returncode": None,
            "reason": "no_typeb_seq01_rows_with_support_maps_for_requested_chunks",
            "requested_chunks": requested_chunks,
            "eligible_chunks": eligible_chunks,
        }

    action_gate_pass = bool(smoke_summary.get("phaseE_gate_pass"))
    good_pair_coverage = False
    good_pair_note = "not_evaluated_in_targeted_seq01_bad_only_smoke"
    gate_pass = bool(action_gate_pass and good_pair_coverage)
    if not typeb_rows:
        decision = "no_typeb_rows_found_recheck_case_bank"
    elif args.execute_smoke and run_result and int(run_result.get("returncode", 1)) != 0:
        decision = "targeted_smoke_failed_runtime_or_missing_artifact"
    elif action_gate_pass and not good_pair_coverage:
        decision = "bad_case_signal_only_good_pair_gate_missing"
    elif action_gate_pass:
        decision = "unexpected_partial_action_signal_requires_good_pair_validation"
    else:
        decision = "typeb_rescue_no_method_gate_pass_needs_direct_merge_gauge_state_controller"

    summary: dict[str, Any] = {
        "gate_pass": gate_pass,
        "decision": decision,
        "bad_rows_audited": len(all_bad_rows),
        "typeb_rows": len(typeb_rows),
        "smoke_eligible_rows": sum(1 for row in typeb_rows if row.get("smoke_eligible")),
        "requested_chunks": requested_chunks,
        "eligible_chunks": eligible_chunks,
        "executed_chunks": chunks if args.execute_smoke else [],
        "targeted_smoke_executed": bool(args.execute_smoke and chunks and run_result and int(run_result.get("returncode", 1)) == 0),
        "targeted_smoke_root": str(smoke_root),
        "targeted_smoke_run_result": run_result,
        "targeted_smoke_eval_result": eval_result,
        "targeted_smoke_phaseE_gate_pass": smoke_summary.get("phaseE_gate_pass"),
        "targeted_smoke_head_tail_pass_count": smoke_summary.get("head_tail_pass_count"),
        "targeted_smoke_overlap_pass_count": smoke_summary.get("overlap_pass_count"),
        "targeted_smoke_head_tail_median_improvement_vs_baseline_ratio": smoke_summary.get("head_tail_median_improvement_vs_baseline_ratio"),
        "targeted_smoke_overlap_median_improvement_vs_baseline_ratio": smoke_summary.get("overlap_median_improvement_vs_baseline_ratio"),
        "good_pair_coverage": good_pair_coverage,
        "good_pair_note": good_pair_note,
        "support_dir": str(args.support_dir),
        "existing_merge_family_rows": len(merge_audit_rows),
        "phase6_rule": (
            "Type-B pass requires bad-pair J_mid or boundary/future improvement with controls and good-pair worsening <=2%; "
            "this wrapper keeps targeted bad-case smoke separate from full method gate."
        ),
    }
    _write_json(args.out_dir / "typeb_rescue_summary.json", summary)
    _write_report(args.out_dir, summary, typeb_rows)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
