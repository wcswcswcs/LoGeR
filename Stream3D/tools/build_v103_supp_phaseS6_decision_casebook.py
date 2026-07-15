#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_phaseS6_decision_casebook"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_phaseS6_decision_casebook_anchorcov_r1"
DEFAULT_S1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers_anchorcov_r1"
DEFAULT_S2_ROOT = AUDIT_ROOT / "v103_supp_phaseS2_role_aware_affinity_anchorcov_r1"
DEFAULT_S3_ROOT = AUDIT_ROOT / "v103_supp_phaseS3_scaffolded_mask_graph_anchorcov_directpair_c0001_r2"
DEFAULT_S4_ROOT = AUDIT_ROOT / "v103_supp_phaseS4_post_birth_history_inheritance_s3direct_v7_r1"
DEFAULT_S5_ROOT = AUDIT_ROOT / "v103_supp_phaseS5_dual_role_from_s1_anchorcov_r1"
DEFAULT_PHASE9E_ROOT = AUDIT_ROOT / "v103_phase9e_d4rt_anchor_da3_induced_carriers_anchorcov_s5repair_r1"
DEFAULT_PHASE9E_BASELINE_ROOT = (
    AUDIT_ROOT / "v103_phase9e_d4rt_anchor_da3_induced_carriers_suppS1_d4rt48mix_s5repair_r1"
)


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _decision(summary: dict[str, Any]) -> str:
    return str(summary.get("decision", summary.get("phase_decision", "")))


def _first(rows: list[dict[str, str]], **where: str) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in where.items()):
            return row
    return {}


def _root_summary_row(stage_id: str, root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_phaseS6_evidence_row_v1",
        "phase_id": PHASE_ID,
        "evidence_id": f"{stage_id}_summary",
        "stage_id": stage_id,
        "root": root,
        "exists": root.exists(),
        "decision": _decision(summary),
        "failure_count": summary.get("failure_count", ""),
        "uses_gt_for_prediction": summary.get("uses_gt_for_prediction", False),
        "uses_future": summary.get("uses_future", False),
        "note": "summary-level evidence only; this script does not recompute metrics",
    }


def _case_row(
    case_id: str,
    stage_id: str,
    finding: str,
    evidence: str,
    blocker: str,
    action: str,
) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_phaseS6_casebook_row_v1",
        "phase_id": PHASE_ID,
        "case_id": case_id,
        "stage_id": stage_id,
        "finding": finding,
        "evidence": evidence,
        "blocker": blocker,
        "recommended_action": action,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate v103 supplement S1-S5 evidence into a decision/casebook artifact."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phaseS1-root", default=str(DEFAULT_S1_ROOT))
    parser.add_argument("--phaseS2-root", default=str(DEFAULT_S2_ROOT))
    parser.add_argument("--phaseS3-root", default=str(DEFAULT_S3_ROOT))
    parser.add_argument("--phaseS4-root", default=str(DEFAULT_S4_ROOT))
    parser.add_argument("--phaseS5-root", default=str(DEFAULT_S5_ROOT))
    parser.add_argument("--phase9e-root", default=str(DEFAULT_PHASE9E_ROOT))
    parser.add_argument("--phase9e-baseline-root", default=str(DEFAULT_PHASE9E_BASELINE_ROOT))
    parser.add_argument("--local-delta-stop-min", type=float, default=0.002)
    parser.add_argument("--scene-control-delta-min", type=float, default=0.006)
    parser.add_argument(
        "--repair-family-id",
        default="anchorcov",
        help="Short label for the S1/S3/S4/Phase9e repair family being summarized.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    repair_family_id = str(args.repair_family_id)

    roots = {
        "phaseS1": _project(args.phaseS1_root),
        "phaseS2": _project(args.phaseS2_root),
        "phaseS3": _project(args.phaseS3_root),
        "phaseS4": _project(args.phaseS4_root),
        "phaseS5": _project(args.phaseS5_root),
        "phase9e": _project(args.phase9e_root),
        "phase9e_baseline": _project(args.phase9e_baseline_root),
    }
    summaries = {stage_id: _read_json(root / "summary.json") for stage_id, root in roots.items()}

    evidence_rows = [
        _root_summary_row(stage_id, root, summaries[stage_id])
        for stage_id, root in roots.items()
    ]

    s3_summary = summaries["phaseS3"]
    s4_summary = summaries["phaseS4"]
    p9e_summary = summaries["phase9e"]
    p9e_base_summary = summaries["phase9e_baseline"]

    s3_variant_rows = _read_csv(roots["phaseS3"] / "variant_metric_rows.csv")
    s3_control_rows = _read_csv(roots["phaseS3"] / "control_rows.csv")
    s4_gate_rows_in = _read_csv(roots["phaseS4"] / "gate_rows.csv")
    p9e_scene_rows = _read_csv(roots["phase9e"] / "scene_summary_rows.csv")
    p9e_base_scene_rows = _read_csv(roots["phase9e_baseline"] / "scene_summary_rows.csv")

    s3_best_delta = _num(s3_summary.get("best_minus_baseline_MV_AP_window", ""))
    s3_best_mv_ap = _num(s3_summary.get("best_MV_AP_window", ""))
    s3_best_ap50 = _num(s3_summary.get("best_MV_AP50_window", ""))
    s3_best_variant = str(s3_summary.get("best_variant_id", ""))
    s3_v7 = _first(s3_variant_rows, variant_id="S3_V7_direct_pair_rel070_support_veto")
    s3_v7_control = _first(s3_control_rows, variant_id="S3_V7_direct_pair_rel070_support_veto")
    s3_v8_control = _first(s3_control_rows, variant_id="S3_V8_anchor_or_direct_pair_rel070_veto")

    s4_real_minus_shuffled = _num(s4_summary.get("real_minus_shuffled_MV_AP_scene", ""))
    s4_real_minus_stale = _num(s4_summary.get("real_minus_stale_MV_AP_scene", ""))
    s4_real_minus_semantic = _num(s4_summary.get("real_minus_semantic_MV_AP_scene", ""))
    failed_s4_controls = [
        row.get("gate_id", "")
        for row in s4_gate_rows_in
        if str(row.get("severity", "")) == "control" and not _truth(row.get("pass", ""))
    ]

    p9e_clean = int(_num(p9e_summary.get("clean_induction_scene_count", ""), -1))
    p9e_base_clean = int(_num(p9e_base_summary.get("clean_induction_scene_count", ""), -1))
    p9e_pass = int(_num(p9e_summary.get("pass_scene_count", ""), -1))
    p9e_scene0050 = _first(p9e_scene_rows, scene_id="scene0050_00")
    p9e_base_scene0050 = _first(p9e_base_scene_rows, scene_id="scene0050_00")
    p9e_scene0011 = _first(p9e_scene_rows, scene_id="scene0011_00")

    s3_family_stop = s3_best_delta < args.local_delta_stop_min
    s4_control_bias = (
        s4_real_minus_shuffled < args.scene_control_delta_min
        or s4_real_minus_stale < args.scene_control_delta_min
        or s4_real_minus_semantic < 0.003
    )
    p9e_clean_regressed = p9e_clean < p9e_base_clean
    p9e_no_clean = p9e_clean <= 0
    method_no_go = s3_family_stop and s4_control_bias and p9e_no_clean

    gate_rows = [
        {
            "schema_version": "stream4d_v103_supp_phaseS6_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "s3_best_local_delta_meets_stop_rule",
            "pass": not s3_family_stop,
            "observed": s3_best_delta,
            "required": f">= {args.local_delta_stop_min}",
            "source_root": roots["phaseS3"],
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
        },
        {
            "schema_version": "stream4d_v103_supp_phaseS6_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "s4_real_beats_controls",
            "pass": not s4_control_bias,
            "observed": (
                f"real_minus_shuffled={s4_real_minus_shuffled}; "
                f"real_minus_stale={s4_real_minus_stale}; "
                f"real_minus_semantic={s4_real_minus_semantic}"
            ),
            "required": f"shuffled/stale >= {args.scene_control_delta_min}; semantic >= 0.003",
            "source_root": roots["phaseS4"],
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
        },
        {
            "schema_version": "stream4d_v103_supp_phaseS6_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": f"phase9e_{repair_family_id}_keeps_clean_induction",
            "pass": not p9e_clean_regressed and p9e_clean > 0,
            "observed": f"{repair_family_id}_clean={p9e_clean}; baseline_clean={p9e_base_clean}",
            "required": f"{repair_family_id} clean induction scene count >= baseline and > 0",
            "source_root": roots["phase9e"],
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
        {
            "schema_version": "stream4d_v103_supp_phaseS6_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "eligible_for_holdout",
            "pass": False,
            "observed": "dev repair gates did not pass",
            "required": "local safety + method gain + control attribution pass before holdout",
            "source_root": out,
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
        },
    ]

    case_rows = [
        _case_row(
            "s3_local_gain_too_small",
            "phaseS3",
            "Scaffolded mask graph did not clear the local method gain gate.",
            (
                f"best_variant={s3_best_variant}; MV_AP_window={s3_best_mv_ap}; "
                f"MV_AP50_window={s3_best_ap50}; delta={s3_best_delta}"
            ),
            "role-aware anchor/direct-pair evidence is real but too sparse for AP gate",
            "stop this S3 repair family; do not tune clustering eps to hide sparse evidence",
        ),
        _case_row(
            "s3_direct_pair_sparse",
            "phaseS3",
            "Direct-pair witness is clean enough for diagnostics but not strong enough as a method lever.",
            (
                f"V7_edges={s3_v7.get('added_anchor_edge_count', '')}; "
                f"V7_delta={s3_v7_control.get('real_minus_control_MV_AP_window', '')}; "
                f"V8_delta={s3_v8_control.get('real_minus_control_MV_AP_window', '')}"
            ),
            "direct-pair support exists but coverage/control delta is insufficient",
            "use direct-pair evidence as a casebook witness, not as a passed birth operator",
        ),
        _case_row(
            "s4_control_bias",
            "phaseS4",
            "Post-birth history inheritance failed attribution controls.",
            (
                f"real_minus_shuffled={s4_real_minus_shuffled}; "
                f"real_minus_stale={s4_real_minus_stale}; "
                f"real_minus_semantic={s4_real_minus_semantic}; "
                f"failed_control_gates={failed_s4_controls}"
            ),
            "history signal is not history-specific enough; stale/shuffled/semantic controls explain it",
            "do not freeze history inheritance; improve current object evidence before history id write",
        ),
        _case_row(
            f"phase9e_{repair_family_id}_clean_induction_regression",
            "phase9e",
            f"Phase9e {repair_family_id} repair did not improve DA3-induced reliable carrier expansion.",
            (
                f"{repair_family_id}_clean_scene_count={p9e_clean}; "
                f"baseline_clean_scene_count={p9e_base_clean}; "
                f"{repair_family_id}_pass_scene_count={p9e_pass}"
            ),
            "current repair family did not produce enough new reliable carrier support",
            "prioritize anchor purity/role selection or a genuinely better provider before adding more carrier volume",
        ),
        _case_row(
            "phase9e_scene0011_no_unanchored_induction",
            "phase9e",
            "scene0011 passes formal bridge but induces no unanchored mask observations.",
            (
                f"formal_bridge={p9e_scene0011.get('formal_bridge_gate_pass', '')}; "
                f"clean_induction={p9e_scene0011.get('clean_induction_gate_pass', '')}; "
                f"induced_unanchored={p9e_scene0011.get('best_induced_unanchored_mask_observation_count', '')}"
            ),
            "DA3 bridge score does not translate into new reliable current coverage for this scene",
            "treat DA3 as diagnostic until a provider path shows both bridge quality and induced coverage",
        ),
        _case_row(
            f"phase9e_scene0050_{repair_family_id}_clean_variant_check",
            "phase9e",
            f"scene0050 clean induction status after {repair_family_id} S1.",
            (
                f"old_clean_variant={p9e_base_scene0050.get('best_clean_variant_id', '')}; "
                f"old_clean_induced={p9e_base_scene0050.get('best_clean_induced_unanchored_mask_observation_count', '')}; "
                f"new_clean_variant={p9e_scene0050.get('best_clean_variant_id', '')}; "
                f"new_clean_induced={p9e_scene0050.get('best_clean_induced_unanchored_mask_observation_count', '')}"
            ),
            "scene0050 clean induction is the minimum sanity check for this repair family",
            "compare anchor purity distributions and induced coverage before adding more carrier volume",
        ),
    ]

    failure_rows: list[dict[str, Any]] = []
    for stage_id, root in roots.items():
        if not root.exists() or not (root / "summary.json").exists():
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_supp_phaseS6_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "stage_id": stage_id,
                    "failure_id": "missing_input_summary",
                    "root": root,
                    "uses_gt_for_prediction": False,
                }
            )

    if failure_rows:
        decision = "NO_GO_PHASES6_MISSING_INPUT"
    elif method_no_go:
        decision = "NO_GO_CURRENT_V103_SUPP_MULTIROLE_SCAFFOLD_DA3_REPAIR"
    else:
        decision = "PARTIAL_PHASES6_REVIEW_REQUIRED"

    failure_md = [
        "# v103 supplement PhaseS6 decision casebook",
        "",
        f"Decision: `{decision}`",
        "",
        "This artifact aggregates existing S1-S5 outputs only. It does not recompute AP and does not use GT for prediction.",
        "",
        "## Key evidence",
        "",
        f"- S3 best local delta: `{s3_best_delta}` from `{s3_best_variant}`.",
        (
            "- S4 controls: "
            f"`real_minus_shuffled={s4_real_minus_shuffled}`, "
            f"`real_minus_stale={s4_real_minus_stale}`, "
            f"`real_minus_semantic={s4_real_minus_semantic}`."
        ),
        f"- Phase9e {repair_family_id} clean induction scenes: `{p9e_clean}`; baseline clean induction scenes: `{p9e_base_clean}`.",
        "",
        "## Current stop reason",
        "",
        (
            "The current repair family should stop because the S3 local gain is below the stop-rule floor, "
            f"S4 history attribution is control-biased, and {repair_family_id} S5/Phase9e did not clear the clean DA3 induction gate."
        ),
        "",
        "## Allowed next actions",
        "",
        "- Do not claim method completion or holdout readiness.",
        "- Do not add more same-family S5 coverage variants without a new provider or anchor-purity diagnosis.",
        "- If continuing, use a new repair family: anchor purity attribution, provider replacement audit, or a DA3 primitive-provider path that still emits B_i,a -> z_i -> Phi_a.",
        "",
    ]
    (out / "failure_decomposition.md").write_text("\n".join(failure_md), encoding="utf-8")

    summary = {
        "schema_version": "stream4d_v103_supp_phaseS6_decision_casebook_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - started,
        "decision": decision,
        "repair_family_id": repair_family_id,
        "failure_count": len(failure_rows),
        "method_goal_achieved": False,
        "eligible_for_holdout": False,
        "repair_family_stop": method_no_go,
        "inputs": roots,
        "s3_best_variant_id": s3_best_variant,
        "s3_best_MV_AP_window": s3_best_mv_ap,
        "s3_best_MV_AP50_window": s3_best_ap50,
        "s3_best_minus_baseline_MV_AP_window": s3_best_delta,
        "s4_real_minus_shuffled_MV_AP_scene": s4_real_minus_shuffled,
        "s4_real_minus_stale_MV_AP_scene": s4_real_minus_stale,
        "s4_real_minus_semantic_MV_AP_scene": s4_real_minus_semantic,
        "phase9e_repair_family_clean_induction_scene_count": p9e_clean,
        "phase9e_repair_family_pass_scene_count": p9e_pass,
        "phase9e_anchorcov_clean_induction_scene_count": p9e_clean,
        "phase9e_baseline_clean_induction_scene_count": p9e_base_clean,
        "phase9e_anchorcov_pass_scene_count": p9e_pass,
        "outputs": {
            "evidence_rows": out / "evidence_rows.csv",
            "gate_rows": out / "gate_rows.csv",
            "casebook_rows": out / "casebook_rows.csv",
            "failure_rows": out / "failure_rows.csv",
            "failure_decomposition": out / "failure_decomposition.md",
            "last_command": out / "last_command.txt",
            "summary": out / "summary.json",
        },
        "truthfulness_note": (
            "This decision artifact aggregates previously generated diagnostics. "
            "It does not freeze a method config, does not run holdout, and does not claim AP success."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_gt_for_diagnostic_labels": True,
    }

    _write_csv(out / "evidence_rows.csv", evidence_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "casebook_rows.csv", case_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if decision.startswith("PARTIAL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
