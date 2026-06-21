#!/usr/bin/env python3
"""Summarize current ACL2 v70-v2 RADIO plan gate status from landed artifacts.

This tool is intentionally read-only.  It does not recompute metrics; it only
collects gate/decision fields from existing experiment summaries so the final
status can be audited without manually opening many JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_REPORT_ROOT = Path("results/kitti01_hmc_v2/acl2_v70_geometry_first_semantic_trust/report_final")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(value)


def _path_status(path: Path, data: Optional[Dict[str, Any]], fields: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": str(path),
        "exists": data is not None,
    }
    if data is None:
        out["status"] = "missing"
        return out
    for field in fields:
        if field in data:
            out[field] = data.get(field)
    return out


def _prefer_existing(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _merge_gate(root: Path) -> Dict[str, Any]:
    variants: List[Dict[str, Any]] = []
    merge_paths = sorted(
        {
            *root.glob("phaseR5_radio_merge_oracle*/radio_merge_oracle_summary.json"),
            *root.glob("phaseR5_radseg*merge_oracle*/radio_merge_oracle_summary.json"),
        }
    )
    for path in merge_paths:
        data = _read_json(path)
        if data is None:
            continue
        counts = data.get("counts") or {}
        candidate_counts = data.get("candidate_counts") or {}
        radio_gate_rows = int(data.get("radio_gate_rows") or counts.get("radio_gate_rows") or len(data.get("radio_gate_chunks") or []))
        control_gate_rows = int(data.get("control_gate_rows") or counts.get("control_gate_rows") or 0)
        baseline_gate_rows = int(data.get("baseline_gate_rows") or counts.get("baseline_gate_rows") or len(data.get("baseline_gate_chunks") or []))
        variants.append(
            {
                "path": str(path),
                "decision": data.get("decision"),
                "r5_merge_oracle_gate_pass": bool(data.get("r5_merge_oracle_gate_pass")),
                "r6_online_allowed_by_this_oracle": bool(data.get("r6_online_allowed_by_this_oracle")),
                "radio_gate_rows": radio_gate_rows,
                "control_gate_rows": control_gate_rows,
                "baseline_gate_rows": baseline_gate_rows,
                "radio_gate_chunks": data.get("radio_gate_chunks") or [],
                "radio_beats_controls_chunks": data.get("radio_beats_controls_chunks") or [],
                "median_best_radio_gate_mechanism_improvement": data.get("median_best_radio_gate_mechanism_improvement"),
                "rows": data.get("rows"),
                "missing_sidecar_chunks": data.get("missing_sidecar_chunks") or [],
                "candidate_counts": {
                    key: value
                    for key, value in candidate_counts.items()
                    if key.startswith("radio_") or key in {"geometry_only", "current_label_confidence_only"}
                },
            }
        )
    if not variants:
        status = "missing"
    elif any(bool(v.get("r5_merge_oracle_gate_pass")) for v in variants):
        status = "pass"
    elif all(int(v.get("radio_gate_rows") or 0) <= 0 for v in variants):
        status = "no_go_zero_radio_gate_rows"
    else:
        status = "no_go_partial_radio_gate"
    return {
        "status": status,
        "variant_count": len(variants),
        "total_radio_gate_rows": sum(int(v.get("radio_gate_rows") or 0) for v in variants),
        "total_control_gate_rows": sum(int(v.get("control_gate_rows") or 0) for v in variants),
        "total_baseline_gate_rows": sum(int(v.get("baseline_gate_rows") or 0) for v in variants),
        "all_radio_gate_chunks": sorted({int(c) for v in variants for c in (v.get("radio_gate_chunks") or [])}),
        "all_radio_beats_controls_chunks": sorted({int(c) for v in variants for c in (v.get("radio_beats_controls_chunks") or [])}),
        "variants": variants,
    }


def _read_online_gate(root: Path) -> Dict[str, Any]:
    paths = [
        root / "phaseR5_radio_read_online_smoke_r3/radio_read_online_smoke_summary.json",
        root / "phaseR5_radio_read_online_smoke_r4_geomblend/radio_read_online_smoke_summary.json",
        root / "phaseR5_radio_read_online_smoke_r5_pair_geomblend/radio_read_online_smoke_summary.json",
        root / "phaseR5_radio_read_online_smoke_r6_crossrisk_key/radio_read_online_smoke_summary.json",
    ]
    rows: List[Dict[str, Any]] = []
    for path in paths:
        data = _read_json(path)
        item = _path_status(path, data, ("read_online_gate_pass", "candidate_pass_chunks", "candidate_hook_active_chunks", "failed_jobs", "rows"))
        rows.append(item)
    any_pass = any(_bool(row.get("read_online_gate_pass")) for row in rows if row.get("exists"))
    return {"status": "pass" if any_pass else ("missing" if not any(row.get("exists") for row in rows) else "no_go"), "runs": rows}


def _swa_online_gate(root: Path) -> Dict[str, Any]:
    paths = [
        root / "phaseR5_radio_swa_online_smoke_r1_serial/radio_swa_online_smoke_summary.json",
        root / "phaseR5_radio_swa_online_smoke_r2_twochunk_full/radio_swa_online_smoke_summary.json",
        root / "phaseR5_radio_swa_online_smoke_r2_twochunk_hookcheck/radio_swa_online_smoke_summary.json",
        root / "phaseR5_radio_swa_online_smoke_r3_current_candidate/radio_swa_online_smoke_summary.json",
        root / "phaseR5_radio_swa_online_smoke_r4_intersection_candidate/radio_swa_online_smoke_summary.json",
        root / "phaseR5_radio_swa_online_smoke_r5_gate_intersection_candidate/radio_swa_online_smoke_summary.json",
        root / "phaseR5_radio_swa_online_smoke_r6_gate_current_candidate/radio_swa_online_smoke_summary.json",
        root / "phaseR5_radio_swa_online_smoke_r7_gate_current_strong_candidate/radio_swa_online_smoke_summary.json",
    ]
    rows: List[Dict[str, Any]] = []
    for path in paths:
        data = _read_json(path)
        rows.append(_path_status(path, data, ("swa_online_gate_pass", "candidate_pass_chunks", "candidate_hook_active_chunks", "failed_jobs", "rows")))
    any_pass = any(_bool(row.get("swa_online_gate_pass")) for row in rows if row.get("exists"))
    return {"status": "pass" if any_pass else ("missing" if not any(row.get("exists") for row in rows) else "no_go"), "runs": rows}


def _ttt_gate(root: Path) -> Dict[str, Any]:
    diag_path = root / "phaseR5_radio_ttt_spatial_delta_diagnostic_v1/radio_ttt_spatial_delta_summary.json"
    smoke_path = root / "phaseR5_radio_ttt_online_smoke_r3_cross_suppress_sweep_chunk08_10/phaseR5_cross_suppress_sweep_metrics.json"
    diag = _read_json(diag_path)
    smoke = _read_json(smoke_path)
    smoke_decisions = (smoke or {}).get("decisions") or {}
    any_smoke_pass = any(_bool(dec.get("phaseD_gate_pass")) for dec in smoke_decisions.values())
    return {
        "status": "no_go" if diag or smoke else "missing",
        "diagnostic": _path_status(
            diag_path,
            diag,
            (
                "decision",
                "rows",
                "missing_spatial_chunks",
                "missing_sidecar_chunks",
                "native_action_cosine_map_status",
                "official_ttt_online_gate_evaluated",
                "r6_online_allowed_by_this_diagnostic",
                "median_radio_prior_group_change_abs",
                "median_radio_post_delta_group_change_abs",
            ),
        ),
        "online_smoke": {
            **_path_status(smoke_path, smoke, ("decision",)),
            "any_phaseD_gate_pass": bool(any_smoke_pass),
            "decisions": smoke_decisions,
        },
    }


def build_summary(root: Path, plan_path: Path) -> Dict[str, Any]:
    r2_path = _prefer_existing(
        [
            root / "phaseR2_radseg_loger_alignment_slide336_stride224/radio_alignment_gate_summary.json",
            root / "phaseR2_radio_loger_alignment/radio_alignment_gate_summary.json",
        ]
    )
    attention_path = root / "phaseR5_radio_attention_oracle/radio_attention_oracle_summary.json"
    read_oracle_path = root / "phaseR5_radio_read_oracle_fulltoken_spotcheck/radio_read_oracle_summary.json"
    swa_oracle_path = _prefer_existing(
        [
            root / "phaseR5_radseg_crossframe_localmatch_swa_oracle_v5_featureonly_cos050/radio_swa_oracle_summary.json",
            root / "phaseR5_radseg_crossframe_localmatch_swa_oracle_v4_featureonly_stronger/radio_swa_oracle_summary.json",
            root / "phaseR5_radseg_crossframe_localmatch_swa_oracle_v3_featurelabel_stronger/radio_swa_oracle_summary.json",
            root / "phaseR5_radseg_overlap_swa_oracle_v2_stronger/radio_swa_oracle_summary.json",
            root / "phaseR5_radio_swa_oracle_v2_stronger/radio_swa_oracle_summary.json",
        ]
    )

    r2 = _read_json(r2_path)
    attention = _read_json(attention_path)
    read_oracle = _read_json(read_oracle_path)
    swa_oracle = _read_json(swa_oracle_path)
    merge = _merge_gate(root)
    read_online = _read_online_gate(root)
    swa_online = _swa_online_gate(root)
    ttt = _ttt_gate(root)

    r5_action_pass = any(
        [
            _bool((read_oracle or {}).get("r5_read_oracle_gate_pass")),
            _bool((swa_oracle or {}).get("r5_swa_oracle_gate_pass")),
            merge.get("status") == "pass",
            _bool(((ttt.get("diagnostic") or {}).get("r6_online_allowed_by_this_diagnostic"))),
        ]
    )
    online_pass = any([read_online.get("status") == "pass", swa_online.get("status") == "pass", (ttt.get("online_smoke") or {}).get("any_phaseD_gate_pass")])
    method_success = False

    blockers: List[str] = []
    if not r5_action_pass:
        blockers.append("R5 action/method gate has no passing RADIO path.")
    if merge.get("status") == "no_go_zero_radio_gate_rows":
        blockers.append("MERGE RADIO variants have zero radio_gate_rows across available oracle variants.")
    elif merge.get("status") == "no_go_partial_radio_gate":
        blockers.append(
            "MERGE has partial RADIO gate rows but fails the R5 pass rule "
            "(needs >=4 chunks and RADIO beating controls)."
        )
    if _bool((read_oracle or {}).get("read_attention_proxy_gate_pass")) and not _bool((read_oracle or {}).get("r5_read_oracle_gate_pass")):
        blockers.append("READ has proxy attention pass but no trajectory/future mechanism gate.")
    if _bool((swa_oracle or {}).get("swa_residual_proxy_gate_pass")) is False:
        blockers.append("SWA residual proxy does not satisfy the R5 gate.")
    if (ttt.get("diagnostic") or {}).get("native_action_cosine_map_status") == "projection_only_from_layer_branch_scalar_cosines":
        blockers.append("TTT lacks raw spatial native_action_cosine_map; current TTT evidence is diagnostic/projection-only.")
    if not online_pass:
        blockers.append("No online READ/SWA/TTT smoke gate passes.")

    summary = {
        "schema": "acl2_v70_v2_plan_gate_status_v1",
        "plan": str(plan_path),
        "report_root": str(root),
        "r2_radio_alignment": _path_status(
            r2_path,
            r2,
            (
                "gate_pass",
                "median_radio_boundary_contrast",
                "median_radio_minus_label_boundary_contrast",
                "radio_beats_label_contrast_any_chunk",
                "pass_chunks",
            ),
        ),
        "r5_attention_proxy": _path_status(
            attention_path,
            attention,
            ("gate_pass", "control_proxy_rows_pass", "control_proxy_rows_count"),
        ),
        "r5_read_oracle": _path_status(
            read_oracle_path,
            read_oracle,
            (
                "decision",
                "read_attention_proxy_gate_pass",
                "r5_read_oracle_gate_pass",
                "r6_online_allowed_by_this_oracle",
                "radio_beats_controls_chunks",
                "median_best_radio_proxy_score",
            ),
        ),
        "r5_swa_oracle": _path_status(
            swa_oracle_path,
            swa_oracle,
            (
                "decision",
                "swa_residual_proxy_gate_pass",
                "r5_swa_oracle_gate_pass",
                "r6_online_allowed_by_this_oracle",
                "radio_beats_controls_chunks",
                "median_best_radio_proxy_improvement",
            ),
        ),
        "r5_merge_oracles": merge,
        "read_online_smokes": read_online,
        "swa_online_smokes": swa_online,
        "ttt": ttt,
        "aggregated_decision": {
            "r5_action_or_method_gate_pass": bool(r5_action_pass),
            "online_smoke_gate_pass": bool(online_pass),
            "method_success": bool(method_success),
            "official_704f_or_full_allowed": False,
            "overall_status": "not_achieved",
            "remaining_plan_backed_blockers": blockers,
        },
    }
    return summary


def _write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    dec = summary["aggregated_decision"]
    lines = [
        "# ACL2 v70-v2 Plan Gate Status Audit",
        "",
        f"- Plan: `{summary['plan']}`",
        f"- Report root: `{summary['report_root']}`",
        f"- Overall status: `{dec['overall_status']}`",
        f"- R5 action/method gate pass: `{dec['r5_action_or_method_gate_pass']}`",
        f"- Online smoke gate pass: `{dec['online_smoke_gate_pass']}`",
        f"- Method success: `{dec['method_success']}`",
        f"- Official 704F/full allowed: `{dec['official_704f_or_full_allowed']}`",
        "",
        "## Gate Summary",
        "",
        f"- R2 RADIO alignment gate: `{summary['r2_radio_alignment'].get('gate_pass')}`",
        f"- R5 attention proxy gate: `{summary['r5_attention_proxy'].get('gate_pass')}` (diagnostic proxy only)",
        f"- R5 READ oracle gate: `{summary['r5_read_oracle'].get('r5_read_oracle_gate_pass')}`",
        f"- R5 SWA oracle gate: `{summary['r5_swa_oracle'].get('r5_swa_oracle_gate_pass')}`",
        f"- R5 MERGE oracle status: `{summary['r5_merge_oracles'].get('status')}`",
        f"- READ online smoke status: `{summary['read_online_smokes'].get('status')}`",
        f"- SWA online smoke status: `{summary['swa_online_smokes'].get('status')}`",
        f"- TTT status: `{summary['ttt'].get('status')}`",
        "",
        "## Remaining Blockers",
        "",
    ]
    for blocker in dec.get("remaining_plan_backed_blockers", []):
        lines.append(f"- {blocker}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--plan", type=Path, default=Path("docs/ACL2_v70_v2_GeometryFirst_RADIO_Sidecar_AttentionCorrection_Plan.md"))
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    out_json = args.out_json or args.report_root / "v70_plan_gate_status_audit.json"
    out_md = args.out_md or args.report_root / "v70_plan_gate_status_audit.md"
    summary = build_summary(args.report_root, args.plan)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(out_md, summary)
    print(json.dumps(summary["aggregated_decision"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={out_json}")
    print(f"wrote_md={out_md}")


if __name__ == "__main__":
    main()
