#!/usr/bin/env python3
"""Build the v79 semantic-goal lock artifacts.

This Phase-0 tool is audit-only.  It reads the v79 plan and v78 evidence logs,
then writes the semantic contribution schema, a candidate contract scaffold,
and a concise v78 evidence lock.  It does not run LoGeR, tune thresholds, or
claim semantic success.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONTRACT_FIELDS: list[tuple[str, str]] = [
    ("candidate_id", "Stable candidate identifier."),
    ("semantic_source", "dense_label_conf, thingstuff_state, RADIO_topology, LoGeR_internal_semgeo, combined, or none for controls."),
    ("semantic_role_definition", "Stable/harm/context/low-observability/regime-shift role definition used by the action."),
    ("memory_body", "Short READ/global/frame, mid SWA/overlap, long TTT write/update, or cross-memory handshake."),
    ("memory_timescale", "short, mid, long, or cross."),
    ("tap_layer_or_path", "Layer, tap, route, or memory path affected by the candidate."),
    ("action_point", "Where the memory behavior is changed."),
    ("actuator_family", "Read/source weighting, SWA Q/K/V route gate, TTT write/update gate, or handshake."),
    ("runtime_trigger_features", "No-GT features that trigger the action."),
    ("no_gt_runtime_features_only", "true only if no GT labels/poses select the action at runtime."),
    ("geometry_only_control", "Matching geometry-only control id or explicit none."),
    ("semantic_shuffle_controls", "Required label/conf/RADIO shuffle controls."),
    ("random_controls", "Required same-mass or group-stratified random controls."),
    ("expected_geometry_metric", "J_single, J_adj, J_5win, J_memory, or official ATE target."),
    ("action_fidelity_metric", "Mass/alignment/state-change evidence proving the intended memory behavior changed."),
    ("promotion_gate", "Plan gate required before promotion."),
]


SEMANTIC_ROLES: dict[str, str] = {
    "STABLE_GEOMETRY_EVIDENCE": "High semantic trust, static structure or object interior, and low geometry/motion/residual risk.",
    "HARMFUL_TRANSIENT_EVIDENCE": "Dynamic thing, sky-for-scale, lowtrust stuff, or boundary evidence plus high geometry/motion/residual risk.",
    "CONTEXT_ONLY_EVIDENCE": "Sky, far road/background, vegetation context; layout context only, not scale refresh.",
    "LOW_OBSERVABILITY_REGIME": "Stable evidence is low and context dominates, making scale/gauge evidence weak.",
    "REGIME_SHIFT_EVIDENCE": "Long-window shadow/exposure/corridor/road-edge/structure continuity changes.",
}


ACTION_FAMILIES: list[dict[str, str]] = [
    {
        "family_id": "READ",
        "memory_body": "global_attention/frame_attention/current_chunk_read",
        "memory_timescale": "short",
        "semantic_role_definition": "STABLE_GEOMETRY_EVIDENCE minus HARMFUL_TRANSIENT_EVIDENCE with CONTEXT_ONLY_EVIDENCE as a small floor.",
        "geometry_only_control": "READ6_GEOMETRY_ONLY_CONTROL",
        "candidate_examples": "READ1,READ2,READ3,READ4,READ5",
    },
    {
        "family_id": "SWA",
        "memory_body": "swa_cache/overlap_handoff/qkv_route",
        "memory_timescale": "mid",
        "semantic_role_definition": "Stable semantic evidence is carried; harmful transient evidence is vetoed; context has a floor.",
        "geometry_only_control": "SWA7_GEOMETRY_ONLY_ROUTE",
        "candidate_examples": "SWA1,SWA2,SWA3,SWA4,SWA5,SWA6",
    },
    {
        "family_id": "TTT",
        "memory_body": "ttt_fast_weight_write_update",
        "memory_timescale": "long",
        "semantic_role_definition": "Stable regimes persist; harmful/regime-shift evidence is short-lived or downweighted; context is neutral.",
        "geometry_only_control": "TTT8_GEOMETRY_ONLY_WRITE",
        "candidate_examples": "TTT1,TTT2,TTT3,TTT4,TTT5,TTT6,TTT7",
    },
    {
        "family_id": "HANDSHAKE",
        "memory_body": "read_swa_ttt_role_alignment",
        "memory_timescale": "cross",
        "semantic_role_definition": "The same stable/harm/context roles must align across READ, SWA, and TTT paths.",
        "geometry_only_control": "HS8_GEOMETRY_ONLY_HANDSHAKE",
        "candidate_examples": "HS4,HS5,HS6,HS7",
    },
]


CANDIDATE_ROWS: list[dict[str, str]] = [
    {
        "candidate_id": "READ1_L07_SEMANTIC_LAYOUT_SELECT",
        "semantic_source": "dense_label_conf + LoGeR_internal_semgeo",
        "semantic_role_definition": "STABLE_GEOMETRY_EVIDENCE,HARMFUL_TRANSIENT_EVIDENCE,CONTEXT_ONLY_EVIDENCE",
        "memory_body": "global_attention_key_layout",
        "memory_timescale": "short",
        "tap_layer_or_path": "Global-K L07",
        "action_point": "read/source layout selection",
        "actuator_family": "semantic read role mask",
        "runtime_trigger_features": "semantic role mass,D_geo,Gram-motion,overlap residual",
        "no_gt_runtime_features_only": "true",
        "geometry_only_control": "READ6_GEOMETRY_ONLY_CONTROL",
        "semantic_shuffle_controls": "READ7_LABEL_SHUFFLE,READ8_CONFIDENCE_SHUFFLE",
        "random_controls": "READ9_SAME_READ_MASS_RANDOM,READ10_GROUP_STRATIFIED_RANDOM",
        "expected_geometry_metric": "J_single,head_to_tail,scale_cv",
        "action_fidelity_metric": "stable/harm/context_read_mass_before_after,attention_entropy",
        "promotion_gate": "J_single>=5pct or head_to_tail/scale_cv>=10pct and beats controls",
    },
    {
        "candidate_id": "READ4_L07_TO_L13_SEMANTIC_CONTRAST",
        "semantic_source": "dense_label_conf + thingstuff_state + LoGeR_internal_semgeo",
        "semantic_role_definition": "L07 selects stable/harm/context; L13 protects stable and damps harm.",
        "memory_body": "global_attention_value_source",
        "memory_timescale": "short",
        "tap_layer_or_path": "Global-K L07 -> Global-V L13",
        "action_point": "source-side value protect/damp",
        "actuator_family": "semantic contrast read gate",
        "runtime_trigger_features": "semantic role mass,D_geo,Gram-motion,L07 layout strength,L13 value action strength",
        "no_gt_runtime_features_only": "true",
        "geometry_only_control": "READ6_GEOMETRY_ONLY_CONTROL",
        "semantic_shuffle_controls": "READ7_LABEL_SHUFFLE,READ8_CONFIDENCE_SHUFFLE",
        "random_controls": "READ9_SAME_READ_MASS_RANDOM,READ10_GROUP_STRATIFIED_RANDOM",
        "expected_geometry_metric": "J_single,head_to_tail,scale_cv",
        "action_fidelity_metric": "selected_semantic_composition,selected_D_geo_distribution,same-read-mass random composition",
        "promotion_gate": "J_single>=5pct or head_to_tail/scale_cv>=10pct and beats controls",
    },
    {
        "candidate_id": "SWA3_SEMANTIC_DUAL_GATE",
        "semantic_source": "dense_label_conf + RADIO_topology + LoGeR_internal_semgeo",
        "semantic_role_definition": "V-side stable protect plus K-side harmful veto with context floor.",
        "memory_body": "swa_qkv_overlap_handoff",
        "memory_timescale": "mid",
        "tap_layer_or_path": "SWA Q/K/V route, L26 diagnostic gate",
        "action_point": "SWA value protect and key risk veto",
        "actuator_family": "semantic dual gate",
        "runtime_trigger_features": "selected_source_quality,V_L26_selected_minus_random,K_L26_selected_minus_random,semantic agreement,RADIO same-object ratio",
        "no_gt_runtime_features_only": "true",
        "geometry_only_control": "SWA7_GEOMETRY_ONLY_ROUTE",
        "semantic_shuffle_controls": "SWA10_LABEL_CONF_SHUFFLE,SWA11_RADIO_SHUFFLE",
        "random_controls": "SWA8_HEAD_RANDOM_SAME_MASS,SWA9_ROUTE_RANDOM_SAME_MASS",
        "expected_geometry_metric": "J_adj,future_after_overlap,head_to_tail,scale_cv",
        "action_fidelity_metric": "SWA_keep_mass,SWA_replace_mass,SWA_gate_mass,stable/harm/context_route_mass",
        "promotion_gate": "J_adj>=5pct or future/head_tail/scale_cv>=10pct and beats controls",
    },
    {
        "candidate_id": "TTT6_REGIME_SHIFT_ONE_HOP",
        "semantic_source": "dense_label_conf + thingstuff_state + LoGeR_internal_semgeo",
        "semantic_role_definition": "REGIME_SHIFT_EVIDENCE is allowed for one hop, then decayed at next commit.",
        "memory_body": "ttt_fast_weight_write_update",
        "memory_timescale": "long",
        "tap_layer_or_path": "TTT write/update role mass and branch/layer update gate",
        "action_point": "TTT one-hop write persistence",
        "actuator_family": "semantic TTT write/update gate",
        "runtime_trigger_features": "regime shift score,low observability,TTT update conflict,post_zp_delta",
        "no_gt_runtime_features_only": "true",
        "geometry_only_control": "TTT8_GEOMETRY_ONLY_WRITE",
        "semantic_shuffle_controls": "TTT10_LABEL_CONF_SHUFFLE,TTT11_RADIO_SHUFFLE",
        "random_controls": "TTT9_SAME_WRITE_MASS_RANDOM",
        "expected_geometry_metric": "J_5win,window5_joint_sim3,subchunk_scale_cv,downstream_future",
        "action_fidelity_metric": "write_mass_role_before_after,post_zp_delta,next_probe_state_hash,branch_layer_update_mass",
        "promotion_gate": "intended write mass changes>=20pct, state changes, J_5win>=5pct or component>=10pct, beats controls",
    },
    {
        "candidate_id": "HS7_READ_SWA_TTT_FULL",
        "semantic_source": "combined",
        "semantic_role_definition": "Stable/harm/context roles align across READ, SWA, and TTT.",
        "memory_body": "read_swa_ttt_role_alignment",
        "memory_timescale": "cross",
        "tap_layer_or_path": "READ role masks, SWA carry/reject masks, TTT write roles",
        "action_point": "cross-memory semantic handshake",
        "actuator_family": "role alignment",
        "runtime_trigger_features": "READ-to-SWA,READ-to-TTT,SWA-to-TTT alignment",
        "no_gt_runtime_features_only": "true",
        "geometry_only_control": "HS8_GEOMETRY_ONLY_HANDSHAKE",
        "semantic_shuffle_controls": "label/conf/RADIO shuffles inherited from component paths",
        "random_controls": "HS9_RANDOM_ROLE_HANDSHAKE",
        "expected_geometry_metric": "J_memory",
        "action_fidelity_metric": "alignment score increases>=20pct",
        "promotion_gate": "J_memory>=5pct, beats best single path, beats controls",
    },
]


V78_EVIDENCE_MARKERS: list[tuple[str, str, str]] = [
    (
        "v78_no_success_claim",
        "v78 仍未达成",
        "v79 must not inherit a success claim from v78.",
    ),
    (
        "five_chunk_regime_shift",
        "long-window appearance / geometry regime shift",
        "Long-window TTT targets should include semantic/appearance/geometric regime shift diagnosis.",
    ),
    (
        "weak_state_energy_signal",
        "LW11",
        "State/delta direction can be a weak mechanism signal, but it is not semantic success.",
    ),
    (
        "tail_soft_no_promotion",
        "No-Go for LW13 / LW14 promotion",
        "Tail-risk soft directional variants did not pass held-out gates.",
    ),
    (
        "kv_centering_negative",
        "weak K/V 和 weak V-only 均未放大",
        "Do not repeat unconditional K/V or V-only frame-static centering as a semantic candidate.",
    ),
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _find_marker(path: Path, marker: str) -> dict[str, Any]:
    text = _read_text(path)
    if not text:
        return {"found": False, "line": None, "excerpt": "", "source_file": str(path)}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if marker in line:
            return {
                "found": True,
                "line": int(lineno),
                "excerpt": line.strip()[:240],
                "source_file": str(path),
            }
    return {"found": False, "line": None, "excerpt": "", "source_file": str(path)}


def _plan_has(plan_text: str, needle: str) -> bool:
    return needle in plan_text


def _build_v78_rows(v78_recap: Path, v78_execution_log: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evidence_id, marker, implication in V78_EVIDENCE_MARKERS:
        recap_hit = _find_marker(v78_recap, marker)
        log_hit = _find_marker(v78_execution_log, marker)
        hit = recap_hit if recap_hit["found"] else log_hit
        rows.append(
            {
                "evidence_id": evidence_id,
                "marker": marker,
                "found": bool(hit["found"]),
                "source_file": hit["source_file"],
                "line": hit["line"] or "",
                "excerpt": hit["excerpt"],
                "implication_for_v79": implication,
                "claim_boundary": "evidence_lock_only_not_v79_success",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/ACL2_v79TF_Revised_SemanticThreeMemoryControl_Plan.md"))
    parser.add_argument("--v78-recap", type=Path, default=Path("docs/ACL2_v78TF_v3_PCA_Grounded_AuditableVisualRediscovery_实验结果复盘.md"))
    parser.add_argument("--v78-execution-log", type=Path, default=Path("docs/ACL2_v78TF_v3_PCA_Grounded_AuditableVisualRediscovery_执行日志.md"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/"
            "report_final/phase0_semantic_goal_lock"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_text = _read_text(args.plan)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_rows = [
        {"field": field, "required": "true", "description": desc}
        for field, desc in CONTRACT_FIELDS
    ]
    _write_csv(
        out_dir / "required_semantic_contract_schema.csv",
        schema_rows,
        fieldnames=["field", "required", "description"],
    )

    _write_csv(
        out_dir / "semantic_memory_action_contract.csv",
        CANDIDATE_ROWS,
        fieldnames=[field for field, _desc in CONTRACT_FIELDS],
    )

    action_family_rows = []
    for row in ACTION_FAMILIES:
        action_family_rows.append(
            {
                **row,
                "semantic_role_present_in_plan": str(
                    any(role in plan_text for role in SEMANTIC_ROLES)
                ).lower(),
                "geometry_only_is_control": str("GEOMETRY_ONLY" in row["geometry_only_control"]).lower(),
            }
        )
    _write_csv(out_dir / "planned_action_family_memory_map.csv", action_family_rows)

    role_rows = [
        {"semantic_role": role, "definition": definition, "present_in_plan": str(_plan_has(plan_text, role)).lower()}
        for role, definition in SEMANTIC_ROLES.items()
    ]
    _write_csv(out_dir / "semantic_role_table.csv", role_rows)

    v78_rows = _build_v78_rows(args.v78_recap, args.v78_execution_log)
    _write_csv(
        out_dir / "v78_evidence_summary.csv",
        v78_rows,
        fieldnames=[
            "evidence_id",
            "marker",
            "found",
            "source_file",
            "line",
            "excerpt",
            "implication_for_v79",
            "claim_boundary",
        ],
    )

    forbidden_lines = [
        "# v79 Forbidden Non-Semantic Repeats",
        "",
        "These are not forbidden as controls, but they cannot be claimed as semantic-memory candidates:",
        "",
        "- Unconditional K/V or V-only frame-static centering.",
        "- Per-window or per-chunk scalar tuning.",
        "- Geometry-only target mining without semantic diagnosis fields.",
        "- TTT global write-strength sweeps when future metrics do not move.",
        "- Claims of semantic success when geometry-only or random/shuffle controls match or beat semantic variants.",
        "- KITTI09 retuning after a KITTI01 rule is selected.",
        "",
        "v78 evidence boundary:",
        "",
    ]
    for row in v78_rows:
        forbidden_lines.append(
            f"- {row['evidence_id']}: found={row['found']} source={row['source_file']}:{row['line']} "
            f"implication={row['implication_for_v79']}"
        )
    (out_dir / "forbidden_nonsemantic_repeats.md").write_text("\n".join(forbidden_lines) + "\n", encoding="utf-8")

    gate_checks = {
        "contract_schema_exists": (out_dir / "required_semantic_contract_schema.csv").is_file(),
        "semantic_memory_action_contract_exists": (out_dir / "semantic_memory_action_contract.csv").is_file(),
        "every_planned_action_family_maps_to_memory": all(
            bool(row["memory_body"]) and bool(row["memory_timescale"]) for row in ACTION_FAMILIES
        ),
        "every_planned_action_family_has_semantic_role_definition": all(
            bool(row["semantic_role_definition"]) for row in ACTION_FAMILIES
        ),
        "geometry_only_action_explicitly_control": all(
            "GEOMETRY_ONLY" in row["geometry_only_control"] for row in ACTION_FAMILIES
        ),
        "semantic_roles_present_in_plan": all(_plan_has(plan_text, role) for role in SEMANTIC_ROLES),
        "v78_evidence_markers_checked": all(bool(row["found"]) for row in v78_rows),
    }
    gate_pass = all(gate_checks.values())

    _write_json(
        out_dir / "semantic_goal_lock.json",
        {
            "phase": "phase0_semantic_goal_lock",
            "diagnostic_only": True,
            "training_free": True,
            "plan": args.plan,
            "v78_recap": args.v78_recap,
            "v78_execution_log": args.v78_execution_log,
            "out_dir": out_dir,
            "semantic_goal": "semantic controls short READ, mid SWA handoff, and long TTT write/update memory behavior",
            "gate_checks": gate_checks,
            "gate_pass": gate_pass,
            "success_claim": False,
            "next_required_phase": "phase1_current_bad_target_mining_with_semantic_diagnosis",
            "outputs": {
                "required_semantic_contract_schema": out_dir / "required_semantic_contract_schema.csv",
                "semantic_memory_action_contract": out_dir / "semantic_memory_action_contract.csv",
                "planned_action_family_memory_map": out_dir / "planned_action_family_memory_map.csv",
                "semantic_role_table": out_dir / "semantic_role_table.csv",
                "v78_evidence_summary": out_dir / "v78_evidence_summary.csv",
                "forbidden_nonsemantic_repeats": out_dir / "forbidden_nonsemantic_repeats.md",
            },
        },
    )
    print(f"wrote {out_dir / 'semantic_goal_lock.json'}")
    print(f"gate_pass={gate_pass}")


if __name__ == "__main__":
    main()
