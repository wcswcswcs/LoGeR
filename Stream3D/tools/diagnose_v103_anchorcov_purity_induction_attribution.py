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

PHASE_ID = "v103_anchorcov_purity_induction_attribution"
DEFAULT_OUT = AUDIT_ROOT / "v103_anchorcov_purity_induction_attribution_r1"
DEFAULT_OLD_S1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers"
DEFAULT_ANCHORCOV_S1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers_anchorcov_r1"
DEFAULT_OLD_PHASE9E_ROOT = AUDIT_ROOT / "v103_phase9e_d4rt_anchor_da3_induced_carriers_suppS1_d4rt48mix_s5repair_r1"
DEFAULT_ANCHORCOV_PHASE9E_ROOT = AUDIT_ROOT / "v103_phase9e_d4rt_anchor_da3_induced_carriers_anchorcov_s5repair_r1"
DEFAULT_PHASE2_SCENE0011 = (
    AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
)
DEFAULT_PHASE2_SCENE0050 = (
    AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
)
DEFAULT_OLD32_SCENE0011 = (
    AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384"
)
DEFAULT_OLD32_SCENE0050 = (
    AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384"
)

SCENES = ("scene0011_00", "scene0050_00")


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


def _first(rows: list[dict[str, str]], **where: str) -> dict[str, str]:
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in where.items()):
            return row
    return {}


def _metric(rows: list[dict[str, str]], scene_id: str, role_name: str = "A_anchor") -> dict[str, str]:
    return _first(rows, scene_id=scene_id, role_name=role_name)


def _overlap(rows: list[dict[str, str]], scene_id: str, metric_name: str) -> dict[str, str]:
    return _first(rows, scene_id=scene_id, metric_name=metric_name)


def _provider_name(summary: dict[str, Any]) -> str:
    diag = summary.get("d4rt_infer_diagnostics", {})
    clip_frames = str(diag.get("clip_frames", ""))
    size = str(diag.get("checkpoint_size_bytes", ""))
    if clip_frames == "48" or size == "13950737434":
        return "OpenD4RT_48CLIP_9Mix_NoCropAUG"
    if clip_frames == "32" or size == "13950006682":
        return "OpenD4RT_32CLIP_9Dataset_NoAUG"
    return "unknown"


def _phase2_row(label: str, scene_id: str, root: Path) -> dict[str, Any]:
    summary = _read_json(root / "summary.json")
    diag = summary.get("d4rt_infer_diagnostics", {})
    return {
        "schema_version": "stream4d_v103_anchorcov_provider_row_v1",
        "phase_id": PHASE_ID,
        "provider_label": label,
        "scene_id": scene_id,
        "root": root,
        "exists": bool(summary),
        "provider_name": _provider_name(summary) if summary else "",
        "clip_frames": diag.get("clip_frames", ""),
        "checkpoint_size_bytes": diag.get("checkpoint_size_bytes", ""),
        "query_count_per_frame": summary.get("query_count_per_frame", ""),
        "query_chunk_size": diag.get("query_chunk_size", ""),
        "source_count": summary.get("source_count", ""),
        "mask_balanced_points_per_mask": summary.get("query_generation_policy", {}).get(
            "mask_balanced_view_probe_points_per_mask", ""
        ),
        "fresh_d4rt_decode": summary.get("carrier_batch_cache", {}).get("fresh_d4rt_decode", ""),
        "uses_gt_for_query_selection": summary.get("uses_gt_for_query_selection", ""),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attribute v103 anchor coverage regression against D4RT provider and S1 role selection evidence."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--old-s1-root", default=str(DEFAULT_OLD_S1_ROOT))
    parser.add_argument("--anchorcov-s1-root", default=str(DEFAULT_ANCHORCOV_S1_ROOT))
    parser.add_argument("--old-phase9e-root", default=str(DEFAULT_OLD_PHASE9E_ROOT))
    parser.add_argument("--anchorcov-phase9e-root", default=str(DEFAULT_ANCHORCOV_PHASE9E_ROOT))
    parser.add_argument("--phase2-scene0011-root", default=str(DEFAULT_PHASE2_SCENE0011))
    parser.add_argument("--phase2-scene0050-root", default=str(DEFAULT_PHASE2_SCENE0050))
    parser.add_argument("--old32-scene0011-root", default=str(DEFAULT_OLD32_SCENE0011))
    parser.add_argument("--old32-scene0050-root", default=str(DEFAULT_OLD32_SCENE0050))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")

    old_s1 = _project(args.old_s1_root)
    anchor_s1 = _project(args.anchorcov_s1_root)
    old_p9e = _project(args.old_phase9e_root)
    anchor_p9e = _project(args.anchorcov_phase9e_root)
    phase2_roots = {
        "scene0011_00": _project(args.phase2_scene0011_root),
        "scene0050_00": _project(args.phase2_scene0050_root),
    }
    old32_roots = {
        "scene0011_00": _project(args.old32_scene0011_root),
        "scene0050_00": _project(args.old32_scene0050_root),
    }

    old_s1_summary = _read_json(old_s1 / "summary.json")
    anchor_s1_summary = _read_json(anchor_s1 / "summary.json")
    old_p9e_summary = _read_json(old_p9e / "summary.json")
    anchor_p9e_summary = _read_json(anchor_p9e / "summary.json")

    old_metric_rows = _read_csv(old_s1 / "carrier_role_metric_rows.csv")
    anchor_metric_rows = _read_csv(anchor_s1 / "carrier_role_metric_rows.csv")
    old_overlap_rows = _read_csv(old_s1 / "role_overlap_rows.csv")
    anchor_overlap_rows = _read_csv(anchor_s1 / "role_overlap_rows.csv")
    old_scene_rows = _read_csv(old_p9e / "scene_summary_rows.csv")
    anchor_scene_rows = _read_csv(anchor_p9e / "scene_summary_rows.csv")

    provider_rows = []
    for scene_id, root in phase2_roots.items():
        provider_rows.append(_phase2_row("current_48mix_maskbalanced8", scene_id, root))
    for scene_id, root in old32_roots.items():
        provider_rows.append(_phase2_row("older_32clip_same_q5c", scene_id, root))

    delta_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    same_provider_phase9e = True
    clean_regression_scenes = 0
    coverage_gain_scenes = 0
    anchor_overlap_v_increase_scenes = 0

    for scene_id in SCENES:
        old_a = _metric(old_metric_rows, scene_id, "A_anchor")
        new_a = _metric(anchor_metric_rows, scene_id, "A_anchor")
        old_overlap_v = _overlap(old_overlap_rows, scene_id, "A_overlap_V_rate")
        new_overlap_v = _overlap(anchor_overlap_rows, scene_id, "A_overlap_V_rate")
        old_p = _first(old_scene_rows, scene_id=scene_id)
        new_p = _first(anchor_scene_rows, scene_id=scene_id)

        old_clean = _truth(old_p.get("clean_induction_gate_pass", ""))
        new_clean = _truth(new_p.get("clean_induction_gate_pass", ""))
        clean_regressed = old_clean and not new_clean
        clean_regression_scenes += int(clean_regressed)
        phase9e_same_phase2_root = str(old_p.get("phase2_root", "")) == str(new_p.get("phase2_root", ""))
        same_provider_phase9e = same_provider_phase9e and phase9e_same_phase2_root

        old_count = _num(old_a.get("carrier_count", ""))
        new_count = _num(new_a.get("carrier_count", ""))
        old_cov = _num(old_a.get("mask_support_coverage_rate", ""))
        new_cov = _num(new_a.get("mask_support_coverage_rate", ""))
        old_overlap_v_rate = _num(old_overlap_v.get("rate", ""))
        new_overlap_v_rate = _num(new_overlap_v.get("rate", ""))
        coverage_gain = new_cov > old_cov or new_count > old_count
        coverage_gain_scenes += int(coverage_gain)
        anchor_overlap_v_increase_scenes += int(new_overlap_v_rate > old_overlap_v_rate)

        old_clean_induced = _num(old_p.get("best_clean_induced_unanchored_mask_observation_count", ""), 0.0)
        new_clean_induced = _num(new_p.get("best_clean_induced_unanchored_mask_observation_count", ""), 0.0)
        old_anchor_obs = _num(old_p.get("d4rt_positive_anchor_observation_count", ""))
        new_anchor_obs = _num(new_p.get("d4rt_positive_anchor_observation_count", ""))

        delta_rows.append(
            {
                "schema_version": "stream4d_v103_anchorcov_scene_delta_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene_id,
                "old_s1_variant": old_a.get("variant_id", ""),
                "anchorcov_s1_variant": new_a.get("variant_id", ""),
                "old_A_anchor_count": old_count,
                "anchorcov_A_anchor_count": new_count,
                "A_anchor_count_delta": new_count - old_count,
                "old_A_mask_support_coverage_rate": old_cov,
                "anchorcov_A_mask_support_coverage_rate": new_cov,
                "A_mask_support_coverage_delta": new_cov - old_cov,
                "old_A_broad_mask_participation_rate": _num(old_a.get("broad_mask_participation_rate", "")),
                "anchorcov_A_broad_mask_participation_rate": _num(new_a.get("broad_mask_participation_rate", "")),
                "old_A_competing_mask_conflict_rate": _num(old_a.get("competing_mask_conflict_rate", "")),
                "anchorcov_A_competing_mask_conflict_rate": _num(new_a.get("competing_mask_conflict_rate", "")),
                "old_A_semantic_contradiction_rate": _num(
                    old_a.get("short_range_semantic_contradiction_rate", "")
                ),
                "anchorcov_A_semantic_contradiction_rate": _num(
                    new_a.get("short_range_semantic_contradiction_rate", "")
                ),
                "old_A_overlap_V_rate": old_overlap_v_rate,
                "anchorcov_A_overlap_V_rate": new_overlap_v_rate,
                "A_overlap_V_rate_delta": new_overlap_v_rate - old_overlap_v_rate,
                "old_phase9e_clean_induction_gate_pass": old_clean,
                "anchorcov_phase9e_clean_induction_gate_pass": new_clean,
                "phase9e_clean_induction_regressed": clean_regressed,
                "old_clean_induced_unanchored_mask_observation_count": old_clean_induced,
                "anchorcov_clean_induced_unanchored_mask_observation_count": new_clean_induced,
                "clean_induced_unanchored_delta": new_clean_induced - old_clean_induced,
                "old_d4rt_positive_anchor_observation_count": old_anchor_obs,
                "anchorcov_d4rt_positive_anchor_observation_count": new_anchor_obs,
                "d4rt_positive_anchor_observation_delta": new_anchor_obs - old_anchor_obs,
                "same_phase2_root_between_old_and_anchorcov_phase9e": phase9e_same_phase2_root,
                "phase2_root": new_p.get("phase2_root", ""),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic_labels": True,
            }
        )

        if clean_regressed:
            case_rows.append(
                {
                    "schema_version": "stream4d_v103_anchorcov_casebook_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene_id,
                    "case_id": "anchorcov_clean_induction_regression",
                    "finding": "Anchor coverage increased or changed under the same D4RT48Mix Phase2 root, but clean DA3 induction was lost.",
                    "evidence": (
                        f"A_count {old_count}->{new_count}; coverage {old_cov}->{new_cov}; "
                        f"clean_induced {old_clean_induced}->{new_clean_induced}; "
                        f"A_overlap_V {old_overlap_v_rate}->{new_overlap_v_rate}"
                    ),
                    "proximate_cause": "S1 role selection / anchor purity regression, not a D4RT checkpoint change",
                    "recommended_action": "Use purity-aware anchor selection or compare anchor strata before adding more A_anchor coverage.",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic_labels": True,
                }
            )

    current_48_ready = all(
        row["provider_name"] == "OpenD4RT_48CLIP_9Mix_NoCropAUG" and _truth(row["uses_gt_for_query_selection"]) is False
        for row in provider_rows
        if row["provider_label"] == "current_48mix_maskbalanced8"
    )
    old32_available = all(row["provider_name"] == "OpenD4RT_32CLIP_9Dataset_NoAUG" for row in provider_rows if row["provider_label"] == "older_32clip_same_q5c")
    anchorcov_clean_count = int(_num(anchor_p9e_summary.get("clean_induction_scene_count", ""), -1))
    old_clean_count = int(_num(old_p9e_summary.get("clean_induction_scene_count", ""), -1))

    gate_rows = [
        {
            "schema_version": "stream4d_v103_anchorcov_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "new_d4rt48mix_weight_already_used",
            "pass": current_48_ready,
            "observed": "current Phase2 roots report clip_frames=48 and checkpoint_size_bytes=13950737434",
            "required": "both scenes use OpenD4RT_48CLIP_9Mix_NoCropAUG without GT query selection",
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v103_anchorcov_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "older_32clip_reference_available",
            "pass": old32_available,
            "observed": "32CLIP q5c Phase2 roots exist for both scenes" if old32_available else "missing 32CLIP reference",
            "required": "32CLIP reference roots available for provider attribution context",
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v103_anchorcov_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "phase9e_regression_uses_same_d4rt_provider",
            "pass": same_provider_phase9e,
            "observed": f"same_phase2_root_all_scenes={same_provider_phase9e}",
            "required": "old S1 and anchorcov S1 Phase9e rows use the same Phase2 provider roots",
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v103_anchorcov_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "anchorcov_clean_induction_not_regressed",
            "pass": anchorcov_clean_count >= old_clean_count and anchorcov_clean_count > 0,
            "observed": f"anchorcov_clean={anchorcov_clean_count}; old_clean={old_clean_count}; clean_regression_scenes={clean_regression_scenes}",
            "required": "anchorcov clean scene count >= old S1 clean scene count and > 0",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
        {
            "schema_version": "stream4d_v103_anchorcov_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "coverage_gain_translates_to_clean_induction",
            "pass": coverage_gain_scenes > 0 and clean_regression_scenes == 0,
            "observed": f"coverage_gain_scenes={coverage_gain_scenes}; clean_regression_scenes={clean_regression_scenes}",
            "required": "coverage increase must not reduce clean induction",
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic_labels": True,
        },
    ]

    if current_48_ready and same_provider_phase9e and clean_regression_scenes > 0:
        decision = "ANCHOR_SELECTION_REGRESSION_NOT_D4RT_WEIGHT_CHANGE"
        provider_problem = "not_proximate_for_anchorcov_regression"
        filter_problem = "proximate"
    elif not current_48_ready:
        decision = "D4RT48MIX_PROVIDER_EVIDENCE_MISSING"
        provider_problem = "unknown"
        filter_problem = "unknown"
    else:
        decision = "ANCHORCOV_ATTRIBUTION_REVIEW_REQUIRED"
        provider_problem = "possible"
        filter_problem = "possible"

    summary = {
        "schema_version": "stream4d_v103_anchorcov_purity_induction_attribution_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - started,
        "decision": decision,
        "failure_count": 0,
        "d4rt48mix_provider_used": current_48_ready,
        "old32_reference_available": old32_available,
        "same_d4rt_provider_between_old_s1_and_anchorcov_phase9e": same_provider_phase9e,
        "old_clean_induction_scene_count": old_clean_count,
        "anchorcov_clean_induction_scene_count": anchorcov_clean_count,
        "clean_regression_scene_count": clean_regression_scenes,
        "coverage_gain_scene_count": coverage_gain_scenes,
        "anchor_overlap_v_increase_scene_count": anchor_overlap_v_increase_scenes,
        "provider_problem_for_anchorcov_regression": provider_problem,
        "filter_or_role_selection_problem_for_anchorcov_regression": filter_problem,
        "inputs": {
            "old_s1_root": old_s1,
            "anchorcov_s1_root": anchor_s1,
            "old_phase9e_root": old_p9e,
            "anchorcov_phase9e_root": anchor_p9e,
            "phase2_roots": phase2_roots,
            "old32_reference_roots": old32_roots,
        },
        "outputs": {
            "provider_rows": out / "provider_rows.csv",
            "scene_delta_rows": out / "scene_delta_rows.csv",
            "gate_rows": out / "gate_rows.csv",
            "casebook_rows": out / "casebook_rows.csv",
            "summary": out / "summary.json",
            "last_command": out / "last_command.txt",
        },
        "truthfulness_note": (
            "This attribution is diagnostic-only. It uses GT labels only where Phase9e already reports diagnostic clean/false bridge metrics; "
            "it does not select thresholds, does not emit predictions, and does not rerun D4RT."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }

    _write_csv(out / "provider_rows.csv", provider_rows)
    _write_csv(out / "scene_delta_rows.csv", delta_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "casebook_rows.csv", case_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
