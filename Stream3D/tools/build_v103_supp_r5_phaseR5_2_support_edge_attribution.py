#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_r5_phaseR5_2_support_edge_attribution"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r5_support_edge_attribution"
DEFAULT_FEATURE_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_affinity"
DEFAULT_LOCAL_AP_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_local_ap_diag"
DEFAULT_ANCHOR_ONLY_ROOT = AUDIT_ROOT / "v103_supp_r5_anchor_only_local_ap_diag"
DEFAULT_CURRENT_PHASE6D_ROOT = AUDIT_ROOT / "v103_phase6d_f2_skeleton_affinity_merge_phase9n_suppS1_d4rt48mix_s5repair_r4_directpair_guard"


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _feature_summary_map(feature_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    df = _read_csv(feature_root / "role_feature_summary_rows.csv")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, rec in df.iterrows():
        out[(str(rec.get("variant_id", "")), str(rec.get("scene_id", "")))] = rec.to_dict()
    return out


def _run_roots(local_ap_root: Path, anchor_only_root: Path, current_root: Path) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = [
        {
            "root_role": "current_phase6d_d9_lock",
            "r5_feature_variant_id": "current_phase6d_locked_D9",
            "root": current_root,
        }
    ]
    for root, role in [(anchor_only_root, "anchor_only_r5_4_diag"), (local_ap_root, "support_weighted_r5_4_diag")]:
        run_parent = root / "phase6d_runs"
        if not run_parent.exists():
            continue
        for child in sorted(run_parent.iterdir()):
            if child.is_dir():
                roots.append({"root_role": role, "r5_feature_variant_id": child.name, "root": child})
    return roots


def _edge_family(row: dict[str, Any]) -> str:
    variant_id = str(row.get("variant_id", ""))
    if _bool(row.get("shuffle_affinity", False)) or variant_id.startswith("R"):
        return "shuffled_control"
    if int(_num(row.get("direct_pair_support_min_count"), 0)) > 0:
        return "DA3_direct_pair_guarded_support"
    if _bool(row.get("broad_support_veto", False)):
        return "support_broad_veto_guarded_affinity"
    return "primitive_affinity"


def _support_edge_rows(roots: list[dict[str, Any]], feature_summary: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root_info in roots:
        root = Path(root_info["root"])
        edge_df = _read_csv(root / "merge_edge_rows.csv")
        if edge_df.empty:
            continue
        for _, rec in edge_df.iterrows():
            data = rec.to_dict()
            r5_variant = str(root_info["r5_feature_variant_id"])
            scene = str(data.get("scene_id", ""))
            fs = feature_summary.get((r5_variant, scene), {})
            support_count = int(_num(data.get("direct_pair_support_count"), 0))
            same_gt = int(_num(data.get("direct_pair_diagnostic_same_gt_count"), 0))
            diff_gt = int(_num(data.get("direct_pair_diagnostic_different_gt_count"), 0))
            edge_rank = int(_num(data.get("edge_rank"), -1))
            phase6d_variant = str(data.get("variant_id", ""))
            accepted = _bool(data.get("accepted_union", False))
            support_ratio = _num(fs.get("support_contribution_ratio"), 0.0)
            veto_ratio = _num(fs.get("veto_overlap_contribution_ratio"), 0.0)
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_2_support_edge_attribution_row_v1",
                    "phase_id": PHASE_ID,
                    "edge_id": f"{r5_variant}:{phase6d_variant}:{scene}:{edge_rank}",
                    "scene_id": scene,
                    "r5_feature_variant_id": r5_variant,
                    "phase6d_variant_id": phase6d_variant,
                    "source_root_role": root_info["root_role"],
                    "source_phase6d_root": _rel(root),
                    "source_object_or_mask_a": str(data.get("object_a", "")),
                    "source_object_or_mask_b": str(data.get("object_b", "")),
                    "edge_family": _edge_family(data),
                    "has_A_anchor": bool(phase6d_variant != "D0_f2_original_replay"),
                    "A_anchor_count": "",
                    "A_anchor_reliability_mean": "",
                    "S_support_count": support_count,
                    "S_support_weighted_mass": support_ratio,
                    "S_support_cosine": _num(data.get("affinity"), 0.0),
                    "S_support_broad_mass": support_count * _num(data.get("direct_pair_broad_risk_max"), 0.0),
                    "S_support_veto_overlap_mass": veto_ratio,
                    "S_support_semantic_filtered_mass": support_ratio if "semantic_filtered" in r5_variant else 0.0,
                    "has_F2_skeleton_edge": True,
                    "has_DA3_direct_pair": support_count >= max(1, int(_num(data.get("direct_pair_support_min_count"), 0))),
                    "V_veto_score": _num(data.get("direct_pair_broad_risk_max"), 0.0),
                    "semantic_similarity_calibrated": _num(data.get("affinity"), 0.0) if "semantic_filtered" in r5_variant else "",
                    "accepted_by_variant": accepted,
                    "diagnostic_same_gt": bool(same_gt > 0 and same_gt >= diff_gt),
                    "diagnostic_diff_gt": bool(diff_gt > 0),
                    "diagnostic_same_gt_count": same_gt,
                    "diagnostic_diff_gt_count": diff_gt,
                    "diagnostic_same_gt_rate": _num(data.get("direct_pair_diagnostic_same_gt_rate"), 0.0),
                    "diagnostic_diff_gt_rate": _num(data.get("direct_pair_diagnostic_different_gt_rate"), 0.0),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                    "uses_future": False,
                }
            )
    return rows


def _metric_rows(roots: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for root_info in roots:
        root = Path(root_info["root"])
        df = _read_csv(root / "merge_metric_rows.csv")
        if df.empty:
            failures.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_2_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "blocker": "MERGE_METRIC_ROWS_MISSING",
                    "detail": _rel(root / "merge_metric_rows.csv"),
                    "repair_direction": "Rerun the corresponding Phase6d diagnostic before edge attribution.",
                }
            )
            continue
        for _, rec in df.iterrows():
            data = rec.to_dict()
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_phaseR5_2_leave_one_family_metric_row_v1",
                    "phase_id": PHASE_ID,
                    "source_root_role": root_info["root_role"],
                    "source_phase6d_root": _rel(root),
                    "r5_feature_variant_id": str(root_info["r5_feature_variant_id"]),
                    "phase6d_variant_id": str(data.get("variant_id", "")),
                    "leave_one_family_variant": _map_leave_one_family(str(root_info["r5_feature_variant_id"]), str(data.get("variant_id", ""))),
                    "MV_AP_window": _num(data.get("MV_AP_window"), 0.0),
                    "MV_AP50_window": _num(data.get("MV_AP50_window"), 0.0),
                    "ScoreFreeMatch50_window": _num(data.get("ScoreFreeMatch50_window"), 0.0),
                    "accepted_edge_count": int(_num(data.get("accepted_merge_count"), 0)),
                    "candidate_edge_count": int(_num(data.get("candidate_edge_count"), 0)),
                    "edge_family_removed": _removed_family(str(root_info["r5_feature_variant_id"]), str(data.get("variant_id", ""))),
                    "same_frame_collision_count": int(_num(data.get("same_frame_collision_count"), 0)),
                    "pixel_collision_rate": _num(data.get("pixel_collision_rate"), 0.0),
                    "missing_mask_raster_count": int(_num(data.get("missing_mask_raster_count"), 0)),
                    "dataset_split": str(data.get("dataset_split", "")),
                    "chunk_id": str(data.get("chunk_id", "")),
                    "uses_gt_for_prediction": _bool(data.get("uses_gt_for_prediction", False)),
                    "uses_gt_for_eval": True,
                    "uses_future": _bool(data.get("uses_future", False)),
                }
            )
    return rows, failures


def _map_leave_one_family(r5_variant: str, phase6d_variant: str) -> str:
    if phase6d_variant == "D0_f2_original_replay":
        return "L0_D9_replay_or_F2_skeleton_replay"
    if r5_variant == "F0_anchor_only":
        return "L1_anchor_feature_only"
    if "semantic_filtered" in r5_variant:
        return "L5_semantic_support_filter_present"
    if phase6d_variant.startswith("R"):
        return "control_shuffled_affinity"
    if "direct_pair" in phase6d_variant:
        return "L6_DA3_direct_pair_present"
    if "score" in phase6d_variant:
        return "L7_score_repair_present"
    if r5_variant.startswith("F"):
        return "L2_support_weighted_feature_present"
    return "L0_current_locked_D9"


def _removed_family(r5_variant: str, phase6d_variant: str) -> str:
    if r5_variant == "F0_anchor_only":
        return "S_support_feature"
    if phase6d_variant == "D0_f2_original_replay":
        return "all_affinity_merge_edges"
    if phase6d_variant.startswith("R"):
        return "real_affinity_alignment_shuffled_control"
    if "direct_pair" in phase6d_variant:
        return "none_direct_pair_guard_present"
    if "score" in phase6d_variant:
        return "none_score_repair_present"
    return "none_support_weighted_feature_present"


def _compare_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["r5_feature_variant_id"], r["phase6d_variant_id"]): r for r in metric_rows}
    replay_by_root: dict[str, dict[str, Any]] = {}
    shuffled_by_root: dict[str, dict[str, Any]] = {}
    for row in metric_rows:
        if row["phase6d_variant_id"] == "D0_f2_original_replay":
            replay_by_root[row["r5_feature_variant_id"]] = row
        if row["phase6d_variant_id"] == "R5_shuffled_affinity_merge_tau065_top1_broad_support_veto_control":
            shuffled_by_root[row["r5_feature_variant_id"]] = row
    out: list[dict[str, Any]] = []
    for row in metric_rows:
        if row["phase6d_variant_id"] != "D9_affinity_merge_tau065_top1_broad_support_veto":
            continue
        replay = replay_by_root.get(row["r5_feature_variant_id"], {})
        shuffled = shuffled_by_root.get(row["r5_feature_variant_id"], {})
        out.append(
            {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_2_support_family_compare_row_v1",
                "phase_id": PHASE_ID,
                "r5_feature_variant_id": row["r5_feature_variant_id"],
                "phase6d_variant_id": row["phase6d_variant_id"],
                "MV_AP_window": row["MV_AP_window"],
                "MV_AP50_window": row["MV_AP50_window"],
                "replay_MV_AP_window": replay.get("MV_AP_window", ""),
                "replay_MV_AP50_window": replay.get("MV_AP50_window", ""),
                "shuffled_MV_AP_window": shuffled.get("MV_AP_window", ""),
                "real_minus_replay_MV_AP_window": _num(row["MV_AP_window"]) - _num(replay.get("MV_AP_window"), 0.0) if replay else "",
                "real_minus_shuffled_MV_AP_window": _num(row["MV_AP_window"]) - _num(shuffled.get("MV_AP_window"), 0.0) if shuffled else "",
                "classification": _classify_compare(row, replay, shuffled),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
        )
    if ("F0_anchor_only", "D9_affinity_merge_tau065_top1_broad_support_veto") in by_key:
        anchor = by_key[("F0_anchor_only", "D9_affinity_merge_tau065_top1_broad_support_veto")]
        for key, support in sorted(by_key.items()):
            if key[0].startswith("F") and key[0] != "F0_anchor_only" and key[1] == "D9_affinity_merge_tau065_top1_broad_support_veto":
                out.append(
                    {
                        "schema_version": "stream4d_v103_supp_r5_phaseR5_2_support_family_compare_row_v1",
                        "phase_id": PHASE_ID,
                        "r5_feature_variant_id": key[0],
                        "phase6d_variant_id": key[1],
                        "MV_AP_window": support["MV_AP_window"],
                        "MV_AP50_window": support["MV_AP50_window"],
                        "anchor_only_MV_AP_window": anchor["MV_AP_window"],
                        "support_minus_anchor_only_MV_AP_window": _num(support["MV_AP_window"]) - _num(anchor["MV_AP_window"]),
                        "classification": "support_vs_anchor_only_no_subset_gate_pass",
                        "uses_gt_for_prediction": False,
                        "uses_gt_for_eval": True,
                        "uses_future": False,
                    }
                )
    return out


def _classify_compare(row: dict[str, Any], replay: dict[str, Any], shuffled: dict[str, Any]) -> str:
    real = _num(row.get("MV_AP_window"), 0.0)
    ap50 = _num(row.get("MV_AP50_window"), 0.0)
    rep = _num(replay.get("MV_AP_window"), 0.0) if replay else 0.0
    rep50 = _num(replay.get("MV_AP50_window"), 0.0) if replay else 0.0
    shuf = _num(shuffled.get("MV_AP_window"), 0.0) if shuffled else 0.0
    if replay and shuffled and real >= rep + 0.005 and ap50 >= rep50 + 0.010 and real >= shuf + 0.003:
        return "subset_gate_like_signal_present"
    if replay and real < rep:
        return "support_weighted_feature_hurts_replay"
    if shuffled and real < shuf + 0.003:
        return "real_minus_shuffled_too_small"
    return "partial_control_signal_but_subset_gate_failed"


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    feature_root = _project(args.feature_root)
    local_ap_root = _project(args.local_ap_root)
    anchor_root = _project(args.anchor_only_root)
    current_root = _project(args.current_phase6d_root)

    feature_summary = _feature_summary_map(feature_root)
    roots = _run_roots(local_ap_root, anchor_root, current_root)
    edge_rows = _support_edge_rows(roots, feature_summary)
    metric_rows, failure_rows = _metric_rows(roots)
    compare_rows = _compare_rows(metric_rows)

    accepted_edges = [r for r in edge_rows if bool(r["accepted_by_variant"])]
    accepted_diff_gt = sum(1 for r in accepted_edges if bool(r["diagnostic_diff_gt"]))
    d9_compare = [r for r in compare_rows if r.get("phase6d_variant_id") == "D9_affinity_merge_tau065_top1_broad_support_veto"]
    no_subset_gate = all(str(r.get("classification")) != "subset_gate_like_signal_present" for r in d9_compare)
    if no_subset_gate:
        failure_rows.append(
            {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_2_failure_row_v1",
                "phase_id": PHASE_ID,
                "blocker": "SUPPORT_COVERAGE_NOT_USED_FOR_SUBSET_LOCAL_AP",
                "detail": "D9 support-weighted variants did not satisfy replay/AP50 subset gates; see support_family_compare_rows.csv.",
                "repair_direction": "Do not promote to full-dev; inspect GT coverage and same-object fragmentation diagnostics.",
            }
        )
    if accepted_diff_gt > 0:
        failure_rows.append(
            {
                "schema_version": "stream4d_v103_supp_r5_phaseR5_2_failure_row_v1",
                "phase_id": PHASE_ID,
                "blocker": "SUPPORT_EDGE_FALSE_BRIDGE_DIAGNOSTIC_NONZERO",
                "detail": f"accepted_edges_with_diagnostic_diff_gt={accepted_diff_gt}",
                "repair_direction": "Keep support guarded by anchor/skeleton/veto; do not allow support-only union.",
            }
        )

    summary = {
        "schema_version": "stream4d_v103_supp_r5_phaseR5_2_summary_v1",
        "phase_id": PHASE_ID,
        "decision": "NO_GO_ENTER_R5_3_DIAGNOSTIC_ONLY" if failure_rows else "PASS_ENTER_R5_3_DIAGNOSTIC_ONLY",
        "phase_r5_2_pass": not failure_rows,
        "edge_row_count": len(edge_rows),
        "accepted_edge_row_count": len(accepted_edges),
        "accepted_diff_gt_edge_count": accepted_diff_gt,
        "metric_row_count": len(metric_rows),
        "support_family_compare_row_count": len(compare_rows),
        "failure_count": len(failure_rows),
        "roots": [_rel(r["root"]) for r in roots],
        "outputs": {
            "support_edge_attribution_rows": _rel(out / "support_edge_attribution_rows.csv"),
            "leave_one_family_metric_rows": _rel(out / "leave_one_family_metric_rows.csv"),
            "support_family_compare_rows": _rel(out / "support_family_compare_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "summary": _rel(out / "summary.json"),
        },
        "truthfulness_note": "R5-2 reads existing Phase6d edge/metric artifacts. It does not run AP, tune thresholds, or use GT for prediction. GT labels in direct-pair fields are diagnostic-only.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "runtime_sec": time.time() - t0,
    }

    _write_csv(out / "support_edge_attribution_rows.csv", edge_rows)
    _write_csv(out / "leave_one_family_metric_rows.csv", metric_rows)
    _write_csv(out / "support_family_compare_rows.csv", compare_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--local-ap-root", default=str(DEFAULT_LOCAL_AP_ROOT))
    parser.add_argument("--anchor-only-root", default=str(DEFAULT_ANCHOR_ONLY_ROOT))
    parser.add_argument("--current-phase6d-root", default=str(DEFAULT_CURRENT_PHASE6D_ROOT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = build(args)
    return 0 if bool(summary.get("phase_r5_2_pass")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
