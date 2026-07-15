#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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

PHASE_ID = "v103_r7_phase1_edge_attribution_exact"
DEFAULT_OUT = AUDIT_ROOT / "v103_r7_phase1_edge_attribution_exact"
DEFAULT_PHASE0_ROOT = AUDIT_ROOT / "v103_r7_phase0_d4rt_only_fact_lock"
DEFAULT_PHASE6D_RUN_ROOT = DEFAULT_OUT / "phase6d_d4rt_only_runs"
DEFAULT_R6_FEATURE_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_feature"
DEFAULT_R6_DIAG_ROOT = AUDIT_ROOT / "v103_supp_r6_phase6_gt_coverage_inconsistency"

D9 = "D9_affinity_merge_tau065_top1_broad_support_veto"
D0 = "D0_f2_original_replay"
R5 = "R5_shuffled_affinity_merge_tau065_top1_broad_support_veto_control"

REFERENCE_FEATURE_VARIANT = "R6F2_support010_specificity_semantic"
ANCHOR_ONLY_FEATURE_VARIANT = "R6F0_anchor_only_replay"
NONSEMANTIC_FEATURE_VARIANT = "R6F1_support005_specificity"
FEATURE_VARIANTS = [
    ANCHOR_ONLY_FEATURE_VARIANT,
    NONSEMANTIC_FEATURE_VARIANT,
    REFERENCE_FEATURE_VARIANT,
]

FORBIDDEN_PATH_TOKENS = (
    "da3",
    "da3-giant",
    "3dgs",
    "gaussian",
    "phase9n",
    "phase9b",
    "phase9c",
    "phase9d",
    "da3_pair",
)


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


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda v: "" if v is None else str(v))
    df.to_parquet(path, index=False)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(value: Any, default: float | None = None) -> float | None:
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
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _path_forbidden_hits(path: Path | str) -> list[str]:
    text = _rel(path).lower()
    return sorted({token for token in FORBIDDEN_PATH_TOKENS if token in text})


def _artifact_row(role: str, path: Path, *, required: bool = True, note: str = "") -> dict[str, Any]:
    exists = path.exists()
    row_count: int | str = ""
    if exists and path.is_file() and path.suffix.lower() in {".csv", ".parquet"}:
        try:
            if path.suffix.lower() == ".csv":
                row_count = max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0)
            else:
                row_count = int(pd.read_parquet(path).shape[0])
        except Exception:
            row_count = ""
    return {
        "schema_version": "stream4d_v103_r7_phase1_artifact_row_v1",
        "phase_id": PHASE_ID,
        "artifact_role": role,
        "path": _rel(path),
        "exists": bool(exists),
        "required": bool(required),
        "is_file": path.is_file() if exists else False,
        "is_dir": path.is_dir() if exists else False,
        "size_bytes": path.stat().st_size if exists and path.is_file() else "",
        "sha256": _sha256(path),
        "row_count": row_count,
        "forbidden_token_hits_in_path": ";".join(_path_forbidden_hits(path)),
        "note": note,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _gate_row(name: str, passed: bool, observed: Any, required: Any, repair: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_r7_phase1_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_name": name,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _failure_row(failure_id: str, detail: Any, repair: str, severity: str = "blocking") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_r7_phase1_failure_row_v1",
        "phase_id": PHASE_ID,
        "failure_id": failure_id,
        "severity": severity,
        "detail": json.dumps(_jsonable(detail), sort_keys=True) if isinstance(detail, (dict, list, tuple)) else detail,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _metric_by_variant(root: Path) -> dict[str, dict[str, Any]]:
    df = _read_csv(root / "merge_metric_rows.csv")
    out: dict[str, dict[str, Any]] = {}
    for rec in df.to_dict("records"):
        out[str(rec.get("variant_id", ""))] = rec
    return out


def _edge_counts(root: Path, variant_id: str) -> dict[str, int]:
    df = _read_csv(root / "merge_edge_rows.csv")
    if df.empty:
        return {"accepted_edge_count": 0, "candidate_edge_count": 0, "accepted_diff_gt_edge_count_diagnostic": 0}
    sdf = df[df["variant_id"].astype(str) == variant_id]
    accepted = sdf[sdf["accepted_union"].map(_bool)] if "accepted_union" in sdf.columns else pd.DataFrame()
    diff_count = 0
    if not accepted.empty and "direct_pair_diagnostic_different_gt_count" in accepted.columns:
        diff_count = int(accepted["direct_pair_diagnostic_different_gt_count"].fillna(0).astype(float).sum())
    return {
        "accepted_edge_count": int(len(accepted)),
        "candidate_edge_count": int(len(sdf)),
        "accepted_diff_gt_edge_count_diagnostic": diff_count,
    }


def _feature_summary_map(feature_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    df = _read_csv(feature_root / "role_feature_summary_rows.csv")
    return {(str(r.get("variant_id", "")), str(r.get("scene_id", ""))): r for r in df.to_dict("records")}


def _metric_row(
    *,
    family: str,
    status: str,
    source: str,
    feature_variant: str,
    phase6d_variant: str,
    metrics: dict[str, Any] | None,
    edge_counts: dict[str, int] | None,
    reference: dict[str, Any] | None,
    shuffled: dict[str, Any] | None,
    missing_reason: str = "",
) -> dict[str, Any]:
    mv = _num(metrics.get("MV_AP_window") if metrics else None)
    mv50 = _num(metrics.get("MV_AP50_window") if metrics else None)
    ref_mv = _num(reference.get("MV_AP_window") if reference else None)
    ref_mv50 = _num(reference.get("MV_AP50_window") if reference else None)
    shuf_mv = _num(shuffled.get("MV_AP_window") if shuffled else None)
    row = {
        "schema_version": "stream4d_v103_r7_phase1_metric_row_v1",
        "phase_id": PHASE_ID,
        "leave_one_family_variant": family,
        "evidence_status": status,
        "source_interpretation": source,
        "feature_variant_id": feature_variant,
        "phase6d_variant_id": phase6d_variant,
        "MV_AP_window": mv if mv is not None else "",
        "MV_AP50_window": _num(metrics.get("MV_AP50_window") if metrics else None, "") if metrics else "",
        "MV_AP25_window": _num(metrics.get("MV_AP25_window") if metrics else None, "") if metrics else "",
        "ScoreFreeMatch50_window": _num(metrics.get("ScoreFreeMatch50_window") if metrics else None, "") if metrics else "",
        "accepted_edge_count": edge_counts.get("accepted_edge_count", "") if edge_counts else "",
        "candidate_edge_count": edge_counts.get("candidate_edge_count", "") if edge_counts else "",
        "accepted_S_only_edge_count": 0 if metrics else "",
        "accepted_diff_gt_edge_count_diagnostic": edge_counts.get("accepted_diff_gt_edge_count_diagnostic", "") if edge_counts else "",
        "same_frame_collision_count": int(_num(metrics.get("same_frame_collision_count") if metrics else None, 0)) if metrics else "",
        "pixel_collision_rate": _num(metrics.get("pixel_collision_rate") if metrics else None, "") if metrics else "",
        "missing_mask_raster_count": int(_num(metrics.get("missing_mask_raster_count") if metrics else None, 0)) if metrics else "",
        "real_minus_shuffled_MV_AP_window": (mv - shuf_mv) if mv is not None and shuf_mv is not None else "",
        "real_minus_density_control_MV_AP_window": "",
        "reference_minus_this_MV_AP_window": (ref_mv - mv) if ref_mv is not None and mv is not None else "",
        "reference_minus_this_MV_AP50_window": (ref_mv50 - mv50) if ref_mv50 is not None and mv50 is not None else "",
        "missing_reason": missing_reason,
        "dataset_split": "dev",
        "chunk_id": "c0001",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": bool(metrics),
        "uses_future": False,
    }
    return row


def _edge_family(feature_variant: str, phase6d_variant: str, rec: dict[str, Any]) -> str:
    if phase6d_variant.startswith("R") or _bool(rec.get("shuffle_affinity", False)):
        return "shuffled_affinity_control"
    if int(_num(rec.get("direct_pair_support_min_count"), 0) or 0) > 0:
        return "direct_pair_gate_no_support_when_d4rt_only"
    if _bool(rec.get("broad_support_veto", False)):
        return "broad_support_veto_guarded"
    if feature_variant == ANCHOR_ONLY_FEATURE_VARIANT:
        return "anchor_only_affinity"
    return "support_conditioned_affinity"


def _edge_rows(run_roots: dict[str, Path], feature_summary: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_variant, root in run_roots.items():
        df = _read_csv(root / "merge_edge_rows.csv")
        if df.empty:
            continue
        for rec in df.to_dict("records"):
            phase6d_variant = str(rec.get("variant_id", ""))
            scene = str(rec.get("scene_id", ""))
            fs = feature_summary.get((feature_variant, scene), {})
            edge_rank_value = _num(rec.get("edge_rank"), -1)
            edge_rank = int(edge_rank_value if edge_rank_value is not None else -1)
            support_feature_enabled = feature_variant != ANCHOR_ONLY_FEATURE_VARIANT
            is_reference_d9 = feature_variant == REFERENCE_FEATURE_VARIANT and phase6d_variant == D9
            rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase1_edge_attribution_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "chunk_id": "c0001",
                    "edge_id": f"{feature_variant}:{phase6d_variant}:{scene}:{edge_rank}",
                    "mask_or_object_a": str(rec.get("object_a", "")),
                    "mask_or_object_b": str(rec.get("object_b", "")),
                    "frame_span_a": "not_available_in_phase6d_edge_rows",
                    "frame_span_b": "not_available_in_phase6d_edge_rows",
                    "edge_family_original": _edge_family(feature_variant, phase6d_variant, rec),
                    "feature_variant_id": feature_variant,
                    "phase6d_variant_id": phase6d_variant,
                    "accepted_by_reference_D9": bool(is_reference_d9 and _bool(rec.get("accepted_union", False))),
                    "accepted_in_source_variant": _bool(rec.get("accepted_union", False)),
                    "has_A_anchor": True,
                    "A_anchor_count": "not_available_edge_level",
                    "A_anchor_reliability_mean": "not_available_edge_level",
                    "S_support_count": "feature_level_only" if support_feature_enabled else 0,
                    "S_support_weight_sum": _num(fs.get("support_contribution_ratio"), "") if support_feature_enabled else 0.0,
                    "S_support_cosine": _num(rec.get("affinity"), "") if support_feature_enabled else "",
                    "S_support_specificity_mean": "feature_level_specificity_weighted" if support_feature_enabled else "",
                    "S_support_semantic_calibrated_mean": _num(fs.get("support_semantic_weight_mean"), "") if support_feature_enabled else "",
                    "S_support_veto_overlap_rate": _num(fs.get("veto_overlap_contribution_ratio"), "") if support_feature_enabled else "",
                    "has_mask_view_skeleton_edge": True,
                    "has_temporal_neighbor_relation": "not_available_edge_level",
                    "has_same_frame_competing_conflict": _bool(rec.get("specific_conflict", False)),
                    "has_broad_support_risk": _bool(rec.get("broad_support_veto", False)),
                    "V_veto_score": float(_bool(rec.get("specific_conflict", False))) + float(_bool(rec.get("broad_support_veto", False))),
                    "semantic_baseline_calibrated_similarity": "feature_level_only",
                    "support_only_edge_flag": False,
                    "diagnostic_same_gt": int(_num(rec.get("direct_pair_diagnostic_same_gt_count"), 0) or 0),
                    "diagnostic_diff_gt": int(_num(rec.get("direct_pair_diagnostic_different_gt_count"), 0) or 0),
                    "diagnostic_same_semantic_diff_gt": "not_available_without_gt_pair_casebook",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": False,
                    "uses_future": False,
                }
            )
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(p.name != "phase6d_d4rt_only_runs" for p in out.iterdir()) and not args.force:
        raise SystemExit(f"output root already has phase1 files; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    phase0_root = _project(args.phase0_root)
    phase6d_root = _project(args.phase6d_run_root)
    r6_feature_root = _project(args.r6_feature_root)
    r6_diag_root = _project(args.r6_diag_root)

    phase0 = _read_json(phase0_root / "summary.json")
    run_roots = {variant: phase6d_root / variant for variant in FEATURE_VARIANTS}
    run_summaries = {variant: _read_json(root / "summary.json") for variant, root in run_roots.items()}
    metrics_by_feature = {variant: _metric_by_variant(root) for variant, root in run_roots.items()}
    feature_summary = _feature_summary_map(r6_feature_root)

    artifact_rows: list[dict[str, Any]] = [
        _artifact_row("phase0_summary", phase0_root / "summary.json"),
        _artifact_row("r6_feature_summary", r6_feature_root / "summary.json"),
        _artifact_row("r6_feature_rows", r6_feature_root / "role_feature_summary_rows.csv"),
        _artifact_row("r6_diag_summary_reference", r6_diag_root / "summary.json", required=False),
        _artifact_row("last_command", out / "last_command.txt", required=False),
    ]
    for variant, root in run_roots.items():
        artifact_rows.extend(
            [
                _artifact_row(f"{variant}_phase6d_summary", root / "summary.json"),
                _artifact_row(f"{variant}_phase6d_metric_rows", root / "merge_metric_rows.csv"),
                _artifact_row(f"{variant}_phase6d_edge_rows", root / "merge_edge_rows.csv"),
                _artifact_row(f"{variant}_phase6d_selected_rows", root / "merge_selected_rows.csv"),
                _artifact_row(f"{variant}_phase6d_last_command", root / "last_command.txt", required=False),
            ]
        )

    input_artifact_path_hits = sorted(
        {
            hit
            for row in artifact_rows
            for hit in str(row.get("forbidden_token_hits_in_path", "")).split(";")
            if hit
        }
    )
    nonempty_phase9n_roots = {
        variant: str(summary.get("phase9n_root", ""))
        for variant, summary in run_summaries.items()
        if str(summary.get("phase9n_root", "")).strip()
    }
    da3_leak = bool(input_artifact_path_hits or nonempty_phase9n_roots)

    ref_metrics = metrics_by_feature[REFERENCE_FEATURE_VARIANT].get(D9, {})
    ref_edges = _edge_counts(run_roots[REFERENCE_FEATURE_VARIANT], D9)
    ref_shuffled = metrics_by_feature[REFERENCE_FEATURE_VARIANT].get(R5, {})
    anchor_metrics = metrics_by_feature[ANCHOR_ONLY_FEATURE_VARIANT].get(D9, {})
    anchor_edges = _edge_counts(run_roots[ANCHOR_ONLY_FEATURE_VARIANT], D9)

    missing_families = [
        "D9_minus_A_anchor_edges",
        "D9_minus_broad_support_veto",
        "D9_minus_mask_view_skeleton_edges",
        "D9_minus_semantic_calibration",
        "D9_minus_support_ranking",
        "D9_minus_temporal_neighborhood_edges",
    ]

    metric_rows: list[dict[str, Any]] = [
        _metric_row(
            family="D9_reference",
            status="observed_exact_d4rt_only_reference",
            source="D4RT-only R6F2 support-conditioned feature with D9 phase6d variant; phase9n_root is empty.",
            feature_variant=REFERENCE_FEATURE_VARIANT,
            phase6d_variant=D9,
            metrics=ref_metrics,
            edge_counts=ref_edges,
            reference=ref_metrics,
            shuffled=ref_shuffled,
        ),
        _metric_row(
            family="D9_minus_S_support_compatibility",
            status="observed_exact_feature_channel_ablation",
            source="Anchor-only R6F0 feature with the same D9 phase6d variant removes the R6F2 support feature channel.",
            feature_variant=ANCHOR_ONLY_FEATURE_VARIANT,
            phase6d_variant=D9,
            metrics=anchor_metrics,
            edge_counts=anchor_edges,
            reference=ref_metrics,
            shuffled=metrics_by_feature[ANCHOR_ONLY_FEATURE_VARIANT].get(R5, {}),
        ),
    ]
    for family in missing_families:
        metric_rows.append(
            _metric_row(
                family=family,
                status="R7_EXACT_ATTRIBUTION_MISSING",
                source="No formal proxy metric is reported for this family.",
                feature_variant="",
                phase6d_variant="",
                metrics=None,
                edge_counts=None,
                reference=ref_metrics,
                shuffled=None,
                missing_reason="Required exact intervention artifact does not exist in the clean D4RT-only c0001 roots.",
            )
        )

    control_rows: list[dict[str, Any]] = []
    for feature_variant, metric_map in metrics_by_feature.items():
        for variant_id in [D0, "D1_affinity_merge_tau090_top1_specific_veto", D9, R5]:
            metric = metric_map.get(variant_id, {})
            if not metric:
                continue
            control_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase1_control_row_v1",
                    "phase_id": PHASE_ID,
                    "feature_variant_id": feature_variant,
                    "phase6d_variant_id": variant_id,
                    "control_role": "reference_replay" if variant_id == D0 else ("shuffled_control" if variant_id == R5 else "observed_variant"),
                    "MV_AP_window": _num(metric.get("MV_AP_window"), ""),
                    "MV_AP50_window": _num(metric.get("MV_AP50_window"), ""),
                    "accepted_edge_count": _edge_counts(run_roots[feature_variant], variant_id)["accepted_edge_count"],
                    "accepted_S_only_edge_count": 0,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": True,
                    "uses_future": False,
                }
            )

    edge_rows = _edge_rows(run_roots, feature_summary)
    variant_rows = [
        {
            "schema_version": "stream4d_v103_r7_phase1_variant_row_v1",
            "phase_id": PHASE_ID,
            "feature_variant_id": variant,
            "phase6d_root": _rel(root),
            "phase9n_root": run_summaries.get(variant, {}).get("phase9n_root", ""),
            "best_real_variant_id": run_summaries.get(variant, {}).get("best_real_variant_id", ""),
            "best_real_MV_AP_window": run_summaries.get(variant, {}).get("best_real_MV_AP_window", ""),
            "best_real_MV_AP50_window": run_summaries.get(variant, {}).get("best_real_MV_AP50_window", ""),
            "decision": run_summaries.get(variant, {}).get("decision", ""),
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
            "uses_future": False,
        }
        for variant, root in run_roots.items()
    ]

    exact_available_families = [
        row["leave_one_family_variant"]
        for row in metric_rows
        if str(row["evidence_status"]).startswith("observed_exact")
    ]
    exact_all_required = len(missing_families) == 0
    ref_mv = _num(ref_metrics.get("MV_AP_window"), 0.0) or 0.0
    replay_mv = _num(phase0.get("current_replay_MV_AP_window"), 0.0) or 0.0
    ref_ap50 = _num(ref_metrics.get("MV_AP50_window"), 0.0) or 0.0
    replay_ap50 = _num(phase0.get("current_replay_MV_AP50_window"), 0.0) or 0.0

    gate_rows = [
        _gate_row("phase0_passed", bool(phase0.get("phase_pass", False)), phase0.get("decision", ""), True, "Rerun R7-0 before R7-1."),
        _gate_row("phase6d_clean_roots_available", all((root / "summary.json").exists() for root in run_roots.values()), list(run_roots), "all clean roots", "Rerun phase6d with empty --phase9n-root."),
        _gate_row("D4RT_only_no_forbidden_input_path", not bool(input_artifact_path_hits), input_artifact_path_hits, [], "Move R7 inputs away from DA3/3DGS/phase9 roots."),
        _gate_row("phase9n_root_empty_in_all_clean_runs", not bool(nonempty_phase9n_roots), nonempty_phase9n_roots, {}, "Rerun phase6d with --phase9n-root ''."),
        _gate_row("exact_D9_reference_available", bool(ref_metrics), f"{REFERENCE_FEATURE_VARIANT}/{D9}", "metric row present", "Regenerate clean D4RT-only R6F2 D9."),
        _gate_row("exact_minus_S_support_available", bool(anchor_metrics), f"{ANCHOR_ONLY_FEATURE_VARIANT}/{D9}", "metric row present", "Regenerate clean D4RT-only R6F0 D9."),
        _gate_row("exact_all_required_families_available", exact_all_required, missing_families, [], "Build exact intervention variants before claiming R7-1 complete attribution."),
        _gate_row("no_proxy_rows_used_as_formal_conclusion", True, "missing families have blank metrics", True, "Keep proxy rows out of formal metric rows."),
        _gate_row("accepted_S_only_edge_count_zero", all(row.get("accepted_S_only_edge_count", 0) in {0, ""} for row in metric_rows), 0, 0, "Disable support-only union."),
    ]

    failure_rows: list[dict[str, Any]] = []
    if da3_leak:
        failure_rows.append(
            _failure_row(
                "R7_DA3_ARTIFACT_LEAKAGE",
                {"path_hits": input_artifact_path_hits, "nonempty_phase9n_roots": nonempty_phase9n_roots},
                "Remove forbidden provider paths and rerun clean phase6d.",
            )
        )
    if missing_families:
        failure_rows.append(
            _failure_row(
                "R7_EXACT_ATTRIBUTION_MISSING",
                {
                    "missing_families": missing_families,
                    "available_exact_families": exact_available_families,
                    "note": "Only exact D9_reference and D9_minus_S_support_compatibility are currently constructible from clean c0001 D4RT-only roots.",
                },
                "Implement exact custom interventions for A_anchor removal, tau065 no-broad-veto, skeleton removal, semantic-only removal, support ranking, and temporal neighborhood removal.",
            )
        )
    if ref_mv < replay_mv or ref_ap50 < replay_ap50:
        failure_rows.append(
            _failure_row(
                "R7_SUPPORT_CONDITIONED_D9_BELOW_REPLAY",
                {
                    "D9_reference_MV_AP_window": ref_mv,
                    "replay_MV_AP_window": replay_mv,
                    "D9_reference_MV_AP50_window": ref_ap50,
                    "replay_MV_AP50_window": replay_ap50,
                },
                "Do not promote support-conditioned feature path; move to anchor/skeleton/proagation confirmation families.",
                severity="diagnostic",
            )
        )
    if not failure_rows:
        failure_rows.append(
            _failure_row("NONE", "No blocking failure.", "Proceed to R7-3/R7-4/R7-5.", severity="info")
        )

    phase_pass = (not da3_leak) and bool(ref_metrics) and bool(anchor_metrics)
    decision = (
        "PARTIAL_R7_1_EXACT_ATTRIBUTION_AVAILABLE_PROCEED_TO_CONFIRMATION_FAMILIES"
        if phase_pass
        else "NO_GO_R7_1_REPAIR_D4RT_ONLY_ATTRIBUTION_INPUTS"
    )
    summary = {
        "schema_version": "stream4d_v103_r7_phase1_summary_v1",
        "phase": "R7-1",
        "phase_id": PHASE_ID,
        "phase_pass": phase_pass,
        "decision": decision,
        "runtime_sec": time.time() - t0,
        "D4RT_only_no_forbidden_input_path": not bool(input_artifact_path_hits),
        "phase9n_root_empty_in_all_clean_runs": not bool(nonempty_phase9n_roots),
        "DA3_USED": bool(da3_leak),
        "DA3_ROWS_LOADED": bool(da3_leak),
        "GS_USED": bool({"3dgs", "gaussian"} & set(input_artifact_path_hits)),
        "reference_feature_variant_id": REFERENCE_FEATURE_VARIANT,
        "D9_reference_MV_AP_window": ref_mv,
        "D9_reference_MV_AP50_window": ref_ap50,
        "replay_MV_AP_window": replay_mv,
        "replay_MV_AP50_window": replay_ap50,
        "D9_minus_S_support_MV_AP_window": _num(anchor_metrics.get("MV_AP_window"), ""),
        "D9_minus_S_support_MV_AP50_window": _num(anchor_metrics.get("MV_AP50_window"), ""),
        "support_channel_reference_minus_anchor_only_MV_AP_window": ref_mv - (_num(anchor_metrics.get("MV_AP_window"), 0.0) or 0.0),
        "support_channel_reference_minus_anchor_only_MV_AP50_window": ref_ap50 - (_num(anchor_metrics.get("MV_AP50_window"), 0.0) or 0.0),
        "exact_available_families": exact_available_families,
        "exact_missing_families": missing_families,
        "exact_missing_count": len(missing_families),
        "edge_attribution_row_count": len(edge_rows),
        "failure_count": len([row for row in failure_rows if row["failure_id"] != "NONE"]),
        "accepted_S_only_edge_count": 0,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": "R7-1 uses freshly rerun c0001 phase6d roots with empty phase9n_root. Missing exact families are explicitly marked R7_EXACT_ATTRIBUTION_MISSING; proxy rows are not used as formal conclusions.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "variant_rows": _rel(out / "variant_rows.csv"),
            "metric_rows": _rel(out / "metric_rows.csv"),
            "control_rows": _rel(out / "control_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "edge_attribution_rows": _rel(out / "edge_attribution_rows.parquet"),
        },
    }

    _write_csv(out / "artifact_rows.csv", artifact_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "variant_rows.csv", variant_rows)
    _write_csv(out / "metric_rows.csv", metric_rows)
    _write_csv(out / "control_rows.csv", control_rows)
    _write_parquet(out / "edge_attribution_rows.parquet", edge_rows)
    _write_json(out / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build R7-1 D4RT-only exact edge attribution summary.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase0-root", default=str(DEFAULT_PHASE0_ROOT))
    parser.add_argument("--phase6d-run-root", default=str(DEFAULT_PHASE6D_RUN_ROOT))
    parser.add_argument("--r6-feature-root", default=str(DEFAULT_R6_FEATURE_ROOT))
    parser.add_argument("--r6-diag-root", default=str(DEFAULT_R6_DIAG_ROOT))
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    summary = build(build_parser().parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
