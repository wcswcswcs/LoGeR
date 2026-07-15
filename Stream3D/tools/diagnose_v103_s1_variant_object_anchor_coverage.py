#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_v103_supp_phaseS1_multirole_carriers import (  # noqa: E402
    ROLE_VARIANTS,
    _anchor_score,
    _top_subset,
)
from diagnose_v103_object_specific_carrier_support import (  # noqa: E402
    DEFAULT_SCENE0011_PHASE2,
    DEFAULT_SCENE0050_PHASE2,
    DEFAULT_S1_ROOT,
    DEFAULT_S3_ROOT,
    PHASE_ID as OBJECT_SUPPORT_PHASE_ID,
    _compute_role_hits,
    _current_object_rows,
    _jsonable,
    _load_phase2_scene_fast,
    _project,
    _project_labels_for_indices_fast,
    _rel,
    _write_csv,
    _write_json,
)


PHASE_ID = "v103_s1_variant_object_anchor_coverage_diagnostic"
SCHEMA_PREFIX = "stream4d_v103_s1_variant_object_anchor_coverage"
DEFAULT_OUT = AUDIT_ROOT / "v103_s1_variant_object_anchor_coverage_purityfirst_r1"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _scene_phase2_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }


def _arrays_from_role_frame(scene_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    cols = [
        "carrier_id",
        "query_source_code",
        "src_frame",
        "visibility_rate",
        "in_image_rate",
        "object_like_mask_rate",
        "broad_mask_participation_rate",
        "semantic_contradiction_rate",
        "competing_mask_conflict_rate",
        "normalized_jitter",
        "source_risk_score",
        "reliability_s2",
    ]
    arrays: dict[str, np.ndarray] = {}
    for col in cols:
        if col not in scene_frame:
            raise RuntimeError(f"missing S1 carrier_role_rows column: {col}")
        arrays[col] = scene_frame[col].to_numpy(copy=False)
    return arrays


def _variant_anchor_mask(variant: dict[str, Any], arrays: dict[str, np.ndarray]) -> np.ndarray:
    source = np.asarray(arrays["query_source_code"], dtype=np.int16)
    visibility = np.asarray(arrays["visibility_rate"], dtype=np.float64)
    in_image = np.asarray(arrays["in_image_rate"], dtype=np.float64)
    object_rate = np.asarray(arrays["object_like_mask_rate"], dtype=np.float64)
    broad = np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)
    sem = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)
    competing = np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float64)
    jitter = np.asarray(arrays["normalized_jitter"], dtype=np.float64)
    source_risk = np.asarray(arrays["source_risk_score"], dtype=np.float64)
    allowed_anchor_sources = np.isin(source, np.asarray([1, 2, 3, 7, 8], dtype=np.int16))
    anchor_pred = (
        allowed_anchor_sources
        & (in_image > 0.0)
        & (visibility >= float(variant["anchor_min_visibility"]))
        & (in_image >= float(variant["anchor_min_in_image"]))
        & (object_rate >= float(variant["anchor_min_object_like_rate"]))
        & (broad <= float(variant["anchor_max_broad"]))
        & (sem <= float(variant["anchor_max_semantic_contradiction"]))
        & (competing <= float(variant["anchor_max_competing"]))
        & (jitter <= float(variant["anchor_max_jitter"]))
    )
    anchor_raw = _top_subset(anchor_pred, _anchor_score(arrays), float(variant["anchor_top_rate"]), min_count=256)
    veto_raw = (
        (in_image > 0.0)
        & (
            (broad > float(variant["veto_max_safe_broad"]))
            | (competing >= float(variant["veto_min_competing"]))
            | (sem > float(variant["veto_min_semantic_contradiction"]))
            | (jitter >= float(variant["veto_min_jitter"]))
            | (source_risk >= 0.75)
        )
    )
    return anchor_raw & ~veto_raw


def _per_frame_p10(arrays: dict[str, np.ndarray], mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    labels, counts = np.unique(np.asarray(arrays["src_frame"], dtype=np.int16)[mask], return_counts=True)
    if labels.size == 0:
        return 0.0
    return float(np.percentile(counts.astype(np.float64), 10))


def _role_metric(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {
            "carrier_count": 0,
            "per_frame_carrier_count_p10": 0.0,
            "broad_mask_participation_rate": 0.0,
            "short_range_semantic_contradiction_rate": 0.0,
            "competing_mask_conflict_rate": 0.0,
            "jitter_norm_p90": 0.0,
        }
    return {
        "carrier_count": int(np.count_nonzero(mask)),
        "per_frame_carrier_count_p10": _per_frame_p10(arrays, mask),
        "broad_mask_participation_rate": float(np.mean(np.asarray(arrays["broad_mask_participation_rate"], dtype=np.float64)[mask])),
        "short_range_semantic_contradiction_rate": float(
            np.mean(np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)[mask])
        ),
        "competing_mask_conflict_rate": float(np.mean(np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float64)[mask])),
        "jitter_norm_p90": float(np.percentile(np.asarray(arrays["normalized_jitter"], dtype=np.float64)[mask], 90)),
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["min_scene_anchor_object_hit_rate"]),
        float(row["mean_scene_anchor_object_hit_rate"]),
        -float(row["max_scene_broad_mask_participation_rate"]),
        -float(row["max_scene_semantic_contradiction_rate"]),
        -float(row["max_scene_competing_mask_conflict_rate"]),
        -float(row["max_scene_jitter_norm_p90"]),
        float(row["min_scene_anchor_carrier_count"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose S1 R0-R4 object-specific A_anchor coverage.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phaseS1-root", default=str(DEFAULT_S1_ROOT))
    parser.add_argument("--phaseS3-root", default=str(DEFAULT_S3_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--phaseS3-variant-id", default="S3_V1_anchor_positive_only")
    parser.add_argument("--chunk-id", default="c0001")
    parser.add_argument("--anchor-object-hit-rate-floor", type=float, default=0.50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")

    phaseS1_root = _project(args.phaseS1_root)
    phaseS3_root = _project(args.phaseS3_root)
    role_frame = pd.read_parquet(phaseS1_root / "carrier_role_rows.parquet")
    current_rows = _current_object_rows(phaseS3_root, str(args.phaseS3_variant_id), str(args.chunk_id))
    scene_phase2_roots = _scene_phase2_roots(args)

    scene_variant_rows: list[dict[str, Any]] = []
    object_variant_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []

    for scene, phase2_root in scene_phase2_roots.items():
        print(f"[{PHASE_ID}] scene={scene} load mmap phase2", file=sys.stderr, flush=True)
        scene_current = current_rows[current_rows["scene_id"].astype(str) == scene].copy()
        scene_roles = role_frame[role_frame["scene_id"].astype(str) == scene].reset_index(drop=True)
        arrays = _arrays_from_role_frame(scene_roles)
        summary, masks, batch, batch_backend = _load_phase2_scene_fast(phase2_root)
        batch_carrier_ids = np.asarray(batch["carrier_id"], dtype=np.int64)
        role_carrier_ids_all = np.asarray(arrays["carrier_id"], dtype=np.int64)
        if not np.array_equal(batch_carrier_ids, role_carrier_ids_all):
            lookup = pd.Series(np.arange(batch_carrier_ids.shape[0], dtype=np.int64), index=batch_carrier_ids)
            role_to_batch = lookup.reindex(role_carrier_ids_all).to_numpy(dtype=np.int64)
        else:
            role_to_batch = np.arange(batch_carrier_ids.shape[0], dtype=np.int64)

        for variant in ROLE_VARIANTS:
            variant_id = str(variant["variant_id"])
            anchor_mask = _variant_anchor_mask(variant, arrays)
            role_indices = role_to_batch[np.flatnonzero(anchor_mask)]
            role_carrier_ids = role_carrier_ids_all[anchor_mask]
            metric = _role_metric(arrays, anchor_mask)
            print(
                f"[{PHASE_ID}] scene={scene} variant={variant_id} A_anchor carriers={role_carrier_ids.shape[0]}",
                file=sys.stderr,
                flush=True,
            )
            labels, ok, weights, backend, runtime = _project_labels_for_indices_fast(
                batch=batch,
                masks=masks,
                carrier_indices=role_indices.astype(np.int64, copy=False),
                batch_backend=batch_backend,
            )
            role_hits, hit_meta = _compute_role_hits(
                scene=scene,
                role_name="A_anchor",
                labels=labels,
                ok=ok,
                weights=weights,
                role_carrier_ids=role_carrier_ids,
                scene_rows=scene_current,
            )
            object_count = int(scene_current["mv_object_id"].astype(str).nunique())
            anchor_object_hit_rate = float(hit_meta["object_with_role_hit_rate"])
            scene_variant_rows.append(
                {
                    "schema_version": f"{SCHEMA_PREFIX}_scene_variant_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "variant_id": variant_id,
                    "object_count": object_count,
                    "A_anchor_object_hit_rate": anchor_object_hit_rate,
                    "A_anchor_object_hit_count": int(round(anchor_object_hit_rate * object_count)),
                    "A_anchor_object_unique_carrier_count_p10": hit_meta["object_role_unique_carrier_count_p10"],
                    "A_anchor_object_unique_carrier_count_median": hit_meta["object_role_unique_carrier_count_median"],
                    **metric,
                    "projection_backend": backend,
                    "projection_runtime_sec": float(runtime),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": False,
                }
            )
            projection_rows.append(
                {
                    "schema_version": f"{SCHEMA_PREFIX}_projection_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "variant_id": variant_id,
                    "phase2_root": _rel(phase2_root),
                    "projection_backend": backend,
                    "projection_runtime_sec": float(runtime),
                    "carrier_count": int(role_carrier_ids.shape[0]),
                    "uses_gt_for_prediction": False,
                }
            )
            for oid, row in role_hits.items():
                object_variant_rows.append(
                    {
                        "schema_version": f"{SCHEMA_PREFIX}_object_variant_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene,
                        "variant_id": variant_id,
                        "source_local_object_id": oid,
                        **row,
                        "uses_gt_for_prediction": False,
                    }
                )

    by_variant: list[dict[str, Any]] = []
    scene_df = pd.DataFrame(scene_variant_rows)
    for variant_id, group in scene_df.groupby("variant_id", sort=True):
        by_variant.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}_variant_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": str(variant_id),
                "scene_count": int(len(group)),
                "min_scene_anchor_object_hit_rate": float(group["A_anchor_object_hit_rate"].min()),
                "mean_scene_anchor_object_hit_rate": float(group["A_anchor_object_hit_rate"].mean()),
                "max_scene_broad_mask_participation_rate": float(group["broad_mask_participation_rate"].max()),
                "max_scene_semantic_contradiction_rate": float(group["short_range_semantic_contradiction_rate"].max()),
                "max_scene_competing_mask_conflict_rate": float(group["competing_mask_conflict_rate"].max()),
                "max_scene_jitter_norm_p90": float(group["jitter_norm_p90"].max()),
                "min_scene_anchor_carrier_count": int(group["carrier_count"].min()),
                "all_scenes_anchor_object_hit_rate_ge_floor": bool(
                    (group["A_anchor_object_hit_rate"] >= float(args.anchor_object_hit_rate_floor)).all()
                ),
                "all_scenes_broad_le_0p15": bool((group["broad_mask_participation_rate"] <= 0.15).all()),
                "all_scenes_semantic_contradiction_le_0p20": bool(
                    (group["short_range_semantic_contradiction_rate"] <= 0.20).all()
                ),
                "all_scenes_competing_le_0p15": bool((group["competing_mask_conflict_rate"] <= 0.15).all()),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": False,
            }
        )
    selected = max(by_variant, key=_selection_key)
    gate_rows = [
        {
            "schema_version": f"{SCHEMA_PREFIX}_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "selected_variant_all_scenes_anchor_object_hit_rate_ge_floor",
            "pass": bool(selected["all_scenes_anchor_object_hit_rate_ge_floor"]),
            "observed": {
                row["scene_id"]: row["A_anchor_object_hit_rate"]
                for row in scene_variant_rows
                if row["variant_id"] == selected["variant_id"]
            },
            "required": f">= {args.anchor_object_hit_rate_floor}",
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": False,
        }
    ]
    decision = "S1_VARIANT_OBJECT_ANCHOR_COVERAGE_REPAIR_CANDIDATE"
    if not bool(selected["all_scenes_anchor_object_hit_rate_ge_floor"]):
        decision = "S1_VARIANT_OBJECT_ANCHOR_COVERAGE_STILL_SPARSE"
    summary = {
        "schema_version": f"{SCHEMA_PREFIX}_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - started,
        "decision": decision,
        "selected_variant_id": selected["variant_id"],
        "selected_variant_min_scene_anchor_object_hit_rate": selected["min_scene_anchor_object_hit_rate"],
        "selected_variant_mean_scene_anchor_object_hit_rate": selected["mean_scene_anchor_object_hit_rate"],
        "inputs": {
            "phaseS1_root": phaseS1_root,
            "phaseS3_root": phaseS3_root,
            "phase2_roots": scene_phase2_roots,
            "phaseS3_variant_id": str(args.phaseS3_variant_id),
        },
        "outputs": {
            "summary": out / "summary.json",
            "scene_variant_rows": out / "scene_variant_rows.csv",
            "variant_rows": out / "variant_rows.csv",
            "object_variant_rows": out / "object_variant_rows.csv",
            "projection_rows": out / "projection_rows.csv",
            "gate_rows": out / "gate_rows.csv",
            "last_command": out / "last_command.txt",
        },
        "truthfulness_note": (
            "This diagnostic replays the pre-registered S1 A_anchor rules using S1 carrier reliability columns, "
            "then projects candidate anchors onto current S3 object masks through Phase2 mmap cache. "
            "It does not alter S1 artifacts and does not use GT for prediction."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": False,
    }
    _write_csv(out / "scene_variant_rows.csv", scene_variant_rows)
    _write_csv(out / "variant_rows.csv", by_variant)
    _write_csv(out / "object_variant_rows.csv", object_variant_rows)
    _write_csv(out / "projection_rows.csv", projection_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
