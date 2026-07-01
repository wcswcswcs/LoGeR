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


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v103_phase3_fast_carrier_reliability_filter import _compute_scene_arrays, _ensure_mmap_cache, _load_cached, _project  # noqa: E402
from diagnose_v103_phase3_reliable_carrier_gt import (  # noqa: E402
    SOURCE_CODEBOOK,
    _carrier_gt_stats,
    _load_gt_stack,
    _retained_phase3_semantics,
    _variant_by_id,
)


PHASE_ID = "v103_phase3_filter_failure_attribution"
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase3_filter_failure_attribution_r1"
DEFAULT_PHASE3_ROOT = AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_q5c_objlike16384_source_balanced_repair3"
DEFAULT_SCENE0011_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_SCENE0050_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_first32"


METRIC_KEYS = [
    ("reliability_s2", "higher"),
    ("r_geo", "higher"),
    ("r_mask", "higher"),
    ("r_sem", "higher"),
    ("semantic_short_range_stability", "higher"),
    ("semantic_contradiction_rate", "lower"),
    ("competing_mask_conflict_rate", "lower"),
    ("combined_semantic_competing_conflict_rate", "lower"),
    ("mask_boundary_hit_rate", "lower"),
    ("source_risk_score", "lower"),
    ("semantic_pair_count", "higher"),
    ("broad_mask_participation_rate", "lower"),
    ("normalized_jitter", "lower"),
    ("object_like_mask_rate", "higher"),
    ("visibility_rate", "higher"),
    ("in_image_rate", "higher"),
]


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
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_rate(num: int | float, den: int | float) -> float:
    return float(num) / max(float(den), 1.0)


def _stat(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values)[np.asarray(mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
    }


def _retention_row(scene: str, variant_id: str, retained: np.ndarray, clean: np.ndarray, multi: np.ndarray, eligible: np.ndarray) -> dict[str, Any]:
    retained = np.asarray(retained, dtype=bool)
    clean = np.asarray(clean, dtype=bool)
    multi = np.asarray(multi, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    clean_total = int(np.count_nonzero(clean))
    multi_total = int(np.count_nonzero(multi))
    retained_clean = int(np.count_nonzero(retained & clean))
    retained_multi = int(np.count_nonzero(retained & multi))
    retained_eligible = int(np.count_nonzero(retained & eligible))
    clean_retention = _safe_rate(retained_clean, clean_total)
    multi_retention = _safe_rate(retained_multi, multi_total)
    return {
        "schema_version": "stream4d_v103_phase3_failure_retention_row_v1",
        "phase_id": PHASE_ID,
        "scene_id": scene,
        "variant_id": variant_id,
        "total_carrier_count": int(retained.shape[0]),
        "retained_carrier_count": int(np.count_nonzero(retained)),
        "eligible_gt_carrier_count": int(np.count_nonzero(eligible)),
        "clean_single_gt_total": clean_total,
        "multi_gt_total": multi_total,
        "retained_clean_single_gt_count": retained_clean,
        "retained_multi_gt_count": retained_multi,
        "clean_retention_rate": clean_retention,
        "multi_gt_retention_rate": multi_retention,
        "good_filtered_out_rate": 1.0 - clean_retention,
        "bad_filtered_out_rate": 1.0 - multi_retention,
        "multi_over_clean_retention_ratio": multi_retention / max(clean_retention, 1e-12),
        "retained_clean_precision": _safe_rate(retained_clean, retained_eligible),
        "retained_multi_gt_rate": _safe_rate(retained_multi, retained_eligible),
        "failure_mode_hint": (
            "bad_retained_more_than_good"
            if multi_retention > clean_retention
            else "good_and_bad_filtered_similarly_or_good_retained_more"
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "diagnostic_only": True,
    }


def _score_rows(scene: str, variant_id: str, retained: np.ndarray, arrays: dict[str, np.ndarray], clean: np.ndarray, multi: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    retained = np.asarray(retained, dtype=bool)
    for key, direction in METRIC_KEYS:
        values = np.asarray(arrays[key])
        clean_stats = _stat(values, clean)
        multi_stats = _stat(values, multi)
        retained_clean_stats = _stat(values, retained & clean)
        retained_multi_stats = _stat(values, retained & multi)
        if direction == "higher":
            clean_advantage = clean_stats["p50"] - multi_stats["p50"]
        else:
            clean_advantage = multi_stats["p50"] - clean_stats["p50"]
        rows.append(
            {
                "schema_version": "stream4d_v103_phase3_failure_score_separation_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "variant_id": variant_id,
                "metric_key": key,
                "clean_direction": direction,
                "clean_p50": clean_stats["p50"],
                "multi_gt_p50": multi_stats["p50"],
                "clean_minus_multi_good_direction_p50": clean_advantage,
                "clean_mean": clean_stats["mean"],
                "multi_gt_mean": multi_stats["mean"],
                "retained_clean_p50": retained_clean_stats["p50"],
                "retained_multi_gt_p50": retained_multi_stats["p50"],
                "retained_clean_mean": retained_clean_stats["mean"],
                "retained_multi_gt_mean": retained_multi_stats["mean"],
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
                "diagnostic_only": True,
            }
        )
    return rows


def _source_rows(scene: str, variant_id: str, retained: np.ndarray, arrays: dict[str, np.ndarray], clean: np.ndarray, multi: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = np.asarray(arrays["query_source_code"], dtype=np.int16)
    retained = np.asarray(retained, dtype=bool)
    for code, name in SOURCE_CODEBOOK.items():
        src = source == int(code)
        if not np.any(src):
            continue
        clean_total = int(np.count_nonzero(src & clean))
        multi_total = int(np.count_nonzero(src & multi))
        retained_clean = int(np.count_nonzero(src & clean & retained))
        retained_multi = int(np.count_nonzero(src & multi & retained))
        rows.append(
            {
                "schema_version": "stream4d_v103_phase3_failure_source_retention_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "variant_id": variant_id,
                "query_source_code": int(code),
                "query_source": name,
                "source_total": int(np.count_nonzero(src)),
                "source_clean_total": clean_total,
                "source_multi_gt_total": multi_total,
                "source_retained_total": int(np.count_nonzero(src & retained)),
                "source_retained_clean_count": retained_clean,
                "source_retained_multi_gt_count": retained_multi,
                "source_clean_retention_rate": _safe_rate(retained_clean, clean_total),
                "source_multi_gt_retention_rate": _safe_rate(retained_multi, multi_total),
                "source_retained_multi_gt_rate": _safe_rate(retained_multi, int(np.count_nonzero(src & retained & (clean | multi)))),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": True,
                "diagnostic_only": True,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GT-only attribution of v103 Phase3 carrier filter failure modes.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase3-root", default=str(DEFAULT_PHASE3_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--scene0011-extra-variant-id", action="append", default=[])
    parser.add_argument("--scene0050-extra-variant-id", action="append", default=[])
    parser.add_argument("--diagnose-all-phase3-variants", action="store_true")
    parser.add_argument("--visible-threshold", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-gt-positive-obs", type=int, default=2)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase3_root = _project(args.phase3_root)
    phase3_summary = _read_json(phase3_root / "summary.json")
    selected_by_scene = {str(k): str(v) for k, v in dict(phase3_summary["selected_variant_by_scene"]).items()}
    all_variant_ids = [str(v) for v in phase3_summary.get("variant_ids", phase3_summary.get("evaluated_variant_ids", []))]
    specs = {
        "scene0011_00": {
            "phase2_root": _project(args.scene0011_phase2_root),
            "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_features.npz",
            "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
            "extra_variant_ids": [str(v) for v in args.scene0011_extra_variant_id],
        },
        "scene0050_00": {
            "phase2_root": _project(args.scene0050_phase2_root),
            "semantic_npz": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_features.npz",
            "semantic_rows": AUDIT_ROOT / "v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
            "extra_variant_ids": [str(v) for v in args.scene0050_extra_variant_id],
        },
    }

    retention_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    implementation_rows: list[dict[str, Any]] = []
    diagnosed_variants: dict[str, list[str]] = {}
    for scene, spec in specs.items():
        scene_out = out / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        diag, _unused_a, _unused_b, arrays = _compute_scene_arrays(scene, spec, scene_out, int(args.cupy_device_id))
        cache_dir, _manifest = _ensure_mmap_cache(spec["phase2_root"])
        batch = _load_cached(cache_dir)
        gt_stack = _load_gt_stack(scene, [int(v) for v in diag["frame_ids"]], tuple(diag["masks"].shape[1:]))
        gt_stats = _carrier_gt_stats(
            batch=batch,
            gt_stack=gt_stack,
            visible_threshold=float(args.visible_threshold),
            confidence_threshold=float(args.confidence_threshold),
        )
        positive = np.asarray(gt_stats["gt_positive_count"], dtype=np.int64)
        unique_gt = np.asarray(gt_stats["unique_gt_count"], dtype=np.int64)
        eligible = positive >= int(args.min_gt_positive_obs)
        clean = eligible & (unique_gt == 1)
        multi = eligible & (unique_gt >= 2)

        variant_ids = [selected_by_scene[scene]]
        if args.diagnose_all_phase3_variants:
            variant_ids = [*all_variant_ids, *variant_ids]
        variant_ids.extend(spec["extra_variant_ids"])
        variant_ids = list(dict.fromkeys(variant_ids))
        diagnosed_variants[scene] = variant_ids

        sem = np.asarray(arrays["semantic_contradiction_rate"], dtype=np.float64)
        comp = np.asarray(arrays["competing_mask_conflict_rate"], dtype=np.float64)
        sem_pair = np.asarray(arrays["semantic_pair_count"], dtype=np.int64)
        r_sem = np.asarray(arrays["r_sem"], dtype=np.float64)
        implementation_rows.append(
            {
                "schema_version": "stream4d_v103_phase3_failure_implementation_audit_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "audit_id": "competing_mask_conflict_rate_independent_from_semantic_contradiction_rate",
                "observed": float(np.max(np.abs(comp - sem))) if comp.size else 0.0,
                "expected_for_independent_competing_signal": ">0 on scenes with distinct same-frame mask-boundary topology",
                "interpretation": (
                    "PASS: competing_mask_conflict_rate is independently computed from same-frame mask-boundary topology"
                    if (float(np.max(np.abs(comp - sem))) if comp.size else 0.0) > 0.0
                    else "FAIL: competing_mask_conflict_rate is still indistinguishable from semantic contradiction"
                ),
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": False,
            }
        )
        no_sem = (sem_pair == 0) & (sem == 0.0) & (r_sem == 1.0)
        implementation_rows.append(
            {
                "schema_version": "stream4d_v103_phase3_failure_implementation_audit_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "audit_id": "no_semantic_pair_treated_as_clean",
                "observed_count": int(np.count_nonzero(no_sem)),
                "observed_rate": float(np.mean(no_sem)) if no_sem.size else 0.0,
                "interpretation": "carriers without short-range semantic pairs receive semantic_contradiction=0 and r_sem=1",
                "uses_gt_for_prediction": False,
                "uses_gt_for_diagnostic": False,
            }
        )

        for variant_id in variant_ids:
            retained, _meta = _retained_phase3_semantics(_variant_by_id(variant_id), arrays, diag)
            retention_rows.append(_retention_row(scene, variant_id, retained, clean, multi, eligible))
            score_rows.extend(_score_rows(scene, variant_id, retained, arrays, clean, multi))
            source_rows.extend(_source_rows(scene, variant_id, retained, arrays, clean, multi))

    _write_csv(out / "retention_attribution_rows.csv", retention_rows)
    _write_csv(out / "score_separation_rows.csv", score_rows)
    _write_csv(out / "source_retention_rows.csv", source_rows)
    _write_csv(out / "implementation_audit_rows.csv", implementation_rows)

    summary = {
        "schema_version": "stream4d_v103_phase3_filter_failure_attribution_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "DIAGNOSTIC_ONLY_PHASE3_FAILURE_ATTRIBUTION",
        "phase3_root": _rel(phase3_root),
        "selected_variant_by_scene": selected_by_scene,
        "diagnosed_variant_ids_by_scene": diagnosed_variants,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": True,
        "diagnostic_only": True,
        "truthfulness_note": "GT labels are used only to attribute why a preselected GT-free Phase3 filter failed; they are not used to tune thresholds or select method parameters.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "retention_attribution_rows": _rel(out / "retention_attribution_rows.csv"),
            "score_separation_rows": _rel(out / "score_separation_rows.csv"),
            "source_retention_rows": _rel(out / "source_retention_rows.csv"),
            "implementation_audit_rows": _rel(out / "implementation_audit_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
