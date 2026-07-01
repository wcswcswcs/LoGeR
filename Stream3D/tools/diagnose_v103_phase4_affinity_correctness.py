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
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v103_phase3_fast_carrier_reliability_filter import _compute_scene_arrays, _ensure_mmap_cache, _load_cached, _project  # noqa: E402
from build_v103_phase4_primitive_affinity_feature import _carrier_affinity_risk_weight, _retained_for_variant, _variant_by_id  # noqa: E402


PHASE_ID = "v103_phase4_affinity_correctness_diagnostic"
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
DEFAULT_OUT = AUDIT_ROOT / "v103_phase4_affinity_correctness_r1"
DEFAULT_PHASE3_ROOT = AUDIT_ROOT / "v103_phase3_carrier_reliability_filter_q5c_objlike16384_source_balanced_repair3"
DEFAULT_PHASE4_ROOT = AUDIT_ROOT / "v103_phase4_primitive_affinity_q5c_repair3_r9_phase3parity"
DEFAULT_SCENE0011_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_first32"
DEFAULT_SCENE0050_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_first32"


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


def _scene_specs(scene0011_phase2: str, scene0050_phase2: str) -> dict[str, dict[str, Path]]:
    audit = STREAM3D_ROOT / "outputs/audit"
    return {
        "scene0011_00": {
            "phase2_root": _project(scene0011_phase2),
            "semantic_npz": audit / "v91_radio_mask_features_npz_scene0011/mask_features.npz",
            "semantic_rows": audit / "v91_radio_mask_features_npz_scene0011/mask_feature_rows.csv",
        },
        "scene0050_00": {
            "phase2_root": _project(scene0050_phase2),
            "semantic_npz": audit / "v91_radio_mask_features_npz_scene0050/mask_features.npz",
            "semantic_rows": audit / "v91_radio_mask_features_npz_scene0050/mask_feature_rows.csv",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit v103 Phase4 primitive affinity incidence and feature arithmetic.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phase3-root", default=str(DEFAULT_PHASE3_ROOT))
    parser.add_argument("--phase4-root", default=str(DEFAULT_PHASE4_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--scene0011-selected-variant-id", default="")
    parser.add_argument("--scene0050-selected-variant-id", default="")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    return parser


def main() -> int:
    t0 = time.time()
    args = build_parser().parse_args()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")
    phase3_root = _project(args.phase3_root)
    phase4_root = _project(args.phase4_root)
    phase3_summary = _read_json(phase3_root / "summary.json")
    selected_by_scene = {str(k): str(v) for k, v in dict(phase3_summary["selected_variant_by_scene"]).items()}
    selected_override_by_scene = {
        "scene0011_00": str(args.scene0011_selected_variant_id),
        "scene0050_00": str(args.scene0050_selected_variant_id),
    }
    selected_override_by_scene = {scene: variant for scene, variant in selected_override_by_scene.items() if variant}
    selected_by_scene.update(selected_override_by_scene)
    specs = _scene_specs(args.scene0011_phase2_root, args.scene0050_phase2_root)

    audit_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for scene, spec in specs.items():
        diag, _unused_a, _unused_b, arrays = _compute_scene_arrays(scene, spec, out / scene, int(args.cupy_device_id))
        variant_id = selected_by_scene[scene]
        variant = _variant_by_id(variant_id)
        retained, _meta = _retained_for_variant(variant, arrays, diag)
        cache_dir, _manifest = _ensure_mmap_cache(spec["phase2_root"])
        batch = _load_cached(cache_dir)
        carrier_ids = np.asarray(batch["carrier_id"], dtype=np.int64)
        retained_ids = carrier_ids[np.flatnonzero(retained)]
        id_to_global = {int(v): i for i, v in enumerate(carrier_ids.tolist())}

        inc_path = phase4_root / scene / "primitive_incidence_sparse.pt"
        feat_path = phase4_root / scene / "primitive_affinity_feature.pt"
        if not inc_path.exists() or not feat_path.exists():
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_phase4_affinity_correctness_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "scene_id": scene,
                    "failure_id": "phase4_artifact_missing",
                    "severity": "blocking",
                    "evidence": f"incidence_exists={inc_path.exists()} feature_exists={feat_path.exists()}",
                }
            )
            continue
        incidence = torch.load(inc_path, map_location="cpu")
        feature = torch.load(feat_path, map_location="cpu")
        affinity_risk_mode = str(incidence.get("affinity_risk_mode", "base"))
        phase4_ids = incidence["carrier_id"].cpu().numpy().astype(np.int64)
        feature_ids = feature["carrier_id"].cpu().numpy().astype(np.int64)
        carrier_exact = bool(np.array_equal(phase4_ids, retained_ids) and np.array_equal(feature_ids, retained_ids))

        local_idx = incidence["carrier_local_index"].cpu().numpy().astype(np.int64)
        frame_idx = incidence["frame_local_index"].cpu().numpy().astype(np.int64)
        mask_id = incidence["mask_id"].cpu().numpy().astype(np.int64)
        stored_b = incidence["B_ia"].cpu().numpy().astype(np.float64)
        local_in_range = (local_idx >= 0) & (local_idx < phase4_ids.shape[0])
        global_idx = np.asarray([id_to_global.get(int(phase4_ids[int(li)]), -1) if ok else -1 for li, ok in zip(local_idx.tolist(), local_in_range.tolist())], dtype=np.int64)
        valid_global = global_idx >= 0
        label_match = np.zeros_like(valid_global, dtype=bool)
        expected_b = np.zeros_like(stored_b, dtype=np.float64)
        if np.any(valid_global):
            gi = global_idx[valid_global]
            fi = frame_idx[valid_global]
            label_match[valid_global] = np.asarray(diag["labels"])[fi, gi].astype(np.int64) == mask_id[valid_global]
            affinity_weight = _carrier_affinity_risk_weight(variant, arrays, affinity_risk_mode)
            expected_b[valid_global] = (
                np.asarray(arrays["reliability_s2"], dtype=np.float64)[gi]
                * np.asarray(batch["visibility_prob"], dtype=np.float64)[fi, gi]
                * np.asarray(batch["confidence_prob"], dtype=np.float64)[fi, gi]
                * np.asarray(affinity_weight, dtype=np.float64)[gi]
            )
        b_err = np.abs(expected_b[valid_global] - stored_b[valid_global]) if np.any(valid_global) else np.zeros((0,), dtype=np.float64)
        feat = feature["feature"].cpu().numpy().astype(np.float32)
        feat_norm = np.linalg.norm(feat, axis=1) if feat.size else np.zeros((0,), dtype=np.float32)
        source_penalty_present = bool(variant.get("source_penalty") or variant.get("allowed_query_sources"))

        row = {
            "schema_version": "stream4d_v103_phase4_affinity_correctness_row_v1",
            "phase_id": PHASE_ID,
            "scene_id": scene,
            "selected_phase3_variant": variant_id,
            "carrier_id_exact_match_phase3_phase4": carrier_exact,
            "phase3_retained_count": int(retained_ids.shape[0]),
            "phase4_incidence_carrier_count": int(phase4_ids.shape[0]),
            "phase4_feature_carrier_count": int(feature_ids.shape[0]),
            "incidence_row_count": int(local_idx.shape[0]),
            "local_index_in_range_rate": float(np.mean(local_in_range)) if local_in_range.size else 1.0,
            "carrier_id_map_valid_rate": float(np.mean(valid_global)) if valid_global.size else 1.0,
            "incidence_label_match_rate": float(np.mean(label_match[valid_global])) if np.any(valid_global) else 1.0,
            "B_ia_max_abs_error_vs_recomputed": float(np.max(b_err)) if b_err.size else 0.0,
            "B_ia_p95_abs_error_vs_recomputed": float(np.percentile(b_err, 95)) if b_err.size else 0.0,
            "feature_norm_min_float16_saved": float(np.min(feat_norm)) if feat_norm.size else 0.0,
            "feature_norm_mean_float16_saved": float(np.mean(feat_norm)) if feat_norm.size else 0.0,
            "feature_norm_max_float16_saved": float(np.max(feat_norm)) if feat_norm.size else 0.0,
            "feature_zero_norm_count": int(np.count_nonzero(feat_norm == 0.0)),
            "source_penalty_or_source_veto_present_in_phase3_variant": source_penalty_present,
            "affinity_risk_mode": affinity_risk_mode,
            "B_ia_uses_base_reliability_s2_not_variant_score": affinity_risk_mode == "base",
            "design_note": (
                "Arithmetic audit recomputes the implemented formula. In base mode B_ia=reliability_s2*visibility*confidence; "
                "risk modes additionally multiply carrier-level source/competing risk weights."
            ),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": False,
        }
        audit_rows.append(row)
        checks = [
            ("carrier_id_exact_match_phase3_phase4", carrier_exact),
            ("local_index_in_range_rate_eq_1", row["local_index_in_range_rate"] == 1.0),
            ("carrier_id_map_valid_rate_eq_1", row["carrier_id_map_valid_rate"] == 1.0),
            ("incidence_label_match_rate_eq_1", row["incidence_label_match_rate"] == 1.0),
            ("B_ia_max_abs_error_le_1e_minus_6", row["B_ia_max_abs_error_vs_recomputed"] <= 1e-6),
            ("feature_zero_norm_count_eq_0", row["feature_zero_norm_count"] == 0),
        ]
        for name, ok in checks:
            if not ok:
                failure_rows.append(
                    {
                        "schema_version": "stream4d_v103_phase4_affinity_correctness_failure_row_v1",
                        "phase_id": PHASE_ID,
                        "scene_id": scene,
                        "failure_id": name,
                        "severity": "blocking_arithmetic_or_provenance_bug",
                        "evidence": json.dumps(_jsonable(row), sort_keys=True),
                    }
                )

    _write_csv(out / "affinity_correctness_rows.csv", audit_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    summary = {
        "schema_version": "stream4d_v103_phase4_affinity_correctness_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - t0,
        "decision": "PASS_PHASE4_AFFINITY_ARITHMETIC_AUDIT" if not failure_rows else "NO_GO_PHASE4_AFFINITY_ARITHMETIC_AUDIT",
        "failure_count": len(failure_rows),
        "phase3_root": _rel(phase3_root),
        "phase4_root": _rel(phase4_root),
        "selected_variant_by_scene": selected_by_scene,
        "selected_variant_override_by_scene": selected_override_by_scene,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic": False,
        "truthfulness_note": "This audit checks implemented Phase4 arithmetic and provenance, not whether the D4RT carriers are semantically correct objects.",
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "affinity_correctness_rows": _rel(out / "affinity_correctness_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "last_command": _rel(out / "last_command.txt"),
        },
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if not failure_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
