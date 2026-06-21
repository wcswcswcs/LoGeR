#!/usr/bin/env python3
"""Phase 2 geometry-memory cue ledger for ACL2 v73."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from v73_semantic_memory_common import (
    TARGET_CHUNKS,
    best_auc,
    load_json,
    load_jsonl,
    nested_get,
    parse_chunks,
    rotation_deg_from_trace,
    safe_float,
    spearman,
    torch_load,
    utc_now,
    write_csv,
    write_json,
)


DEFAULT_OUT = Path("results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final/phase2_geometry_memory_cue_ledger")
DEFAULT_REPORT = Path("results/kitti01_hmc_v2/acl2_v73_semantic_memory_control/report_final")
DEFAULT_H35 = Path(
    "results/kitti01_hmc_v2/acl2_v67_dense_semantic_reconstruction/"
    "phaseO2_h35_trace_geom_merge_full/rollouts/V67S_H35_TRACE_GEOM_MERGE_FULL_H35_PARITY"
)
DEFAULT_FEATURES = Path("results/kitti01_hmc_v2/acl2_v68_integrated_cueconstruction/phaseC_target_feature_dumps/features")


FEATURE_PATHS = {
    "short": [
        "D_geo_mean_patch",
        "D_geo_q90_patch",
        "dynamic_mass_D_gt_050",
        "global_k_layer5_gram_motion",
        "global_k_layer7_gram_motion",
        "frame_attention_abs_bias",
        "chunk_attention_abs_bias",
    ],
    "mid": [
        "raw_overlap_residual_rmse",
        "raw_overlap_residual_mean",
        "overlap_semantic_nonvoid_ratio",
        "merge_transform_abs_log_scale",
        "merge_transform_rotation_deg",
        "merge_transform_translation_norm",
        "swa_mean_gate",
        "swa_overlap_replace_alpha",
    ],
    "long": [
        "ttt_write_score_mean",
        "ttt_selected_mass",
        "ttt_post_delta_norm_mean",
        "ttt_native_cosine_mean",
        "ttt_tri_neg_mass_mean",
    ],
}


def _hmc_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if "chunk_idx" in row:
            out[int(row["chunk_idx"])] = row
    return out


def _trace_by_chunk(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if "chunk_idx" in row:
            out[int(row["chunk_idx"])] = row
    return out


def _feature_motion(feature_path: Path) -> dict[str, Any]:
    if not feature_path.exists():
        return {"global_k_layer5_gram_motion": None, "global_k_layer7_gram_motion": None, "global_k_feature_available": False}
    try:
        payload = torch_load(feature_path)
        tensor = payload.get("tap::global_k_raw_patchvec_layers") if isinstance(payload, dict) else None
        if not hasattr(tensor, "shape") or int(tensor.shape[0]) < 2:
            return {"global_k_layer5_gram_motion": None, "global_k_layer7_gram_motion": None, "global_k_feature_available": False}
        arr = tensor.float().numpy()
        out: dict[str, Any] = {"global_k_feature_available": True}
        for pos, layer in enumerate((5, 7)):
            if pos >= arr.shape[1]:
                out[f"global_k_layer{layer}_gram_motion"] = None
                continue
            x = arr[:, pos].reshape(arr.shape[0], -1, arr.shape[-1])
            x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
            cos = np.sum(x[1:] * x[:-1], axis=-1)
            out[f"global_k_layer{layer}_gram_motion"] = float(np.mean(1.0 - cos))
        return out
    except Exception as exc:
        return {
            "global_k_layer5_gram_motion": None,
            "global_k_layer7_gram_motion": None,
            "global_k_feature_available": False,
            "global_k_feature_error": type(exc).__name__,
        }


def _overlap_features(run_dir: Path, chunk_id: int) -> dict[str, Any]:
    pair = run_dir / "overlap_pairs" / f"chunk_{chunk_id - 1:03d}_{chunk_id:03d}.pt"
    out: dict[str, Any] = {
        "overlap_pair_path": str(pair),
        "overlap_pair_available": pair.exists(),
        "raw_overlap_residual_rmse": None,
        "raw_overlap_residual_mean": None,
        "overlap_semantic_nonvoid_ratio": None,
        "overlap_valid_pair_count": None,
    }
    if not pair.exists():
        return out
    try:
        payload = torch_load(pair)
    except Exception as exc:
        out["overlap_pair_error"] = type(exc).__name__
        return out
    out["raw_overlap_residual_rmse"] = safe_float(payload.get("raw_residual_rmse"))
    out["raw_overlap_residual_mean"] = safe_float(payload.get("raw_residual_mean"))
    out["overlap_semantic_nonvoid_ratio"] = safe_float(payload.get("semantic_nonvoid_ratio"))
    out["overlap_valid_pair_count"] = safe_float(payload.get("valid_pair_count"))
    return out


def _predictiveness(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred_rows: list[dict[str, Any]] = []
    labels = {
        "short": "Y_short",
        "mid": "Y_mid",
        "long": "Y_long",
        "scale": "Y_scale_drift",
    }
    for path, features in FEATURE_PATHS.items():
        label_key = labels.get(path, "Y_scale_drift")
        if path == "long":
            label_key = "Y_scale_drift"
        for feature in features:
            values = [row.get(feature) for row in rows]
            for label_name in dict.fromkeys((label_key, "Y_scale_drift")):
                auc = best_auc(values, [row.get(label_name) for row in rows])
                rho = spearman(values, [row.get(label_name) for row in rows])
                gate = bool((rho is not None and abs(rho) >= 0.30) or (auc.get("best_auc") is not None and auc["best_auc"] >= 0.65))
                pred_rows.append(
                    {
                        "memory_path": path,
                        "feature": feature,
                        "label": label_name,
                        "spearman": rho,
                        "auc": auc.get("auc"),
                        "best_auc": auc.get("best_auc"),
                        "auc_direction": auc.get("direction"),
                        "gate_pass": gate,
                    }
                )
    summary = {
        "schema": "acl2_v73_phase2_cue_predictiveness_v1",
        "gate_rule": "At least one cue per memory path has abs Spearman >=0.30 OR best-direction AUC >=0.65 on target chunks.",
        "path_gate": {},
    }
    for path in ("short", "mid", "long"):
        candidates = [row for row in pred_rows if row["memory_path"] == path]
        summary["path_gate"][path] = {
            "gate_pass": any(bool(row.get("gate_pass")) for row in candidates),
            "best_rows": sorted(
                candidates,
                key=lambda r: max(abs(safe_float(r.get("spearman")) or 0.0), safe_float(r.get("best_auc")) or 0.0),
                reverse=True,
            )[:5],
        }
    return pred_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--h35-run-dir", type=Path, default=DEFAULT_H35)
    parser.add_argument("--v68-feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--target-chunks", default=",".join(map(str, TARGET_CHUNKS)))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ledger_payload = load_json(args.report_root / "phase1_scale_drift_ledger" / "scale_drift_ledger.json") or {}
    ledger_rows = ledger_payload.get("rows", []) if isinstance(ledger_payload, dict) else []
    ledger_by_chunk = {int(row["chunk_id"]): row for row in ledger_rows if row.get("chunk_id") is not None}
    hmc_rows = _hmc_by_chunk(args.h35_run_dir / "hmc_state_hash.jsonl")
    trace_rows = _trace_by_chunk(args.h35_run_dir / "merge_state_trace.jsonl")
    rows: list[dict[str, Any]] = []
    for chunk_id in parse_chunks(args.target_chunks):
        base = dict(ledger_by_chunk.get(chunk_id, {"chunk_id": chunk_id}))
        hmc = hmc_rows.get(chunk_id, {})
        trace = trace_rows.get(chunk_id, {})
        hook = nested_get(hmc, ["control_trace", "hook_effect_summary"], {}) or {}
        frame_attn = hook.get("frame_attention", {}) if isinstance(hook, dict) else {}
        chunk_attn = hook.get("chunk_attention", {}) if isinstance(hook, dict) else {}
        swa_read = hook.get("swa_read", {}) if isinstance(hook, dict) else {}
        row: dict[str, Any] = {
            "chunk_id": chunk_id,
            **base,
            "D_geo_mean_patch": safe_float(hmc.get("prior_mean_D_patch")),
            "D_geo_q90_patch": safe_float(hmc.get("prior_q90_D_patch")),
            "dynamic_mass_D_gt_050": safe_float(hmc.get("prior_dynamic_mass_D_gt_050")),
            "frame_attention_abs_bias": safe_float(frame_attn.get("mean_abs_bias")),
            "chunk_attention_abs_bias": safe_float(chunk_attn.get("mean_abs_bias")),
            "swa_mean_gate": safe_float(swa_read.get("mean_swa_gate")),
            "swa_overlap_replace_alpha": safe_float(swa_read.get("mean_swa_overlap_source_replace_alpha")),
            "ttt_write_score_mean": safe_float(hmc.get("prior_hmc_write_score_mean")),
            "ttt_selected_mass": safe_float(hmc.get("prior_hmc_write_selected_mass")),
            "ttt_post_delta_norm_mean": safe_float(hmc.get("probe_ttt_write_post_delta_norm_mean")),
            "ttt_native_cosine_mean": safe_float(hmc.get("probe_ttt_write_native_cosine_mean")),
            "ttt_tri_neg_mass_mean": safe_float(hmc.get("probe_ttt_write_tri_neg_mass_mean")),
            "merge_transform_abs_log_scale": None,
            "merge_transform_rotation_deg": rotation_deg_from_trace(trace.get("transform_rot_trace")),
            "merge_transform_translation_norm": safe_float(trace.get("transform_trans_norm")),
            "transform_reason": trace.get("transform_reason"),
        }
        scale_value = safe_float(trace.get("transform_scale_value"))
        if scale_value is not None and scale_value > 0:
            row["merge_transform_abs_log_scale"] = abs(float(math.log(scale_value)))
        row.update(_feature_motion(args.v68_feature_dir / f"chunk_{chunk_id:03d}.pt"))
        row.update(_overlap_features(args.h35_run_dir, chunk_id))
        rows.append(row)
    pred_rows, pred_summary = _predictiveness(rows)
    summary = {
        "schema": "acl2_v73_phase2_geometry_memory_cue_ledger_v1",
        "created_at": utc_now(),
        "h35_run_dir": str(args.h35_run_dir),
        "v68_feature_dir": str(args.v68_feature_dir),
        "target_chunks": parse_chunks(args.target_chunks),
        "rows": len(rows),
        "cue_predictiveness": pred_summary,
        "diagnostic_scope": "target chunks only; online action fidelity is not implied by cue predictiveness.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "geometry_cue_by_chunk.csv", rows)
    write_json(args.out_dir / "geometry_cue_by_chunk.json", {"summary": summary, "rows": rows})
    write_csv(args.out_dir / "cue_predictiveness_rows.csv", pred_rows)
    write_json(args.out_dir / "cue_predictiveness_summary.json", pred_summary)
    print({"out_dir": str(args.out_dir), "rows": len(rows), "path_gate": pred_summary["path_gate"]})


if __name__ == "__main__":
    main()
