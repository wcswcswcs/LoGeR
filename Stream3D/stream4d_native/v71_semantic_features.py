from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d.scannet_stream import ScanNetStream  # noqa: E402
from stream4d_native.frozen_feature_adapter import FrozenFeatureAdapter, locate_default_dinov2_checkpoint  # noqa: E402
from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _rel  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import DEFAULT_SCENES, _discover_pipeline_root, _mask_dir_from_pipeline, _parse_csv_list  # noqa: E402


MASK_FEATURE_FIELDS = [
    "scene_id",
    "chunk_id",
    "frame_id",
    "mask_id",
    "mask_observation_id",
    "semantic_backend",
    "feature_layer",
    "feature_resolution",
    "feature_pooling_method",
    "feature_available",
    "feature_norm",
    "feature_nan_count",
    "feature_dim",
    "semantic_prototype_id",
    "semantic_prototype_margin",
    "semantic_entropy",
    "semantic_intra_variance",
    "semantic_boundary_variance",
    "semantic_texture_score",
    "semantic_background_score_proxy",
    "used_token_count",
    "used_pixel_count",
    "broad_background_risk",
    "uses_gt_for_prediction",
]


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_json(value: Any, default: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _read_mask_label(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., 0]
    if image.shape[:2] != shape_hw:
        image = cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(image, dtype=np.int64)


def _resize_binary(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.shape != (height, width):
        mask_u8 = cv2.resize(mask_u8, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    out = mask_u8.astype(bool)
    if not np.any(out) and np.any(mask):
        ys, xs = np.nonzero(mask)
        cy = int(np.clip(round(float(ys.mean()) * float(height) / max(mask.shape[0], 1)), 0, int(height) - 1))
        cx = int(np.clip(round(float(xs.mean()) * float(width) / max(mask.shape[1], 1)), 0, int(width) - 1))
        out[cy, cx] = True
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    out[1:, :] |= mask[1:, :] != mask[:-1, :]
    out[:-1, :] |= mask[:-1, :] != mask[1:, :]
    out[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    out[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return out & mask


def _entropy_from_abs_feature(feature: np.ndarray) -> float:
    vals = np.abs(np.asarray(feature, dtype=np.float32).reshape(-1))
    denom = float(vals.sum())
    if denom <= 1e-12:
        return 0.0
    probs = vals / denom
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return float(entropy / max(1e-12, math.log(max(2, vals.size))))


def _prototype_from_feature(feature: np.ndarray) -> tuple[str, float]:
    vec = np.asarray(feature, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return "", 0.0
    order = np.argsort(np.abs(vec))[-4:][::-1]
    pieces = []
    for idx in order[:3]:
        sign = "p" if vec[int(idx)] >= 0 else "n"
        pieces.append(f"{sign}{int(idx)}")
    margin = 0.0
    if len(order) >= 2:
        margin = float(abs(vec[int(order[0])]) - abs(vec[int(order[1])]))
    return "dino|" + "|".join(pieces), margin


def _feature_sha256(feature: np.ndarray) -> str:
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _row_key(row: dict[str, str]) -> tuple[str, int, int]:
    return str(row["scene_id"]), int(float(row["frame_id"])), int(float(row["mask_id"]))


def _load_candidate_index(candidate_rows: Path, scenes: list[str], max_unique_masks: int | None = None) -> dict[tuple[str, int], dict[int, dict[str, str]]]:
    scene_set = set(scenes)
    by_frame: dict[tuple[str, int], dict[int, dict[str, str]]] = defaultdict(dict)
    with candidate_rows.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = str(row.get("scene_id") or "")
            if scene not in scene_set:
                continue
            frame_id = int(float(row.get("frame_id") or 0))
            mask_id = int(float(row.get("mask_id") or 0))
            key = (scene, frame_id)
            if mask_id not in by_frame[key]:
                by_frame[key][mask_id] = row
                if max_unique_masks is not None and sum(len(v) for v in by_frame.values()) >= max_unique_masks:
                    return by_frame
    return by_frame


def _load_edge_metrics(edge_metric_rows: Path) -> list[dict[str, Any]]:
    if not edge_metric_rows.exists():
        return []
    with edge_metric_rows.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_diagnostic_metric(metrics: list[dict[str, Any]], preferred: str) -> dict[str, Any]:
    for row in metrics:
        if row.get("edge_type") == preferred:
            return dict(row)
    for row in metrics:
        if row.get("edge_type") == "E_appearance_abs":
            return dict(row)
    return {}


def _write_mask_features(
    *,
    candidate_index: dict[tuple[str, int], dict[int, dict[str, str]]],
    output_csv: Path,
    scenes: list[str],
    backend: str,
    device: str,
    checkpoint: str | None,
    entropy_variance_scale: float,
    write_feature_json: bool,
) -> dict[str, Any]:
    adapter = FrozenFeatureAdapter(backend=backend, device=device, checkpoint=checkpoint) if backend else None
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_requested = sum(len(v) for v in candidate_index.values())
    valid_count = 0
    nan_rows = 0
    missing_frames: list[dict[str, Any]] = []
    entropy_values: list[float] = []
    intra_values: list[float] = []
    broad_entropy: list[float] = []
    clean_entropy: list[float] = []
    prototype_counter: Counter[str] = Counter()
    frame_count = 0
    fields = list(MASK_FEATURE_FIELDS)
    if write_feature_json:
        fields.extend(["feature_sha256", "feature_head_json", "feature_json"])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scene in scenes:
            pipeline_root = _discover_pipeline_root(scene)
            if pipeline_root is None:
                missing_frames.append({"scene_id": scene, "missing": "pipeline_root"})
                continue
            mask_dir = _mask_dir_from_pipeline(pipeline_root)
            stream = ScanNetStream(scene, root=ROOT / "data/scannet/processed")
            frame_items = sorted((frame_id, masks) for (scene_key, frame_id), masks in candidate_index.items() if scene_key == scene)
            for frame_id, masks_by_id in frame_items:
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"[v71-semantic] frame_count={frame_count} scene={scene} frame={frame_id} masks={len(masks_by_id)}", file=sys.stderr, flush=True)
                try:
                    rgb = stream.load_rgb(int(frame_id))
                except FileNotFoundError:
                    missing_frames.append({"scene_id": scene, "frame_id": int(frame_id), "missing": "rgb"})
                    continue
                mask_label = _read_mask_label(mask_dir / f"{int(frame_id)}.png", rgb.shape[:2])
                if mask_label is None:
                    missing_frames.append({"scene_id": scene, "frame_id": int(frame_id), "missing": "mask_png"})
                    continue
                try:
                    feature_map = adapter.extract_dense_features(rgb) if adapter is not None else None
                except Exception as exc:
                    missing_frames.append({"scene_id": scene, "frame_id": int(frame_id), "missing": f"feature_extract_failed:{type(exc).__name__}:{exc}"})
                    continue
                assert feature_map is not None
                features = np.asarray(feature_map.features, dtype=np.float32)
                feature_h, feature_w, feature_dim = int(features.shape[0]), int(features.shape[1]), int(features.shape[2])
                for mask_id, source_row in sorted(masks_by_id.items()):
                    binary = mask_label == int(mask_id)
                    used_pixel_count = int(binary.sum())
                    small = _resize_binary(binary, feature_h, feature_w)
                    token_values = features[small]
                    feature_available = bool(token_values.size > 0 and used_pixel_count >= 1)
                    pooled = np.zeros((feature_dim,), dtype=np.float32)
                    feature_norm = 0.0
                    feature_nan_count = 0
                    entropy = ""
                    intra = ""
                    boundary_variance = ""
                    prototype_id = ""
                    prototype_margin = ""
                    used_token_count = int(token_values.shape[0]) if token_values.ndim == 2 else 0
                    if feature_available:
                        pooled = token_values.mean(axis=0).astype(np.float32)
                        feature_nan_count = int(np.isnan(pooled).sum())
                        if feature_nan_count == 0:
                            feature_norm = float(np.linalg.norm(pooled))
                            if feature_norm > 1e-8:
                                pooled = pooled / feature_norm
                            intra_value = float(np.mean(np.var(token_values.astype(np.float32), axis=0))) if token_values.shape[0] > 1 else 0.0
                            boundary_small = _boundary(small)
                            if np.any(boundary_small):
                                boundary_tokens = features[boundary_small]
                                boundary_variance_value = float(np.mean(np.var(boundary_tokens.astype(np.float32), axis=0))) if boundary_tokens.shape[0] > 1 else 0.0
                            else:
                                boundary_variance_value = 0.0
                            entropy_value = float(1.0 - math.exp(-intra_value / max(1e-12, entropy_variance_scale)))
                            prototype_id, prototype_margin_value = _prototype_from_feature(pooled)
                            entropy = entropy_value
                            intra = intra_value
                            boundary_variance = boundary_variance_value
                            prototype_margin = prototype_margin_value
                            valid_count += 1
                            entropy_values.append(entropy_value)
                            intra_values.append(intra_value)
                            prototype_counter[prototype_id] += 1
                            if _bool(source_row.get("broad_background_risk")):
                                broad_entropy.append(entropy_value)
                            else:
                                clean_entropy.append(entropy_value)
                        else:
                            nan_rows += 1
                    out_row = {
                        "scene_id": scene,
                        "chunk_id": source_row.get("chunk_id", ""),
                        "frame_id": int(frame_id),
                        "mask_id": int(mask_id),
                        "mask_observation_id": source_row.get("mask_observation_id", f"{scene}:{frame_id}:{mask_id}"),
                        "semantic_backend": backend,
                        "feature_layer": "x_norm_patchtokens",
                        "feature_resolution": f"{feature_h}x{feature_w}",
                        "feature_pooling_method": "mask_token_mean",
                        "feature_available": bool(feature_available and feature_nan_count == 0),
                        "feature_norm": feature_norm,
                        "feature_nan_count": feature_nan_count,
                        "feature_dim": feature_dim,
                        "semantic_prototype_id": prototype_id,
                        "semantic_prototype_margin": prototype_margin,
                        "semantic_entropy": entropy,
                        "semantic_intra_variance": intra,
                        "semantic_boundary_variance": boundary_variance,
                        "semantic_texture_score": intra,
                        "semantic_background_score_proxy": source_row.get("broad_background_risk", ""),
                        "used_token_count": used_token_count,
                        "used_pixel_count": used_pixel_count,
                        "broad_background_risk": _bool(source_row.get("broad_background_risk")),
                        "uses_gt_for_prediction": False,
                    }
                    if write_feature_json:
                        if feature_available and feature_nan_count == 0:
                            out_row["feature_sha256"] = _feature_sha256(pooled)
                            out_row["feature_head_json"] = json.dumps([float(v) for v in pooled[:8]], separators=(",", ":"))
                            out_row["feature_json"] = json.dumps([float(v) for v in pooled], separators=(",", ":"))
                        else:
                            out_row["feature_sha256"] = ""
                            out_row["feature_head_json"] = ""
                            out_row["feature_json"] = ""
                    writer.writerow(out_row)
    success_rate = float(valid_count / max(1, total_requested))
    broad_mean = float(np.mean(broad_entropy)) if broad_entropy else None
    clean_mean = float(np.mean(clean_entropy)) if clean_entropy else None
    return {
        "requested_unique_mask_observation_count": int(total_requested),
        "mask_feature_row_count": int(total_requested),
        "valid_feature_count": int(valid_count),
        "semantic_feature_success_rate": success_rate,
        "semantic_nan_rate": float(nan_rows / max(1, total_requested)),
        "semantic_entropy_mean": float(np.mean(entropy_values)) if entropy_values else None,
        "semantic_entropy_p90": float(np.quantile(np.asarray(entropy_values, dtype=np.float32), 0.90)) if entropy_values else None,
        "semantic_intra_variance_mean": float(np.mean(intra_values)) if intra_values else None,
        "semantic_prototype_count": int(len(prototype_counter)),
        "diagnostic_broad_mask_Hsem_mean": broad_mean,
        "diagnostic_clean_mask_Hsem_mean": clean_mean,
        "diagnostic_broad_minus_clean_Hsem": None if broad_mean is None or clean_mean is None else float(broad_mean - clean_mean),
        "missing_frame_count": int(len(missing_frames)),
        "missing_frame_examples": missing_frames[:20],
        "feature_frame_count": int(frame_count),
        "semantic_entropy_definition": "1-exp(-mask_token_covariance_trace/entropy_variance_scale)",
        "semantic_entropy_variance_scale": float(entropy_variance_scale),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_rows = _rooted(args.candidate_rows)
    edge_summary_path = _rooted(args.edge_summary)
    edge_metric_rows_path = _rooted(args.edge_metric_rows)
    missing = []
    for name, path in [
        ("candidate_rows", candidate_rows),
        ("edge_summary", edge_summary_path),
        ("edge_metric_rows", edge_metric_rows_path),
    ]:
        if not path.exists():
            missing.append({"name": name, "path": _rel(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v71_semantic_features",
            "decision": "FAIL_MISSING_INPUTS",
            "gate": {"pass": False, "all_inputs_present": False},
            "missing_inputs": missing,
        }
        _write_json(output_root / "semantic_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    scenes = _parse_csv_list(args.scenes)
    checkpoint = args.checkpoint or locate_default_dinov2_checkpoint()
    candidate_index = _load_candidate_index(candidate_rows, scenes, args.max_unique_masks)
    feature_stats = _write_mask_features(
        candidate_index=candidate_index,
        output_csv=output_root / "mask_feature_rows.csv",
        scenes=scenes,
        backend=str(args.backend),
        device=str(args.device),
        checkpoint=checkpoint,
        entropy_variance_scale=float(args.entropy_variance_scale),
        write_feature_json=bool(args.write_feature_json),
    )
    edge_summary = _load_json(edge_summary_path)
    edge_metrics = _load_edge_metrics(edge_metric_rows_path)
    diagnostic = _select_diagnostic_metric(edge_metrics, "E_combined_frozen_appearance")
    semantic_edge_rows = []
    for row in edge_metrics:
        out = dict(row)
        out["source"] = _rel(edge_metric_rows_path)
        out["uses_gt_for_prediction"] = False
        out["uses_gt_for_diagnostic_labels"] = True
        out["diagnostic_only"] = True
        out["forbidden_for_method_table"] = True
        semantic_edge_rows.append(out)
    _write_csv(output_root / "semantic_edge_metric_rows.csv", semantic_edge_rows)
    _write_csv(output_root / "atom_feature_rows.csv", [])
    _write_csv(output_root / "missing_input_rows.csv", [])
    edge_auc = _float(diagnostic.get("edge_AUC"))
    top1 = _float(diagnostic.get("top1_precision"))
    hard_negative = _float(diagnostic.get("hard_negative_precision")) or _float((edge_summary.get("combined_metrics") or {}).get("hard_negative_precision"))
    broad_delta = feature_stats.get("diagnostic_broad_minus_clean_Hsem")
    gate = {
        "all_inputs_present": True,
        "semantic_feature_success_rate_ge_0p95": feature_stats["semantic_feature_success_rate"] >= 0.95,
        "semantic_nan_rate_eq_0": feature_stats["semantic_nan_rate"] == 0.0,
        "DINO_or_RADIO_edge_AUC_diagnostic_ge_0p75": edge_auc is not None and edge_auc >= 0.75,
        "DINO_or_RADIO_top1_precision_diagnostic_ge_0p75": top1 is not None and top1 >= 0.75,
        "hard_negative_precision_diagnostic_ge_0p90": hard_negative is not None and hard_negative >= 0.90,
        "semantic_entropy_broad_ge_clean_plus_0p10": broad_delta is not None and broad_delta >= 0.10,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    if gate["pass"]:
        decision = "PASS_V71_SEMANTIC_FEATURES"
    elif gate["semantic_feature_success_rate_ge_0p95"] and gate["DINO_or_RADIO_edge_AUC_diagnostic_ge_0p75"]:
        decision = "PARTIAL_V71_SEMANTIC_FEATURES_ENTROPY_BLOCKER"
    else:
        decision = "NO_GO_V71_SEMANTIC_FEATURES"
    summary = {
        "phase": "v71_semantic_features",
        "decision": decision,
        "semantic_backend": str(args.backend),
        "checkpoint": str(checkpoint or ""),
        "device": str(args.device),
        "source_artifacts": {
            "candidate_rows": _rel(candidate_rows),
            "edge_summary": _rel(edge_summary_path),
            "edge_metric_rows": _rel(edge_metric_rows_path),
        },
        "gate": gate,
        "key_metrics": {
            **feature_stats,
            "DINO_edge_AUC_diagnostic": edge_auc,
            "DINO_top1_precision_diagnostic": top1,
            "hard_negative_precision_diagnostic": hard_negative,
            "RADIO_unavailable": True,
        },
        "rows": {
            "semantic_summary_json": _rel(output_root / "semantic_summary.json"),
            "mask_feature_rows_csv": _rel(output_root / "mask_feature_rows.csv"),
            "atom_feature_rows_csv": _rel(output_root / "atom_feature_rows.csv"),
            "semantic_edge_metric_rows_csv": _rel(output_root / "semantic_edge_metric_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
        "notes": [
            "Mask semantic rows are computed with frozen DINOv2 mask-token mean pooling from RGB only.",
            "Semantic entropy is the normalized DINO token covariance trace, following the plan repair direction when abs-feature entropy failed to separate broad masks.",
            "No GT, depth, pose, mesh, or RGB-D backprojection is used for feature extraction.",
            "Semantic edge diagnostic metrics are imported from the v68 DINO edge audit and remain diagnostic_only=true.",
            "RADIO is recorded unavailable in this run; no RADIO feature is fabricated.",
        ],
        "write_feature_json": bool(args.write_feature_json),
    }
    _write_json(output_root / "semantic_summary.json", summary)
    sha_rows = []
    for path in [
        output_root / "semantic_summary.json",
        output_root / "mask_feature_rows.csv",
        output_root / "atom_feature_rows.csv",
        output_root / "semantic_edge_metric_rows.csv",
        output_root / "missing_input_rows.csv",
    ]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v71 Phase 3 semantic feature audit.")
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--candidate-rows", default="outputs/audit/v71_candidate_bank/candidate_mask_rows.csv")
    parser.add_argument("--edge-summary", default="outputs/audit/v68_edge_audit_dinov2/edge_audit_summary.json")
    parser.add_argument("--edge-metric-rows", default="outputs/audit/v68_edge_audit_dinov2/edge_metric_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v71_semantic_features")
    parser.add_argument("--backend", default="dinov2_timm", choices=["dinov2_timm", "rgb_stats"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max-unique-masks", type=int, default=None)
    parser.add_argument("--entropy-variance-scale", type=float, default=0.001)
    parser.add_argument("--write-feature-json", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
