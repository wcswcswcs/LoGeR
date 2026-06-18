from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from sklearn.metrics import roc_auc_score

from stream4d.scannet_stream import ScanNetStream
from stream4d_native.d4rt_scene_builder import D4RTNativeSceneBuilder
from tools.run_v26_object_quality_diagnostics import _json_safe, _read_split, _write_csv, assign_gt_labels
from tools.run_v28_proposal_oracle import _cluster_metrics
from tools.run_v28_proposal_selection import _parse_core_tube_ids
from tools.trace_v25_real_geometry_flow import chunks_to_records, load_scene_chunks_from_cache


LOCAL_GATE = {
    "local_ARI": 0.40,
    "local_purity": 0.85,
    "local_completeness": 0.50,
    "unknown_tube_ratio_max": 0.40,
    "scene0081_local_ARI": 0.20,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean(values: list[Any]) -> float | None:
    vals = [_float(value) for value in values]
    vals = [float(value) for value in vals if value is not None]
    return float(np.mean(vals)) if vals else None


def _gate_status(row: dict[str, Any]) -> dict[str, Any]:
    ari = _float(row.get("local_ARI"))
    purity = _float(row.get("local_purity"))
    completeness = _float(row.get("local_completeness"))
    unknown = _float(row.get("unknown_tube_ratio"))
    scene0081 = _float(row.get("scene0081_local_ARI"))
    checks = {
        "ari_pass": ari is not None and ari >= LOCAL_GATE["local_ARI"],
        "purity_pass": purity is not None and purity >= LOCAL_GATE["local_purity"],
        "completeness_pass": completeness is not None and completeness >= LOCAL_GATE["local_completeness"],
        "unknown_pass": unknown is not None and unknown <= LOCAL_GATE["unknown_tube_ratio_max"],
        "scene0081_pass": scene0081 is not None and scene0081 >= LOCAL_GATE["scene0081_local_ARI"],
    }
    return {**checks, "local_gate_pass": bool(all(checks.values())), "local_gate_thresholds": dict(LOCAL_GATE)}


def _make_model(args: argparse.Namespace) -> torch.nn.Module:
    model = timm.create_model(str(args.backbone), pretrained=False, num_classes=0)
    state = torch.load(str(args.checkpoint), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    model.to(args.device)
    setattr(model, "_v35_missing_keys", list(missing)[:20])
    setattr(model, "_v35_unexpected_keys", list(unexpected)[:20])
    return model


def _preprocess(rgb: np.ndarray, image_size: int, device: str) -> torch.Tensor:
    resized = cv2.resize(rgb, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (resized - mean[None, None, :]) / std[None, None, :]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor


@torch.no_grad()
def _patch_tokens(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    out = model.forward_features(tensor)
    if isinstance(out, dict):
        if "x_norm_patchtokens" in out:
            tokens = out["x_norm_patchtokens"]
        elif "x" in out:
            tokens = out["x"][:, 1:, :]
        else:
            first = next(value for value in out.values() if torch.is_tensor(value))
            tokens = first[:, 1:, :] if first.ndim == 3 else first.reshape(first.shape[0], -1, first.shape[-1])
    else:
        tokens = out[:, 1:, :] if out.ndim == 3 else out.reshape(out.shape[0], -1, out.shape[-1])
    tokens = F.normalize(tokens.float(), dim=-1)
    return tokens.squeeze(0).detach().cpu()


def _visible_index(tube: Any, frame_id: int, min_visibility: float, min_confidence: float) -> int | None:
    frames = np.asarray(tube.target_frames_global, dtype=np.int64).reshape(-1)
    matches = np.flatnonzero(frames == int(frame_id))
    if matches.size == 0:
        return None
    idx = int(matches[0])
    visibility = np.asarray(tube.visibility, dtype=np.float32).reshape(-1)
    confidence = np.asarray(tube.confidence, dtype=np.float32).reshape(-1)
    if visibility[idx] < float(min_visibility) or confidence[idx] < float(min_confidence):
        return None
    return idx


def _extract_scene_embeddings(args: argparse.Namespace, model: torch.nn.Module, scene: str) -> tuple[dict[int, np.ndarray], dict[int, int], dict[str, Any]]:
    chunks, _ = load_scene_chunks_from_cache(
        Path(args.cache_root) / scene,
        max_tubes_per_window=int(args.max_tubes_per_window),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
    )
    builder = D4RTNativeSceneBuilder(object(), {"model": {"input": {"clip_frames": 32}}}, temporal_chunk_size=32, temporal_chunk_stride=16)
    records = chunks_to_records(builder.stitch_to_canonical(chunks))
    stream = ScanNetStream(seq_name=scene)
    gt_labels = assign_gt_labels(records, stream=stream, min_visibility=float(args.min_visibility), min_confidence=float(args.min_confidence))
    frame_ids = sorted({int(frame) for tube in records for frame in np.asarray(tube.target_frames_global, dtype=np.int64).tolist()})
    if len(frame_ids) > int(args.max_frames_per_scene):
        keep = np.linspace(0, len(frame_ids) - 1, num=int(args.max_frames_per_scene), dtype=np.int64)
        frame_ids = [frame_ids[int(idx)] for idx in keep.tolist()]
    tube_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
    frame_diag = []
    grid = int(args.image_size) // int(args.patch_size)
    by_id = {int(tube.tube_id): tube for tube in records}
    for frame_id in frame_ids:
        rgb = stream.load_rgb(frame_id)
        tokens = _patch_tokens(model, _preprocess(rgb, int(args.image_size), str(args.device)))
        if tokens.shape[0] != grid * grid:
            grid = int(round(math.sqrt(tokens.shape[0])))
        for tube in records:
            idx = _visible_index(tube, frame_id, float(args.min_visibility), float(args.min_confidence))
            if idx is None:
                continue
            uv = np.asarray(tube.uv, dtype=np.float32).reshape(-1, 2)[idx]
            if not np.isfinite(uv).all():
                continue
            px = int(np.clip(round(float(uv[0]) * (grid - 1)), 0, grid - 1))
            py = int(np.clip(round(float(uv[1]) * (grid - 1)), 0, grid - 1))
            tube_vectors[int(tube.tube_id)].append(tokens[py * grid + px].numpy())
        frame_diag.append({"scene": scene, "frame_id": int(frame_id), "token_count": int(tokens.shape[0]), "grid": int(grid)})
    embeddings = {
        int(tid): np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        for tid, vecs in tube_vectors.items()
        if vecs
    }
    embeddings = {tid: vec / max(float(np.linalg.norm(vec)), 1e-8) for tid, vec in embeddings.items()}
    diag = {
        "scene": scene,
        "record_count": int(len(records)),
        "embedded_tube_count": int(len(embeddings)),
        "labeled_tube_count": int(sum(1 for value in gt_labels.values() if int(value) > 0)),
        "frame_count": int(len(frame_ids)),
        "frames": frame_ids,
        "frame_diag": frame_diag,
    }
    return embeddings, gt_labels, diag


def _balanced_pairs(embeddings: dict[int, np.ndarray], gt_labels: dict[int, int], max_pairs: int, seed: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    tids = sorted(tid for tid in embeddings if int(gt_labels.get(int(tid), 0)) > 0)
    pos: list[tuple[int, int]] = []
    neg: list[tuple[int, int]] = []
    for i, a in enumerate(tids):
        ga = int(gt_labels[a])
        for b in tids[i + 1 :]:
            gb = int(gt_labels[b])
            (pos if ga == gb else neg).append((a, b))
    rng = np.random.default_rng(seed)
    per = min(len(pos), len(neg), max(1, int(max_pairs) // 2))
    if per == 0:
        pairs = pos[: int(max_pairs)] + neg[: int(max_pairs)]
    else:
        pos_idx = rng.choice(len(pos), size=per, replace=False)
        neg_idx = rng.choice(len(neg), size=per, replace=False)
        pairs = [pos[int(idx)] for idx in pos_idx] + [neg[int(idx)] for idx in neg_idx]
    labels = np.asarray([int(gt_labels[a] == gt_labels[b]) for a, b in pairs], dtype=np.int64)
    return pairs, labels


def _pair_scores(embeddings: dict[int, np.ndarray], pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([float(np.dot(embeddings[a], embeddings[b])) for a, b in pairs], dtype=np.float64)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(labels) == 0 or len(set(labels.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _proposal_embedding_auc(rows: list[dict[str, Any]], embeddings_by_scene: dict[str, dict[int, np.ndarray]]) -> dict[str, Any]:
    labels = []
    scores = []
    for row in rows:
        scene = str(row.get("scene") or "")
        embeddings = embeddings_by_scene.get(scene, {})
        tube_ids = [tid for tid in _parse_core_tube_ids(row) if int(tid) in embeddings]
        if len(tube_ids) < 3:
            continue
        purity = _float(row.get("proposal_purity"))
        if purity is None:
            continue
        vecs = np.stack([embeddings[int(tid)] for tid in tube_ids], axis=0)
        centroid = vecs.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
        compactness = float(np.mean(vecs @ centroid))
        labels.append(int(purity >= 0.85))
        scores.append(compactness)
    if len(set(labels)) < 2:
        return {"mixed_region_AUC": None, "region_count": int(len(labels))}
    return {"mixed_region_AUC": float(roc_auc_score(np.asarray(labels), np.asarray(scores))), "region_count": int(len(labels))}


def _labels_from_embedding_graph(
    embeddings: dict[int, np.ndarray],
    gt_labels: dict[int, int],
    threshold: float,
    min_component_tubes: int,
    pair_boost: dict[tuple[int, int], float] | None = None,
) -> tuple[dict[int, int], int, int]:
    tids = sorted(tid for tid in embeddings if int(gt_labels.get(int(tid), 0)) > 0)
    parent = {tid: tid for tid in tids}

    def find(tid: int) -> int:
        cur = int(tid)
        while parent[cur] != cur:
            parent[cur] = parent[parent[cur]]
            cur = parent[cur]
        return cur

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_count = 0
    for i, a in enumerate(tids):
        va = embeddings[a]
        for b in tids[i + 1 :]:
            score = float(np.dot(va, embeddings[b]))
            if pair_boost:
                key = (a, b) if a < b else (b, a)
                score += float(pair_boost.get(key, 0.0))
            if score >= float(threshold):
                union(a, b)
                edge_count += 1
    comps: dict[int, list[int]] = defaultdict(list)
    for tid in tids:
        comps[find(tid)].append(tid)
    labels_pred: dict[int, int] = {}
    next_label = 0
    unknown_count = 0
    for comp in sorted(comps.values(), key=lambda values: (len(values), values[0]), reverse=True):
        if len(comp) >= int(min_component_tubes):
            for tid in comp:
                labels_pred[int(tid)] = next_label
            next_label += 1
        else:
            for tid in comp:
                labels_pred[int(tid)] = next_label
                next_label += 1
                unknown_count += 1
    return labels_pred, unknown_count, edge_count


def _d4rt_pair_boost(rows: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    counts: Counter[tuple[int, int]] = Counter()
    for row in rows:
        ptype = str(row.get("proposal_type") or "")
        if not ptype.startswith(("R3_", "R5_", "R8_", "R9_", "R10_", "R11_", "R12_")):
            continue
        ids = list(_parse_core_tube_ids(row))
        if len(ids) > 120:
            ids = ids[:120]
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                key = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
                counts[key] += 1
    return {key: min(0.10, 0.02 * math.log1p(count)) for key, count in counts.items()}


def _evaluate_variant(
    scene: str,
    variant: str,
    embeddings: dict[int, np.ndarray],
    gt_labels: dict[int, int],
    threshold: float,
    min_component_tubes: int,
    pair_boost: dict[tuple[int, int], float] | None,
) -> dict[str, Any]:
    labels_pred, unknown_count, edge_count = _labels_from_embedding_graph(
        embeddings,
        gt_labels,
        threshold=float(threshold),
        min_component_tubes=int(min_component_tubes),
        pair_boost=pair_boost,
    )
    metrics = _cluster_metrics(labels_pred, gt_labels)
    labeled = int(sum(1 for value in gt_labels.values() if int(value) > 0 and int(value) in embeddings))
    row = {
        "scene": scene,
        "variant": variant,
        "threshold": float(threshold),
        "min_component_tubes": int(min_component_tubes),
        "embedded_labeled_tube_count": labeled,
        "edge_count": int(edge_count),
        "unknown_tube_count": int(unknown_count),
        "unknown_tube_ratio": float(unknown_count / max(labeled, 1)),
        "local_ARI": metrics["ari"],
        "local_purity": metrics["purity"],
        "local_completeness": metrics["completeness"],
        "local_overmerge": metrics["overmerge"],
        "local_oversplit": metrics["oversplit"],
        "is_method_result": True,
        "is_diagnostic_only": False,
        "forbidden_for_method_table": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_rgbd_for_prediction": False,
        "uses_pose_for_prediction": False,
        "uses_scannet_mesh_for_prediction": False,
        "uses_eval_sim3_for_prediction": False,
        "uses_d4rt_self_sim3": True,
        "uses_frozen_visual_backbone": True,
        "visual_backbone_name": "DINOv2",
        "visual_backbone_checkpoint": "",
        "mask_source": "prepared Cropformer masks",
        "geometry_field": "D4RT uv/visibility/confidence for tube-to-patch embedding sampling",
        "coordinate_frame": "image patch grid and D4RT canonical tube ids",
        "alignment_source": "D4RT self-Sim3 inherited from cached records",
    }
    return {**row, **_gate_status(row)}


def _aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    row = {
        "scene": "ALL",
        "variant": variant,
        "threshold": rows[0].get("threshold") if rows else None,
        "min_component_tubes": rows[0].get("min_component_tubes") if rows else None,
        "embedded_labeled_tube_count": int(sum(int(r.get("embedded_labeled_tube_count") or 0) for r in rows)),
        "edge_count": int(sum(int(r.get("edge_count") or 0) for r in rows)),
        "unknown_tube_count": int(sum(int(r.get("unknown_tube_count") or 0) for r in rows)),
        "unknown_tube_ratio": _mean([r.get("unknown_tube_ratio") for r in rows]),
        "local_ARI": _mean([r.get("local_ARI") for r in rows]),
        "local_purity": _mean([r.get("local_purity") for r in rows]),
        "local_completeness": _mean([r.get("local_completeness") for r in rows]),
        "local_overmerge": _mean([r.get("local_overmerge") for r in rows]),
        "local_oversplit": _mean([r.get("local_oversplit") for r in rows]),
        "scene0081_local_ARI": next((r.get("local_ARI") for r in rows if r.get("scene") == "scene0081_01"), None),
        "is_method_result": True,
        "is_diagnostic_only": False,
        "forbidden_for_method_table": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "uses_frozen_visual_backbone": True,
        "visual_backbone_name": "DINOv2",
        "visual_backbone_checkpoint": "",
    }
    return {**row, **_gate_status(row)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    args.device = "cuda:0" if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    model = _make_model(args)
    proposal_rows = _read_json(Path(args.proposal_root) / f"{args.proposal_label}_proposal_rows.json")
    rows_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in proposal_rows:
        rows_by_scene[str(row.get("scene") or "")].append(row)
    scenes = _read_split(Path(args.split))
    embeddings_by_scene: dict[str, dict[int, np.ndarray]] = {}
    gt_by_scene: dict[str, dict[int, int]] = {}
    scene_diag = []
    feature_rows = []
    for scene in scenes:
        embeddings, gt_labels, diag = _extract_scene_embeddings(args, model, scene)
        embeddings_by_scene[scene] = embeddings
        gt_by_scene[scene] = gt_labels
        scene_diag.append(diag)
        pairs, labels = _balanced_pairs(embeddings, gt_labels, int(args.max_feature_pairs_per_scene), int(args.seed))
        scores = _pair_scores(embeddings, pairs)
        feature_rows.append(
            {
                "scene": scene,
                "same_GT_pair_AUC": _auc(labels, scores),
                "pair_count": int(len(pairs)),
                "positive_pair_count": int(labels.sum()) if len(labels) else 0,
                "embedded_tube_count": int(len(embeddings)),
                "labeled_tube_count": int(sum(1 for value in gt_labels.values() if int(value) > 0)),
                "scene0081_feature_AUC": _auc(labels, scores) if scene == "scene0081_01" else None,
            }
        )
    mixed_auc = _proposal_embedding_auc(proposal_rows, embeddings_by_scene)
    variants = {
        "B1_frozen_embedding_only": {"threshold": float(args.embedding_threshold), "min_component_tubes": 2, "boost": False},
        "B3_embedding_d4rt_uv_support": {"threshold": float(args.embedding_threshold), "min_component_tubes": 2, "boost": True},
        "B4_embedding_d4rt_boundary_unknown": {"threshold": float(args.embedding_unknown_threshold), "min_component_tubes": 3, "boost": True},
    }
    summary_rows = []
    scene_metric_rows = []
    for variant, params in variants.items():
        rows = []
        for scene in scenes:
            boost = _d4rt_pair_boost(rows_by_scene.get(scene, [])) if params["boost"] else None
            row = _evaluate_variant(
                scene,
                variant,
                embeddings_by_scene.get(scene, {}),
                gt_by_scene.get(scene, {}),
                float(params["threshold"]),
                int(params["min_component_tubes"]),
                boost,
            )
            rows.append(row)
            scene_metric_rows.append(row)
        summary_rows.append(_aggregate(rows, variant))
    feature_all = {
        "scene": "ALL",
        "same_GT_pair_AUC": _mean([row.get("same_GT_pair_AUC") for row in feature_rows]),
        "mixed_region_AUC": mixed_auc.get("mixed_region_AUC"),
        "mixed_region_count": mixed_auc.get("region_count"),
        "scene0081_feature_AUC": next((row.get("same_GT_pair_AUC") for row in feature_rows if row["scene"] == "scene0081_01"), None),
        "feature_gate_pass": False,
    }
    feature_all["feature_gate_pass"] = bool(
        (feature_all["same_GT_pair_AUC"] or 0.0) >= 0.75
        and (feature_all["mixed_region_AUC"] or 0.0) >= 0.75
        and (feature_all["scene0081_feature_AUC"] or 0.0) >= 0.65
    )
    checkpoint_manifest = {
        "backbone": str(args.backbone),
        "checkpoint": str(args.checkpoint),
        "checkpoint_exists": Path(args.checkpoint).exists(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(args.device),
        "model_missing_keys_preview": getattr(model, "_v35_missing_keys", []),
        "model_unexpected_keys_preview": getattr(model, "_v35_unexpected_keys", []),
        "uses_frozen_visual_backbone": True,
    }
    _write_csv(output_root / "routeB_feature_metrics.csv", feature_rows + [feature_all])
    _write_csv(output_root / "routeB_object_metrics.csv", summary_rows)
    _write_json(
        output_root / "routeB_summary.json",
        {
            "checkpoint_manifest": checkpoint_manifest,
            "scene_diag": scene_diag,
            "feature_rows": feature_rows,
            "feature_all": feature_all,
            "summary_rows": summary_rows,
            "scene_metric_rows": scene_metric_rows,
        },
    )
    _write_json(output_root / "method_manifest.json", checkpoint_manifest)
    print(json.dumps(_json_safe({"output_root": str(output_root), "feature_gate_pass": feature_all["feature_gate_pass"], "summary_count": len(summary_rows)}), indent=2))
    return {"summary_rows": summary_rows, "feature_all": feature_all}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stream4D v35 frozen DINOv2 embedding route.")
    parser.add_argument("--cache-root", default="outputs/stream4d_debug_v26_occupancy_d5_warmstart64_probe5_topup20_patch2")
    parser.add_argument("--split", default="splits/scannet_v6_probe5.txt")
    parser.add_argument("--proposal-root", default="outputs/audit/v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2")
    parser.add_argument("--proposal-label", default="v28_proposal_oracle_repair_temporal_track_640_pruned_shared2_t20_guard5_probe5_r2")
    parser.add_argument("--output-root", default="outputs/audit/v35_routeB_visual_embedding")
    parser.add_argument("--backbone", default="vit_small_patch14_dinov2")
    parser.add_argument("--checkpoint", default="/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--max-frames-per-scene", type=int, default=8)
    parser.add_argument("--max-tubes-per-window", type=int, default=640)
    parser.add_argument("--image-width", type=int, default=1296)
    parser.add_argument("--image-height", type=int, default=968)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-feature-pairs-per-scene", type=int, default=60000)
    parser.add_argument("--embedding-threshold", type=float, default=0.78)
    parser.add_argument("--embedding-unknown-threshold", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=3502)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
