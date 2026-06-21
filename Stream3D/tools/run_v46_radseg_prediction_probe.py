from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from evaluation.constants import SCANNET_LABELS, SCANNETPP_LABELS
from stream4d_native.frozen_feature_adapter import locate_default_radio_checkpoint
from tools.run_v46_raw_carrier_incidence_repair import ROOT, _json_safe, _load_mask_label


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in keys})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_rgb(scene: str, frame_id: int) -> np.ndarray:
    path = ROOT / "data/scannet/processed" / str(scene) / "color" / f"{int(frame_id)}.jpg"
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"missing color frame: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _class_list(source: str, explicit: str, limit: int) -> list[str]:
    if explicit.strip():
        labels = [item.strip() for item in explicit.split(",") if item.strip()]
    elif source == "scannetpp":
        labels = list(SCANNETPP_LABELS)
    else:
        labels = list(SCANNET_LABELS)
    if int(limit) > 0:
        labels = labels[: int(limit)]
    if not labels:
        raise ValueError("empty RADSeg class prompt list")
    return labels


def _select_edges(path: Path, group: str, max_edges: int) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    selected = rows[: max(0, int(max_edges))]
    for rank, row in enumerate(selected, start=1):
        row["_edge_group"] = group
        row["_edge_rank"] = rank
    return selected


def _load_node_map(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    rows = _read_csv(path)
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[int(float(row["node_id"]))] = row
    return out


def _node_refs(edge_rows: list[dict[str, Any]], node_map: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in edge_rows:
        scene = str(row["scene"])
        for side in ["left", "right"]:
            frame_id = int(float(row[f"{side}_frame_id"]))
            node_id = int(float(row[f"{side}_node_id"]))
            mask_value = row.get(f"{side}_mask_id")
            if mask_value in {None, ""}:
                node_row = node_map.get(node_id, {})
                mask_value = node_row.get("mask_id")
                if node_row.get("frame_id") not in {None, ""}:
                    frame_id = int(float(node_row["frame_id"]))
            if mask_value in {None, ""}:
                raise ValueError(f"missing mask_id for node_id={node_id}; pass --node-rows-csv")
            mask_id = int(float(mask_value))
            key = (scene, frame_id, mask_id, node_id)
            keyed[key] = {
                "scene": scene,
                "frame_id": frame_id,
                "mask_id": mask_id,
                "node_id": node_id,
            }
    return list(keyed.values())


def _load_encoder(
    *,
    checkpoint: str | None,
    device: str,
    lang_model: str,
    classes: list[str],
    text_query_mode: str,
    amp: bool,
    prediction_thresh: float,
    prompt_denoising_thresh: float,
    slide_crop: int,
    slide_stride: int,
) -> Any:
    import torch
    from vipe.priors.embedding.radseg_encoder import RADSegEncoder

    model_version = checkpoint or locate_default_radio_checkpoint()
    if model_version is None:
        raise FileNotFoundError("no local RADIO/RADSeg checkpoint found")

    original_load = torch.load

    def _compat_load(*args: Any, **kwargs: Any) -> Any:
        if args and str(args[0]) == str(model_version) and "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _compat_load
    try:
        encoder = RADSegEncoder(
            device=device,
            model_version=str(model_version),
            lang_model=str(lang_model),
            return_radio_features=True,
            compile=False,
            amp=bool(amp),
            predict=True,
            classes=classes,
            text_query_mode=str(text_query_mode),
            prediction_thresh=float(prediction_thresh),
            prompt_denoising_thresh=float(prompt_denoising_thresh),
            slide_crop=int(slide_crop),
            slide_stride=int(slide_stride),
            sam_refinement=False,
        )
    finally:
        torch.load = original_load
    return encoder


def _softmax_scaled(scores: np.ndarray, scale: float = 100.0) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64) * float(scale)
    values = values - float(np.max(values)) if values.size else values
    exp = np.exp(values)
    denom = float(exp.sum())
    if denom <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return exp / denom


def _resize_rgb_max_side(rgb: np.ndarray, max_image_side: int) -> np.ndarray:
    if int(max_image_side) <= 0:
        return rgb
    height, width = rgb.shape[:2]
    longer = max(int(height), int(width))
    if longer <= int(max_image_side):
        return rgb
    scale = float(max_image_side) / float(longer)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _frame_prediction(
    encoder: Any,
    rgb: np.ndarray,
    *,
    device: str,
    probe_mode: str,
    max_image_side: int,
) -> dict[str, np.ndarray | str | int | list[int]]:
    import torch

    original_shape = [int(rgb.shape[0]), int(rgb.shape[1])]
    rgb = _resize_rgb_max_side(rgb, int(max_image_side))
    resized_shape = [int(rgb.shape[0]), int(rgb.shape[1])]
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0).float()[None].to(device)
    with torch.inference_mode():
        if probe_mode == "pixel_prediction":
            probs, pred = encoder.encode_image_to_feat_map(tensor, orig_img_size=rgb.shape[:2], return_preds=True)
            return {
                "probe_mode": "pixel_prediction",
                "original_shape": original_shape,
                "resized_shape": resized_shape,
                "probs": probs.squeeze(0).detach().cpu().float().numpy(),
                "pred": pred.squeeze(0).squeeze(0).detach().cpu().numpy(),
            }
        old_predict = bool(encoder.predict)
        encoder.predict = False
        try:
            feat = encoder.encode_image_to_feat_map(tensor)
        finally:
            encoder.predict = old_predict
        aligned = encoder.align_spatial_features_with_language(feat, onehot=False)
        features = aligned.squeeze(0).permute(1, 2, 0).contiguous().float().detach().cpu().numpy()
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    features = features / np.maximum(norm, 1e-8)
    return {
        "probe_mode": "mask_pooled_logits",
        "original_shape": original_shape,
        "resized_shape": resized_shape,
        "features": features.astype(np.float32),
    }


def _top_probs(mean_probs: np.ndarray, labels: list[str], top_k: int) -> list[dict[str, Any]]:
    class_probs = np.asarray(mean_probs[1:], dtype=np.float64)
    order = np.argsort(-class_probs)[: int(top_k)]
    return [
        {
            "class_index": int(idx + 1),
            "label": labels[int(idx)],
            "prob": float(class_probs[int(idx)]),
        }
        for idx in order
    ]


def _entropy(probs: np.ndarray) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)) / max(math.log(float(arr.size)), 1e-12))


def _mask_summary(
    *,
    node: dict[str, Any],
    labels: list[str],
    frame_payload: dict[str, np.ndarray | str],
    text_embeds: np.ndarray,
) -> dict[str, Any]:
    scene = str(node["scene"])
    frame_id = int(node["frame_id"])
    mask_id = int(node["mask_id"])
    label_img = _load_mask_label(scene, frame_id)
    if label_img is None:
        raise FileNotFoundError(f"missing prepared mask label for {scene} frame {frame_id}")
    full_mask = label_img == mask_id
    pixel_count = int(full_mask.sum())
    probe_mode = str(frame_payload["probe_mode"])
    resized_shape = frame_payload.get("resized_shape")
    token_count = None
    if probe_mode == "pixel_prediction":
        probs = np.asarray(frame_payload["probs"], dtype=np.float32)
        pred = np.asarray(frame_payload["pred"])
        if label_img.shape[:2] != pred.shape[:2]:
            label_img = cv2.resize(label_img, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = label_img == mask_id
        token_count = int(mask.sum())
        if token_count <= 0:
            mean_probs = np.zeros((len(labels) + 1,), dtype=np.float32)
            mode_label = ""
            mode_fraction = 0.0
        else:
            mean_probs = probs[:, mask].mean(axis=1)
            pred_values = pred[mask].astype(np.int64, copy=False)
            pred_values = pred_values[(pred_values > 0) & (pred_values <= len(labels))]
            if pred_values.size:
                counts = Counter(int(v) for v in pred_values.tolist())
                mode_index, mode_count = counts.most_common(1)[0]
                mode_label = labels[int(mode_index) - 1]
                mode_fraction = float(mode_count / max(token_count, 1))
            else:
                mode_label = ""
                mode_fraction = 0.0
    else:
        features = np.asarray(frame_payload["features"], dtype=np.float32)
        small_mask = cv2.resize(
            full_mask.astype(np.uint8),
            (features.shape[1], features.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        token_count = int(small_mask.sum())
        if token_count <= 0:
            mean_probs = np.zeros((len(labels) + 1,), dtype=np.float32)
            mode_label = ""
            mode_fraction = 0.0
        else:
            token_features = features[small_mask]
            pooled = token_features.mean(axis=0)
            pooled = pooled / max(float(np.linalg.norm(pooled)), 1e-8)
            scores = np.asarray(text_embeds, dtype=np.float32) @ pooled.astype(np.float32)
            class_probs = _softmax_scaled(scores)
            mean_probs = np.concatenate([np.zeros((1,), dtype=np.float64), class_probs], axis=0).astype(np.float32)
            token_scores = token_features @ np.asarray(text_embeds, dtype=np.float32).T
            token_pred = np.argmax(token_scores, axis=1) + 1
            counts = Counter(int(v) for v in token_pred.tolist())
            mode_index, mode_count = counts.most_common(1)[0]
            mode_label = labels[int(mode_index) - 1]
            mode_fraction = float(mode_count / max(token_count, 1))
    top = _top_probs(mean_probs, labels, top_k=5)
    return {
        "scene": scene,
        "node_id": int(node["node_id"]),
        "frame_id": frame_id,
        "mask_id": mask_id,
        "mask_pixel_count": pixel_count,
        "radseg_probe_mode": probe_mode,
        "radseg_resized_shape": resized_shape,
        "radseg_probe_token_count": token_count,
        "radseg_mode_label": mode_label,
        "radseg_mode_fraction": mode_fraction,
        "radseg_top1_label": top[0]["label"] if top else "",
        "radseg_top1_prob": top[0]["prob"] if top else 0.0,
        "radseg_top2_label": top[1]["label"] if len(top) > 1 else "",
        "radseg_top2_prob": top[1]["prob"] if len(top) > 1 else 0.0,
        "radseg_top5": top,
        "radseg_mean_prob_entropy": _entropy(mean_probs[1:]),
        "radseg_mean_probs": mean_probs.astype(np.float32).tolist(),
        "uses_rgb_for_prediction": True,
        "uses_frozen_radseg_prediction": True,
        "uses_gt_for_prediction": False,
    }


def _cosine01(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, 0.5 * (float(np.dot(a, b) / denom) + 1.0))))


def _edge_summary(row: dict[str, Any], mask_by_node: dict[int, dict[str, Any]]) -> dict[str, Any]:
    left = mask_by_node[int(float(row["left_node_id"]))]
    right = mask_by_node[int(float(row["right_node_id"]))]
    left_probs = np.asarray(left["radseg_mean_probs"], dtype=np.float32)[1:]
    right_probs = np.asarray(right["radseg_mean_probs"], dtype=np.float32)[1:]
    return {
        "edge_group": row.get("_edge_group"),
        "edge_rank": int(row.get("_edge_rank", 0)),
        "scene": row.get("scene"),
        "left_node_id": int(float(row["left_node_id"])),
        "right_node_id": int(float(row["right_node_id"])),
        "left_gt": row.get("left_gt"),
        "right_gt": row.get("right_gt"),
        "diagnostic_same_gt": _parse_bool(row.get("diagnostic_same_gt")),
        "P5_p4_semantic_boost_capped": _parse_float(row.get("P5_p4_semantic_boost_capped")),
        "P6_feature_only": _parse_float(row.get("P6_feature_only")),
        "left_radseg_top1_label": left["radseg_top1_label"],
        "right_radseg_top1_label": right["radseg_top1_label"],
        "radseg_top1_label_same": bool(left["radseg_top1_label"] == right["radseg_top1_label"]),
        "left_radseg_mode_label": left["radseg_mode_label"],
        "right_radseg_mode_label": right["radseg_mode_label"],
        "radseg_mode_label_same": bool(left["radseg_mode_label"] == right["radseg_mode_label"]),
        "radseg_mean_prob_cosine01": _cosine01(left_probs, right_probs),
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }


def _free_text_model(encoder: Any) -> bool:
    try:
        import torch

        if getattr(getattr(encoder, "lang_adaptor", None), "text_model", None) is None:
            return False
        encoder.lang_adaptor.text_model = None
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False


def _group_summary(edge_rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    rows = [row for row in edge_rows if row["edge_group"] == group]
    if not rows:
        return {
            "edge_group": group,
            "edge_count": 0,
            "radseg_top1_label_same_rate": None,
            "radseg_mode_label_same_rate": None,
            "radseg_mean_prob_cosine01_mean": None,
        }
    return {
        "edge_group": group,
        "edge_count": len(rows),
        "radseg_top1_label_same_rate": float(sum(1 for row in rows if row["radseg_top1_label_same"]) / len(rows)),
        "radseg_mode_label_same_rate": float(sum(1 for row in rows if row["radseg_mode_label_same"]) / len(rows)),
        "radseg_mean_prob_cosine01_mean": float(np.mean([float(row["radseg_mean_prob_cosine01"]) for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe real RADSeg prediction semantics on v46 false-positive/negative mask edges.")
    parser.add_argument("--false-positive-csv", required=True)
    parser.add_argument("--false-negative-csv", required=True)
    parser.add_argument("--node-rows-csv", default="")
    parser.add_argument("--max-edges-per-group", type=int, default=10)
    parser.add_argument("--class-source", choices=["scannet", "scannetpp"], default="scannet")
    parser.add_argument("--classes", default="")
    parser.add_argument("--class-limit", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lang-model", default="siglip2")
    parser.add_argument("--text-query-mode", choices=["labels", "prompts"], default="prompts")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--probe-mode", choices=["mask_pooled_logits", "pixel_prediction"], default="mask_pooled_logits")
    parser.add_argument("--max-image-side", type=int, default=0)
    parser.add_argument("--keep-text-model", action="store_true")
    parser.add_argument("--prediction-thresh", type=float, default=0.0)
    parser.add_argument("--prompt-denoising-thresh", type=float, default=0.0)
    parser.add_argument("--slide-crop", type=int, default=0)
    parser.add_argument("--slide-stride", type=int, default=224)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    false_positive_edges = _select_edges(Path(args.false_positive_csv), "top_false_positive", int(args.max_edges_per_group))
    false_negative_edges = _select_edges(Path(args.false_negative_csv), "top_false_negative", int(args.max_edges_per_group))
    selected_edges = false_positive_edges + false_negative_edges
    node_map = _load_node_map(Path(args.node_rows_csv) if str(args.node_rows_csv).strip() else None)
    labels = _class_list(str(args.class_source), str(args.classes), int(args.class_limit))
    checkpoint = str(args.checkpoint).strip() or locate_default_radio_checkpoint()
    encoder = _load_encoder(
        checkpoint=checkpoint,
        device=str(args.device),
        lang_model=str(args.lang_model),
        classes=labels,
        text_query_mode=str(args.text_query_mode),
        amp=bool(args.amp),
        prediction_thresh=float(args.prediction_thresh),
        prompt_denoising_thresh=float(args.prompt_denoising_thresh),
        slide_crop=int(args.slide_crop),
        slide_stride=int(args.slide_stride),
    )
    text_embeds = np.asarray(encoder.text_embeds.detach().cpu().float().numpy(), dtype=np.float32)
    text_embeds = text_embeds / np.maximum(np.linalg.norm(text_embeds, axis=1, keepdims=True), 1e-8)
    text_model_freed = False if bool(args.keep_text_model) else _free_text_model(encoder)

    frame_cache: dict[tuple[str, int], dict[str, np.ndarray | str]] = {}
    mask_rows: list[dict[str, Any]] = []
    for node in _node_refs(selected_edges, node_map):
        key = (str(node["scene"]), int(node["frame_id"]))
        if key not in frame_cache:
            frame_cache[key] = _frame_prediction(
                encoder,
                _load_rgb(key[0], key[1]),
                device=str(args.device),
                probe_mode=str(args.probe_mode),
                max_image_side=int(args.max_image_side),
            )
        mask_rows.append(_mask_summary(node=node, labels=labels, frame_payload=frame_cache[key], text_embeds=text_embeds))
    mask_by_node = {int(row["node_id"]): row for row in mask_rows}
    edge_rows = [_edge_summary(row, mask_by_node) for row in selected_edges]
    summary_rows = [_group_summary(edge_rows, "top_false_positive"), _group_summary(edge_rows, "top_false_negative")]

    payload = {
        "phase": "v46_radseg_prediction_probe",
        "created_at": _utc_now(),
        "false_positive_csv": str(args.false_positive_csv),
        "false_negative_csv": str(args.false_negative_csv),
        "node_rows_csv": str(args.node_rows_csv),
        "max_edges_per_group": int(args.max_edges_per_group),
        "class_source": str(args.class_source),
        "class_count": len(labels),
        "classes": labels,
        "checkpoint": checkpoint,
        "device": str(args.device),
        "lang_model": str(args.lang_model),
        "text_query_mode": str(args.text_query_mode),
        "amp": bool(args.amp),
        "probe_mode": str(args.probe_mode),
        "max_image_side": int(args.max_image_side),
        "text_model_freed_after_embedding": bool(text_model_freed),
        "prediction_thresh": float(args.prediction_thresh),
        "prompt_denoising_thresh": float(args.prompt_denoising_thresh),
        "slide_crop": int(args.slide_crop),
        "slide_stride": int(args.slide_stride),
        "frame_prediction_count": len(frame_cache),
        "mask_count": len(mask_rows),
        "edge_count": len(edge_rows),
        "summary_rows": summary_rows,
        "uses_rgb_for_prediction": True,
        "uses_frozen_radseg_prediction": True,
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
        "diagnostic_only": True,
    }
    out = ROOT / str(args.output_root)
    _write_json(out / "radseg_prediction_probe.json", payload)
    _write_csv(out / "radseg_mask_rows.csv", mask_rows)
    _write_csv(out / "radseg_edge_rows.csv", edge_rows)
    _write_csv(out / "radseg_summary_rows.csv", summary_rows)
    print(json.dumps({"summary": str(out / "radseg_prediction_probe.json"), "summary_rows": summary_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
