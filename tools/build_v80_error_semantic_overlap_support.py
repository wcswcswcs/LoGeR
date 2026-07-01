#!/usr/bin/env python3
"""Build v80 geometry-error semantic overlap support for V68 merge weighting."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_RISK_LABELS = {
    "void",
    "sky",
    "tree",
    "grass",
    "other_plant",
    "flower",
    "car",
    "truck",
    "bicycle",
    "motorcycle",
    "person",
}
DEFAULT_STABLE_LABELS = {
    "road",
    "ground",
    "path",
    "building",
    "house",
    "other_construction",
    "pole",
    "handrail_or_fence",
    "traffic sign",
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_clean(value), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if torch.is_tensor(value):
        return _clean(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _find_chunk_masklet(stage_c_cache_dir: Path, chunk: int) -> Path:
    matches = sorted(stage_c_cache_dir.glob(f"chunk_{int(chunk):03d}_*/masklet.pt"))
    if not matches:
        raise FileNotFoundError(f"Cannot find stage-C masklet for chunk {chunk}: {stage_c_cache_dir}")
    return matches[0]


def _read_geometry_deltas(path: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                frame = int(row.get("frame", ""))
            except ValueError:
                continue
            parsed: Dict[str, float] = {}
            for key in (
                "baseline_error_m",
                "candidate_error_m",
                "control_error_m",
                "delta_error_vs_baseline_m",
                "delta_error_vs_control_m",
            ):
                try:
                    parsed[key] = float(row.get(key, "nan"))
                except ValueError:
                    parsed[key] = float("nan")
            out[frame] = parsed
    return out


def _label_ids(label_names: Sequence[str], names: Iterable[str]) -> List[int]:
    wanted = {str(x).strip().lower().replace(" ", "_") for x in names}
    out: List[int] = []
    for idx, raw in enumerate(label_names):
        norm = str(raw).strip().lower().replace(" ", "_")
        if norm in wanted:
            out.append(int(idx))
    return out


def _isin(labels: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    if not ids:
        return torch.zeros_like(labels, dtype=torch.bool)
    ids_t = torch.tensor(sorted(set(int(x) for x in ids)), dtype=labels.dtype, device=labels.device)
    return (labels[..., None] == ids_t).any(dim=-1)


def _top_labels(labels: torch.Tensor, label_names: Sequence[str], limit: int = 8) -> List[Dict[str, Any]]:
    vals, counts = torch.unique(labels.reshape(-1), return_counts=True)
    pairs = sorted(zip(vals.tolist(), counts.tolist()), key=lambda item: int(item[1]), reverse=True)
    total = float(labels.numel()) if labels.numel() else 1.0
    rows: List[Dict[str, Any]] = []
    for label_id, count in pairs[:limit]:
        name = label_names[int(label_id)] if 0 <= int(label_id) < len(label_names) else f"id_{label_id}"
        rows.append(
            {
                "label_id": int(label_id),
                "label": str(name),
                "pixels": int(count),
                "ratio": float(count) / total,
            }
        )
    return rows


def _resize_patch(value: torch.Tensor, patch_grid: Tuple[int, int], mode: str) -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(value[:, None].float(), size=patch_grid, mode="nearest").squeeze(1)
    return F.interpolate(value[:, None].float(), size=patch_grid, mode="bilinear", align_corners=False).squeeze(1)


def build(args: argparse.Namespace) -> Dict[str, Any]:
    masklet_path = _find_chunk_masklet(Path(args.stage_c_cache_dir), int(args.chunk))
    chunk_payload = torch.load(masklet_path, map_location="cpu", weights_only=False)
    if not isinstance(chunk_payload, dict):
        raise RuntimeError(f"Unsupported masklet payload: {masklet_path}")
    sem = chunk_payload.get("semantic_segmentation")
    if not isinstance(sem, dict) or "label_maps" not in sem:
        raise RuntimeError(f"Masklet missing semantic label maps: {masklet_path}")

    label_maps = sem["label_maps"].detach().cpu() if torch.is_tensor(sem["label_maps"]) else torch.as_tensor(sem["label_maps"])
    confidence_maps = sem.get("confidence_maps")
    if torch.is_tensor(confidence_maps):
        conf = confidence_maps.detach().cpu().float()
    else:
        conf = torch.ones_like(label_maps, dtype=torch.float32)
    label_names = [str(x) for x in sem.get("label_names", [])]
    start_frame = int(sem.get("global_start_frame", chunk_payload.get("manifest", {}).get("start_frame", 0)))
    overlap = min(int(args.overlap), int(label_maps.shape[0]))
    patch_grid = (int(args.patch_grid[0]), int(args.patch_grid[1]))
    risk_ids = _label_ids(label_names, DEFAULT_RISK_LABELS)
    stable_ids = _label_ids(label_names, DEFAULT_STABLE_LABELS)
    geometry = _read_geometry_deltas(Path(args.geometry_error_csv))
    bad_delta_key = str(args.bad_delta_key or "delta_error_vs_baseline_m")

    label_overlap = label_maps[:overlap].long()
    conf_overlap = conf[:overlap].float().clamp(0.0, 1.0)
    label_patch = _resize_patch(label_overlap.float(), patch_grid, "nearest").round().long()
    conf_patch = _resize_patch(conf_overlap, patch_grid, "bilinear").clamp(0.0, 1.0)

    risk = _isin(label_patch, risk_ids).float()
    stable = _isin(label_patch, stable_ids).float()
    low_conf = (conf_patch < float(args.low_conf_threshold)).float()
    badness_source_note = (
        "absolute geometry error diagnostic"
        if not bad_delta_key.startswith("delta_error_")
        else "candidate-minus-reference geometry error delta"
    )

    scores: List[torch.Tensor] = []
    frame_rows: List[Dict[str, Any]] = []
    for local_idx in range(overlap):
        frame = int(start_frame + local_idx)
        geom_row = geometry.get(frame, {})
        delta = float(geom_row.get(bad_delta_key, 0.0))
        badness = max(0.0, min(1.0, delta / max(float(args.bad_delta_scale), 1e-6)))
        risk_penalty = risk[local_idx] * (float(args.risk_penalty_min) + (1.0 - float(args.risk_penalty_min)) * badness)
        low_conf_penalty = low_conf[local_idx] * float(args.low_conf_penalty) * badness
        stable_bonus = stable[local_idx] * float(args.stable_bonus) * (1.0 - risk[local_idx])
        score = (1.0 - risk_penalty - low_conf_penalty + stable_bonus).clamp(0.0, 1.0)
        scores.append(score)
        raw_labels = label_overlap[local_idx]
        frame_rows.append(
            {
                "frame": frame,
                "local_frame": int(local_idx),
                "delta_error_vs_baseline_m": float(geom_row.get("delta_error_vs_baseline_m", float("nan"))),
                "bad_delta_key": bad_delta_key,
                "bad_delta_value_m": delta,
                "delta_error_vs_control_m": float(geom_row.get("delta_error_vs_control_m", float("nan"))),
                "geometry_badness": float(badness),
                "risk_patch_ratio": float(risk[local_idx].mean().item()),
                "stable_patch_ratio": float(stable[local_idx].mean().item()),
                "low_conf_patch_ratio": float(low_conf[local_idx].mean().item()),
                "support_mean": float(score.mean().item()),
                "support_q10": float(torch.quantile(score.reshape(-1), 0.10).item()),
                "support_q50": float(torch.quantile(score.reshape(-1), 0.50).item()),
                "support_q90": float(torch.quantile(score.reshape(-1), 0.90).item()),
                "top_labels_fullres": _top_labels(raw_labels, label_names, limit=8),
            }
        )

    score_overlap = torch.stack(scores, dim=0).reshape(overlap, int(patch_grid[0]) * int(patch_grid[1])).float()
    control_gen = torch.Generator().manual_seed(int(args.seed))
    control_overlap = torch.rand(score_overlap.shape, generator=control_gen, dtype=torch.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{int(args.chunk):03d}_swa_overlap_{args.kind}_layer_{int(args.layer_idx):02d}.pt"
    payload = {
        "schema": "acl2_v68_swa_overlap_feature_map_v1",
        "artifact": "ACL2_V80_ERROR_SEMANTIC_OVERLAP_SUPPORT",
        "kind": str(args.kind),
        "mode": "geometry_error_semantic_risk_downweight",
        "chunk_idx": int(args.chunk),
        "swa_layer_idx": int(args.layer_idx),
        "batch_size": 1,
        "frame_num": int(label_maps.shape[0]),
        "tokens_per_frame": int(score_overlap.shape[-1]),
        "history_tokens": 0,
        "source_start": 0,
        "source_end": int(score_overlap.numel()),
        "source_tokens": int(score_overlap.numel()),
        "overlap_frames_effective": int(overlap),
        "runtime_swa_overlap_feature_not_qk_proxy": True,
        "Dq_overlap": score_overlap.unsqueeze(0),
        "Ds_overlap": (1.0 - score_overlap).unsqueeze(0),
        "score_overlap": score_overlap.unsqueeze(0),
        "control_overlap": control_overlap.unsqueeze(0),
        "score_mean": float(score_overlap.mean().item()),
        "score_q10": float(torch.quantile(score_overlap.reshape(-1), 0.10).item()),
        "score_q50": float(torch.quantile(score_overlap.reshape(-1), 0.50).item()),
        "score_q90": float(torch.quantile(score_overlap.reshape(-1), 0.90).item()),
        "control_mean": float(control_overlap.mean().item()),
        "control_q90": float(torch.quantile(control_overlap.reshape(-1), 0.90).item()),
        "bad_delta_key": bad_delta_key,
        "badness_source_note": badness_source_note,
        "source_stage_c_masklet": str(masklet_path),
        "source_geometry_error_csv": str(args.geometry_error_csv),
        "source_ttt_attribution_summary": str(args.ttt_attribution_summary),
    }
    torch.save(payload, out_path)

    ttt_summary = _read_json(Path(args.ttt_attribution_summary))
    ttt_decision = ttt_summary.get("decision")
    if not isinstance(ttt_decision, dict):
        ttt_decision = ttt_summary
    summary = {
        "schema": "acl2_v80_error_semantic_overlap_support_summary_v1",
        "support_path": str(out_path),
        "chunk": int(args.chunk),
        "start_frame": int(start_frame),
        "overlap": int(overlap),
        "patch_grid": [int(patch_grid[0]), int(patch_grid[1])],
        "tokens_per_frame": int(score_overlap.shape[-1]),
        "score_mean": float(score_overlap.mean().item()),
        "score_q10": float(torch.quantile(score_overlap.reshape(-1), 0.10).item()),
        "score_q50": float(torch.quantile(score_overlap.reshape(-1), 0.50).item()),
        "score_q90": float(torch.quantile(score_overlap.reshape(-1), 0.90).item()),
        "risk_label_ids": risk_ids,
        "risk_label_names": [label_names[i] for i in risk_ids if 0 <= i < len(label_names)],
        "stable_label_ids": stable_ids,
        "stable_label_names": [label_names[i] for i in stable_ids if 0 <= i < len(label_names)],
        "frame_rows": frame_rows,
        "phase9_ttt_attribution_evidence": {
            "semantic_explains_error_region": ttt_decision.get("semantic_explains_error_region"),
            "ttt_writes_stable_carrier": ttt_decision.get("ttt_writes_stable_carrier"),
            "semantic_ttt_positive_write_available": ttt_decision.get("semantic_ttt_positive_write_available"),
            "random_control_separation": ttt_decision.get("random_control_separation"),
            "recommendation": ttt_decision.get("recommendation"),
        },
        "badness_source_note": badness_source_note,
        "method_note": (
            f"Support is low where overlap frames have {badness_source_note} and semantic risk/low-confidence labels. "
            "It is diagnostic/method-proposal input for V68_OVERLAP_SUPPORT_WEIGHT, not a claimed gate result. "
            "When bad_delta_key is an absolute-error column, do not report it as candidate-regression evidence."
        ),
        "bad_delta_key": bad_delta_key,
    }
    chunk_summary_path = out_dir / f"chunk_{int(args.chunk):03d}_support_map_summary.json"
    summary_path = out_dir / "support_map_summary.json"
    _write_json(chunk_summary_path, summary)
    _write_json(summary_path, summary)
    summary["summary_path"] = str(chunk_summary_path)
    summary["latest_summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-c-cache-dir", required=True, type=Path)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--patch-grid", nargs=2, type=int, default=(19, 66))
    parser.add_argument("--geometry-error-csv", required=True, type=Path)
    parser.add_argument("--ttt-attribution-summary", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--kind", default="source_gate")
    parser.add_argument("--layer-idx", type=int, default=18)
    parser.add_argument("--bad-delta-scale", type=float, default=0.35)
    parser.add_argument(
        "--bad-delta-key",
        default="delta_error_vs_baseline_m",
        choices=(
            "delta_error_vs_baseline_m",
            "delta_error_vs_control_m",
            "baseline_error_m",
            "candidate_error_m",
            "control_error_m",
        ),
        help=(
            "Geometry column treated as badness source; default preserves prior delta behavior. "
            "Absolute-error columns are diagnostic and must be reported as such."
        ),
    )
    parser.add_argument("--low-conf-threshold", type=float, default=0.45)
    parser.add_argument("--risk-penalty-min", type=float, default=0.15)
    parser.add_argument("--low-conf-penalty", type=float, default=0.25)
    parser.add_argument("--stable-bonus", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=80622)
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
