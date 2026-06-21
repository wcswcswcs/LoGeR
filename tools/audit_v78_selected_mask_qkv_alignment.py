#!/usr/bin/env python3
"""Audit selected-mask-conditioned compact SWA K/V overlap alignment.

This is diagnostic-only.  The SWA overlap mask dump has per-frame full tokens
including Pi3 special/register tokens, while compact PCA K/V dumps contain only
patch-grid tensors.  This script uses the auditable Pi3 layout relation

    tokens_per_frame - patch_grid[0] * patch_grid[1] == 6

to skip the six non-patch tokens before mapping selected/random masks to the
compact [T, H, W, C] K/V tensors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final"
)
PHASE9_ROOT = REPORT_ROOT / "phase9_swa_cache_value_carryover"
DEFAULT_MASK_DIR = PHASE9_ROOT / "selected_mask_materialization_v1/masks"
DEFAULT_OUT_DIR = PHASE9_ROOT / "selected_mask_qkv_alignment_v1"


CASES: list[dict[str, Any]] = [
    {
        "suite": "KITTI01_chunk06_P9_34_all_heads_topq80",
        "sequence": "01",
        "chunk": 6,
        "action": "P9_34",
        "action_label": "weak_positive_boundary",
        "qkv_feature_dir": PHASE9_ROOT
        / "qkv_tiny_smoke_chunk06_p9_34_36_38_beta070_v2/chunk06/"
        / "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST/"
        / "v68_layer_pca_features",
    },
    {
        "suite": "KITTI01_chunk06_P9_36_head6_topq80",
        "sequence": "01",
        "chunk": 6,
        "action": "P9_36",
        "action_label": "weak_negative_default",
        "qkv_feature_dir": PHASE9_ROOT
        / "qkv_tiny_smoke_chunk06_p9_34_36_38_beta070_v2/chunk06/"
        / "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST/"
        / "v68_layer_pca_features",
    },
    {
        "suite": "KITTI01_chunk06_P9_38_heads0_6_8_topq80",
        "sequence": "01",
        "chunk": 6,
        "action": "P9_38",
        "action_label": "weak_negative_overlap",
        "qkv_feature_dir": PHASE9_ROOT
        / "qkv_tiny_smoke_chunk06_p9_34_36_38_beta070_v2/chunk06/"
        / "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST/"
        / "v68_layer_pca_features",
    },
    {
        "suite": "KITTI02_chunk14_P9_34_all_heads_topq80",
        "sequence": "02",
        "chunk": 14,
        "action": "P9_34",
        "action_label": "weak_negative_boundary",
        "qkv_feature_dir": Path(
            "results/kitti02_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
            "phase9_swa_cache_value_carryover/qkv_tiny_smoke_chunk14_p9_34_36_38_beta070_v2/chunk14/"
            "P9_34_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_MASS_AUDIT_LAST/"
            "v68_layer_pca_features"
        ),
    },
    {
        "suite": "KITTI02_chunk14_P9_36_head6_topq80",
        "sequence": "02",
        "chunk": 14,
        "action": "P9_36",
        "action_label": "weak_positive_default",
        "qkv_feature_dir": Path(
            "results/kitti02_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
            "phase9_swa_cache_value_carryover/qkv_tiny_smoke_chunk14_p9_34_36_38_beta070_v2/chunk14/"
            "P9_36_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEAD6_LAST/"
            "v68_layer_pca_features"
        ),
    },
    {
        "suite": "KITTI02_chunk14_P9_38_heads0_6_8_topq80",
        "sequence": "02",
        "chunk": 14,
        "action": "P9_38",
        "action_label": "weak_negative_overlap",
        "qkv_feature_dir": Path(
            "results/kitti02_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
            "phase9_swa_cache_value_carryover/qkv_tiny_smoke_chunk14_p9_34_36_38_beta070_v2/chunk14/"
            "P9_38_ATTENTION_BIAS_STABLE_AGREEMENT_TOPQ80_HEADS0_6_8_LAST/"
            "v68_layer_pca_features"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overlap-frames", type=int, default=3)
    parser.add_argument("--patch-random-seed", type=int, default=7802)
    return parser.parse_args()


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _stable_seed(name: str, base_seed: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(base_seed) + (int(digest[:8], 16) % 1_000_000)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=True)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_feature_payloads(feature_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(feature_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            out.append((path, payload))
    return sorted(out, key=lambda item: int(item[1].get("chunk_idx", -1)))


def _layer_ids(payload: dict[str, Any], tap: str) -> list[int]:
    tensor = payload.get(f"layer_ids::{tap}")
    if torch.is_tensor(tensor):
        return [int(x) for x in tensor.detach().cpu().reshape(-1).tolist()]
    meta = payload.get("taps", {}).get(tap, {})
    return [int(x) for x in meta.get("selected_layer_ids", [])]


def _tap_tensor(payload: dict[str, Any], kind: str, source: str) -> torch.Tensor | None:
    value = payload.get(f"tap::pca_swa_{source}_{kind}_layers")
    return value.detach().cpu().float() if torch.is_tensor(value) else None


def _metric_stats(values: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    if values.shape != mask.shape:
        raise ValueError(f"metric/mask shape mismatch: {tuple(values.shape)} vs {tuple(mask.shape)}")
    finite = torch.isfinite(values)
    use = mask & finite
    selected = values[use]
    if selected.numel() == 0:
        return {
            "count": 0,
            "fraction": 0.0,
            "mean": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(selected.numel()),
        "fraction": float(use.float().mean().item()),
        "mean": float(selected.mean().item()),
        "min": float(selected.min().item()),
        "p05": float(torch.quantile(selected.float(), 0.05).item()),
        "p50": float(torch.quantile(selected.float(), 0.50).item()),
        "p95": float(torch.quantile(selected.float(), 0.95).item()),
        "max": float(selected.max().item()),
    }


def _same_mass_patch_random(mask: torch.Tensor, *, seed: int) -> torch.Tensor:
    flat = mask.reshape(-1)
    n = int(flat.numel())
    k = int(mask.sum().item())
    out = torch.zeros(n, dtype=torch.bool)
    if k > 0:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        out[torch.randperm(n, generator=generator)[:k]] = True
    return out.reshape_as(mask)


def _prepare_patch_masks(
    *,
    mask_payload: dict[str, Any],
    patch_grid: tuple[int, int],
    overlap_frames: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    selected_full = mask_payload.get("selected_mask_topq")
    random_full = mask_payload.get("random_same_mass_mask")
    score_full = mask_payload.get("score_overlap")
    if not torch.is_tensor(selected_full) or not torch.is_tensor(random_full):
        raise RuntimeError("mask artifact lacks selected_mask_topq/random_same_mass_mask")
    selected_full = selected_full.detach().cpu().bool()
    random_full = random_full.detach().cpu().bool()
    score_full_t = score_full.detach().cpu().float() if torch.is_tensor(score_full) else None
    if selected_full.ndim == 3 and int(selected_full.shape[0]) == 1:
        selected_full = selected_full[0]
    if random_full.ndim == 3 and int(random_full.shape[0]) == 1:
        random_full = random_full[0]
    if score_full_t is not None and score_full_t.ndim == 3 and int(score_full_t.shape[0]) == 1:
        score_full_t = score_full_t[0]
    if selected_full.ndim != 2 or random_full.ndim != 2:
        raise RuntimeError(f"unexpected mask shapes: selected={tuple(selected_full.shape)} random={tuple(random_full.shape)}")
    ov = min(int(overlap_frames), int(selected_full.shape[0]), int(random_full.shape[0]))
    h, w = patch_grid
    patch_tokens = int(h) * int(w)
    tokens_per_frame = int(selected_full.shape[-1])
    patch_start = tokens_per_frame - patch_tokens
    if patch_start < 0:
        raise RuntimeError(
            f"mask has fewer tokens than compact patch grid: tokens={tokens_per_frame} patches={patch_tokens}"
        )
    patch_end = patch_start + patch_tokens
    selected_patch = selected_full[:ov, patch_start:patch_end].reshape(ov, h, w)
    random_saved_patch = random_full[:ov, patch_start:patch_end].reshape(ov, h, w)
    random_patch_same_mass = _same_mass_patch_random(selected_patch, seed=seed)
    masks = {
        "selected": selected_patch,
        "random_saved_patch": random_saved_patch,
        "random_patch_same_mass": random_patch_same_mass,
        "all_patch": torch.ones_like(selected_patch, dtype=torch.bool),
    }
    meta: dict[str, Any] = {
        "overlap_frames_used": int(ov),
        "tokens_per_frame": int(tokens_per_frame),
        "patch_grid": [int(h), int(w)],
        "patch_tokens_per_frame": int(patch_tokens),
        "patch_start": int(patch_start),
        "patch_end": int(patch_end),
        "special_tokens_per_frame_inferred": int(patch_start),
        "selected_full_count": int(selected_full[:ov].sum().item()),
        "random_saved_full_count": int(random_full[:ov].sum().item()),
        "selected_patch_count": int(selected_patch.sum().item()),
        "random_saved_patch_count": int(random_saved_patch.sum().item()),
        "random_patch_same_mass_count": int(random_patch_same_mass.sum().item()),
        "selected_special_count_dropped": int(selected_full[:ov, :patch_start].sum().item()) if patch_start else 0,
        "random_saved_special_count_dropped": int(random_full[:ov, :patch_start].sum().item()) if patch_start else 0,
        "patch_random_seed": int(seed),
    }
    if score_full_t is not None:
        score_patch = score_full_t[:ov, patch_start:patch_end].reshape(ov, h, w)
        meta.update(
            {
                "score_patch_selected_mean": _metric_stats(score_patch, selected_patch)["mean"],
                "score_patch_random_same_mass_mean": _metric_stats(score_patch, random_patch_same_mass)["mean"],
                "score_patch_selected_minus_random_same_mass_mean": (
                    (_metric_stats(score_patch, selected_patch)["mean"] or 0.0)
                    - (_metric_stats(score_patch, random_patch_same_mass)["mean"] or 0.0)
                ),
            }
        )
    return masks, meta


def _build_rows_for_case(
    *,
    case: dict[str, Any],
    mask_dir: Path,
    overlap_frames: int,
    patch_random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suite = str(case["suite"])
    mask_path = mask_dir / f"{_safe_name(suite)}_selected_random_masks.pt"
    feature_dir = Path(case["qkv_feature_dir"])
    case_meta: dict[str, Any] = {
        "suite": suite,
        "sequence": case["sequence"],
        "chunk": int(case["chunk"]),
        "action": case["action"],
        "action_label": case["action_label"],
        "mask_path": str(mask_path),
        "qkv_feature_dir": str(feature_dir),
        "available": False,
    }
    if not mask_path.is_file():
        case_meta["reason"] = "missing_mask_artifact"
        return [], case_meta
    payloads = _load_feature_payloads(feature_dir)
    if len(payloads) < 2:
        case_meta["reason"] = "missing_two_qkv_feature_payloads"
        return [], case_meta

    pair: tuple[tuple[Path, dict[str, Any]], tuple[Path, dict[str, Any]]] | None = None
    for prev, cur in zip(payloads, payloads[1:]):
        prev_chunk = int(prev[1].get("chunk_idx", -1))
        cur_chunk = int(cur[1].get("chunk_idx", -1))
        if cur_chunk == prev_chunk + 1 and cur_chunk == int(case["chunk"]):
            pair = (prev, cur)
            break
    if pair is None:
        case_meta["reason"] = "missing_consecutive_prev_cur_pair"
        return [], case_meta
    (prev_file, prev_payload), (cur_file, cur_payload) = pair
    patch_grid_raw = cur_payload.get("patch_grid") or prev_payload.get("patch_grid")
    if not isinstance(patch_grid_raw, (list, tuple)) or len(patch_grid_raw) != 2:
        case_meta["reason"] = "missing_patch_grid"
        return [], case_meta
    patch_grid = (int(patch_grid_raw[0]), int(patch_grid_raw[1]))
    mask_payload = torch.load(mask_path, map_location="cpu", weights_only=False)
    seed = _stable_seed(suite, int(patch_random_seed))
    masks, mapping_meta = _prepare_patch_masks(
        mask_payload=mask_payload,
        patch_grid=patch_grid,
        overlap_frames=overlap_frames,
        seed=seed,
    )
    ov = int(mapping_meta["overlap_frames_used"])
    rows: list[dict[str, Any]] = []
    for kind in ("k", "v"):
        for source_pair in (
            ("current", "current"),
            ("cache", "current"),
            ("current", "cache"),
            ("cache", "cache"),
        ):
            prev_tensor = _tap_tensor(prev_payload, kind, source_pair[0])
            cur_tensor = _tap_tensor(cur_payload, kind, source_pair[1])
            if prev_tensor is None or cur_tensor is None:
                rows.append(
                    {
                        **case_meta,
                        **mapping_meta,
                        "available": False,
                        "kind": kind,
                        "source_pair": f"prev_{source_pair[0]}__cur_{source_pair[1]}",
                        "reason": "missing_tap",
                    }
                )
                continue
            if prev_tensor.ndim < 5 or tuple(prev_tensor.shape[2:4]) != patch_grid:
                rows.append(
                    {
                        **case_meta,
                        **mapping_meta,
                        "available": False,
                        "kind": kind,
                        "source_pair": f"prev_{source_pair[0]}__cur_{source_pair[1]}",
                        "reason": "unexpected_prev_shape",
                    }
                )
                continue
            if cur_tensor.ndim < 5 or tuple(cur_tensor.shape[2:4]) != patch_grid:
                rows.append(
                    {
                        **case_meta,
                        **mapping_meta,
                        "available": False,
                        "kind": kind,
                        "source_pair": f"prev_{source_pair[0]}__cur_{source_pair[1]}",
                        "reason": "unexpected_cur_shape",
                    }
                )
                continue
            layer_ids = _layer_ids(prev_payload, f"pca_swa_{source_pair[0]}_{kind}_layers")
            layer_count = min(int(prev_tensor.shape[1]), int(cur_tensor.shape[1]))
            for layer_pos in range(layer_count):
                prev_tail = prev_tensor[-ov:, layer_pos].float()
                cur_head = cur_tensor[:ov, layer_pos].float()
                cos = F.cosine_similarity(
                    prev_tail.reshape(-1, prev_tail.shape[-1]),
                    cur_head.reshape(-1, cur_head.shape[-1]),
                    dim=-1,
                ).reshape(ov, patch_grid[0], patch_grid[1])
                diff = prev_tail - cur_head
                rmse = torch.sqrt((diff.float() ** 2).mean(dim=-1))
                mean_abs = diff.abs().mean(dim=-1)
                prev_norm = torch.linalg.norm(prev_tail, dim=-1)
                cur_norm = torch.linalg.norm(cur_head, dim=-1)
                row: dict[str, Any] = {
                    **case_meta,
                    **mapping_meta,
                    "available": True,
                    "prev_feature_file": str(prev_file),
                    "cur_feature_file": str(cur_file),
                    "prev_chunk_idx": int(prev_payload.get("chunk_idx")),
                    "cur_chunk_idx": int(cur_payload.get("chunk_idx")),
                    "prev_tail_start_frame": int(prev_payload.get("end_frame")) - ov,
                    "prev_tail_end_frame": int(prev_payload.get("end_frame")),
                    "cur_head_start_frame": int(cur_payload.get("start_frame")),
                    "cur_head_end_frame": int(cur_payload.get("start_frame")) + ov,
                    "kind": kind,
                    "source_pair": f"prev_{source_pair[0]}__cur_{source_pair[1]}",
                    "layer_pos": int(layer_pos),
                    "layer_id": int(layer_ids[layer_pos]) if layer_pos < len(layer_ids) else None,
                }
                for mask_name, mask in masks.items():
                    cos_stats = _metric_stats(cos, mask)
                    rmse_stats = _metric_stats(rmse, mask)
                    mean_abs_stats = _metric_stats(mean_abs, mask)
                    prev_norm_stats = _metric_stats(prev_norm, mask)
                    cur_norm_stats = _metric_stats(cur_norm, mask)
                    prefix = mask_name
                    row.update(
                        {
                            f"{prefix}_count": cos_stats["count"],
                            f"{prefix}_fraction": cos_stats["fraction"],
                            f"{prefix}_cosine_mean": cos_stats["mean"],
                            f"{prefix}_cosine_p05": cos_stats["p05"],
                            f"{prefix}_cosine_p50": cos_stats["p50"],
                            f"{prefix}_cosine_p95": cos_stats["p95"],
                            f"{prefix}_rmse_mean": rmse_stats["mean"],
                            f"{prefix}_mean_abs_diff": mean_abs_stats["mean"],
                            f"{prefix}_prev_norm_mean": prev_norm_stats["mean"],
                            f"{prefix}_cur_norm_mean": cur_norm_stats["mean"],
                        }
                    )
                sel_cos = _finite(row.get("selected_cosine_mean"))
                rand_cos = _finite(row.get("random_patch_same_mass_cosine_mean"))
                sel_rmse = _finite(row.get("selected_rmse_mean"))
                rand_rmse = _finite(row.get("random_patch_same_mass_rmse_mean"))
                row["selected_minus_random_patch_same_mass_cosine_mean"] = (
                    sel_cos - rand_cos if sel_cos is not None and rand_cos is not None else None
                )
                row["selected_minus_random_patch_same_mass_rmse_mean"] = (
                    sel_rmse - rand_rmse if sel_rmse is not None and rand_rmse is not None else None
                )
                rows.append(row)
    case_meta.update(
        {
            "available": True,
            **mapping_meta,
            "prev_feature_file": str(prev_file),
            "cur_feature_file": str(cur_file),
            "num_rows": len(rows),
        }
    )
    return rows, case_meta


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case in CASES:
        case_rows, case_meta = _build_rows_for_case(
            case=case,
            mask_dir=args.mask_dir,
            overlap_frames=args.overlap_frames,
            patch_random_seed=args.patch_random_seed,
        )
        rows.extend(case_rows)
        case_summaries.append(case_meta)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = args.out_dir / "selected_mask_qkv_alignment_rows.csv"
    summary_json = args.out_dir / "selected_mask_qkv_alignment_summary.json"
    _write_csv(rows_csv, rows)

    available_rows = [row for row in rows if row.get("available") is True]
    key_rows = [
        row
        for row in available_rows
        if row.get("source_pair") == "prev_current__cur_current"
    ]
    by_case_kind_layer = {
        f"{row['suite']}:{row['kind']}:L{row['layer_id']}": {
            "selected_cosine_mean": row.get("selected_cosine_mean"),
            "random_patch_same_mass_cosine_mean": row.get("random_patch_same_mass_cosine_mean"),
            "selected_minus_random_patch_same_mass_cosine_mean": row.get(
                "selected_minus_random_patch_same_mass_cosine_mean"
            ),
            "selected_rmse_mean": row.get("selected_rmse_mean"),
            "random_patch_same_mass_rmse_mean": row.get("random_patch_same_mass_rmse_mean"),
            "selected_minus_random_patch_same_mass_rmse_mean": row.get(
                "selected_minus_random_patch_same_mass_rmse_mean"
            ),
        }
        for row in key_rows
    }
    deltas = [
        _finite(row.get("selected_minus_random_patch_same_mass_cosine_mean"))
        for row in key_rows
    ]
    deltas = [x for x in deltas if x is not None]
    _write_json(
        summary_json,
        {
            "schema": "acl2_v78_selected_mask_qkv_alignment_v1",
            "diagnostic_only": True,
            "method_gate_claimed": False,
            "rows_csv": str(rows_csv),
            "mask_dir": str(args.mask_dir),
            "cases": case_summaries,
            "num_rows": len(rows),
            "num_available_rows": len(available_rows),
            "key_source_pair": "prev_current__cur_current",
            "selected_minus_random_patch_same_mass_cosine_mean_by_case_kind_layer": by_case_kind_layer,
            "key_delta_count": len(deltas),
            "key_delta_positive_count": int(sum(1 for x in deltas if x > 0.0)),
            "key_delta_mean": float(sum(deltas) / len(deltas)) if deltas else None,
            "interpretation": [
                "The selected SWA overlap source mask can be mapped to compact K/V patch tensors by dropping the six Pi3 special/register tokens per frame.",
                "Rows compare previous-tail source K/V to aligned current-head K/V on selected patch tokens versus deterministic patch-space same-mass random tokens.",
                "The six-suite beta0.70 audit is intended to compare weak-positive and weak-negative action/window cases.",
                "This is still offline diagnostic evidence; it does not prove a runtime selector or held-out improvement.",
            ],
            "limitations": [
                "The current default cases cover six beta0.70 compact Q/K/V tiny-smoke action/window dumps.",
                "The mapping uses patch_start=tokens_per_frame-H*W and is valid only when it equals the known Pi3 patch_start_idx=6.",
                "Patch-space random is a deterministic diagnostic control because full-token random can select non-patch special tokens absent from compact Q/K/V dumps.",
                "These compact Q/K/V features are mostly window-level; they do not by themselves explain head/action-specific outcome differences.",
            ],
        },
    )
    print(json.dumps({"rows": str(rows_csv), "summary": str(summary_json)}, indent=2))


if __name__ == "__main__":
    main()
