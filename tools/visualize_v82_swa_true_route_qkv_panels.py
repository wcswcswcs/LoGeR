#!/usr/bin/env python3
"""Render v82 SWA true-route/QKV panels from runtime route dumps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


DEFAULT_BANK = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase2_swa_pair_bank_v2/swa_pair_bank_v2.csv"
)
DEFAULT_ROUTE_ROOT = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase3_swa_true_route_visual_confirmation/route_dump"
)
DEFAULT_QKV_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS3_swa_visual_confirmation/qkv_prefix_runs"
)
DEFAULT_OUT_DIR = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase3_swa_true_route_visual_confirmation"
)

ROUTE_CASES = {
    "source_replace": "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
    "source_gate": "P9_6_SOURCE_GATE_ROLE_NEGATIVE_V_LAST",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_map(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _title(img: Image.Image, title: str) -> Image.Image:
    img = img.convert("RGB")
    if not title:
        return img
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, img.width, 18), fill=(0, 0, 0))
    draw.text((4, 4), title[:90], fill=(255, 255, 255), font=font)
    return img


def _text(lines: list[str], size: tuple[int, int], title: str) -> Image.Image:
    img = Image.new("RGB", size, (247, 247, 247))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, size[0], 20), fill=(0, 0, 0))
    draw.text((4, 4), title[:80], fill=(255, 255, 255), font=font)
    y = 28
    for line in lines:
        draw.text((6, y), str(line)[:96], fill=(20, 20, 20), font=font)
        y += 14
        if y > size[1] - 14:
            break
    return img


def _heat(arr: np.ndarray, size: tuple[int, int], title: str) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((3, 1260), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    while arr.ndim > 2:
        arr = arr.mean(axis=0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    amin = float(arr.min())
    amax = float(arr.max())
    norm = (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr)
    rgb = np.stack(
        [
            norm * 255.0,
            (1.0 - np.abs(norm - 0.5) * 2.0) * 190.0,
            (1.0 - norm) * 255.0,
        ],
        axis=-1,
    )
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
    return _title(img, title)


def _bar(values: dict[str, float], size: tuple[int, int], title: str) -> Image.Image:
    img = Image.new("RGB", size, (250, 250, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, size[0], 20), fill=(0, 0, 0))
    draw.text((4, 4), title[:80], fill=(255, 255, 255), font=font)
    y = 36
    colors = {
        "B0_high": (38, 120, 78),
        "B1_mixed": (180, 125, 40),
        "B2_zero": (160, 42, 42),
        "B3_high_res_low_conf": (86, 82, 170),
    }
    for key in ["B0_high", "B1_mixed", "B2_zero", "B3_high_res_low_conf"]:
        value = float(values.get(key, 0.0) or 0.0)
        width = int((size[0] - 130) * max(0.0, min(1.0, value)))
        draw.text((8, y), key, fill=(20, 20, 20), font=font)
        draw.rectangle((128, y, 128 + width, y + 12), fill=colors.get(key, (80, 80, 80)))
        draw.text((136 + width, y), f"{value:.4f}", fill=(20, 20, 20), font=font)
        y += 24
    return img


def _route_file(route_root: Path, seq: str, chunk: int, case: str) -> Path | None:
    pattern = f"seq{seq}_*/chunk{chunk:02d}/{case}/swa_overlap_feature_maps/*.pt"
    matches = sorted(route_root.glob(pattern))
    return matches[0] if matches else None


def _load_route(path: Path | None) -> tuple[dict[str, Any], dict[str, float]]:
    if path is None or not path.is_file():
        return {}, {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}, {}
    score = payload.get("score_overlap")
    control = payload.get("control_overlap")
    dq = payload.get("Dq_overlap")
    ds = payload.get("Ds_overlap")
    stats: dict[str, float] = {}
    for name, tensor in {
        "score": score,
        "control": control,
        "Dq": dq,
        "Ds": ds,
    }.items():
        if torch.is_tensor(tensor):
            x = tensor.detach().cpu().float()
            stats[f"{name}_mean"] = float(x.mean().item())
            stats[f"{name}_q90"] = float(torch.quantile(x.flatten(), 0.90).item())
            stats[f"{name}_nonzero_ratio"] = float((x > 0).float().mean().item())
    meta = {
        "path": str(path),
        "schema": payload.get("schema", ""),
        "artifact": payload.get("artifact", ""),
        "kind": payload.get("kind", ""),
        "mode": payload.get("mode", ""),
        "chunk_idx": payload.get("chunk_idx", ""),
        "swa_layer_idx": payload.get("swa_layer_idx", ""),
        "tokens_per_frame": payload.get("tokens_per_frame", ""),
        "overlap_frames_effective": payload.get("overlap_frames_effective", ""),
        "runtime_swa_overlap_feature_not_qk_proxy": bool(
            payload.get("runtime_swa_overlap_feature_not_qk_proxy", False)
        ),
        "score_overlap": score.detach().cpu().float().numpy() if torch.is_tensor(score) else np.zeros((3, 1260)),
        "control_overlap": control.detach().cpu().float().numpy() if torch.is_tensor(control) else np.zeros((3, 1260)),
    }
    return meta, stats


def _random_same_values(arr: np.ndarray, seed_text: str) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return np.zeros((3, 1260), dtype=np.float32)
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return flat[rng.permutation(flat.size)].reshape(np.asarray(arr).shape)


def _qkv_map(path: Path, tap: str, size: tuple[int, int], title: str) -> tuple[Image.Image, bool]:
    if not path.is_file():
        return _text([f"missing {path}"], size, title), False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload.get(f"tap::{tap}") if isinstance(payload, dict) else None
    if not torch.is_tensor(tensor):
        return _text([f"missing tap {tap}", str(path)], size, title), False
    x = tensor.detach().cpu().float()
    if x.ndim >= 1:
        x = x[0]
    while x.ndim > 3:
        x = x[0]
    if x.ndim == 3:
        x = torch.linalg.norm(x, dim=-1)
    return _heat(x.numpy(), size, title), True


def _route_stats_row(prefix: str, stats: dict[str, float]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def _make_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    seq = str(row["seq"]).zfill(2)
    prev_chunk = int(row["prev_chunk"])
    curr_chunk = int(row["curr_chunk"])
    tile = (380, 190)

    replace_file = _route_file(args.route_root, seq, curr_chunk, ROUTE_CASES["source_replace"])
    gate_file = _route_file(args.route_root, seq, curr_chunk, ROUTE_CASES["source_gate"])
    replace_meta, replace_stats = _load_route(replace_file)
    gate_meta, gate_stats = _load_route(gate_file)
    replace_score = np.asarray(replace_meta.get("score_overlap", np.zeros((3, 1260))), dtype=np.float32)
    gate_score = np.asarray(gate_meta.get("score_overlap", np.zeros((3, 1260))), dtype=np.float32)
    replace_random = _random_same_values(replace_score, f"{seq}-{curr_chunk}-replace")
    actual_random_l1 = float(np.mean(np.abs(replace_score - replace_random))) if replace_score.size else 0.0

    qkv_dir = args.qkv_root / f"seq{seq}_native_prefix641" / "v68_layer_pca_features"
    prev_qkv = qkv_dir / f"chunk_{prev_chunk:03d}.pt"
    curr_qkv = qkv_dir / f"chunk_{curr_chunk:03d}.pt"

    q_img, q_ok = _qkv_map(curr_qkv, "global_q_raw_patchvec_layers", tile, "Current Q norm")
    k_img, k_ok = _qkv_map(prev_qkv, "global_k_raw_patchvec_layers", tile, "Cache K norm")
    v_img, v_ok = _qkv_map(prev_qkv, "global_v_raw_patchvec_layers", tile, "Cache V norm")

    true_route_panel = Image.new("RGB", (tile[0] * 2, tile[1] * 2), (255, 255, 255))
    for img, xy in [
        (_heat(replace_score, tile, "P9_40 runtime source-replace score"), (0, 0)),
        (_heat(gate_score, tile, "P9_6 runtime source-gate score"), (tile[0], 0)),
        (_heat(np.asarray(replace_meta.get("control_overlap", np.zeros((3, 1260)))), tile, "P9_40 control route feature"), (0, tile[1])),
        (
            _text(
                [
                    f"seq={seq} pair={prev_chunk}->{curr_chunk}",
                    f"case={row.get('case_type')} base={row.get('base_case_type')}",
                    f"replace_file={bool(replace_file)} gate_file={bool(gate_file)}",
                    f"runtime_not_qk_proxy={replace_meta.get('runtime_swa_overlap_feature_not_qk_proxy')} / {gate_meta.get('runtime_swa_overlap_feature_not_qk_proxy')}",
                    f"replace_score_mean={replace_stats.get('score_mean')} gate_score_mean={gate_stats.get('score_mean')}",
                    f"stable={row.get('stable_overlap_mass')} harm={row.get('harm_overlap_mass')}",
                ],
                tile,
                "Runtime route provenance",
            ),
            (tile[0], tile[1]),
        ),
    ]:
        true_route_panel.paste(img, xy)

    qkv_panel = Image.new("RGB", (tile[0] * 3, tile[1]), (255, 255, 255))
    for img, xy in [(q_img, (0, 0)), (k_img, (tile[0], 0)), (v_img, (tile[0] * 2, 0))]:
        qkv_panel.paste(img, xy)

    actual_random_panel = Image.new("RGB", (tile[0] * 3, tile[1]), (255, 255, 255))
    for img, xy in [
        (_heat(replace_score, tile, "Actual runtime route score"), (0, 0)),
        (_heat(replace_random, tile, "Deterministic same-values random"), (tile[0], 0)),
        (_heat(np.abs(replace_score - replace_random), tile, f"Abs diff mean={actual_random_l1:.6f}"), (tile[0] * 2, 0)),
    ]:
        actual_random_panel.paste(img, xy)

    bin_values = {
        "B0_high": _float(row.get("high_conf_pair_count")) or 0.0,
        "B1_mixed": _float(row.get("mixed_conf_pair_count")) or 0.0,
        "B2_zero": _float(row.get("zero_conf_pair_count")) or 0.0,
        "B3_high_res_low_conf": _float(row.get("high_res_low_conf_pair_count")) or 0.0,
    }
    total = sum(bin_values.values())
    bin_ratios = {key: (value / total if total > 0 else 0.0) for key, value in bin_values.items()}
    confidence_panel = Image.new("RGB", (tile[0] * 2, tile[1]), (255, 255, 255))
    confidence_panel.paste(_bar(bin_ratios, tile, "Confidence-bin support ratios"), (0, 0))
    confidence_panel.paste(
        _text(
            [
                f"quality={row.get('quality_type')} source={row.get('quality_source')}",
                f"saved high/mixed/zero/B3={int(bin_values['B0_high'])}/{int(bin_values['B1_mixed'])}/{int(bin_values['B2_zero'])}/{int(bin_values['B3_high_res_low_conf'])}",
                f"either_zero={row.get('either_zero_ratio')} both_zero={row.get('both_zero_ratio')}",
                f"semantic_conf={row.get('semantic_confidence_mean')}",
                f"artifact_risk={row.get('artifact_quality_risk')}",
            ],
            tile,
            "Pair quality context",
        ),
        (tile[0], 0),
    )

    stem = f"seq{seq}_chunk{prev_chunk:03d}_{curr_chunk:03d}_{row.get('case_type')}"
    out_paths: dict[str, Path] = {}
    for dirname, image in [
        ("true_route_panels", true_route_panel),
        ("qkv_head_layer_panels", qkv_panel),
        ("actual_vs_random_panels", actual_random_panel),
        ("confidence_bin_panels", confidence_panel),
    ]:
        target = args.out_dir / dirname / f"{stem}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
        out_paths[dirname] = target

    has_true_route = bool(
        replace_meta.get("runtime_swa_overlap_feature_not_qk_proxy")
        and gate_meta.get("runtime_swa_overlap_feature_not_qk_proxy")
    )
    has_qkv = bool(q_ok and k_ok and v_ok)
    missing_overlay = not (has_true_route and has_qkv and all(path.is_file() for path in out_paths.values()))
    manifest = {
        "seq": seq,
        "prev_chunk": prev_chunk,
        "curr_chunk": curr_chunk,
        "case_type": row.get("case_type", ""),
        "base_case_type": row.get("base_case_type", ""),
        "quality_type": row.get("quality_type", ""),
        "quality_source": row.get("quality_source", ""),
        "visual_file": str(out_paths["true_route_panels"]),
        "true_route_panel": str(out_paths["true_route_panels"]),
        "qkv_head_layer_panel": str(out_paths["qkv_head_layer_panels"]),
        "actual_vs_random_panel": str(out_paths["actual_vs_random_panels"]),
        "confidence_bin_panel": str(out_paths["confidence_bin_panels"]),
        "true_route_panel_sha256": _sha256(out_paths["true_route_panels"]),
        "qkv_panel_sha256": _sha256(out_paths["qkv_head_layer_panels"]),
        "actual_vs_random_panel_sha256": _sha256(out_paths["actual_vs_random_panels"]),
        "confidence_bin_panel_sha256": _sha256(out_paths["confidence_bin_panels"]),
        "has_actual_route_mask": has_true_route,
        "has_source_replace_route": bool(replace_file),
        "has_source_gate_route": bool(gate_file),
        "source_replace_route_file": str(replace_file or ""),
        "source_gate_route_file": str(gate_file or ""),
        "source_replace_route_schema": replace_meta.get("schema", ""),
        "source_gate_route_schema": gate_meta.get("schema", ""),
        "source_replace_route_kind": replace_meta.get("kind", ""),
        "source_gate_route_kind": gate_meta.get("kind", ""),
        "source_replace_runtime_not_qk_proxy": bool(replace_meta.get("runtime_swa_overlap_feature_not_qk_proxy")),
        "source_gate_runtime_not_qk_proxy": bool(gate_meta.get("runtime_swa_overlap_feature_not_qk_proxy")),
        "source_replace_swa_layer_idx": replace_meta.get("swa_layer_idx", ""),
        "source_gate_swa_layer_idx": gate_meta.get("swa_layer_idx", ""),
        "actual_vs_random_l1": actual_random_l1,
        "actual_vs_random_difference_reviewed": actual_random_l1 > 0.0,
        "has_qkv_maps": has_qkv,
        "qkv_prev_file": str(prev_qkv),
        "qkv_curr_file": str(curr_qkv),
        "missing_overlay": missing_overlay,
        "high_conf_pair_count": row.get("high_conf_pair_count", ""),
        "mixed_conf_pair_count": row.get("mixed_conf_pair_count", ""),
        "zero_conf_pair_count": row.get("zero_conf_pair_count", ""),
        "high_res_low_conf_pair_count": row.get("high_res_low_conf_pair_count", ""),
        "either_zero_ratio": row.get("either_zero_ratio", ""),
        "semantic_confidence_mean": row.get("semantic_confidence_mean", ""),
        "stable_overlap_mass": row.get("stable_overlap_mass", ""),
        "harm_overlap_mass": row.get("harm_overlap_mass", ""),
        "context_overlap_mass": row.get("context_overlap_mass", ""),
        "source_path": row.get("source_path", ""),
        "artifact_quality_risk": row.get("artifact_quality_risk", ""),
    }
    manifest.update(_route_stats_row("source_replace", replace_stats))
    manifest.update(_route_stats_row("source_gate", gate_stats))
    # Carry the row's machine-readable bin JSON forward if future audits need it.
    for key, value in _json_map(row.get("stable_mass_by_bin", "")).items():
        manifest[f"stable_mass_{key}"] = value
    for key, value in _json_map(row.get("harm_mass_by_bin", "")).items():
        manifest[f"harm_mass_{key}"] = value
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--route-root", type=Path, default=DEFAULT_ROUTE_ROOT)
    parser.add_argument("--qkv-root", type=Path, default=DEFAULT_QKV_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = _read_csv(args.bank)
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    manifest = [_make_row(row, args) for row in rows]
    _write_csv(args.out_dir / "visual_manifest.csv", manifest)
    summary = {
        "rows": len(manifest),
        "out_dir": str(args.out_dir),
        "true_route_rows": sum(1 for row in manifest if row.get("has_actual_route_mask")),
        "qkv_rows": sum(1 for row in manifest if row.get("has_qkv_maps")),
        "missing_overlay_count": sum(1 for row in manifest if row.get("missing_overlay")),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
