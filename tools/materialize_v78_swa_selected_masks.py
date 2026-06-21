#!/usr/bin/env python3
"""Materialize v78 SWA selected masks and same-mass random controls.

This is diagnostic-only.  It reconstructs the top-q selected source mask from
the existing SWA overlap feature dumps and stores a deterministic same-mass
random mask beside it.  The saved masks make the source-quality audit
reproducible without claiming direct Q/K/V alignment.
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

from audit_v78_swa_action_conditioned_signal import SUITES


DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase9_swa_cache_value_carryover/selected_mask_materialization_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--base-seed", type=int, default=7801)
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _stable_seed(name: str, base_seed: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return base_seed + (int(digest[:8], 16) % 1_000_000)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _score_stats(score: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    values = score[mask]
    return {
        "count": int(mask.sum().item()),
        "fraction": float(mask.float().mean().item()),
        "score_mean": float(values.mean().item()) if values.numel() else None,
        "score_min": float(values.min().item()) if values.numel() else None,
        "score_max": float(values.max().item()) if values.numel() else None,
    }


def _per_frame_rows(
    suite: dict[str, Any],
    score: torch.Tensor,
    selected_mask: torch.Tensor,
    random_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    if score.ndim != 3:
        return []
    rows: list[dict[str, Any]] = []
    for frame_idx in range(int(score.shape[1])):
        score_frame = score[:, frame_idx, :]
        selected_frame = selected_mask[:, frame_idx, :]
        random_frame = random_mask[:, frame_idx, :]
        selected_stats = _score_stats(score_frame, selected_frame)
        random_stats = _score_stats(score_frame, random_frame)
        rows.append(
            {
                "suite": suite["suite"],
                "sequence": suite["sequence"],
                "chunk": suite["chunk"],
                "action": suite["action"],
                "window_key": suite["window_key"],
                "overlap_frame_index": frame_idx,
                "score_mean": float(score_frame.mean().item()),
                "selected_count": selected_stats["count"],
                "selected_fraction": selected_stats["fraction"],
                "selected_score_mean": selected_stats["score_mean"],
                "random_same_mass_count": random_stats["count"],
                "random_same_mass_fraction": random_stats["fraction"],
                "random_same_mass_score_mean": random_stats["score_mean"],
                "selected_minus_random_same_mass_score_mean": (
                    selected_stats["score_mean"] - random_stats["score_mean"]
                    if selected_stats["score_mean"] is not None
                    and random_stats["score_mean"] is not None
                    else None
                ),
            }
        )
    return rows


def _materialize_suite(
    suite: dict[str, Any],
    out_dir: Path,
    quantile: float,
    base_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_dump = Path(suite["feature_dump"])
    obj = torch.load(feature_dump, map_location="cpu")
    score = obj.get("score_overlap")
    if not torch.is_tensor(score) or score.numel() == 0:
        raise ValueError(f"missing score_overlap: {feature_dump}")

    score_f = score.detach().cpu().float()
    flat = score_f.reshape(-1)
    threshold = torch.quantile(flat, quantile)
    selected_mask = score_f >= threshold
    selected_count = int(selected_mask.sum().item())

    seed = _stable_seed(str(suite["suite"]), base_seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_flat = torch.zeros(flat.numel(), dtype=torch.bool)
    if selected_count:
        random_flat[torch.randperm(flat.numel(), generator=generator)[:selected_count]] = True
    random_mask = random_flat.reshape_as(score_f)

    selected_stats = _score_stats(score_f, selected_mask)
    random_stats = _score_stats(score_f, random_mask)
    all_mean = float(score_f.mean().item())
    safe_suite = _safe_name(str(suite["suite"]))
    mask_path = out_dir / "masks" / f"{safe_suite}_selected_random_masks.pt"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "acl2_v78_swa_selected_mask_materialization_v1",
            "diagnostic_only": True,
            "method_gate_claimed": False,
            "suite": suite["suite"],
            "sequence": suite["sequence"],
            "chunk": suite["chunk"],
            "action": suite["action"],
            "action_label": suite["action_label"],
            "window_key": suite["window_key"],
            "feature_dump": str(feature_dump),
            "feature_schema": obj.get("schema"),
            "feature_runtime_not_qk_proxy": bool(
                obj.get("runtime_swa_overlap_feature_not_qk_proxy", False)
            ),
            "quantile": quantile,
            "threshold": float(threshold.item()),
            "same_mass_random_seed": seed,
            "score_overlap": score.detach().cpu(),
            "selected_mask_topq": selected_mask.cpu(),
            "random_same_mass_mask": random_mask.cpu(),
        },
        mask_path,
    )

    row = {
        "suite": suite["suite"],
        "sequence": suite["sequence"],
        "chunk": suite["chunk"],
        "action": suite["action"],
        "action_label": suite["action_label"],
        "window_key": suite["window_key"],
        "feature_dump": str(feature_dump),
        "mask_artifact": str(mask_path),
        "feature_schema": obj.get("schema"),
        "feature_runtime_not_qk_proxy": bool(
            obj.get("runtime_swa_overlap_feature_not_qk_proxy", False)
        ),
        "quantile": quantile,
        "threshold": float(threshold.item()),
        "score_mean": all_mean,
        "selected_count": selected_stats["count"],
        "selected_fraction": selected_stats["fraction"],
        "selected_score_mean": selected_stats["score_mean"],
        "selected_quality_lift_vs_all_mean": (
            selected_stats["score_mean"] - all_mean
            if selected_stats["score_mean"] is not None
            else None
        ),
        "random_same_mass_seed": seed,
        "random_same_mass_count": random_stats["count"],
        "random_same_mass_fraction": random_stats["fraction"],
        "random_same_mass_score_mean": random_stats["score_mean"],
        "random_same_mass_lift_vs_all_mean": (
            random_stats["score_mean"] - all_mean
            if random_stats["score_mean"] is not None
            else None
        ),
        "selected_minus_random_same_mass_score_mean": (
            selected_stats["score_mean"] - random_stats["score_mean"]
            if selected_stats["score_mean"] is not None
            and random_stats["score_mean"] is not None
            else None
        ),
    }
    return row, _per_frame_rows(suite, score_f, selected_mask, random_mask)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    for suite in SUITES:
        row, frame_rows = _materialize_suite(
            suite, args.out_dir, args.quantile, args.base_seed
        )
        rows.append(row)
        per_frame_rows.extend(frame_rows)

    rows_csv = args.out_dir / "selected_mask_materialization_rows.csv"
    frames_csv = args.out_dir / "selected_mask_materialization_per_frame.csv"
    summary_json = args.out_dir / "selected_mask_materialization_summary.json"
    _write_csv(rows_csv, rows)
    _write_csv(frames_csv, per_frame_rows)
    _write_json(
        summary_json,
        {
            "schema": "acl2_v78_swa_selected_mask_materialization_v1",
            "diagnostic_only": True,
            "method_gate_claimed": False,
            "quantile": args.quantile,
            "base_seed": args.base_seed,
            "num_suites": len(rows),
            "rows_csv": str(rows_csv),
            "per_frame_csv": str(frames_csv),
            "rows": rows,
            "key_findings": {
                "selected_minus_random_same_mass_score_mean_by_suite": {
                    str(row["suite"]): row["selected_minus_random_same_mass_score_mean"]
                    for row in rows
                },
                "selected_quality_lift_vs_all_mean_by_suite": {
                    str(row["suite"]): row["selected_quality_lift_vs_all_mean"]
                    for row in rows
                },
                "random_same_mass_lift_vs_all_mean_by_suite": {
                    str(row["suite"]): row["random_same_mass_lift_vs_all_mean"]
                    for row in rows
                },
            },
            "limitations": [
                "Masks are reconstructed from score_overlap feature dumps, not direct Q/K/V tensors.",
                "Same-mass random masks are deterministic audit controls, not necessarily the exact runtime random masks.",
                "Repeated action suites on the same geometry window can share identical score_overlap maps.",
            ],
        },
    )
    print(json.dumps(_jsonable({"rows": rows_csv, "per_frame": frames_csv, "summary": summary_json}), indent=2))


if __name__ == "__main__":
    main()
