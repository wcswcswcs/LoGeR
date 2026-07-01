#!/usr/bin/env python3
"""Extract v85 anchor-row SWA Q/cache-K PCA features.

This reads Phase1 anchor rows and direct PCA SWA Q/cache-K dump tensors. It
does not fit latent alignment C and does not claim route attention mass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


DEFAULT_ANCHOR_ROWS = Path(
    "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe/anchor_pair_rows.csv"
)
DEFAULT_OUT_DIR = Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase2_qk_feature_bank")
PATCH_GRID = (19, 66)
Q_KEY = "tap::pca_swa_current_q_layers"
K_KEY = "tap::pca_swa_cache_k_layers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-rows", type=Path, default=DEFAULT_ANCHOR_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all anchor rows")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def safe_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def patch_yx_from_id(value: Any) -> tuple[int | None, int | None]:
    pid = safe_int(value)
    if pid is None or pid < 0:
        return None, None
    return pid // PATCH_GRID[1], pid % PATCH_GRID[1]


class FeatureCache:
    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}
        self.load_errors: dict[str, str] = {}

    def get(self, path: str) -> dict[str, Any] | None:
        if not path:
            return None
        if path not in self.cache:
            try:
                payload = torch_load(Path(path))
            except Exception as exc:  # noqa: BLE001
                self.cache[path] = None
                self.load_errors[path] = f"{type(exc).__name__}: {exc}"
            else:
                self.cache[path] = payload if isinstance(payload, dict) else None
        payload = self.cache[path]
        if not isinstance(payload, dict) or Q_KEY not in payload or K_KEY not in payload:
            return None
        return payload


def extract_rows(anchor_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, dict[str, Any]]:
    cache = FeatureCache()
    index_rows: list[dict[str, Any]] = []
    q_features: list[torch.Tensor] = []
    k_features: list[torch.Tensor] = []
    missing_reasons: dict[str, int] = {}

    for row_idx, row in enumerate(anchor_rows):
        if not (parse_bool(row.get("feature_q_available")) and parse_bool(row.get("feature_k_available"))):
            missing_reasons["phase1_feature_flag_false"] = missing_reasons.get("phase1_feature_flag_false", 0) + 1
            continue
        source_path = str(row.get("feature_source_path") or "")
        payload = cache.get(source_path)
        if payload is None:
            missing_reasons["feature_payload_missing_or_schema_bad"] = (
                missing_reasons.get("feature_payload_missing_or_schema_bad", 0) + 1
            )
            continue
        q = payload[Q_KEY].detach().cpu().float()
        k = payload[K_KEY].detach().cpu().float()
        start_frame = safe_int(payload.get("start_frame")) or 0
        curr_frame = safe_int(row.get("curr_frame_id"))
        if curr_frame is None:
            missing_reasons["curr_frame_missing"] = missing_reasons.get("curr_frame_missing", 0) + 1
            continue
        local_frame = curr_frame - start_frame
        py, px = patch_yx_from_id(row.get("curr_patch_id"))
        if py is None or px is None:
            missing_reasons["curr_patch_missing"] = missing_reasons.get("curr_patch_missing", 0) + 1
            continue
        if local_frame < 0 or local_frame >= q.shape[0] or py >= q.shape[2] or px >= q.shape[3]:
            missing_reasons["frame_or_patch_out_of_bounds"] = missing_reasons.get("frame_or_patch_out_of_bounds", 0) + 1
            continue
        layer_ids_q = payload.get(f"layer_ids::{Q_KEY}", torch.arange(q.shape[1]))
        layer_ids_k = payload.get(f"layer_ids::{K_KEY}", torch.arange(k.shape[1]))
        layer_count = min(q.shape[1], k.shape[1])
        for layer_pos in range(layer_count):
            q_vec = q[local_frame, layer_pos, py, px].clone()
            k_vec = k[local_frame, layer_pos, py, px].clone()
            q_features.append(q_vec)
            k_features.append(k_vec)
            layer_id = int(layer_ids_q[layer_pos].item()) if torch.is_tensor(layer_ids_q) else int(layer_pos)
            k_layer_id = int(layer_ids_k[layer_pos].item()) if torch.is_tensor(layer_ids_k) else int(layer_pos)
            index_rows.append(
                {
                    "seq": row.get("seq"),
                    "prev_chunk": row.get("prev_chunk"),
                    "curr_chunk": row.get("curr_chunk"),
                    "pair_id": row.get("pair_id"),
                    "anchor_row_index": row_idx,
                    "case_label": row.get("case_label"),
                    "quality_label": row.get("quality_label"),
                    "anchor_support_class": row.get("anchor_support_class"),
                    "layer_id": layer_id,
                    "k_layer_id": k_layer_id,
                    "head_id": "pooled",
                    "feature_dim_raw": int(q_vec.numel()),
                    "feature_dim_projected": int(q_vec.numel()),
                    "projection_method": "preexisting_pca_swa_dump",
                    "local_frame": local_frame,
                    "curr_patch_id": row.get("curr_patch_id"),
                    "q_norm": float(torch.linalg.norm(q_vec).item()),
                    "k_norm": float(torch.linalg.norm(k_vec).item()),
                    "feature_source_path": source_path,
                    "feature_schema": payload.get("schema", ""),
                    "authority_source": "direct_pca_swa_q_cache_k_dump",
                    "route_mass_available": False,
                }
            )

    if q_features:
        q_tensor = torch.stack(q_features, dim=0)
        k_tensor = torch.stack(k_features, dim=0)
    else:
        q_tensor = torch.empty(0, 0)
        k_tensor = torch.empty(0, 0)
    meta = {
        "anchor_row_count": len(anchor_rows),
        "feature_entry_count": len(index_rows),
        "missing_reasons": missing_reasons,
        "feature_load_error_count": len(cache.load_errors),
        "feature_load_errors": cache.load_errors,
        "note": "Direct PCA SWA Q/cache-K features only; no route mass and no latent C fitting.",
    }
    return index_rows, q_tensor, k_tensor, meta


def write_missing_report(path: Path, meta: Mapping[str, Any]) -> None:
    lines = [
        "# Phase2 Missing Feature Report",
        "",
        f"- anchor rows scanned: {meta['anchor_row_count']}",
        f"- feature entries extracted: {meta['feature_entry_count']}",
        f"- feature load error count: {meta['feature_load_error_count']}",
        "- authority: direct PCA SWA Q/cache-K dump",
        "- true route mass: unavailable in this artifact",
        "",
        "## Missing Reasons",
        "",
    ]
    for key, value in sorted(meta.get("missing_reasons", {}).items()):
        lines.append(f"- {key}: {value}")
    if not meta.get("missing_reasons"):
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    anchor_rows = read_csv(args.anchor_rows)
    if args.max_rows > 0:
        anchor_rows = anchor_rows[: args.max_rows]
    index_rows, q_tensor, k_tensor, meta = extract_rows(anchor_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "acl2_v85_phase2_qk_anchor_features_v1",
            "q_features": q_tensor,
            "k_features": k_tensor,
            "meta": meta,
        },
        args.out_dir / "qk_anchor_features.pt",
    )
    write_csv(args.out_dir / "qk_anchor_feature_index.csv", index_rows)
    write_json(args.out_dir / "extract_summary.json", meta)
    write_missing_report(args.out_dir / "missing_feature_report.md", meta)
    print(f"anchor_row_count={meta['anchor_row_count']}")
    print(f"feature_entry_count={meta['feature_entry_count']}")
    print(f"q_shape={tuple(q_tensor.shape)}")
    print(f"k_shape={tuple(k_tensor.shape)}")
    print(f"missing_reasons={meta['missing_reasons']}")


if __name__ == "__main__":
    main()
