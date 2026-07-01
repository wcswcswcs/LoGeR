#!/usr/bin/env python3
"""Audit v82 confidence-stratified overlap-pair quality.

This Phase1 tool reads existing v81S default and min_conf=0.0 overlap-pair
artifacts. It stratifies confidence bins and writes manifests for downstream
SWA carrier discovery. It does not modify model state or run actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import torch


DEFAULT_DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS1_multiseq_swa_overlap_repair"
)
DEFAULT_MINCONF0_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS1_multiseq_swa_overlap_repair_minconf0"
)
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")
DEFAULT_OUT_DIR = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase1_overlap_quality_stratification"
)

HIGH_CONF_MIN = 0.05
SEQS = ("00", "01", "02", "05")

STABLE_WORDS = (
    "building",
    "house",
    "wall",
    "fence",
    "handrail_or_fence",
    "pole",
    "traffic sign",
    "traffic light",
    "bridge",
    "construction",
    "billboard",
    "pillar",
    "stair",
)
DYNAMIC_WORDS = ("car", "person", "rider", "bicycle", "motorcycle", "bus", "truck", "train", "dog")
LOWTRUST_WORDS = ("tree", "grass", "vegetation", "mountain", "terrain", "void", "unknown", "plant")
CONTEXT_WORDS = ("sky", "road", "ground", "sidewalk", "path", "crosswalk")

PAIR_FIELDS = [
    "seq",
    "prev_chunk",
    "curr_chunk",
    "source",
    "quality_type",
    "source_path",
    "saved_pairs",
    "valid_pairs",
    "semantic_projection_ratio",
    "semantic_nonvoid_ratio",
    "semantic_confidence_mean",
    "both_conf_nonzero_ratio",
    "either_zero_ratio",
    "both_zero_ratio",
    "high_conf_pair_count",
    "mixed_conf_pair_count",
    "zero_conf_pair_count",
    "high_res_low_conf_pair_count",
    "raw_residual_rmse",
    "raw_residual_mean",
    "confidence_weighted_residual",
    "overlap_scale_residual",
    "proxy_scale_residual",
    "residual_by_conf_bin",
    "semantic_role_by_conf_bin",
    "stable_mass_by_bin",
    "harm_mass_by_bin",
    "context_mass_by_bin",
    "seq01_sparse_support_flag",
    "high_quality_usable",
    "low_conf_stress_usable",
    "forbidden_as_stable_evidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-root", type=Path, default=DEFAULT_DEFAULT_ROOT)
    parser.add_argument("--minconf0-root", type=Path, default=DEFAULT_MINCONF0_ROOT)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seqs", default=",".join(SEQS))
    return parser.parse_args()


def parse_seqs(text: str) -> list[str]:
    return [part.strip().zfill(2) for part in str(text).split(",") if part.strip()]


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(jsonable(row.get(key)), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key, "")
                    for key in fields
                }
            )


def load_label_names(preprocess_root: Path, seq: str) -> list[str]:
    path = preprocess_root / seq / "sparse_masklets_with_semantic.pt"
    if not path.is_file():
        return []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sem = payload.get("semantic_segmentation", payload) if isinstance(payload, dict) else {}
    names = sem.get("label_names", []) if isinstance(sem, dict) else []
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(name) for name in names]


def label_family(name: str) -> str:
    lowered = str(name).lower()
    if any(word in lowered for word in DYNAMIC_WORDS):
        return "harm"
    if any(word in lowered for word in LOWTRUST_WORDS):
        return "context"
    if any(word in lowered for word in CONTEXT_WORDS):
        return "context"
    if any(word in lowered for word in STABLE_WORDS):
        return "stable"
    return "context"


def robust_scale(points: torch.Tensor) -> float | None:
    if not torch.is_tensor(points) or points.ndim != 2 or points.shape[0] < 2:
        return None
    pts = points.float()
    centered = pts - pts.mean(dim=0, keepdim=True)
    dist = torch.linalg.norm(centered, dim=1)
    value = float(torch.median(dist).item())
    return value if math.isfinite(value) and value > 1e-12 else None


def mean_or_none(values: torch.Tensor) -> float | None:
    if not torch.is_tensor(values) or values.numel() == 0:
        return None
    out = float(values.float().mean().item())
    return out if math.isfinite(out) else None


def count_mask(mask: torch.Tensor) -> int:
    return int(mask.sum().item()) if torch.is_tensor(mask) else 0


def role_masses(labels: torch.Tensor, mask: torch.Tensor, label_names: list[str]) -> dict[str, float]:
    total = count_mask(mask)
    if total <= 0 or not torch.is_tensor(labels):
        return {"stable": 0.0, "harm": 0.0, "context": 0.0}
    selected = labels.long().reshape(-1)[mask.reshape(-1)]
    counts = {"stable": 0, "harm": 0, "context": 0}
    if selected.numel() == 0:
        return {"stable": 0.0, "harm": 0.0, "context": 0.0}
    for label_id, count in zip(*torch.unique(selected, return_counts=True)):
        idx = int(label_id.item())
        name = label_names[idx] if 0 <= idx < len(label_names) else "unknown"
        counts[label_family(name)] += int(count.item())
    return {key: float(value) / float(total) for key, value in counts.items()}


def residual_stats_by_bin(residual: torch.Tensor, masks: Mapping[str, torch.Tensor]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, mask in masks.items():
        if count_mask(mask) <= 0:
            out[name] = None
            continue
        values = residual[mask]
        out[name] = float(torch.sqrt((values * values).mean()).item()) if values.numel() else None
    return out


def confidence_weighted_rmse(residual: torch.Tensor, weights: torch.Tensor) -> float | None:
    if residual.numel() == 0:
        return None
    weights = weights.float().clamp_min(0.0)
    denom = float(weights.sum().item())
    if denom <= 1e-12:
        return float(torch.sqrt((residual * residual).mean()).item())
    return float(torch.sqrt(((residual * residual) * weights).sum() / weights.sum()).item())


def classify_quality(row: Mapping[str, Any]) -> tuple[str, bool, bool, bool]:
    saved = int(row.get("saved_pairs") or 0)
    high_count = int(row.get("high_conf_pair_count") or 0)
    projection = finite_float(row.get("semantic_projection_ratio")) or 0.0
    both_nonzero = finite_float(row.get("both_conf_nonzero_ratio")) or 0.0
    either_zero = finite_float(row.get("either_zero_ratio")) or 0.0
    raw = finite_float(row.get("raw_residual_rmse"))
    scale = finite_float(row.get("overlap_scale_residual"))
    high_quality = bool(high_count >= 10000 and projection >= 0.90 and both_nonzero >= 0.50 and scale is not None)
    low_conf_stress = bool(saved >= 10000 and projection >= 0.90 and either_zero > 0.30 and raw is not None)
    forbidden_stable = bool((int(row.get("zero_conf_pair_count") or 0) > 0) or either_zero > 0.30)
    if high_quality:
        quality = "high_quality"
    elif low_conf_stress:
        quality = "low_conf_stress"
    elif saved <= 0:
        quality = "missing_or_empty"
    else:
        quality = "insufficient_quality"
    return quality, high_quality, low_conf_stress, forbidden_stable


def audit_pair(path: Path, seq: str, source: str, label_names: list[str]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prev_conf = payload.get("prev_conf")
    curr_conf = payload.get("curr_conf")
    prev_points = payload.get("prev_overlap_points")
    curr_points = payload.get("curr_overlap_points")
    prev_local = payload.get("prev_overlap_local_points")
    curr_local = payload.get("curr_overlap_local_points")
    prev_sem_conf = payload.get("prev_semantic_conf")
    curr_sem_conf = payload.get("curr_semantic_conf")
    labels = payload.get("prev_semantic_labels")

    if not torch.is_tensor(prev_conf) or not torch.is_tensor(curr_conf):
        prev_conf = torch.empty(0)
        curr_conf = torch.empty(0)
    prev_conf = prev_conf.float().reshape(-1)
    curr_conf = curr_conf.float().reshape(-1)
    n = min(int(prev_conf.numel()), int(curr_conf.numel()))
    prev_conf = prev_conf[:n]
    curr_conf = curr_conf[:n]
    min_conf = torch.minimum(prev_conf, curr_conf) if n else torch.empty(0)

    high_mask = (prev_conf >= HIGH_CONF_MIN) & (curr_conf >= HIGH_CONF_MIN)
    mixed_mask = ((prev_conf < HIGH_CONF_MIN) ^ (curr_conf < HIGH_CONF_MIN)) if n else torch.empty(0, dtype=torch.bool)
    zero_mask = (prev_conf < HIGH_CONF_MIN) & (curr_conf < HIGH_CONF_MIN) if n else torch.empty(0, dtype=torch.bool)
    both_zero_mask = (prev_conf == 0) & (curr_conf == 0) if n else torch.empty(0, dtype=torch.bool)
    either_zero_mask = (prev_conf == 0) | (curr_conf == 0) if n else torch.empty(0, dtype=torch.bool)
    both_nonzero_mask = (prev_conf > 0) & (curr_conf > 0) if n else torch.empty(0, dtype=torch.bool)

    if torch.is_tensor(prev_points) and torch.is_tensor(curr_points) and prev_points.numel() and curr_points.numel():
        m = min(int(prev_points.shape[0]), int(curr_points.shape[0]), n)
        residual = torch.linalg.norm(prev_points[:m].float() - curr_points[:m].float(), dim=1)
        high_mask = high_mask[:m]
        mixed_mask = mixed_mask[:m]
        zero_mask = zero_mask[:m]
        min_conf = min_conf[:m]
        either_zero_mask = either_zero_mask[:m]
        both_zero_mask = both_zero_mask[:m]
        both_nonzero_mask = both_nonzero_mask[:m]
        n = m
    else:
        raw_rmse = finite_float(payload.get("raw_residual_rmse"))
        residual = torch.full((n,), float(raw_rmse or 0.0))

    low_conf_mask = ~high_mask if n else torch.empty(0, dtype=torch.bool)
    if residual.numel() and count_mask(low_conf_mask) > 0:
        threshold = float(torch.quantile(residual, 0.90).item())
        high_res_low_conf = low_conf_mask & (residual >= threshold)
    else:
        high_res_low_conf = torch.zeros_like(high_mask)

    scale_prev = robust_scale(prev_local if torch.is_tensor(prev_local) else prev_points)
    scale_curr = robust_scale(curr_local if torch.is_tensor(curr_local) else curr_points)
    if scale_prev and scale_curr:
        overlap_scale = abs(math.log(scale_curr / scale_prev))
    else:
        overlap_scale = finite_float(payload.get("raw_residual_rmse"))

    sem_conf = None
    if torch.is_tensor(prev_sem_conf) and torch.is_tensor(curr_sem_conf):
        m = min(int(prev_sem_conf.numel()), int(curr_sem_conf.numel()), n)
        if m:
            sem_conf = float(torch.minimum(prev_sem_conf[:m].float(), curr_sem_conf[:m].float()).mean().item())

    bin_masks = {
        "B0_high": high_mask,
        "B1_mixed": mixed_mask,
        "B2_zero": zero_mask,
        "B3_high_res_low_conf": high_res_low_conf,
    }
    role_by_bin = {name: role_masses(labels, mask, label_names) for name, mask in bin_masks.items()}
    stable_by_bin = {name: values["stable"] for name, values in role_by_bin.items()}
    harm_by_bin = {name: values["harm"] for name, values in role_by_bin.items()}
    context_by_bin = {name: values["context"] for name, values in role_by_bin.items()}

    raw_residual_rmse = finite_float(payload.get("raw_residual_rmse"))
    if raw_residual_rmse is None and residual.numel():
        raw_residual_rmse = float(torch.sqrt((residual * residual).mean()).item())

    row: dict[str, Any] = {
        "seq": seq,
        "prev_chunk": int(payload.get("prev_chunk", -1)),
        "curr_chunk": int(payload.get("curr_chunk", -1)),
        "source": source,
        "source_path": str(path),
        "saved_pairs": int(payload.get("saved_pair_count", n) or 0),
        "valid_pairs": int(payload.get("valid_pair_count", n) or 0),
        "semantic_projection_ratio": finite_float(payload.get("semantic_label_projected_ratio")),
        "semantic_nonvoid_ratio": finite_float(payload.get("semantic_nonvoid_ratio")),
        "semantic_confidence_mean": sem_conf,
        "both_conf_nonzero_ratio": float(both_nonzero_mask.float().mean().item()) if n else 0.0,
        "either_zero_ratio": float(either_zero_mask.float().mean().item()) if n else 0.0,
        "both_zero_ratio": float(both_zero_mask.float().mean().item()) if n else 0.0,
        "high_conf_pair_count": count_mask(high_mask),
        "mixed_conf_pair_count": count_mask(mixed_mask),
        "zero_conf_pair_count": count_mask(zero_mask),
        "high_res_low_conf_pair_count": count_mask(high_res_low_conf),
        "raw_residual_rmse": raw_residual_rmse,
        "raw_residual_mean": finite_float(payload.get("raw_residual_mean")) or mean_or_none(residual),
        "confidence_weighted_residual": confidence_weighted_rmse(residual, min_conf) if residual.numel() else None,
        "overlap_scale_residual": overlap_scale,
        "proxy_scale_residual": True,
        "residual_by_conf_bin": residual_stats_by_bin(residual, bin_masks),
        "semantic_role_by_conf_bin": role_by_bin,
        "stable_mass_by_bin": stable_by_bin,
        "harm_mass_by_bin": harm_by_bin,
        "context_mass_by_bin": context_by_bin,
        "seq01_sparse_support_flag": False,
    }
    row["seq01_sparse_support_flag"] = bool(seq == "01" and source == "default" and int(row["high_conf_pair_count"]) < 10000)
    quality, high_quality, low_conf_stress, forbidden_stable = classify_quality(row)
    row["quality_type"] = quality
    row["high_quality_usable"] = high_quality
    row["low_conf_stress_usable"] = low_conf_stress
    row["forbidden_as_stable_evidence"] = forbidden_stable
    return row


def collect_rows(default_root: Path, minconf0_root: Path, preprocess_root: Path, seqs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label_cache = {seq: load_label_names(preprocess_root, seq) for seq in seqs}
    for source, root in (("default", default_root), ("minconf0", minconf0_root)):
        for seq in seqs:
            pair_dir = root / "overlap_pairs" / seq
            if not pair_dir.is_dir():
                continue
            for path in sorted(pair_dir.glob("chunk_*.pt")):
                rows.append(audit_pair(path, seq, source, label_cache.get(seq, [])))
    return rows


def summarize_by_seq(rows: list[dict[str, Any]], seqs: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for seq in seqs:
        seq_rows = [row for row in rows if row["seq"] == seq]
        by_source: dict[str, Any] = {}
        for source in ("default", "minconf0"):
            src_rows = [row for row in seq_rows if row["source"] == source]
            high_quality_count = sum(1 for row in src_rows if row.get("high_quality_usable"))
            low_conf_count = sum(1 for row in src_rows if row.get("low_conf_stress_usable"))
            saved_values = [int(row.get("saved_pairs") or 0) for row in src_rows]
            high_values = [int(row.get("high_conf_pair_count") or 0) for row in src_rows]
            zero_values = [float(row.get("either_zero_ratio") or 0.0) for row in src_rows]
            median_high_conf = median(high_values) if high_values else None
            high_quality_row_ratio = high_quality_count / len(src_rows) if src_rows else 0.0
            source_high_quality_usable = bool(
                median_high_conf is not None
                and median_high_conf >= 10000
                and high_quality_row_ratio >= 0.50
            )
            by_source[source] = {
                "pair_rows": len(src_rows),
                "high_quality_pair_rows": high_quality_count,
                "high_quality_pair_row_ratio": high_quality_row_ratio,
                "low_conf_stress_pair_rows": low_conf_count,
                "median_saved_pairs": median(saved_values) if saved_values else None,
                "median_high_conf_pair_count": median_high_conf,
                "median_either_zero_ratio": median(zero_values) if zero_values else None,
                "usable_high_quality": source_high_quality_usable,
                "usable_low_conf_stress": low_conf_count > 0,
            }
        if by_source["default"]["usable_high_quality"]:
            status = "default_high_quality_usable"
        elif by_source["default"]["high_quality_pair_rows"] > 0 and by_source["minconf0"]["usable_low_conf_stress"]:
            status = "default_partial_high_quality_plus_low_conf_stress"
        elif by_source["minconf0"]["usable_low_conf_stress"]:
            status = "low_conf_stress_only"
        elif by_source["minconf0"]["usable_high_quality"]:
            status = "minconf0_high_quality_usable"
        else:
            status = "not_usable"
        summary[seq] = {"status": status, "sources": by_source}
    return summary


def confidence_bin_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["seq"]), str(row["source"]))].append(row)
    out: list[dict[str, Any]] = []
    for (seq, source), group in sorted(grouped.items()):
        saved = sum(int(row.get("saved_pairs") or 0) for row in group)
        high = sum(int(row.get("high_conf_pair_count") or 0) for row in group)
        mixed = sum(int(row.get("mixed_conf_pair_count") or 0) for row in group)
        zero = sum(int(row.get("zero_conf_pair_count") or 0) for row in group)
        high_res_low = sum(int(row.get("high_res_low_conf_pair_count") or 0) for row in group)
        out.append(
            {
                "seq": seq,
                "source": source,
                "pair_rows": len(group),
                "saved_pairs_total": saved,
                "B0_high_count": high,
                "B1_mixed_count": mixed,
                "B2_zero_count": zero,
                "B3_high_res_low_conf_count": high_res_low,
                "B0_high_ratio": high / saved if saved else 0.0,
                "B1_mixed_ratio": mixed / saved if saved else 0.0,
                "B2_zero_ratio": zero / saved if saved else 0.0,
                "B3_high_res_low_conf_ratio": high_res_low / saved if saved else 0.0,
            }
        )
    return out


def write_stratified_manifests(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifests: dict[str, list[dict[str, Any]]] = {
        "high_conf": [],
        "mixed_conf": [],
        "zero_conf": [],
        "high_residual_low_conf": [],
    }
    for row in rows:
        base = {key: row.get(key) for key in PAIR_FIELDS if key not in {"residual_by_conf_bin", "semantic_role_by_conf_bin", "stable_mass_by_bin", "harm_mass_by_bin", "context_mass_by_bin"}}
        if int(row.get("high_conf_pair_count") or 0) > 0:
            manifests["high_conf"].append(base | {"bin_count": row.get("high_conf_pair_count"), "bin_name": "B0_high"})
        if int(row.get("mixed_conf_pair_count") or 0) > 0:
            manifests["mixed_conf"].append(base | {"bin_count": row.get("mixed_conf_pair_count"), "bin_name": "B1_mixed"})
        if int(row.get("zero_conf_pair_count") or 0) > 0:
            manifests["zero_conf"].append(base | {"bin_count": row.get("zero_conf_pair_count"), "bin_name": "B2_zero"})
        if int(row.get("high_res_low_conf_pair_count") or 0) > 0:
            manifests["high_residual_low_conf"].append(
                base | {"bin_count": row.get("high_res_low_conf_pair_count"), "bin_name": "B3_high_res_low_conf"}
            )
    for name, manifest_rows in manifests.items():
        write_csv(out_dir / "stratified_overlap_sets" / name / "manifest.csv", manifest_rows)


def write_seq01_diagnosis(path: Path, summary: Mapping[str, Any], rows: list[dict[str, Any]]) -> None:
    seq01 = summary.get("01", {}) if isinstance(summary.get("01"), Mapping) else {}
    default = (seq01.get("sources") or {}).get("default", {})
    minconf0 = (seq01.get("sources") or {}).get("minconf0", {})
    seq01_default_rows = [row for row in rows if row["seq"] == "01" and row["source"] == "default"]
    seq01_minconf_rows = [row for row in rows if row["seq"] == "01" and row["source"] == "minconf0"]
    lines = [
        "# seq01 Sparse Support Diagnosis",
        "",
        f"- final_status: `{seq01.get('status')}`",
        f"- default_pair_rows: `{default.get('pair_rows')}`",
        f"- default_median_saved_pairs: `{default.get('median_saved_pairs')}`",
        f"- default_median_high_conf_pair_count: `{default.get('median_high_conf_pair_count')}`",
        f"- minconf0_pair_rows: `{minconf0.get('pair_rows')}`",
        f"- minconf0_median_saved_pairs: `{minconf0.get('median_saved_pairs')}`",
        f"- minconf0_median_high_conf_pair_count: `{minconf0.get('median_high_conf_pair_count')}`",
        f"- minconf0_median_either_zero_ratio: `{minconf0.get('median_either_zero_ratio')}`",
        "",
        "Interpretation:",
        "",
        "seq01 default overlap remains sparse under the high-confidence source. The min_conf=0.0 source restores pair count, but most late-overlap rows have zero or near-zero geometry confidence. Therefore v82 may use seq01 minconf0 rows as low-confidence stress / outlier evidence, but not as stable evidence.",
        "",
        "Default per-pair high-conf counts:",
        "",
    ]
    lines.extend(
        f"- chunk_{int(row['prev_chunk']):03d}_{int(row['curr_chunk']):03d}: saved={row['saved_pairs']}, high_conf={row['high_conf_pair_count']}, either_zero={row['either_zero_ratio']}"
        for row in seq01_default_rows
    )
    lines.extend(["", "min_conf=0.0 per-pair high-conf / zero ratios:", ""])
    lines.extend(
        f"- chunk_{int(row['prev_chunk']):03d}_{int(row['curr_chunk']):03d}: saved={row['saved_pairs']}, high_conf={row['high_conf_pair_count']}, either_zero={row['either_zero_ratio']}, both_zero={row['both_zero_ratio']}"
        for row in seq01_minconf_rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_gate(rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> dict[str, Any]:
    usable_seqs = [
        seq
        for seq, payload in summary.items()
        if (payload.get("sources") or {}).get("default", {}).get("usable_high_quality")
        or (payload.get("sources") or {}).get("minconf0", {}).get("usable_low_conf_stress")
        or (payload.get("sources") or {}).get("minconf0", {}).get("usable_high_quality")
    ]
    default_high_quality = [
        seq for seq, payload in summary.items() if (payload.get("sources") or {}).get("default", {}).get("usable_high_quality")
    ]
    return {
        "phase1_gate_pass": bool(
            len(usable_seqs) >= 3
            and "01" in summary
            and all(seq in default_high_quality for seq in ("00", "02", "05"))
            and all(row.get("high_conf_pair_count") is not None for row in rows)
        ),
        "usable_seq_count": len(usable_seqs),
        "usable_seqs": usable_seqs,
        "default_high_quality_seqs": default_high_quality,
        "seq01_status": (summary.get("01") or {}).get("status"),
        "seq01_explicitly_classified": "01" in summary,
        "seq00_02_05_default_high_quality_pass": all(seq in default_high_quality for seq in ("00", "02", "05")),
        "all_pair_rows_have_confidence_bins": all(row.get("high_conf_pair_count") is not None for row in rows),
    }


def main() -> None:
    args = parse_args()
    seqs = parse_seqs(args.seqs)
    rows = collect_rows(args.default_root, args.minconf0_root, args.preprocess_root, seqs)
    summary = summarize_by_seq(rows, seqs)
    bin_rows = confidence_bin_summary(rows)
    gate = build_gate(rows, summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "overlap_quality_by_pair.csv", rows, PAIR_FIELDS)
    write_json(args.out_dir / "overlap_quality_by_seq.json", {"schema": "acl2_v82_overlap_quality_by_seq_v1", "gate": gate, "seqs": summary})
    write_csv(args.out_dir / "confidence_bin_summary.csv", bin_rows)
    write_seq01_diagnosis(args.out_dir / "seq01_sparse_support_diagnosis.md", summary, rows)
    write_stratified_manifests(args.out_dir, rows)

    decision = {
        "schema": "acl2_v82_phase1_overlap_quality_stratification_v1",
        "out_dir": str(args.out_dir),
        "default_root": str(args.default_root),
        "minconf0_root": str(args.minconf0_root),
        "pair_rows": len(rows),
        "confidence_bin_rows": len(bin_rows),
        "gate": gate,
    }
    write_json(args.out_dir / "phase1_overlap_quality_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
