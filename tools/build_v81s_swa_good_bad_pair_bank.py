#!/usr/bin/env python3
"""Build v81S SWA adjacent-pair good/bad case bank from repaired overlap pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


DEFAULT_CASE_METRICS = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase1_three_memory_case_bank/mid_adjacent_pair_candidate_metrics.csv"
)
DEFAULT_OVERLAP_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS1_multiseq_swa_overlap_repair_minconf0"
)
DEFAULT_OUT_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v81s_swa_first_semantic_scale_memory_control/"
    "report_final/phaseS2_swa_good_bad_pair_bank"
)
DEFAULT_PREPROCESS_ROOT = Path("results/kitti_preprocess")

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

FIELDS = [
    "seq",
    "prev_chunk",
    "curr_chunk",
    "frame_start",
    "frame_end",
    "case_type",
    "future_after_overlap",
    "boundary_jump",
    "raw_overlap_residual",
    "overlap_scale_residual",
    "prev_local_sim3",
    "curr_local_sim3",
    "prev_to_curr_scale_jump",
    "stable_overlap_mass",
    "harm_overlap_mass",
    "context_overlap_mass",
    "READ_used_stable_mass",
    "SWA_carried_stable_mass",
    "SWA_carried_harm_mass",
    "V_alignment_delta",
    "K_risk_delta",
    "same_object_overlap_ratio",
    "cross_object_boundary_ratio",
    "RADIO_temporal_stability",
    "has_radio",
    "proxy_scale_residual",
    "target_reason",
    "J_mid",
    "case_rank",
    "sample_policy",
    "overlap_pair_file",
    "saved_pair_count",
    "semantic_label_projected_ratio",
    "semantic_nonvoid_ratio",
    "min_geometry_conf_mean",
    "min_geometry_conf_median",
    "min_geometry_conf_p10",
    "min_geometry_conf_p90",
    "either_zero_geometry_conf_ratio",
    "both_zero_geometry_conf_ratio",
    "artifact_quality_risk",
    "missing_fields",
]


def _parse_seqs(text: str) -> list[str]:
    return [part.strip().zfill(2) for part in str(text).split(",") if part.strip()]


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pair_path(root: Path, seq: str, prev_chunk: int, curr_chunk: int) -> Path:
    return root / "overlap_pairs" / seq / f"chunk_{prev_chunk:03d}_{curr_chunk:03d}.pt"


def _load_label_names(preprocess_root: Path, seq: str) -> list[str]:
    path = preprocess_root / seq / "sparse_masklets_with_semantic.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sem = payload.get("semantic_segmentation", payload) if isinstance(payload, dict) else {}
    names = sem.get("label_names", []) if isinstance(sem, dict) else []
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(name) for name in names]


def _label_family(name: str) -> str:
    lowered = str(name).lower()
    if any(word in lowered for word in DYNAMIC_WORDS):
        return "harm"
    if any(word in lowered for word in CONTEXT_WORDS):
        return "context"
    if any(word in lowered for word in LOWTRUST_WORDS):
        return "context"
    if any(word in lowered for word in STABLE_WORDS):
        return "stable"
    return "context"


def _semantic_masses(labels: torch.Tensor, label_names: list[str]) -> dict[str, float]:
    labels = labels.long().reshape(-1)
    total = int(labels.numel())
    if total <= 0:
        return {"stable_overlap_mass": 0.0, "harm_overlap_mass": 0.0, "context_overlap_mass": 0.0}
    counts = {"stable": 0, "harm": 0, "context": 0}
    for label_id, count in zip(*torch.unique(labels, return_counts=True)):
        idx = int(label_id.item())
        name = label_names[idx] if 0 <= idx < len(label_names) else "unknown"
        counts[_label_family(name)] += int(count.item())
    return {
        "stable_overlap_mass": float(counts["stable"]) / float(total),
        "harm_overlap_mass": float(counts["harm"]) / float(total),
        "context_overlap_mass": float(counts["context"]) / float(total),
    }


def _pair_stats(path: Path, label_names: list[str]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prev_conf = payload.get("prev_conf")
    curr_conf = payload.get("curr_conf")
    labels = payload.get("prev_semantic_labels")
    semantic_masses = (
        _semantic_masses(labels, label_names)
        if torch.is_tensor(labels)
        else {"stable_overlap_mass": "", "harm_overlap_mass": "", "context_overlap_mass": ""}
    )
    if not torch.is_tensor(prev_conf) or not torch.is_tensor(curr_conf) or prev_conf.numel() == 0:
        out = {
            "saved_pair_count": int(payload.get("saved_pair_count", 0) or 0),
            "semantic_label_projected_ratio": payload.get("semantic_label_projected_ratio", ""),
            "semantic_nonvoid_ratio": payload.get("semantic_nonvoid_ratio", ""),
            "overlap_scale_residual": payload.get("raw_residual_rmse", ""),
            "min_geometry_conf_mean": "",
            "min_geometry_conf_median": "",
            "min_geometry_conf_p10": "",
            "min_geometry_conf_p90": "",
            "either_zero_geometry_conf_ratio": "",
            "both_zero_geometry_conf_ratio": "",
        }
        out.update(semantic_masses)
        return out
    prev_conf = prev_conf.float()
    curr_conf = curr_conf.float()
    min_conf = torch.minimum(prev_conf, curr_conf)
    out = {
        "saved_pair_count": int(payload.get("saved_pair_count", int(prev_conf.numel())) or 0),
        "semantic_label_projected_ratio": float(payload.get("semantic_label_projected_ratio", 0.0) or 0.0),
        "semantic_nonvoid_ratio": float(payload.get("semantic_nonvoid_ratio", 0.0) or 0.0),
        "overlap_scale_residual": payload.get("raw_residual_rmse", ""),
        "min_geometry_conf_mean": float(min_conf.mean().item()),
        "min_geometry_conf_median": float(min_conf.median().item()),
        "min_geometry_conf_p10": float(torch.quantile(min_conf, 0.10).item()),
        "min_geometry_conf_p90": float(torch.quantile(min_conf, 0.90).item()),
        "either_zero_geometry_conf_ratio": float(((prev_conf == 0) | (curr_conf == 0)).float().mean().item()),
        "both_zero_geometry_conf_ratio": float(((prev_conf == 0) & (curr_conf == 0)).float().mean().item()),
    }
    out.update(semantic_masses)
    return out


def _read_candidates(path: Path, seqs: set[str], overlap_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = str(row.get("seq", "")).zfill(2)
            if seq not in seqs:
                continue
            try:
                prev_chunk = int(row["prev_chunk"])
                curr_chunk = int(row["curr_chunk"])
            except (KeyError, TypeError, ValueError):
                continue
            pair_path = _pair_path(overlap_root, seq, prev_chunk, curr_chunk)
            if not pair_path.is_file():
                continue
            out = dict(row)
            out["seq"] = seq
            out["prev_chunk"] = prev_chunk
            out["curr_chunk"] = curr_chunk
            out["pair_path"] = pair_path
            out["J_mid_float"] = _float(row.get("J_mid"))
            if out["J_mid_float"] is None:
                continue
            rows.append(out)
    return rows


def _select_by_seq(rows: list[dict[str, Any]], *, case_type: str, per_seq: int) -> list[dict[str, Any]]:
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seq[str(row["seq"])].append(row)
    selected: list[dict[str, Any]] = []
    reverse = case_type == "bad"
    for seq in sorted(by_seq):
        ranked = sorted(by_seq[seq], key=lambda row: float(row["J_mid_float"]), reverse=reverse)
        for rank, row in enumerate(ranked[:per_seq], start=1):
            copy = dict(row)
            copy["case_type"] = case_type
            copy["case_rank"] = rank
            selected.append(copy)
    return selected


def _build_row(row: dict[str, Any], overlap_root: Path, label_names_by_seq: dict[str, list[str]]) -> dict[str, Any]:
    seq = str(row.get("seq", "")).zfill(2)
    stats = _pair_stats(Path(row["pair_path"]), label_names_by_seq.get(seq, []))
    min_conf_median = _float(stats.get("min_geometry_conf_median"))
    either_zero = _float(stats.get("either_zero_geometry_conf_ratio"))
    quality_risk = bool((min_conf_median is not None and min_conf_median <= 0.0) or (either_zero is not None and either_zero >= 0.25))
    out = {
        "seq": row.get("seq", ""),
        "prev_chunk": row.get("prev_chunk", ""),
        "curr_chunk": row.get("curr_chunk", ""),
        "frame_start": row.get("frame_start", ""),
        "frame_end": row.get("frame_end", ""),
        "case_type": row.get("case_type", ""),
        "future_after_overlap": row.get("future_after_overlap", ""),
        "boundary_jump": row.get("boundary_jump", ""),
        "raw_overlap_residual": row.get("raw_overlap_residual", ""),
        "overlap_scale_residual": stats.get("overlap_scale_residual", ""),
        "prev_local_sim3": "",
        "curr_local_sim3": "",
        "prev_to_curr_scale_jump": row.get("scale_cv", ""),
        "stable_overlap_mass": stats.get("stable_overlap_mass", row.get("stable_overlap_mass", "")),
        "harm_overlap_mass": stats.get("harm_overlap_mass", row.get("harm_overlap_mass", "")),
        "context_overlap_mass": stats.get("context_overlap_mass", row.get("context_overlap_mass", "")),
        "READ_used_stable_mass": "",
        "SWA_carried_stable_mass": "",
        "SWA_carried_harm_mass": "",
        "V_alignment_delta": row.get("V_alignment_delta", ""),
        "K_risk_delta": row.get("K_risk_delta", ""),
        "same_object_overlap_ratio": row.get("same_object_overlap_ratio", ""),
        "cross_object_boundary_ratio": row.get("cross_object_boundary_ratio", ""),
        "RADIO_temporal_stability": "",
        "has_radio": False,
        "proxy_scale_residual": True,
        "target_reason": f"{row.get('case_type')}_by_seq_stratified_J_mid_rank; source=v80_mid_candidate_metrics; overlap_root={overlap_root}",
        "J_mid": row.get("J_mid", ""),
        "case_rank": row.get("case_rank", ""),
        "sample_policy": "top_residual_conf_product",
        "overlap_pair_file": str(row["pair_path"]),
        "artifact_quality_risk": quality_risk,
    }
    out.update(stats)
    missing = [key for key in FIELDS if key not in {"missing_fields"} and out.get(key, "") == ""]
    out["missing_fields"] = ";".join(missing)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-metrics", type=Path, default=DEFAULT_CASE_METRICS)
    parser.add_argument("--overlap-root", type=Path, default=DEFAULT_OVERLAP_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--preprocess-root", type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--per-seq-per-type", type=int, default=3)
    args = parser.parse_args()

    seqs = set(_parse_seqs(args.seqs))
    candidates = _read_candidates(args.case_metrics, seqs, args.overlap_root)
    selected = _select_by_seq(candidates, case_type="bad", per_seq=int(args.per_seq_per_type))
    selected.extend(_select_by_seq(candidates, case_type="good", per_seq=int(args.per_seq_per_type)))
    label_names_by_seq = {seq: _load_label_names(args.preprocess_root, seq) for seq in sorted(seqs)}
    rows = [_build_row(row, args.overlap_root, label_names_by_seq) for row in selected]

    case_counts = Counter(row["case_type"] for row in rows)
    seq_coverage = sorted({row["seq"] for row in rows})
    by_case_seq_coverage = {
        case_type: sorted({row["seq"] for row in rows if row["case_type"] == case_type})
        for case_type in ("bad", "good")
    }
    missing_counts: Counter[str] = Counter()
    for row in rows:
        for key in str(row.get("missing_fields", "")).split(";"):
            if key:
                missing_counts[key] += 1
    rows_have_proxy = all(str(row.get("proxy_scale_residual")).lower() == "true" or row.get("overlap_scale_residual") not in ("", None) for row in rows)
    semantic_fields_present = all(row.get("stable_overlap_mass", "") != "" and row.get("harm_overlap_mass", "") != "" and row.get("context_overlap_mass", "") != "" for row in rows)
    gate = {
        "bad_ge_12": case_counts.get("bad", 0) >= 12,
        "good_ge_12": case_counts.get("good", 0) >= 12,
        "coverage_ge_3": len(seq_coverage) >= 3,
        "bad_coverage_ge_3": len(by_case_seq_coverage["bad"]) >= 3,
        "good_coverage_ge_3": len(by_case_seq_coverage["good"]) >= 3,
        "rows_have_overlap_or_proxy_residual": rows_have_proxy,
        "semantic_diagnosis_fields_present": semantic_fields_present,
    }
    gate["phaseS2_gate_pass"] = all(gate.values())
    summary = {
        "schema": "acl2_v81s_swa_good_bad_pair_bank_v1",
        "case_metrics": str(args.case_metrics),
        "overlap_root": str(args.overlap_root),
        "out_dir": str(args.out_dir),
        "candidate_rows_with_repaired_overlap_pairs": len(candidates),
        "rows": len(rows),
        "case_counts": dict(case_counts),
        "seq_coverage": seq_coverage,
        "by_case_seq_coverage": by_case_seq_coverage,
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "artifact_quality_risk_rows": int(sum(1 for row in rows if bool(row.get("artifact_quality_risk")))),
        "gate": gate,
    }
    _write_csv(args.out_dir / "swa_good_bad_pair_bank.csv", rows, FIELDS)
    _write_json(args.out_dir / "swa_good_bad_pair_bank_summary.json", summary)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
