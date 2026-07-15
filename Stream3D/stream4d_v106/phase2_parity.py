from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from .artifacts import file_record, read_json, sha256_file, write_json
from .config import Phase2ArtifactConfig


@dataclass(frozen=True)
class VariantArtifact:
    name: str
    summary_path: Path
    labels_dir: Path


def _load_label(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def _summary_subset(summary: Dict[str, Any]) -> Dict[str, Any]:
    video_bank = summary.get("video_feature_bank") or {}
    bank_summary = video_bank.get("video_bank_summary") if isinstance(video_bank, dict) else {}
    if not isinstance(bank_summary, dict):
        bank_summary = {}
    forward_counter = video_bank.get("forward_counter") if isinstance(video_bank, dict) else {}
    if not isinstance(forward_counter, dict):
        forward_counter = {}
    use_video_feature_bank = summary.get("use_video_feature_bank")
    if use_video_feature_bank is None and isinstance(video_bank, dict):
        use_video_feature_bank = video_bank.get("enabled")
    reuse_video_state_template = summary.get("reuse_video_state_template")
    if reuse_video_state_template is None and str(summary.get("schema_version")) == "stream4d_v105_phase5_frozen_birth_replay_summary_v1":
        reuse_video_state_template = False
    return {
        "schema_version": summary.get("schema_version"),
        "scene_id": summary.get("scene_id"),
        "variant": summary.get("variant"),
        "baseline_id": summary.get("baseline_id"),
        "frame_count": summary.get("frame_count"),
        "frame_ids": summary.get("frame_ids"),
        "total_runtime_sec": summary.get("total_runtime_sec"),
        "tracking_runtime_sec": summary.get("total_tracking_runtime_sec"),
        "gap_or_birth_runtime_sec": summary.get("total_gap_segmentation_runtime_sec")
        if "total_gap_segmentation_runtime_sec" in summary
        else summary.get("total_birth_decode_runtime_sec"),
        "peak_cuda_memory_mb": summary.get("peak_cuda_memory_mb"),
        "reference_x0_total_runtime_sec": summary.get("reference_x0_total_runtime_sec"),
        "reference_x1_total_runtime_sec": summary.get("reference_x1_total_runtime_sec"),
        "use_video_feature_bank": use_video_feature_bank,
        "reuse_video_state_template": reuse_video_state_template,
        "video_state_template_init_sec": summary.get("video_state_template_init_sec"),
        "feature_bank_build_runtime_sec": video_bank.get("build_runtime_sec") if isinstance(video_bank, dict) else None,
        "feature_bank_gpu_bytes": bank_summary.get("feature_bank_gpu_bytes"),
        "feature_bank_cpu_bytes": bank_summary.get("feature_bank_cpu_bytes"),
        "sam2_backbone_forward_count": bank_summary.get("sam2_backbone_forward_count"),
        "feature_bank_hit_count": bank_summary.get("feature_bank_hit_count"),
        "feature_bank_miss_count": bank_summary.get("feature_bank_miss_count"),
        "patched_forward_counter_count": forward_counter.get("count"),
        "state_init_count": summary.get("state_init_count"),
        "state_template_clone_count": summary.get("state_template_clone_count"),
        "birth_count_total": summary.get("birth_count_total", summary.get("birth_record_count")),
        "frame0_seed_meta": summary.get("frame0_seed_meta"),
    }


def _label_files(labels_dir: Path) -> List[Path]:
    return sorted(labels_dir.glob("*.png"))


def _foreground_iou(a: np.ndarray, b: np.ndarray) -> float:
    fg_a = a > 0
    fg_b = b > 0
    union = np.logical_or(fg_a, fg_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(fg_a, fg_b).sum() / union)


def _per_id_iou(a: np.ndarray, b: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ids = sorted(int(v) for v in set(np.unique(a).tolist()) | set(np.unique(b).tolist()) if int(v) > 0)
    records: List[Dict[str, Any]] = []
    for label_id in ids:
        ma = a == label_id
        mb = b == label_id
        union = np.logical_or(ma, mb).sum()
        iou = 1.0 if union == 0 else float(np.logical_and(ma, mb).sum() / union)
        records.append(
            {
                "label_id": label_id,
                "iou": iou,
                "pixels_ref": int(ma.sum()),
                "pixels_cmp": int(mb.sum()),
                "pixels_intersection": int(np.logical_and(ma, mb).sum()),
                "pixels_union": int(union),
            }
        )
    ious = [r["iou"] for r in records]
    return records, {
        "id_count": len(records),
        "mean_iou": statistics.fmean(ious) if ious else 1.0,
        "min_iou": min(ious) if ious else 1.0,
        "zero_iou_count": sum(1 for value in ious if value == 0.0),
    }


def compare_label_dirs(reference_dir: Path, compare_dir: Path, compare_name: str) -> Dict[str, Any]:
    start = time.perf_counter()
    reference_files = _label_files(reference_dir)
    frame_records: List[Dict[str, Any]] = []
    per_id_records: List[Dict[str, Any]] = []
    for ref_file in reference_files:
        cmp_file = compare_dir / ref_file.name
        if not cmp_file.exists():
            frame_records.append(
                {
                    "frame_name": ref_file.name,
                    "compare_name": compare_name,
                    "compare_exists": False,
                    "label_exact": False,
                }
            )
            continue
        ref = _load_label(ref_file)
        cmp = _load_label(cmp_file)
        if ref.shape != cmp.shape:
            frame_records.append(
                {
                    "frame_name": ref_file.name,
                    "compare_name": compare_name,
                    "compare_exists": True,
                    "shape_ref": list(ref.shape),
                    "shape_cmp": list(cmp.shape),
                    "label_exact": False,
                    "error": "shape_mismatch",
                }
            )
            continue
        ids, id_summary = _per_id_iou(ref, cmp)
        for row in ids:
            row["frame_name"] = ref_file.name
            row["compare_name"] = compare_name
        per_id_records.extend(ids)
        frame_records.append(
            {
                "frame_name": ref_file.name,
                "compare_name": compare_name,
                "compare_exists": True,
                "shape": list(ref.shape),
                "dtype_ref": str(ref.dtype),
                "dtype_cmp": str(cmp.dtype),
                "label_exact": bool(np.array_equal(ref, cmp)),
                "pixel_exact_ratio": float((ref == cmp).sum() / ref.size),
                "foreground_union_iou": _foreground_iou(ref, cmp),
                "per_id_iou_summary": id_summary,
                "ref_sha256": sha256_file(ref_file),
                "cmp_sha256": sha256_file(cmp_file),
            }
        )
    fg_ious = [r["foreground_union_iou"] for r in frame_records if "foreground_union_iou" in r]
    pixel_ratios = [r["pixel_exact_ratio"] for r in frame_records if "pixel_exact_ratio" in r]
    summary = {
        "compare_name": compare_name,
        "frame_count": len(reference_files),
        "compared_frame_count": sum(1 for r in frame_records if r.get("compare_exists")),
        "exact_frame_count": sum(1 for r in frame_records if r.get("label_exact")),
        "label_exact_all_frames": bool(frame_records) and all(r.get("label_exact") for r in frame_records),
        "mean_pixel_exact_ratio": statistics.fmean(pixel_ratios) if pixel_ratios else None,
        "mean_foreground_union_iou": statistics.fmean(fg_ious) if fg_ious else None,
        "min_foreground_union_iou": min(fg_ious) if fg_ious else None,
        "comparison_runtime_sec": time.perf_counter() - start,
    }
    return {
        "summary": summary,
        "frame_records": frame_records,
        "per_id_iou_records": per_id_records,
    }


def _birth_schedule_record(repo_root: Path, b2_birth_records: Path, variant_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "b2_birth_records": file_record(repo_root, str(b2_birth_records.relative_to(repo_root)) if b2_birth_records.is_absolute() and b2_birth_records.is_relative_to(repo_root) else str(b2_birth_records), required=True)
        if b2_birth_records.exists()
        else {"path": str(b2_birth_records), "exists": False},
        "variant_birth_sources": {},
        "frozen_schedule_match": None,
        "diagnosis": [],
    }
    b2_payload = read_json(b2_birth_records) if b2_birth_records.exists() else None
    b2_sha = sha256_file(b2_birth_records) if b2_birth_records.exists() else None
    b2_count = b2_payload.get("row_count") if isinstance(b2_payload, dict) else None
    for name, summary in variant_summaries.items():
        meta = summary.get("frame0_seed_meta") or {}
        birth_records_path = meta.get("birth_records_path", summary.get("birth_records_path"))
        birth_records_sha256 = meta.get("birth_records_sha256", summary.get("birth_records_sha256"))
        payload_frame_count = meta.get("birth_records_payload_frame_count")
        if payload_frame_count is None and isinstance(summary.get("frame_ids"), list):
            payload_frame_count = len(summary["frame_ids"])
        birth_count_total = summary.get("birth_count_total", summary.get("birth_record_count"))
        record["variant_birth_sources"][name] = {
            "birth_records_path": birth_records_path,
            "birth_records_sha256": birth_records_sha256,
            "birth_records_frame_id_contract": meta.get("birth_records_frame_id_contract", "full_requested_sequence"),
            "birth_records_payload_frame_count": payload_frame_count,
            "frame0_seed_count": meta.get("frame0_seed_count"),
            "birth_count_total": birth_count_total,
        }
    matches = []
    for name, source in record["variant_birth_sources"].items():
        matches.append(source.get("birth_records_sha256") == b2_sha)
        if source.get("birth_records_sha256") != b2_sha:
            record["diagnosis"].append(
                {
                    "variant": name,
                    "issue": "birth_schedule_sha_mismatch",
                    "b2_birth_sha256": b2_sha,
                    "variant_birth_sha256": source.get("birth_records_sha256"),
                    "b2_row_count": b2_count,
                    "variant_frame0_seed_count": source.get("frame0_seed_count"),
                    "variant_birth_count_total": source.get("birth_count_total"),
                }
            )
    record["frozen_schedule_match"] = bool(matches) and all(matches)
    return record


def _variant_identity_checks(summaries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    expectations = {
        "B3_feature_bank": {
            "use_video_feature_bank": True,
            "reuse_video_state_template": False,
        },
        "B4_state_template": {
            "use_video_feature_bank": True,
            "reuse_video_state_template": True,
        },
    }
    checks: List[Dict[str, Any]] = []
    for name, expected_fields in expectations.items():
        summary = _summary_subset(summaries[name])
        actual = {field: summary.get(field) for field in expected_fields}
        checks.append(
            {
                "name": f"{name}_variant_identity",
                "passes": actual == expected_fields,
                "actual": actual,
                "expected": expected_fields,
            }
        )
    return checks


def run_phase2_artifact_parity(repo_root: Path, config: Phase2ArtifactConfig, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = {
        "B2_exact_reference": VariantArtifact(
            "B2_exact_reference", repo_root / config.b2_summary, repo_root / config.b2_labels_dir
        ),
        "B3_feature_bank": VariantArtifact("B3_feature_bank", repo_root / config.b3_summary, repo_root / config.b3_labels_dir),
        "B4_state_template": VariantArtifact("B4_state_template", repo_root / config.b4_summary, repo_root / config.b4_labels_dir),
    }
    summaries = {name: read_json(artifact.summary_path) for name, artifact in variants.items()}
    anchor_inventory = {
        "scope": "real artifact replay parity; no SAM2 model rerun in this phase2 harness",
        "scene_id": config.scene_id,
        "x0_quality_anchor": {
            "summary": file_record(repo_root, config.x0_summary, required=True),
            "subset": _summary_subset(read_json(repo_root / config.x0_summary)) if (repo_root / config.x0_summary).exists() else None,
        },
        "x1_speed_anchor": {
            "summary": file_record(repo_root, config.x1_summary, required=True),
            "subset": _summary_subset(read_json(repo_root / config.x1_summary)) if (repo_root / config.x1_summary).exists() else None,
        },
        "variants": {
            name: {
                "summary": file_record(repo_root, str(artifact.summary_path.relative_to(repo_root)), required=True),
                "labels_dir": str(artifact.labels_dir.relative_to(repo_root)),
                "label_file_count": len(_label_files(artifact.labels_dir)),
                "summary_subset": _summary_subset(summaries[name]),
            }
            for name, artifact in variants.items()
        },
    }
    b2 = variants["B2_exact_reference"]
    b3 = variants["B3_feature_bank"]
    b4 = variants["B4_state_template"]
    comparisons = {
        "B3_vs_B2": compare_label_dirs(b2.labels_dir, b3.labels_dir, "B3_vs_B2"),
        "B4_vs_B2": compare_label_dirs(b2.labels_dir, b4.labels_dir, "B4_vs_B2"),
        "B4_vs_B3": compare_label_dirs(b3.labels_dir, b4.labels_dir, "B4_vs_B3"),
        "B2_self_control": compare_label_dirs(b2.labels_dir, b2.labels_dir, "B2_self_control"),
    }
    for name, payload in comparisons.items():
        write_json(output_dir / f"{name}_frame_records.json", payload["frame_records"])
        write_json(output_dir / f"{name}_per_id_iou_records.json", payload["per_id_iou_records"])
    label_summary = {name: payload["summary"] for name, payload in comparisons.items()}
    b2_runtime = summaries["B2_exact_reference"].get("total_runtime_sec")
    b3_runtime = summaries["B3_feature_bank"].get("total_runtime_sec")
    b4_runtime = summaries["B4_state_template"].get("total_runtime_sec")
    runtime_records = {
        "B2_exact_reference_runtime_sec": b2_runtime,
        "B3_feature_bank_runtime_sec": b3_runtime,
        "B4_state_template_runtime_sec": b4_runtime,
        "B3_runtime_ratio_vs_B2": b3_runtime / b2_runtime if b2_runtime else None,
        "B4_runtime_ratio_vs_B2": b4_runtime / b2_runtime if b2_runtime else None,
        "B4_runtime_le_0p8_B2": (b4_runtime <= 0.8 * b2_runtime) if b2_runtime and b4_runtime else None,
        "B4_runtime_vs_B3_ratio": b4_runtime / b3_runtime if b3_runtime and b4_runtime else None,
        "peak_cuda_memory_mb": {
            name: summaries[name].get("peak_cuda_memory_mb") for name in summaries
        },
        "feature_bank_bytes": {
            name: _summary_subset(summaries[name]).get("feature_bank_gpu_bytes") for name in summaries
        },
        "forward_count": {
            name: _summary_subset(summaries[name]).get("sam2_backbone_forward_count") for name in summaries
        },
        "state_init_count": {
            name: _summary_subset(summaries[name]).get("state_init_count") for name in summaries
        },
        "state_template_clone_count": {
            name: _summary_subset(summaries[name]).get("state_template_clone_count") for name in summaries
        },
        "missing_counter_note": "Null counters mean the source summary did not expose that counter; no value was inferred.",
    }
    birth_schedule = _birth_schedule_record(
        repo_root,
        repo_root / config.b2_birth_records,
        {
            "B3_feature_bank": summaries["B3_feature_bank"],
            "B4_state_template": summaries["B4_state_template"],
        },
    )
    variant_identity_checks = _variant_identity_checks(summaries)
    gate_checks = [
        {
            "name": "evaluator_self_control_exact",
            "passes": label_summary["B2_self_control"]["label_exact_all_frames"],
            "actual": label_summary["B2_self_control"],
            "expected": "B2 labels compared to themselves are exact",
        },
        *variant_identity_checks,
        {
            "name": "B3_labels_exact_vs_B2",
            "passes": label_summary["B3_vs_B2"]["label_exact_all_frames"],
            "actual": label_summary["B3_vs_B2"],
            "expected": "all frames exact",
        },
        {
            "name": "B4_labels_exact_vs_B2",
            "passes": label_summary["B4_vs_B2"]["label_exact_all_frames"],
            "actual": label_summary["B4_vs_B2"],
            "expected": "all frames exact",
        },
        {
            "name": "B4_runtime_le_0p8_B2",
            "passes": runtime_records["B4_runtime_le_0p8_B2"] is True,
            "actual": runtime_records["B4_runtime_ratio_vs_B2"],
            "expected": "<= 0.8",
        },
        {
            "name": "frozen_birth_schedule_match",
            "passes": birth_schedule["frozen_schedule_match"] is True,
            "actual": birth_schedule,
            "expected": "B3/B4 birth schedule sha matches B2 frozen schedule",
        },
    ]
    suggested_repairs: List[Dict[str, Any]] = []
    if any(not check["passes"] for check in variant_identity_checks):
        suggested_repairs.append(
            {
                "repair_ladder_step": "repair artifact selection before parity interpretation",
                "evidence": "B3 must be feature-bank-only and B4 must enable state template; otherwise label/runtime comparisons are not isolating Phase2 acceleration factors.",
                "next_action": "select or rerun artifacts whose summary flags match the Phase2 B3/B4 definitions, then repeat label parity",
            }
        )
    if not label_summary["B3_vs_B2"]["label_exact_all_frames"] or not label_summary["B4_vs_B2"]["label_exact_all_frames"]:
        b4_vs_b3 = label_summary["B4_vs_B3"]
        suggested_repairs.append(
            {
                "repair_ladder_step": "check image/video transform separation and frozen birth schedule before transform tuning",
                "evidence": {
                    "B3_vs_B2_exact": label_summary["B3_vs_B2"]["label_exact_all_frames"],
                    "B4_vs_B2_exact": label_summary["B4_vs_B2"]["label_exact_all_frames"],
                    "B4_vs_B3_exact": b4_vs_b3["label_exact_all_frames"],
                    "frozen_birth_schedule_match": birth_schedule["frozen_schedule_match"],
                },
                "next_action": "rerun B3/B4 from the B2 frozen birth_records schedule, then repeat label parity before state-template tuning",
            }
        )
    if not label_summary["B4_vs_B3"]["label_exact_all_frames"]:
        suggested_repairs.append(
            {
                "repair_ladder_step": "check state clone clearing fields, runtime local ID order, and autocast/dtype",
                "evidence": "B4 state-template artifact is not label-exact against the feature-bank-only B3 artifact even before comparing to B2.",
                "next_action": "after frozen schedule alignment, compare B4 against B3 on identical inputs; if mismatch persists, audit state template clone fields before escalating specgap",
            }
        )
    if runtime_records["B4_runtime_le_0p8_B2"] is not True:
        suggested_repairs.append(
            {
                "repair_ladder_step": "fallback to feature bank only and disable state template",
                "evidence": "B4 runtime did not satisfy <=0.8*B2 in this artifact set",
                "next_action": "profile B3/B4 prompt decode and state clone critical path",
            }
        )
    phase2_summary = {
        "schema_version": "stream4d_v106_phase2_artifact_parity_summary_v1",
        "scope": "artifact replay over existing v105 outputs; not a fresh SAM2 v106 model run",
        "passes": all(check["passes"] for check in gate_checks),
        "gate_checks": gate_checks,
        "label_summary": label_summary,
        "runtime_records": runtime_records,
        "birth_schedule": birth_schedule,
        "suggested_repairs": suggested_repairs,
    }
    write_json(output_dir / "anchor_inventory.json", anchor_inventory)
    write_json(output_dir / "label_parity_summary.json", label_summary)
    write_json(output_dir / "runtime_vram_forward_records.json", runtime_records)
    write_json(output_dir / "frozen_birth_schedule_audit.json", birth_schedule)
    write_json(output_dir / "phase2_gate_summary.json", phase2_summary)
    return phase2_summary
