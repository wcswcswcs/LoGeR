#!/usr/bin/env python3
"""ACL2 v25B video-masklet cache and implementation audit.

This script audits the deployable predicted semantic source used by v25B:
the offline Stage-C video-masklet cache.  It never treats predicted masks as
GT semantic labels and does not launch HMC candidate rows itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


CHUNK_STARTS = {
    5: 145,
    6: 174,
    10: 290,
    16: 464,
}


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    materialized = list(rows)
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_int_list(text: str, default: Sequence[int]) -> List[int]:
    text = (text or "").strip()
    if not text:
        return list(default)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _selected_cache_chunks(chunks: Sequence[int], horizons: Sequence[int]) -> Set[int]:
    needed: Set[int] = set()
    for chunk in chunks:
        for horizon in horizons:
            for idx in range(int(chunk), int(chunk) + int(horizon) + 1):
                needed.add(idx)
    return needed


def _load_cache_index(cache_dir: Path) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for row in _read_jsonl(cache_dir / "cache_index.jsonl"):
        try:
            idx = int(row.get("chunk_idx"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        out[idx] = row
    return out


def _load_semantic_rows(cache_dir: Path) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    for row in _read_csv(cache_dir / "semantic_audit" / "per_chunk_semantic.csv"):
        try:
            idx = int(row.get("chunk_idx", ""))
        except ValueError:
            continue
        rows[idx] = row
    return rows


def _load_fine_labels(cache_dir: Path, chunks: Set[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in _read_csv(cache_dir / "semantic_audit" / "label_counts_by_chunk.csv"):
        try:
            idx = int(row.get("chunk_idx", ""))
        except ValueError:
            continue
        if chunks and idx not in chunks:
            continue
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        rows.append(
            {
                "chunk_idx": idx,
                "start_frame": row.get("start_frame"),
                "end_frame": row.get("end_frame"),
                "fine_label": label,
                "count": row.get("count", "0"),
                "uses_gt_semantic": False,
                "semantic_source": "video_masklet_frontend_cache",
            }
        )
    return rows


def _read_pose_file(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            rows.append([float(part) for part in line.split()])
    return rows


def _pose_diff_metrics(candidate_path: Path, reference_path: Path) -> Dict[str, object]:
    candidate = _read_pose_file(candidate_path)
    reference = _read_pose_file(reference_path)
    if not candidate or not reference:
        return {
            "pose_compare_available": False,
            "matched_rows": 0,
            "max_translation_abs_diff": float("inf"),
            "max_pose_abs_diff": float("inf"),
            "shape_mismatch": bool(len(candidate) != len(reference)),
        }
    matched = min(len(candidate), len(reference))
    max_translation = 0.0
    max_pose = 0.0
    for cand_row, ref_row in zip(candidate[:matched], reference[:matched]):
        width = min(len(cand_row), len(ref_row))
        for col in range(1, min(width, 4)):
            max_translation = max(max_translation, abs(cand_row[col] - ref_row[col]))
        for col in range(1, min(width, 8)):
            max_pose = max(max_pose, abs(cand_row[col] - ref_row[col]))
    return {
        "pose_compare_available": True,
        "matched_rows": matched,
        "max_translation_abs_diff": max_translation,
        "max_pose_abs_diff": max_pose,
        "shape_mismatch": bool(len(candidate) != len(reference)),
    }


def _resolve_run_file(run_dir: str, phase0_report: Path) -> Path:
    path = Path(run_dir)
    if not path.is_absolute():
        path = (phase0_report.parent.parent.parent.parent / path).resolve()
        if not path.exists():
            path = (Path.cwd() / run_dir).resolve()
    return path / "01.txt"


def _phase0_noop_gate(phase0_report: Optional[Path]) -> Dict[str, object]:
    if phase0_report is None:
        return {
            "phase0_report_provided": False,
            "noop_parity_gate_pass": None,
            "noop_rows_checked": 0,
            "noop_failures": [],
            "noop_gate_method": "not_run",
        }
    path = phase0_report / "candidate_vs_H9_delta_by_horizon.csv"
    rows = _read_csv(path)
    wanted = {
        "K1_H9",
        "P0_01_SEMANTIC_ROLE_NOOP_IGNORED",
        "P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED",
        "P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY",
    }
    checked_external = 0
    external_failures: List[Dict[str, object]] = []
    run_dirs: Dict[str, str] = {}
    for row in rows:
        candidate = row.get("candidate_id", "")
        if candidate not in wanted:
            continue
        checked_external += 1
        run_dirs[candidate] = row.get("run_dir", "")
        ate = _safe_float(row.get("ATE_delta_vs_H9"))
        raw = _safe_float(row.get("raw_trans_max_diff_vs_H9", row.get("raw_trans_max_diff")))
        if not (math.isfinite(ate) and math.isfinite(raw) and ate == 0.0 and raw == 0.0):
            external_failures.append(
                {
                    "candidate_id": candidate,
                    "ATE_delta_vs_H9": row.get("ATE_delta_vs_H9"),
                    "raw_trans_max_diff": row.get("raw_trans_max_diff_vs_H9", row.get("raw_trans_max_diff")),
                }
            )

    direct_checked = 0
    direct_failures: List[Dict[str, object]] = []
    direct_rows: List[Dict[str, object]] = []
    reference_run = run_dirs.get("K1_H9", "")
    reference_path = _resolve_run_file(reference_run, phase0_report) if reference_run else Path()
    if reference_run:
        for candidate in sorted(wanted):
            run_dir = run_dirs.get(candidate, "")
            if not run_dir:
                direct_failures.append({"candidate_id": candidate, "failure": "missing_run_dir"})
                continue
            candidate_path = _resolve_run_file(run_dir, phase0_report)
            metrics = _pose_diff_metrics(candidate_path, reference_path)
            direct_checked += 1
            row = {
                "candidate_id": candidate,
                "reference_candidate_id": "K1_H9",
                "candidate_pose_path": str(candidate_path),
                "reference_pose_path": str(reference_path),
                **metrics,
            }
            direct_rows.append(row)
            if (
                not metrics["pose_compare_available"]
                or metrics["shape_mismatch"]
                or metrics["max_translation_abs_diff"] != 0.0
                or metrics["max_pose_abs_diff"] != 0.0
            ):
                direct_failures.append(row)

    direct_available = bool(reference_run and direct_checked > 0)
    direct_gate = bool(direct_available and direct_checked >= len(wanted) and not direct_failures)
    external_gate = bool(checked_external >= len(wanted) and not external_failures)
    gate = direct_gate if direct_available else external_gate
    return {
        "phase0_report_provided": True,
        "noop_parity_gate_pass": gate,
        "noop_gate_method": "direct_k1_pose_compare" if direct_available else "external_H9_report",
        "noop_rows_checked": direct_checked if direct_available else checked_external,
        "noop_failures": direct_failures if direct_available else external_failures,
        "noop_direct_reference_candidate_id": "K1_H9" if direct_available else "",
        "noop_direct_rows": direct_rows,
        "noop_external_H9_report_gate_pass": external_gate,
        "noop_external_H9_rows_checked": checked_external,
        "noop_external_H9_failures": external_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results/kitti01_hmc_v2/acl2_v25b_videomasklet_semanticprior_allmemory_parallel",
    )
    parser.add_argument(
        "--stage-c-cache-dir",
        default="results/kitti01_hmc_v2/acl2_v6_stage_c_cache_mask2former_cityscapes_full",
    )
    parser.add_argument("--chunks", default="6,10,16")
    parser.add_argument("--horizons", default="10,15")
    parser.add_argument("--phase0-report", default="")
    args = parser.parse_args()

    results = Path(args.results_root).resolve()
    out_dir = results / "implementation_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.stage_c_cache_dir).resolve()
    chunks = _parse_int_list(args.chunks, [6, 10, 16])
    horizons = _parse_int_list(args.horizons, [10, 15])
    needed_chunks = _selected_cache_chunks(chunks, horizons)
    cache_index = _load_cache_index(cache_dir)
    semantic_rows = _load_semantic_rows(cache_dir)

    chunk_rows: List[Dict[str, object]] = []
    coverage_values: List[float] = []
    visible_values: List[float] = []
    masklet_counts: List[int] = []
    hit_count = 0
    for idx in sorted(needed_chunks):
        index_row = cache_index.get(idx)
        sem = semantic_rows.get(idx, {})
        masklet_path = ""
        manifest_path = ""
        file_hit = False
        if index_row is not None:
            chunk_name = str(index_row.get("chunk"))
            masklet_path = str(cache_dir / chunk_name / "masklet.pt")
            manifest_path = str(cache_dir / chunk_name / "manifest.json")
            file_hit = Path(masklet_path).exists() and Path(manifest_path).exists()
        if file_hit:
            hit_count += 1
        coverage = _safe_float(sem.get("coverage_mean"))
        visible = _safe_float(sem.get("visible_masklet_frame_frac"))
        n_masklets = int(float(sem.get("num_masklets", index_row.get("num_masklets", 0) if index_row else 0) or 0))
        if math.isfinite(coverage):
            coverage_values.append(coverage)
        if math.isfinite(visible):
            visible_values.append(visible)
        masklet_counts.append(n_masklets)
        chunk_rows.append(
            {
                "chunk_idx": idx,
                "cache_index_hit": index_row is not None,
                "masklet_file_hit": file_hit,
                "masklet_path": masklet_path,
                "manifest_path": manifest_path,
                "num_masklets": n_masklets,
                "coverage_mean": coverage if math.isfinite(coverage) else "",
                "visible_masklet_frame_frac": visible if math.isfinite(visible) else "",
                "masklet_labels": sem.get("masklet_labels", ""),
            }
        )

    all_sem_rows = list(semantic_rows.values())
    focus_weight = 0.0
    focus_cov_weighted = 0.0
    for row in all_sem_rows:
        overlap = _safe_float(row.get("focus_overlap"))
        cov = _safe_float(row.get("coverage_mean"))
        if math.isfinite(overlap) and math.isfinite(cov) and overlap > 0:
            focus_weight += overlap
            focus_cov_weighted += overlap * cov
    focus_coverage = focus_cov_weighted / focus_weight if focus_weight > 0 else float("nan")

    fine_rows = _load_fine_labels(cache_dir, needed_chunks)
    fine_labels = sorted({str(row["fine_label"]) for row in fine_rows})
    _write_csv(out_dir / "stage_c_cache_hit_audit.csv", chunk_rows)
    _write_csv(out_dir / "fine_label_coverage_by_chunk.csv", fine_rows)

    cache_hit_rate = hit_count / max(1, len(needed_chunks))
    chunks_with_masklets = sum(1 for c in masklet_counts if c > 0)
    chunks_with_masklets_ratio = chunks_with_masklets / max(1, len(needed_chunks))
    mean_coverage = sum(coverage_values) / max(1, len(coverage_values))
    mean_visible = sum(visible_values) / max(1, len(visible_values))
    mean_masklets = sum(masklet_counts) / max(1, len(masklet_counts))
    cache_quality_gate = bool(
        cache_hit_rate == 1.0
        and chunks_with_masklets_ratio >= 0.95
        and mean_coverage >= 0.80
        and math.isfinite(focus_coverage)
        and focus_coverage >= 0.80
    )

    phase0_report = Path(args.phase0_report).resolve() if str(args.phase0_report).strip() else None
    noop_status = _phase0_noop_gate(phase0_report)
    noop_gate = noop_status["noop_parity_gate_pass"]
    all_gate = bool(cache_quality_gate and (noop_gate is True if phase0_report else True))

    failures: List[Dict[str, object]] = []
    if not cache_quality_gate:
        failures.append(
            {
                "gate": "stage_c_cache_quality",
                "failure": "stage_c_cache_quality_below_gate",
                "detail": (
                    f"cache_hit_rate={cache_hit_rate}, chunks_with_masklets_ratio={chunks_with_masklets_ratio}, "
                    f"mean_coverage={mean_coverage}, focus_coverage={focus_coverage}"
                ),
            }
        )
    if phase0_report and noop_gate is not True:
        failures.append(
            {
                "gate": "phase0_noop_parity",
                "failure": "noop_parity_failed_or_incomplete",
                "detail": json.dumps(noop_status, ensure_ascii=False, sort_keys=True),
            }
        )
    _write_jsonl(out_dir / "codex_self_check_failures.jsonl", failures)

    summary: Dict[str, object] = {
        "phase": "v25b_video_masklet_phase0_audit",
        "uses_gt_semantic": False,
        "uses_video_masklet_semantic": True,
        "semantic_source": "video_masklet_frontend_cache",
        "stage_c_cache_dir": str(cache_dir),
        "stage_c_cache_mode": "read",
        "stage_c_cache_require_hit": True,
        "stage_c_inline_when_ignored": False,
        "needed_chunks": sorted(needed_chunks),
        "needed_chunk_count": len(needed_chunks),
        "cache_hit_rate": cache_hit_rate,
        "chunks_with_masklets_ratio": chunks_with_masklets_ratio,
        "chunks_with_masklets": chunks_with_masklets,
        "mean_masklets_per_chunk": mean_masklets,
        "mean_coverage": mean_coverage,
        "mean_visible_masklet_frame_frac": mean_visible,
        "focus_coverage_200_300": focus_coverage,
        "fine_label_available": bool(fine_labels),
        "fine_label_count": len(fine_labels),
        "fine_labels": fine_labels,
        "runtime_fine_role_policy_available": False,
        "runtime_fine_role_policy_note": (
            "Fine labels are present in the video-masklet cache audit, but current runtime semantic role "
            "routing is coarse-group keyed. Fine policies must be marked blocked/coarse fallback."
        ),
        "coarse_group_available": True,
        "cache_quality_gate_pass": cache_quality_gate,
        **noop_status,
        "all_gate_pass": all_gate,
        "selector_allowed": False,
        "full_online_validation_allowed": False,
        "counts_as_deployable_online_success": False,
    }
    (out_dir / "codex_self_check_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# ACL2 v25B Video-Masklet Phase-0 Audit",
        "",
        f"cache_quality_gate_pass = `{str(cache_quality_gate).lower()}`",
        f"all_gate_pass = `{str(all_gate).lower()}`",
        f"cache_hit_rate = `{cache_hit_rate:.10f}`",
        f"chunks_with_masklets_ratio = `{chunks_with_masklets_ratio:.10f}`",
        f"mean_coverage = `{mean_coverage:.10f}`",
        f"focus_coverage_200_300 = `{focus_coverage:.10f}`",
        f"fine_label_available = `{str(bool(fine_labels)).lower()}`",
        f"fine_label_count = `{len(fine_labels)}`",
        f"runtime_fine_role_policy_available = `false`",
        "",
        "This is a predicted video-masklet semantic audit, not a GT semantic audit.",
    ]
    if phase0_report:
        report.extend(
            [
                "",
                f"noop_parity_gate_pass = `{str(noop_gate).lower()}`",
                f"noop_rows_checked = `{noop_status['noop_rows_checked']}`",
            ]
        )
    (out_dir / "codex_self_check_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return 0 if all_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
