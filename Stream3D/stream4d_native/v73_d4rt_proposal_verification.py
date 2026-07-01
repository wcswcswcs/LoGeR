from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v72_d4rt_proposal_verification import run as run_v72_d4rt  # noqa: E402


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_source_mask_id(source_mask_ids: str) -> int:
    try:
        return int(str(source_mask_ids).split(":")[-1])
    except (TypeError, ValueError):
        return -1


def _build_adapter_rows(phase2_rows: Path, adapter_rows: Path, target_variant: str) -> dict[str, Any]:
    adapter: list[dict[str, Any]] = []
    total_target = 0
    with phase2_rows.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") != target_variant:
                continue
            total_target += 1
            seed_type = str(row.get("seed_type") or "")
            if not seed_type.startswith("existing_mask"):
                continue
            source_mask_ids = str(row.get("source_mask_ids") or "")
            adapter.append(
                {
                    "proposal_id": row.get("proposal_id"),
                    "scene_id": row.get("scene_id"),
                    "chunk_id": row.get("chunk_id"),
                    "frame_id": row.get("frame_id"),
                    "source_mask_ids": source_mask_ids,
                    "source_mask_id": _parse_source_mask_id(source_mask_ids),
                    "source_type": seed_type,
                    "proposal_region_ref": "existing_mask_id_from_v73_p5",
                    "proposal_token_grid_shape": "1x1",
                    "proposal_token_coords": "",
                    "proposal_area_ratio": row.get("proposal_area_ratio"),
                    "proposal_bbox": row.get("bbox"),
                    "semantic_backend": row.get("semantic_backend") or "dinov2_timm",
                    "semantic_entropy": row.get("semantic_entropy"),
                    "semantic_intra_variance": row.get("interior_semantic_variance"),
                    "semantic_boundary_divergence": row.get("boundary_contrast"),
                    "semantic_prototype_id": "",
                    "semantic_prototype_margin": row.get("object_extent_score"),
                    "source_broad_large_risk": row.get("source_broad_large_risk"),
                    "source_underseg_proxy": row.get("source_underseg_proxy"),
                    "source_semantic_entropy": row.get("source_semantic_entropy"),
                    "proposal_compactness_score": row.get("object_extent_score"),
                    "proposal_background_proxy_score": row.get("background_proxy_score"),
                    "proposal_same_frame_overlap_count": "",
                    "majority_gt_id_diagnostic": row.get("majority_GT_diagnostic"),
                    "majority_iou_diagnostic": row.get("proposal_majority_IoU_diagnostic"),
                    "source_majority_iou_diagnostic": row.get("proposal_majority_IoU_diagnostic"),
                    "broad_to_subproposal_iou_gain_diagnostic": "",
                    "broad_to_subproposal_entropy_drop": "",
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_evaluation": True,
                    "diagnostic_only": True,
                    "forbidden_for_method_table": True,
                    "variant": "SP0_existing_masks_baseline",
                    "debug_json": json.dumps(
                        {
                            "adapted_from_v73_variant": target_variant,
                            "v73_seed_type": seed_type,
                            "v73_proposal_id": row.get("proposal_id"),
                        },
                        sort_keys=True,
                    ),
                }
            )
    _write_csv(adapter_rows, adapter)
    return {
        "target_variant": target_variant,
        "target_variant_proposal_count": total_target,
        "adapter_existing_subset_count": len(adapter),
        "adapter_existing_subset_rate": float(len(adapter) / max(1, total_target)),
    }


def _copy_alias(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_bytes(src.read_bytes())


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase2_rows = _rooted(args.phase2_proposal_rows)
    phase2_summary = _rooted(args.phase2_summary)
    missing: list[dict[str, Any]] = []
    for name, path in {
        "phase2_proposal_rows": phase2_rows,
        "phase2_summary": phase2_summary,
        "atom_rows": _rooted(args.atom_rows),
        "witness_summary": _rooted(args.witness_summary),
    }.items():
        if not path.exists():
            missing.append({"name": name, "path": _rel(path)})
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v73_phase3_d4rt_proposal_verification",
            "decision": "NO_GO_PHASE3_MISSING_INPUT",
            "gate": {"pass": False},
            "missing_input_count": len(missing),
        }
        _write_json(output_root / "d4rt_proposal_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    adapter_rows = output_root / "p5_existing_adapter_rows.csv"
    adapter_stats = _build_adapter_rows(phase2_rows, adapter_rows, str(args.target_variant))
    v72_args = Namespace(
        proposal_rows=str(adapter_rows),
        atom_rows=str(_rooted(args.atom_rows)),
        witness_summary=str(_rooted(args.witness_summary)),
        output_root=str(output_root),
        target_dense_variant="__v73_no_dense_adapter__",
        d4rt_weight=float(args.d4rt_weight),
        min_d4rt_score=float(args.min_d4rt_score),
        subproposal_membership_from_carrier_uv=True,
        min_carrier_visibility=float(args.min_carrier_visibility),
        min_carrier_confidence=float(args.min_carrier_confidence),
    )
    base_summary = run_v72_d4rt(v72_args)

    semantic_sf50 = _float(base_summary.get("semantic_only_SF50_diagnostic"))
    verified_sf50 = _float(base_summary.get("D4RT_verified_SF50_diagnostic"))
    shuffled_sf50 = _float(base_summary.get("shuffled_carrier_SF50_diagnostic"))
    no_temporal_sf50 = _float(base_summary.get("no_temporal_SF50_diagnostic"))
    hard_sf50 = _float(base_summary.get("D4RT_hard_filter_SF50_diagnostic"))
    hard_background_delta = _float(base_summary.get("hard_filter_background_false_positive_delta"))
    recall_drop_after_hard_filter = max(0.0, semantic_sf50 - hard_sf50)
    real_minus_shuffled = verified_sf50 - shuffled_sf50
    real_minus_no_temporal = verified_sf50 - no_temporal_sf50
    gate = {
        "adapter_existing_subset_rate_ge_0p50": _float(adapter_stats.get("adapter_existing_subset_rate")) >= 0.50,
        "D4RT_verified_SF50_ge_semantic_plus_0p03": verified_sf50 >= semantic_sf50 + 0.03,
        "real_minus_shuffled_SF50_ge_0p03": real_minus_shuffled >= 0.03,
        "real_minus_no_temporal_SF50_ge_0p02": real_minus_no_temporal >= 0.02,
        "hard_filter_background_false_positive_delta_le_minus_0p10": hard_background_delta <= -0.10,
        "recall_drop_after_hard_filter_le_0p10": recall_drop_after_hard_filter <= 0.10,
        "uses_gt_for_prediction_false": True,
    }
    gate["pass"] = bool(
        gate["adapter_existing_subset_rate_ge_0p50"]
        and gate["D4RT_verified_SF50_ge_semantic_plus_0p03"]
        and gate["real_minus_shuffled_SF50_ge_0p03"]
        and gate["real_minus_no_temporal_SF50_ge_0p02"]
        and gate["hard_filter_background_false_positive_delta_le_minus_0p10"]
        and gate["recall_drop_after_hard_filter_le_0p10"]
    )
    decision = "PASS_V73_PHASE3_D4RT_PROPOSAL_VERIFICATION" if gate["pass"] else "NO_GO_PHASE3_D4RT_FUSION_NOT_PROVEN"
    summary = dict(base_summary)
    summary.update(
        {
            "phase": "v73_phase3_d4rt_proposal_verification",
            "schema": "stream4d_v73_phase3_d4rt_proposal_verification_v1",
            "decision": decision,
            "phase2_summary": _rel(phase2_summary),
            "phase2_target_variant": str(args.target_variant),
            "adapter_stats": adapter_stats,
            "full_p5_d4rt_evaluated": bool(_float(adapter_stats.get("adapter_existing_subset_rate")) >= 0.999),
            "dense_or_merged_p5_not_evaluated_reason": "Phase2 rows do not serialize dense/merged full masks; D4RT verification is not extrapolated beyond reconstructable existing-mask P5 subset.",
            "real_minus_shuffled_SF50": real_minus_shuffled,
            "real_minus_no_temporal_SF50": real_minus_no_temporal,
            "recall_drop_after_hard_filter": recall_drop_after_hard_filter,
            "gate": gate,
            "D4RT_contribution_proven": bool(gate["pass"]),
            "D4RT_status_for_phase4": "core_contribution_allowed" if gate["pass"] else "optional_diagnostic_or_soft_veto_only",
            "can_enter_phase4_semantic_only": True,
            "can_claim_D4RT_core_contribution": bool(gate["pass"]),
            "method_boundary": {
                "training_free": True,
                "uses_gt_for_method_prediction": False,
                "gt_used_for_diagnostic_evaluation": True,
                "subproposal_membership_source": "carrier_observation_table_uv_inside_existing_P5_mask",
                "adapter_subset_only": True,
            },
        }
    )
    _write_json(output_root / "d4rt_proposal_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    _copy_alias(output_root / "control_metric_rows.csv", output_root / "metric_rows.csv")
    _copy_alias(output_root / "proposal_verification_rows.csv", output_root / "main_rows.csv")
    _write_csv(output_root / "variant_summary_rows.csv", _load_control_rows(output_root / "control_metric_rows.csv"))
    _write_sha_rows(output_root, [phase2_rows, phase2_summary, _rooted(args.atom_rows), _rooted(args.witness_summary), adapter_rows])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _load_control_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_sha_rows(output_root: Path, inputs: list[Path]) -> None:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        if path.exists() and path.is_file():
            rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "input"})
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "output"})
    _write_csv(output_root / "sha256_rows.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v73 Phase3 D4RT proposal verification.")
    parser.add_argument("--phase2-proposal-rows", default="outputs/audit/v73_phase2_semantic_extent_proposals/proposal_rows.csv")
    parser.add_argument("--phase2-summary", default="outputs/audit/v73_phase2_semantic_extent_proposals/proposal_summary.json")
    parser.add_argument("--target-variant", default="P5_boundary_and_mask_lattice_consensus")
    parser.add_argument("--atom-rows", default="outputs/audit/v71_d4rt_atoms/atom_rows.csv")
    parser.add_argument("--witness-summary", default="outputs/audit/v70_carrier_witness/witness_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v73_phase3_d4rt_proposal_verification")
    parser.add_argument("--d4rt-weight", type=float, default=0.35)
    parser.add_argument("--min-d4rt-score", type=float, default=0.20)
    parser.add_argument("--min-carrier-visibility", type=float, default=0.0)
    parser.add_argument("--min-carrier-confidence", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
