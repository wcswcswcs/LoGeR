from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUTS = {
    "v73_phase0_fact_lock": "outputs/audit/v73_phase0_fact_lock/fact_lock_summary.json",
    "v71_candidate_summary": "outputs/audit/v71_candidate_bank/candidate_bank_summary.json",
    "v71_candidate_rows": "outputs/audit/v71_candidate_bank/candidate_mask_rows.csv",
    "v71_candidate_variant_rows": "outputs/audit/v71_candidate_bank/candidate_variant_summary_rows.csv",
    "v71_semantic_summary": "outputs/audit/v71_semantic_features/semantic_summary.json",
    "v72_phase1_signal": "outputs/audit/v72_phase1_signal_adequacy/signal_adequacy_summary.json",
    "v72_dense_area_summary": "outputs/audit/v72_phase2_dense_token_proposals_smoke10_area_bin1/dense_token_proposal_summary.json",
    "v72_dense_area_variant_rows": "outputs/audit/v72_phase2_dense_token_proposals_smoke10_area_bin1/proposal_variant_summary_rows.csv",
    "v72_dense_area_rows": "outputs/audit/v72_phase2_dense_token_proposals_smoke10_area_bin1/proposal_rows.csv",
    "v72_dense_no_area_summary": "outputs/audit/v72_phase2_dense_token_proposals_smoke10_hybrid_no_area_floor/dense_token_proposal_summary.json",
    "v72_dense_no_area_variant_rows": "outputs/audit/v72_phase2_dense_token_proposals_smoke10_hybrid_no_area_floor/proposal_variant_summary_rows.csv",
    "v72_dense_no_area_rows": "outputs/audit/v72_phase2_dense_token_proposals_smoke10_hybrid_no_area_floor/proposal_rows.csv",
    "v72_semantic_proposal_summary": "outputs/audit/v72_phase2_semantic_proposals/semantic_proposal_summary.json",
    "v72_semantic_proposal_variant_rows": "outputs/audit/v72_phase2_semantic_proposals/proposal_variant_summary_rows.csv",
    "v72_sam2_relaxed_summary": "outputs/audit/v72_phase2_sam2_source_diagnostic_relaxed/sam2_source_diagnostic_summary.json",
    "v72_sam2_relaxed_rows": "outputs/audit/v72_phase2_sam2_source_diagnostic_relaxed/source_proposal_rows.csv",
}


OPTIONAL_INPUTS = {
    "v72_sam2_relaxed_summary",
    "v72_sam2_relaxed_rows",
}


ROW_DEFAULTS = {
    "scene_id": "aggregate",
    "chunk_id": "aggregate",
    "phase": "v73_phase1_source_signal_audit",
    "variant": "",
    "uses_gt_for_prediction": True,
    "uses_gt_for_evaluation": True,
    "diagnostic_only": True,
    "forbidden_for_method_table": True,
    "method_prediction_safe": False,
    "source_artifact": "",
    "score_mode": "diagnostic_source_oracle",
    "support_scope": "source_family_aggregate",
}


SOURCE_FIELDS = [
    "source_name",
    "source_family",
    "source_available",
    "source_available_for_gate",
    "coverage_scope",
    "support_chunk_count",
    "support_frame_count",
    "proposal_count_per_frame_mean",
    "proposal_count_per_chunk_mean",
    "proposal_area_ratio_mean",
    "proposal_area_ratio_p10",
    "proposal_area_ratio_p50",
    "proposal_area_ratio_p90",
    "semantic_feature_success_rate",
    "RADIO_feature_success_rate",
    "D4RT_uv_membership_available_rate",
    "source_oracle_SF50_diagnostic",
    "source_oracle_AP50_diagnostic",
    "source_GT_best_IoU_mean_diagnostic",
    "source_IoU50_coverage_rate_diagnostic",
    "background_proxy_rate",
    "broad_underseg_proxy_rate",
    "clean_objectlike_rate",
    "notes",
]


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
    for preferred in [
        "scene_id",
        "chunk_id",
        "phase",
        "variant",
        *SOURCE_FIELDS,
        "metric",
        "value",
        "expected",
        "pass",
        "uses_gt_for_prediction",
        "uses_gt_for_evaluation",
        "diagnostic_only",
        "forbidden_for_method_table",
        "method_prediction_safe",
        "source_artifact",
        "score_mode",
        "support_scope",
    ]:
        if any(preferred in row for row in rows) and preferred not in fields:
            fields.append(preferred)
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
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _quantiles(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    vals = sorted(values)

    def pick(q: float) -> float:
        idx = int(round((len(vals) - 1) * q))
        idx = max(0, min(len(vals) - 1, idx))
        return float(vals[idx])

    return pick(0.10), pick(0.50), pick(0.90)


def _mean(values: list[float]) -> float | None:
    valid = [value for value in values if math.isfinite(value)]
    return float(sum(valid) / len(valid)) if valid else None


def _row_base(variant: str, source_artifact: str, support_scope: str = "source_family_aggregate") -> dict[str, Any]:
    row = dict(ROW_DEFAULTS)
    row.update({"variant": variant, "source_artifact": source_artifact, "support_scope": support_scope})
    return row


def _variant_row(rows: list[dict[str, str]], variant: str) -> dict[str, str]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return {}


def _area_stats_from_rows(
    path: Path,
    area_field: str,
    variant: str | None = None,
    variants: set[str] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    areas: list[float] = []
    row_count = 0
    frame_keys: set[tuple[str, str]] = set()
    chunk_keys: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if variant is not None and row.get("variant") != variant:
                continue
            if variants is not None and row.get("variant") not in variants:
                continue
            if source is not None and row.get("source") != source:
                continue
            area = _float(row.get(area_field))
            if area is not None:
                areas.append(area)
            row_count += 1
            scene = str(row.get("scene_id") or "")
            frame = str(row.get("frame_id") or "")
            if scene and frame:
                frame_keys.add((scene, frame))
            chunk = str(row.get("chunk_id") or "")
            if chunk:
                chunk_keys.add(chunk)
    p10, p50, p90 = _quantiles(areas)
    return {
        "row_count": row_count,
        "area_mean": _mean(areas),
        "area_p10": p10,
        "area_p50": p50,
        "area_p90": p90,
        "frame_count": len(frame_keys) or None,
        "chunk_count": len(chunk_keys) or None,
    }


def _candidate_area_stats(path: Path) -> dict[str, Any]:
    areas: list[float] = []
    frame_keys: set[tuple[str, str]] = set()
    chunk_keys: set[str] = set()
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            area = _float(row.get("area_ratio"))
            if area is not None:
                areas.append(area)
            count += 1
            scene = str(row.get("scene_id") or "")
            frame = str(row.get("frame_id") or "")
            if scene and frame:
                frame_keys.add((scene, frame))
            chunk = str(row.get("chunk_id") or "")
            if chunk:
                chunk_keys.add(chunk)
    p10, p50, p90 = _quantiles(areas)
    return {
        "row_count": count,
        "area_mean": _mean(areas),
        "area_p10": p10,
        "area_p50": p50,
        "area_p90": p90,
        "frame_count": len(frame_keys) or None,
        "chunk_count": len(chunk_keys) or None,
    }


def _source_row(
    *,
    source_name: str,
    source_family: str,
    source_artifact: str,
    coverage_scope: str,
    source_available: bool = True,
    source_available_for_gate: bool = True,
    support_chunk_count: Any = None,
    support_frame_count: Any = None,
    proposal_count_per_frame_mean: Any = None,
    proposal_count_per_chunk_mean: Any = None,
    proposal_area_ratio_mean: Any = None,
    proposal_area_ratio_p10: Any = None,
    proposal_area_ratio_p50: Any = None,
    proposal_area_ratio_p90: Any = None,
    semantic_feature_success_rate: Any = None,
    radio_feature_success_rate: Any = None,
    d4rt_uv_membership_available_rate: Any = None,
    source_oracle_sf50: Any = None,
    source_oracle_ap50: Any = None,
    source_gt_best_iou_mean: Any = None,
    source_iou50_coverage_rate: Any = None,
    background_proxy_rate: Any = None,
    broad_underseg_proxy_rate: Any = None,
    clean_objectlike_rate: Any = None,
    notes: str = "",
) -> dict[str, Any]:
    row = _row_base(source_name, source_artifact, coverage_scope)
    row.update(
        {
            "source_name": source_name,
            "source_family": source_family,
            "source_available": source_available,
            "source_available_for_gate": source_available_for_gate,
            "coverage_scope": coverage_scope,
            "support_chunk_count": support_chunk_count,
            "support_frame_count": support_frame_count,
            "proposal_count_per_frame_mean": proposal_count_per_frame_mean,
            "proposal_count_per_chunk_mean": proposal_count_per_chunk_mean,
            "proposal_area_ratio_mean": proposal_area_ratio_mean,
            "proposal_area_ratio_p10": proposal_area_ratio_p10,
            "proposal_area_ratio_p50": proposal_area_ratio_p50,
            "proposal_area_ratio_p90": proposal_area_ratio_p90,
            "semantic_feature_success_rate": semantic_feature_success_rate,
            "RADIO_feature_success_rate": radio_feature_success_rate,
            "D4RT_uv_membership_available_rate": d4rt_uv_membership_available_rate,
            "source_oracle_SF50_diagnostic": source_oracle_sf50,
            "source_oracle_AP50_diagnostic": source_oracle_ap50,
            "source_GT_best_IoU_mean_diagnostic": source_gt_best_iou_mean,
            "source_IoU50_coverage_rate_diagnostic": source_iou50_coverage_rate,
            "background_proxy_rate": background_proxy_rate,
            "broad_underseg_proxy_rate": broad_underseg_proxy_rate,
            "clean_objectlike_rate": clean_objectlike_rate,
            "notes": notes,
        }
    )
    return row


def _metric_row(metric: str, value: Any, expected: str, passed: bool | None, source_artifact: str, notes: str = "") -> dict[str, Any]:
    row = _row_base("phase1_gate", source_artifact, "phase1_gate")
    row.update({"metric": metric, "value": value, "expected": expected, "pass": passed, "notes": notes})
    return row


def _input_rows(loaded_paths: dict[str, Path], missing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_names = {str(row.get("input_name")) for row in missing}
    for name, rel_path in DEFAULT_INPUTS.items():
        path = _rooted(rel_path)
        required = name not in OPTIONAL_INPUTS
        row = _row_base("input_presence", rel_path, "input_presence")
        present = path.exists()
        row.update(
            {
                "source_name": name,
                "source_available": present,
                "source_available_for_gate": required,
                "input_name": name,
                "required": required,
                "path": rel_path,
                "bytes": path.stat().st_size if present else None,
                "sha256": _sha256(path) if present else None,
                "pass": present or not required,
                "notes": "optional input missing" if name in missing_names and not required else "",
            }
        )
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    missing: list[dict[str, Any]] = []
    loaded_json: dict[str, dict[str, Any]] = {}
    loaded_csv: dict[str, list[dict[str, str]]] = {}
    loaded_paths: dict[str, Path] = {}
    for name, rel_path in DEFAULT_INPUTS.items():
        path = _rooted(rel_path)
        if not path.exists():
            missing.append(
                {
                    "scene_id": "aggregate",
                    "chunk_id": "aggregate",
                    "phase": "v73_phase1_source_signal_audit",
                    "variant": "input_presence",
                    "input_name": name,
                    "path": rel_path,
                    "required": name not in OPTIONAL_INPUTS,
                    "notes": "optional alternative source" if name in OPTIONAL_INPUTS else "required input missing",
                }
            )
            continue
        loaded_paths[name] = path
        if path.suffix == ".json":
            loaded_json[name] = _load_json(path)
        elif path.suffix == ".csv" and "rows" in name and not name.endswith("_rows"):
            loaded_csv[name] = _load_csv(path)
        elif path.suffix == ".csv" and "variant_rows" in name:
            loaded_csv[name] = _load_csv(path)

    required_missing = [row for row in missing if _bool(row.get("required"))]
    if required_missing:
        source_rows = _input_rows(loaded_paths, missing)
        summary = {
            "phase": "v73_phase1_source_signal_audit",
            "schema": "stream4d_v73_phase1_source_signal_audit_v1",
            "decision": "NO_GO_PHASE1_MISSING_INPUT",
            "missing_required_input_count": len(required_missing),
            "gate": {"all_required_inputs_present": False, "pass": False},
        }
        _write_csv(output_root / "missing_input_rows.csv", missing)
        _write_csv(output_root / "source_rows.csv", source_rows)
        _write_csv(output_root / "main_rows.csv", source_rows)
        _write_csv(output_root / "source_metric_rows.csv", [])
        _write_csv(output_root / "metric_rows.csv", [])
        _write_csv(output_root / "variant_summary_rows.csv", [])
        _write_json(output_root / "summary.json", summary)
        _write_json(output_root / "source_signal_summary.json", summary)
        _write_sha_rows(output_root, DEFAULT_INPUTS)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    phase0 = loaded_json["v73_phase0_fact_lock"]
    v71_candidate = loaded_json["v71_candidate_summary"]
    v71_semantic = loaded_json["v71_semantic_summary"]
    v72_signal = loaded_json["v72_phase1_signal"]
    v72_dense_area = loaded_json["v72_dense_area_summary"]
    v72_dense_no_area = loaded_json["v72_dense_no_area_summary"]
    v72_semantic_proposal = loaded_json["v72_semantic_proposal_summary"]
    sam2_summary = loaded_json.get("v72_sam2_relaxed_summary", {})
    v71_variants = _load_csv(_rooted(DEFAULT_INPUTS["v71_candidate_variant_rows"]))
    dense_area_variants = _load_csv(_rooted(DEFAULT_INPUTS["v72_dense_area_variant_rows"]))
    dense_no_area_variants = _load_csv(_rooted(DEFAULT_INPUTS["v72_dense_no_area_variant_rows"]))
    semantic_variants = _load_csv(_rooted(DEFAULT_INPUTS["v72_semantic_proposal_variant_rows"]))

    candidate_stats = _candidate_area_stats(_rooted(DEFAULT_INPUTS["v71_candidate_rows"]))
    dense_area_variant = str(v72_dense_area.get("best_dense_variant") or "SP7_existing_plus_SP2_DINO_affinity_connected_components")
    dense_no_area_variant = str(v72_dense_no_area.get("best_dense_variant") or "SP7_existing_plus_SP2_DINO_affinity_connected_components")
    dense_area_row = _variant_row(dense_area_variants, dense_area_variant)
    dense_no_area_row = _variant_row(dense_no_area_variants, dense_no_area_variant)
    semantic_best_variant = str(v72_semantic_proposal.get("best_variant") or "SP0_existing_masks_baseline")
    semantic_row = _variant_row(semantic_variants, semantic_best_variant)
    dense_sp7_components = {"SP0_existing_masks_baseline", "SP2_DINO_affinity_connected_components"}
    dense_area_stats = _area_stats_from_rows(
        _rooted(DEFAULT_INPUTS["v72_dense_area_rows"]), "proposal_area_ratio", variants=dense_sp7_components
    )
    dense_no_area_stats = _area_stats_from_rows(
        _rooted(DEFAULT_INPUTS["v72_dense_no_area_rows"]), "proposal_area_ratio", variants=dense_sp7_components
    )
    sam2_stats = (
        _area_stats_from_rows(_rooted(DEFAULT_INPUTS["v72_sam2_relaxed_rows"]), "proposal_area_ratio", source="sam2_filtered_npz")
        if "v72_sam2_relaxed_rows" in loaded_paths
        else {}
    )

    v71_metrics = v71_candidate.get("key_metrics") or {}
    v71_c6 = _variant_row(v71_variants, "C6_union_candidate_bank")
    semantic_metrics = v71_semantic.get("key_metrics") or {}
    signal_metrics = v72_signal.get("key_metrics") or {}

    radio_rate = 0.0 if _bool(semantic_metrics.get("RADIO_unavailable")) or _bool(v72_signal.get("RADIO_unavailable")) else None
    source_rows: list[dict[str, Any]] = [
        _source_row(
            source_name="SRC0_cropformer_flat_candidate_bank_C6",
            source_family="SRC0_CropFormer_flat_masks",
            source_artifact=DEFAULT_INPUTS["v71_candidate_summary"],
            coverage_scope="full_v71_candidate_bank_157_chunks",
            support_chunk_count=v71_c6.get("chunk_count"),
            support_frame_count=candidate_stats.get("frame_count"),
            proposal_count_per_frame_mean=v71_metrics.get("candidate_count_per_frame_mean"),
            proposal_count_per_chunk_mean=v71_metrics.get("candidate_count_per_chunk_mean"),
            proposal_area_ratio_mean=candidate_stats.get("area_mean"),
            proposal_area_ratio_p10=candidate_stats.get("area_p10"),
            proposal_area_ratio_p50=candidate_stats.get("area_p50"),
            proposal_area_ratio_p90=candidate_stats.get("area_p90"),
            semantic_feature_success_rate=v71_metrics.get("semantic_feature_success_rate"),
            radio_feature_success_rate=radio_rate,
            d4rt_uv_membership_available_rate=v71_metrics.get("D4RT_supported_candidate_rate"),
            source_oracle_sf50=v71_metrics.get("C6_union_candidate_bank_oracle_SF50"),
            source_oracle_ap50=v71_metrics.get("C6_union_candidate_bank_AP50"),
            source_gt_best_iou_mean=v71_metrics.get("C6_union_candidate_bank_GT_best_IoU_mean"),
            source_iou50_coverage_rate=_float(v71_c6.get("local_score_free_match50_recall_mean")),
            background_proxy_rate=v71_metrics.get("broad_background_risk_rate"),
            broad_underseg_proxy_rate=max(
                _float(v71_metrics.get("broad_background_risk_rate")) or 0.0,
                _float(v71_metrics.get("underseg_proxy_rate")) or 0.0,
            ),
            clean_objectlike_rate=max(
                0.0,
                1.0
                - max(
                    _float(v71_metrics.get("broad_background_risk_rate")) or 0.0,
                    _float(v71_metrics.get("underseg_proxy_rate")) or 0.0,
                ),
            ),
            notes="Full candidate-bank oracle from v71; diagnostic GT oracle only, not method success.",
        ),
        _source_row(
            source_name="SRC1_v72_dense_DINO_area_bin1_SP7",
            source_family="SRC1_existing_v72_dense_DINO_token_proposals",
            source_artifact=DEFAULT_INPUTS["v72_dense_area_summary"],
            coverage_scope="v72_smoke10_dense_area_bin1",
            support_chunk_count=dense_area_row.get("chunk_count"),
            support_frame_count=v72_dense_area.get("dense_frame_count"),
            proposal_count_per_frame_mean=dense_area_row.get("proposal_count_per_frame_mean"),
            proposal_count_per_chunk_mean=dense_area_row.get("proposal_count_per_chunk_mean"),
            proposal_area_ratio_mean=dense_area_stats.get("area_mean"),
            proposal_area_ratio_p10=dense_area_stats.get("area_p10"),
            proposal_area_ratio_p50=dense_area_stats.get("area_p50"),
            proposal_area_ratio_p90=dense_area_stats.get("area_p90"),
            semantic_feature_success_rate=v71_metrics.get("semantic_feature_success_rate"),
            radio_feature_success_rate=radio_rate,
            d4rt_uv_membership_available_rate=0.0,
            source_oracle_sf50=dense_area_row.get("proposal_oracle_SF50_mean"),
            source_oracle_ap50=dense_area_row.get("proposal_oracle_AP50_mean"),
            source_gt_best_iou_mean=dense_area_row.get("proposal_oracle_GT_best_IoU_mean"),
            source_iou50_coverage_rate=dense_area_row.get("proposal_IoU50_rate_mean"),
            background_proxy_rate=dense_area_row.get("proposal_background_proxy_rate_mean"),
            broad_underseg_proxy_rate=dense_area_row.get("proposal_source_broad_rate_mean"),
            clean_objectlike_rate=max(0.0, 1.0 - (_float(dense_area_row.get("proposal_source_broad_rate_mean")) or 0.0)),
            notes="Imported v72 dense-token smoke10 diagnostic; generated before v73 Phase2, no D4RT verification.",
        ),
        _source_row(
            source_name="SRC4_v72_hybrid_no_area_floor_SP7",
            source_family="SRC4_hybrid_CropFormer_plus_dense_token",
            source_artifact=DEFAULT_INPUTS["v72_dense_no_area_summary"],
            coverage_scope="v72_smoke10_hybrid_no_area_floor",
            support_chunk_count=dense_no_area_row.get("chunk_count"),
            support_frame_count=v72_dense_no_area.get("dense_frame_count"),
            proposal_count_per_frame_mean=dense_no_area_row.get("proposal_count_per_frame_mean"),
            proposal_count_per_chunk_mean=dense_no_area_row.get("proposal_count_per_chunk_mean"),
            proposal_area_ratio_mean=dense_no_area_stats.get("area_mean"),
            proposal_area_ratio_p10=dense_no_area_stats.get("area_p10"),
            proposal_area_ratio_p50=dense_no_area_stats.get("area_p50"),
            proposal_area_ratio_p90=dense_no_area_stats.get("area_p90"),
            semantic_feature_success_rate=v71_metrics.get("semantic_feature_success_rate"),
            radio_feature_success_rate=radio_rate,
            d4rt_uv_membership_available_rate=0.0,
            source_oracle_sf50=dense_no_area_row.get("proposal_oracle_SF50_mean"),
            source_oracle_ap50=dense_no_area_row.get("proposal_oracle_AP50_mean"),
            source_gt_best_iou_mean=dense_no_area_row.get("proposal_oracle_GT_best_IoU_mean"),
            source_iou50_coverage_rate=dense_no_area_row.get("proposal_IoU50_rate_mean"),
            background_proxy_rate=dense_no_area_row.get("proposal_background_proxy_rate_mean"),
            broad_underseg_proxy_rate=dense_no_area_row.get("proposal_source_broad_rate_mean"),
            clean_objectlike_rate=max(0.0, 1.0 - (_float(dense_no_area_row.get("proposal_source_broad_rate_mean")) or 0.0)),
            notes="Hybrid existing+dense smoke10 source; imported as prior diagnostic, not v73 object slot output.",
        ),
        _source_row(
            source_name="SRC4_v72_mask_level_semantic_fallback_SP0",
            source_family="SRC4_mask_level_semantic_fallback",
            source_artifact=DEFAULT_INPUTS["v72_semantic_proposal_summary"],
            coverage_scope="full_v72_mask_level_fallback_157_chunks",
            support_chunk_count=semantic_row.get("chunk_count"),
            support_frame_count=candidate_stats.get("frame_count"),
            proposal_count_per_frame_mean=semantic_row.get("proposal_count_per_frame_mean"),
            proposal_count_per_chunk_mean=semantic_row.get("proposal_count_per_chunk_mean"),
            proposal_area_ratio_mean=candidate_stats.get("area_mean"),
            proposal_area_ratio_p10=candidate_stats.get("area_p10"),
            proposal_area_ratio_p50=candidate_stats.get("area_p50"),
            proposal_area_ratio_p90=candidate_stats.get("area_p90"),
            semantic_feature_success_rate=v71_metrics.get("semantic_feature_success_rate"),
            radio_feature_success_rate=radio_rate,
            d4rt_uv_membership_available_rate=v71_metrics.get("D4RT_supported_candidate_rate"),
            source_oracle_sf50=semantic_row.get("proposal_oracle_SF50_mean"),
            source_oracle_ap50=semantic_row.get("proposal_oracle_AP50_mean"),
            source_gt_best_iou_mean=semantic_row.get("proposal_GT_best_IoU_mean"),
            source_iou50_coverage_rate=semantic_row.get("proposal_IoU50_rate_mean"),
            background_proxy_rate=semantic_row.get("proposal_background_proxy_rate_mean"),
            broad_underseg_proxy_rate=semantic_row.get("proposal_source_broad_rate_mean"),
            clean_objectlike_rate=max(0.0, 1.0 - (_float(semantic_row.get("proposal_source_broad_rate_mean")) or 0.0)),
            notes="v72 full mask-level fallback; useful as source adequacy evidence but failed as proposal generator.",
        ),
        _source_row(
            source_name="SRC2_boundary_aware_semantic_token_proposals",
            source_family="SRC2_boundary_aware_semantic_token_proposals",
            source_artifact="not_generated_until_v73_phase2",
            coverage_scope="not_available_until_phase2",
            source_available=False,
            source_available_for_gate=False,
            notes="Phase1 records this planned source family as unavailable until Phase2 generation.",
        ),
    ]

    if sam2_summary:
        sam2_block = sam2_summary.get("sam2_filtered_npz") or {}
        source_rows.append(
            _source_row(
                source_name="SRC3_sam2_filtered_npz_relaxed_diagnostic",
                source_family="SRC3_optional_SAM2_alternative",
                source_artifact=DEFAULT_INPUTS["v72_sam2_relaxed_summary"],
                coverage_scope="optional_v51_relaxed_20_frame_diagnostic_not_full_v73_source",
                source_available=True,
                source_available_for_gate=False,
                support_chunk_count=None,
                support_frame_count=sam2_block.get("frame_count") or sam2_stats.get("frame_count"),
                proposal_count_per_frame_mean=sam2_block.get("proposal_count_per_frame_mean"),
                proposal_count_per_chunk_mean=None,
                proposal_area_ratio_mean=sam2_block.get("proposal_area_ratio_mean") or sam2_stats.get("area_mean"),
                proposal_area_ratio_p10=sam2_stats.get("area_p10"),
                proposal_area_ratio_p50=sam2_stats.get("area_p50"),
                proposal_area_ratio_p90=sam2_stats.get("area_p90"),
                semantic_feature_success_rate=None,
                radio_feature_success_rate=radio_rate,
                d4rt_uv_membership_available_rate=0.0,
                source_oracle_sf50=sam2_block.get("source_oracle_SF50_diagnostic"),
                source_oracle_ap50=sam2_block.get("source_oracle_AP50_diagnostic"),
                source_gt_best_iou_mean=sam2_block.get("source_oracle_GT_best_IoU_mean_diagnostic"),
                source_iou50_coverage_rate=sam2_block.get("gt_IoU50_coverage_rate_diagnostic"),
                background_proxy_rate=None,
                broad_underseg_proxy_rate=None,
                clean_objectlike_rate=None,
                notes=str(sam2_summary.get("cannot_replace_reason") or "Optional SAM2 diagnostic imported; not full source replacement."),
            )
        )

    gate_candidates = [row for row in source_rows if _bool(row.get("source_available_for_gate")) and _float(row.get("source_oracle_SF50_diagnostic")) is not None]
    best_source = max(gate_candidates, key=lambda row: _float(row.get("source_oracle_SF50_diagnostic")) or -1.0)
    best_source_oracle = _float(best_source.get("source_oracle_SF50_diagnostic"))
    best_source_gt = _float(best_source.get("source_GT_best_IoU_mean_diagnostic"))
    best_source_count = _float(best_source.get("proposal_count_per_frame_mean"))
    best_source_semantic_success = _float(best_source.get("semantic_feature_success_rate"))
    best_source_risk = _float(best_source.get("broad_underseg_proxy_rate"))
    area_auc = _float(signal_metrics.get("area_only_control_AUC_iou50"))
    semantic_auc = _float(signal_metrics.get("best_semantic_AUC_iou50"))

    gate = {
        "all_required_inputs_present": True,
        "phase0_passed": phase0.get("decision") == "PASS_V73_PHASE0_FACT_LOCK",
        "best_source_name": best_source.get("source_name"),
        "best_source_oracle_SF50": best_source_oracle,
        "best_source_GT_best_IoU_mean": best_source_gt,
        "best_source_proposal_count_per_frame_mean": best_source_count,
        "best_source_semantic_feature_success_rate": best_source_semantic_success,
        "best_source_oracle_SF50_ge_0p50": best_source_oracle is not None and best_source_oracle >= 0.50,
        "best_source_GT_best_IoU_mean_ge_0p35": best_source_gt is not None and best_source_gt >= 0.35,
        "best_source_proposal_count_per_frame_mean_le_200": best_source_count is not None and best_source_count <= 200.0,
        "best_source_semantic_feature_success_rate_ge_0p95": best_source_semantic_success is not None and best_source_semantic_success >= 0.95,
        "best_source_broad_underseg_proxy_rate_gt_0p60": best_source_risk is not None and best_source_risk > 0.60,
        "area_only_control_AUC_iou50": area_auc,
        "best_semantic_AUC_iou50": semantic_auc,
        "area_only_control_AUC_gt_best_semantic_plus_0p05": area_auc is not None and semantic_auc is not None and area_auc > semantic_auc + 0.05,
        "SRC2_boundary_aware_source_available": False,
        "SAM2_optional_full_source_replacement_available": bool(sam2_summary.get("can_replace_v72_phase2_full_source")),
    }
    gate["source_safety_warning"] = bool(
        gate["best_source_broad_underseg_proxy_rate_gt_0p60"] or gate["area_only_control_AUC_gt_best_semantic_plus_0p05"]
    )
    gate["pass"] = bool(
        gate["phase0_passed"]
        and gate["best_source_oracle_SF50_ge_0p50"]
        and gate["best_source_GT_best_IoU_mean_ge_0p35"]
        and gate["best_source_proposal_count_per_frame_mean_le_200"]
        and gate["best_source_semantic_feature_success_rate_ge_0p95"]
    )

    decision = "PASS_V73_PHASE1_SOURCE_HEADROOM_WITH_SAFETY_WARNING" if gate["pass"] and gate["source_safety_warning"] else ""
    if gate["pass"] and not gate["source_safety_warning"]:
        decision = "PASS_V73_PHASE1_SOURCE_ADEQUACY"
    if not gate["pass"]:
        decision = "NO_GO_PHASE1_SOURCE_SIGNAL_INSUFFICIENT"

    metric_rows = [
        _metric_row("phase0_passed", gate["phase0_passed"], "true", bool(gate["phase0_passed"]), DEFAULT_INPUTS["v73_phase0_fact_lock"]),
        _metric_row("best_source_name", gate["best_source_name"], "record", None, best_source.get("source_artifact", "")),
        _metric_row("best_source_oracle_SF50", best_source_oracle, ">=0.50", bool(gate["best_source_oracle_SF50_ge_0p50"]), best_source.get("source_artifact", "")),
        _metric_row("best_source_GT_best_IoU_mean", best_source_gt, ">=0.35", bool(gate["best_source_GT_best_IoU_mean_ge_0p35"]), best_source.get("source_artifact", "")),
        _metric_row("best_source_proposal_count_per_frame_mean", best_source_count, "<=200", bool(gate["best_source_proposal_count_per_frame_mean_le_200"]), best_source.get("source_artifact", "")),
        _metric_row("best_source_semantic_feature_success_rate", best_source_semantic_success, ">=0.95", bool(gate["best_source_semantic_feature_success_rate_ge_0p95"]), best_source.get("source_artifact", "")),
        _metric_row("best_source_broad_underseg_proxy_rate", best_source_risk, "warning if >0.60", not bool(gate["best_source_broad_underseg_proxy_rate_gt_0p60"]), best_source.get("source_artifact", "")),
        _metric_row("area_only_control_AUC_iou50", area_auc, "warning if > best_semantic_AUC_iou50+0.05", None, DEFAULT_INPUTS["v72_phase1_signal"]),
        _metric_row("best_semantic_AUC_iou50", semantic_auc, "record", None, DEFAULT_INPUTS["v72_phase1_signal"]),
        _metric_row("area_only_control_AUC_gt_best_semantic_plus_0p05", gate["area_only_control_AUC_gt_best_semantic_plus_0p05"], "false for no warning", not bool(gate["area_only_control_AUC_gt_best_semantic_plus_0p05"]), DEFAULT_INPUTS["v72_phase1_signal"]),
        _metric_row("source_safety_warning", gate["source_safety_warning"], "record", None, "phase1_gate"),
        _metric_row("can_enter_phase2", gate["pass"], "true", bool(gate["pass"]), "phase1_gate"),
    ]

    summary = {
        "phase": "v73_phase1_source_signal_audit",
        "schema": "stream4d_v73_phase1_source_signal_audit_v1",
        "decision": decision,
        "inputs": DEFAULT_INPUTS,
        "gate": gate,
        "best_source": {
            key: best_source.get(key)
            for key in [
                "source_name",
                "coverage_scope",
                "proposal_count_per_frame_mean",
                "proposal_count_per_chunk_mean",
                "semantic_feature_success_rate",
                "RADIO_feature_success_rate",
                "D4RT_uv_membership_available_rate",
                "source_oracle_SF50_diagnostic",
                "source_oracle_AP50_diagnostic",
                "source_GT_best_IoU_mean_diagnostic",
                "source_IoU50_coverage_rate_diagnostic",
                "background_proxy_rate",
                "broad_underseg_proxy_rate",
                "clean_objectlike_rate",
            ]
        },
        "can_enter_phase2_boundary_aware_generation": gate["pass"],
        "can_enter_phase4_local_slot_birth": False,
        "source_has_headroom_but_needs_object_extent_decomposition": bool(gate["pass"] and gate["source_safety_warning"]),
        "method_boundary": {
            "training_free": True,
            "uses_gt_for_method_prediction": False,
            "oracle_rows_forbidden_for_method_table": True,
            "gt_used_only_for_source_adequacy_diagnostic": True,
            "RADIO_unavailable": bool(radio_rate == 0.0),
        },
        "notes": [
            "Phase1 imports prior source/proposal diagnostics and does not generate v73 local object slots.",
            "Source oracle metrics are diagnostic only and forbidden for method tables.",
            "SRC2 boundary-aware semantic token proposals are recorded as unavailable until v73 Phase2.",
            "The gate passes on source headroom, but safety warnings indicate object-extent decomposition is still required before local slot birth.",
        ],
    }

    _write_csv(output_root / "source_rows.csv", source_rows)
    _write_csv(output_root / "source_metric_rows.csv", metric_rows)
    _write_csv(output_root / "main_rows.csv", source_rows)
    _write_csv(output_root / "metric_rows.csv", metric_rows)
    _write_csv(output_root / "variant_summary_rows.csv", source_rows)
    _write_csv(output_root / "missing_input_rows.csv", missing)
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "source_signal_summary.json", summary)
    _write_sha_rows(output_root, DEFAULT_INPUTS)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _write_sha_rows(output_root: Path, inputs: dict[str, str]) -> None:
    rows: list[dict[str, Any]] = []
    for name, rel_path in inputs.items():
        path = _rooted(rel_path)
        if path.exists():
            row = _row_base("sha256", _rel(path), "sha256")
            row.update({"name": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
            rows.append(row)
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            row = _row_base("sha256", _rel(path), "sha256")
            row.update({"name": f"output:{path.name}", "bytes": path.stat().st_size, "sha256": _sha256(path)})
            rows.append(row)
    _write_csv(output_root / "sha256_rows.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v73 Phase1 source and signal sufficiency audit.")
    parser.add_argument("--output-root", default="outputs/audit/v73_phase1_source_signal_audit")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
