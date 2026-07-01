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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _read_variants(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["variant"]: row for row in csv.DictReader(handle)}


def _control_row(control: str, source_variant: str, row: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "scene_id": "aggregate",
        "chunk_id": "aggregate",
        "phase": "v73_phase5_local_controls",
        "variant": control,
        "source_variant": source_variant,
        "local_SF50": row.get("proposal_oracle_SF50"),
        "local_AP50": row.get("proposal_oracle_AP50"),
        "GT_best_IoU_mean": row.get("proposal_GT_best_IoU_mean"),
        "unresolved_broad_underseg_rate": row.get("unresolved_broad_underseg_rate"),
        "single_frame_object_rate": "",
        "D4RT_coverage_ratio": "",
        "background_proxy_rate": row.get("background_proxy_rate"),
        "notes": notes,
        "uses_gt_for_prediction": False,
        "uses_gt_for_evaluation": True,
        "diagnostic_only": True,
        "forbidden_for_method_table": True,
        "method_prediction_safe": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase2_variant_rows": _rooted(args.phase2_variant_rows),
        "phase3_summary": _rooted(args.phase3_summary),
        "phase4_summary": _rooted(args.phase4_summary),
    }
    missing = [{"name": name, "path": _rel(path)} for name, path in paths.items() if not path.exists()]
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {"phase": "v73_phase5_local_controls", "decision": "NO_GO_PHASE5_MISSING_INPUT", "gate": {"pass": False}, "missing_input_count": len(missing)}
        _write_json(output_root / "local_control_summary.json", summary)
        _write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    variants = _read_variants(paths["phase2_variant_rows"])
    phase3 = _load_json(paths["phase3_summary"])
    phase4 = _load_json(paths["phase4_summary"])
    p0 = variants.get("P0_existing_CropFormer_baseline", {})
    p1 = variants.get("P1_dense_token_affinity_component_v72_baseline", {})
    p2 = variants.get("P2_boundary_aware_region_grow", {})
    p3 = variants.get("P3_broad_mask_semantic_cut", {})
    p4 = variants.get("P4_multi_seed_object_extent_merge", {})
    p5 = variants.get("P5_boundary_and_mask_lattice_consensus", {})
    controls = [
        _control_row("C0_semantic_only", "P3_broad_mask_semantic_cut", p3, "semantic cut only; no existing mask lattice rescue"),
        _control_row("C2_boundary_or_mask_lattice_only", "P0_existing_CropFormer_baseline", p0, "existing mask lattice / area-heavy control"),
        _control_row("C3_semantic_plus_boundary", "P5_boundary_and_mask_lattice_consensus", p5, "final Phase2/4 proposal source"),
        _control_row("C6_area_only_control", "P0_existing_CropFormer_baseline", p0, "area/lattice-heavy control from source masks"),
        _control_row("C7_dense_affinity_control", "P1_dense_token_affinity_component_v72_baseline", p1, "v72 dense affinity baseline"),
        _control_row("C8_region_grow_control", "P2_boundary_aware_region_grow", p2, "DINO prototype grow control"),
        _control_row("C9_merge_control", "P4_multi_seed_object_extent_merge", p4, "multi-seed merge control"),
    ]
    c3 = _float(p5.get("proposal_oracle_SF50"))
    c2 = _float(p0.get("proposal_oracle_SF50"))
    c6 = _float(p0.get("proposal_oracle_SF50"))
    c0 = _float(p3.get("proposal_oracle_SF50"))
    semantic_contribution = c3 >= c2 + 0.05 and c3 >= c6 + 0.05
    boundary_contribution = c3 >= c0 + 0.05
    d4rt_contribution = bool(phase3.get("D4RT_contribution_proven"))
    area_only_gap = c3 - c6
    gate = {
        "local_gate_pass": bool((phase4.get("gate") or {}).get("pass")),
        "semantic_contribution_proven": semantic_contribution,
        "boundary_proposal_contribution_proven": boundary_contribution,
        "D4RT_contribution_proven": d4rt_contribution,
        "C3_semantic_plus_boundary_SF50": c3,
        "C2_boundary_or_mask_lattice_only_SF50": c2,
        "C6_area_only_control_SF50": c6,
        "C0_semantic_only_SF50": c0,
        "area_only_gap": area_only_gap,
        "pass": bool((phase4.get("gate") or {}).get("pass") and semantic_contribution and boundary_contribution),
    }
    if gate["pass"]:
        decision = "PASS_V73_PHASE5_LOCAL_CONTROLS"
    else:
        decision = "NO_GO_PHASE5_LOCAL_CONTROLS_AREA_LATTICE_BIAS"
    metrics = [
        {"metric": "C3_minus_C2_SF50", "value": c3 - c2, "expected": ">=0.05", "pass": c3 >= c2 + 0.05},
        {"metric": "C3_minus_C6_area_only_SF50", "value": area_only_gap, "expected": ">=0.05", "pass": c3 >= c6 + 0.05},
        {"metric": "C3_minus_C0_semantic_only_SF50", "value": c3 - c0, "expected": ">=0.05", "pass": c3 >= c0 + 0.05},
        {"metric": "D4RT_contribution_proven", "value": d4rt_contribution, "expected": "true if D4RT claimed", "pass": d4rt_contribution},
        {"metric": "phase5_pass", "value": gate["pass"], "expected": "true", "pass": gate["pass"]},
    ]
    for row in metrics:
        row.update(
            {
                "scene_id": "aggregate",
                "chunk_id": "aggregate",
                "phase": "v73_phase5_local_controls",
                "variant": "control_gate",
                "uses_gt_for_prediction": False,
                "uses_gt_for_evaluation": True,
                "diagnostic_only": True,
                "forbidden_for_method_table": True,
                "method_prediction_safe": True,
            }
        )
    _write_csv(output_root / "control_metric_rows.csv", controls)
    _write_csv(output_root / "metric_rows.csv", metrics)
    _write_csv(output_root / "main_rows.csv", controls)
    _write_csv(output_root / "variant_summary_rows.csv", controls)
    _write_csv(output_root / "missing_input_rows.csv", [])
    summary = {
        "phase": "v73_phase5_local_controls",
        "schema": "stream4d_v73_phase5_local_controls_v1",
        "decision": decision,
        "gate": gate,
        "primary_blocker": None if gate["pass"] else "AREA_LATTICE_CONTROL_BEATS_FINAL_P5",
        "secondary_blocker": None if d4rt_contribution else "D4RT_FUSION_NOT_PROVEN",
        "can_claim_semantic_contribution": semantic_contribution,
        "can_claim_boundary_contribution": boundary_contribution,
        "can_claim_D4RT_contribution": d4rt_contribution,
        "can_enter_phase6_scene_eval_as_method": bool(gate["pass"]),
        "can_enter_local2history": False,
        "notes": [
            "P5 passes local gate but does not beat P0/C6 area-lattice control.",
            "Do not claim semantic object birth contribution without repairing area/lattice bias.",
        ],
    }
    _write_json(output_root / "local_control_summary.json", summary)
    _write_json(output_root / "summary.json", summary)
    sha_rows = []
    for path in paths.values():
        sha_rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "input"})
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": "output"})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v73 Phase5 local controls.")
    parser.add_argument("--phase2-variant-rows", default="outputs/audit/v73_phase2_semantic_extent_proposals/proposal_variant_summary_rows.csv")
    parser.add_argument("--phase3-summary", default="outputs/audit/v73_phase3_d4rt_proposal_verification/d4rt_proposal_summary.json")
    parser.add_argument("--phase4-summary", default="outputs/audit/v73_phase4_local_slot_birth/local_slot_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v73_phase5_local_controls")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
