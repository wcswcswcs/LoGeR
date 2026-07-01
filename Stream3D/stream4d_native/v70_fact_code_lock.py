from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v65_soma_pipeline_visualization import _rel  # noqa: E402


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_first(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return dict(row)
    return {}


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _scan_material_closure_source(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": _rel(path),
            "source_available": False,
            "code_reads_carrier_cache_npz": "unknown",
            "code_reads_uv_tracks": "unknown",
            "code_reads_visibility": "unknown",
            "code_computes_R_in_R_out": "unknown",
            "code_uses_component_overlap_proxy": "unknown",
            "code_uses_v68_edge_rows": "unknown",
            "code_uses_tracklet_index": "unknown",
        }
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    reads_npz = "np.load" in text or "carrier_cache" in lowered or ".npz" in lowered
    reads_uv = "uv" in lowered or "projection" in lowered or "project" in lowered
    reads_visibility = "visibility_mean" in text or "carrier_observation_table" in lowered or "visible_carrier" in lowered
    computes_ratios = "inside_ratio" in text and "outside_ratio" in text and "carrier" in lowered
    uses_component = "component_index" in text or "component_jaccard" in text or "component_intersection_count" in text
    uses_edges = "edge_rows" in text or "v68_edge_audit" in text
    uses_tracklet = "tracklet_index" in text or "_tracklet_edges_for_chunk" in text
    return {
        "path": _rel(path),
        "source_available": True,
        "line_count": len(text.splitlines()),
        "code_reads_carrier_cache_npz": reads_npz,
        "code_reads_uv_tracks": reads_uv,
        "code_reads_visibility": reads_visibility,
        "code_computes_R_in_R_out": computes_ratios,
        "code_uses_component_overlap_proxy": uses_component,
        "code_uses_v68_edge_rows": uses_edges,
        "code_uses_tracklet_index": uses_tracklet,
        "scan_basis": "static token scan; uncertain fields are conservative booleans, not execution proof",
    }


def _metric_rows(final: dict[str, Any], anchor: dict[str, Any], material: dict[str, Any]) -> list[dict[str, Any]]:
    final_best_anchor = final.get("selected_final_attempt") or {}
    final_best_material = final.get("best_material_attempt") or {}
    anchor_best = anchor.get("best_anchor_variant") or {}
    material_best = material.get("best_closure_variant") or {}
    return [
        {
            "metric": "v69r2_decision",
            "value": final.get("decision"),
            "source": "v69r2_final_decision",
        },
        {
            "metric": "phase0_v68_fact_lock",
            "value": (final.get("phase_status") or {}).get("phase0_v68_fact_lock"),
            "source": "v69r2_final_decision",
        },
        {
            "metric": "phase1_anchor_bank",
            "value": (final.get("phase_status") or {}).get("phase1_anchor_bank"),
            "source": "v69r2_final_decision",
        },
        {
            "metric": "phase2_material_closure",
            "value": (final.get("phase_status") or {}).get("phase2_material_closure"),
            "source": "v69r2_final_decision",
        },
        {
            "metric": "best_anchor_variant",
            "value": final_best_anchor.get("best_anchor_variant") or anchor_best.get("anchor_variant"),
            "source": "v69r2_anchor_bank",
        },
        {
            "metric": "best_anchor_SF50",
            "value": final_best_anchor.get("anchor_oracle_SF50") or anchor_best.get("anchor_oracle_SF50"),
            "source": "v69r2_anchor_bank",
        },
        {
            "metric": "best_anchor_underseg_rate",
            "value": final_best_anchor.get("anchor_underseg_rate") or anchor_best.get("anchor_underseg_rate"),
            "source": "v69r2_anchor_bank",
        },
        {
            "metric": "best_material_variant",
            "value": final_best_material.get("best_closure_variant") or material_best.get("closure_variant"),
            "source": "v69r2_material_closure",
        },
        {
            "metric": "best_material_SF50",
            "value": final_best_material.get("single_anchor_SF50") or material_best.get("single_anchor_SF50"),
            "source": "v69r2_material_closure",
        },
        {
            "metric": "best_material_AP50",
            "value": final_best_material.get("single_anchor_AP50") or material_best.get("single_anchor_AP50"),
            "source": "v69r2_material_closure",
        },
        {
            "metric": "best_material_GT_best_IoU",
            "value": final_best_material.get("single_anchor_GT_best_IoU_mean") or material_best.get("single_anchor_GT_best_IoU_mean"),
            "source": "v69r2_material_closure",
        },
        {
            "metric": "best_material_single_frame_rate",
            "value": final_best_material.get("single_anchor_single_frame_rate") or material_best.get("single_anchor_single_frame_rate"),
            "source": "v69r2_material_closure",
        },
        {
            "metric": "best_material_real_minus_shuffled",
            "value": final_best_material.get("real_minus_shuffled_SF50") or material.get("real_minus_shuffled_SF50"),
            "source": "v69r2_material_closure",
        },
        {
            "metric": "best_material_real_minus_no_temporal",
            "value": final_best_material.get("real_minus_no_temporal_SF50") or material.get("real_minus_no_temporal_SF50"),
            "source": "v69r2_material_closure",
        },
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    final_path = _rooted(args.final_decision)
    anchor_path = _rooted(args.anchor_summary)
    material_path = _rooted(args.material_summary)
    material_metric_path = _rooted(args.material_metric_rows)
    source_path = _rooted(args.material_closure_source)
    final = _load_json(final_path)
    anchor = _load_json(anchor_path)
    material = _load_json(material_path)
    material_metric = _load_csv_first(material_metric_path)
    code_audit = _scan_material_closure_source(source_path)
    metric_rows = _metric_rows(final, anchor, material)
    code_rows = [
        {
            "field": key,
            "value": value,
            "source": code_audit["path"],
            "audit_type": "static_source_scan",
        }
        for key, value in code_audit.items()
        if key != "path"
    ]
    _write_csv(output_root / "v69r2_metric_rows.csv", metric_rows)
    _write_csv(output_root / "code_audit_rows.csv", code_rows)

    gate = final.get("gate") or {}
    phase_status = final.get("phase_status") or {}
    phase0_pass = (
        final.get("decision") == "NO_GO_MATERIAL_CLOSURE"
        and bool(gate.get("phase1_anchor_bank_pass"))
        and not bool(gate.get("phase2_material_closure_pass"))
        and phase_status.get("phase2_material_closure") == "NO_GO_MATERIAL_CLOSURE"
        and code_audit.get("code_computes_R_in_R_out") is False
        and code_audit.get("code_uses_component_overlap_proxy") is True
    )
    summary = {
        "phase": "v70_phase0_fact_code_lock",
        "decision": "PASS_V69R2_PROXY_CLOSURE_LOCK" if phase0_pass else "NO_GO_V69R2_FACT_CODE_LOCK",
        "gate": {
            "pass": bool(phase0_pass),
            "v69r2_decision_is_no_go_material_closure": final.get("decision") == "NO_GO_MATERIAL_CLOSURE",
            "phase1_anchor_bank_pass": bool(gate.get("phase1_anchor_bank_pass")),
            "phase2_material_closure_pass": bool(gate.get("phase2_material_closure_pass")),
            "can_enter_typed_assignment": bool(gate.get("can_enter_typed_assignment")),
            "can_enter_local2history": bool(gate.get("can_enter_local2history")),
            "code_computes_R_in_R_out": bool(code_audit.get("code_computes_R_in_R_out")) if isinstance(code_audit.get("code_computes_R_in_R_out"), bool) else code_audit.get("code_computes_R_in_R_out"),
            "code_uses_component_overlap_proxy": code_audit.get("code_uses_component_overlap_proxy"),
            "code_uses_v68_edge_rows": code_audit.get("code_uses_v68_edge_rows"),
            "code_uses_tracklet_index": code_audit.get("code_uses_tracklet_index"),
        },
        "v69r2": {
            "decision": final.get("decision"),
            "phase_status": phase_status,
            "selected_final_attempt": final.get("selected_final_attempt"),
            "best_material_attempt": final.get("best_material_attempt"),
        },
        "source_code_audit": code_audit,
        "material_metric_first_row": material_metric,
        "inputs": {
            "final_decision": _rel(final_path),
            "anchor_summary": _rel(anchor_path),
            "material_summary": _rel(material_path),
            "material_metric_rows": _rel(material_metric_path),
            "material_closure_source": _rel(source_path),
        },
        "rows": {
            "v69r2_metric_rows_csv": _rel(output_root / "v69r2_metric_rows.csv"),
            "code_audit_rows_csv": _rel(output_root / "code_audit_rows.csv"),
        },
        "notes": [
            "Phase 0 only locks current facts and implementation shape; it does not create a v70 method result.",
            "Static source scan shows whether v69-r2 computes true carrier projection ratios or proxy material overlap fields.",
        ],
    }
    _write_json(output_root / "fact_code_lock_summary.json", summary)
    sha_rows = []
    for path in [output_root / "fact_code_lock_summary.json", output_root / "v69r2_metric_rows.csv", output_root / "code_audit_rows.csv"]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream4D v70 Phase 0 v69-r2 fact/code lock.")
    parser.add_argument("--output-root", default="outputs/audit/v70_phase0_fact_code_lock")
    parser.add_argument("--final-decision", default="outputs/audit/v69r2_final_decision/final_decision.json")
    parser.add_argument("--anchor-summary", default="outputs/audit/v69r2_anchor_bank_repair5_nogt_underseg/anchor_bank_summary.json")
    parser.add_argument("--material-summary", default="outputs/audit/v69r2_material_closure_repair2_shared_no_bridge_probe5/closure_summary.json")
    parser.add_argument("--material-metric-rows", default="outputs/audit/v69r2_material_closure_repair2_shared_no_bridge_probe5/closure_metric_rows.csv")
    parser.add_argument("--material-closure-source", default="stream4d_native/v69r2_material_closure.py")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
