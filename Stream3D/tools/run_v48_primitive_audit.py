from __future__ import annotations

import argparse
from typing import Any

from stream4d_native.v47_common import write_csv, write_json
from stream4d_native.v48_data_contract import (
    ROOT,
    first_present,
    load_optional_json,
    metric_row,
    nested,
    project_path,
    rel,
    utc_now,
)


def _row(
    *,
    primitive_type: str,
    source: str,
    count: Any = None,
    purity: Any = None,
    completeness: Any = None,
    ari: Any = None,
    real_minus_shuffled: Any = None,
    real_minus_no_temporal: Any = None,
    scene0081_ari: Any = None,
    scene0591_completeness: Any = None,
    status: str = "ok",
    note: str = "",
) -> dict[str, Any]:
    purity_f = _float_or_none(purity)
    completeness_f = _float_or_none(completeness)
    shuffled_f = _float_or_none(real_minus_shuffled)
    gate = {
        "purity_pass": purity_f is not None and purity_f >= 0.88,
        "completeness_pass": completeness_f is not None and completeness_f >= 0.35,
        "real_minus_shuffled_ARI_pass": shuffled_f is not None and shuffled_f >= 0.20,
    }
    gate["pass"] = bool(gate["purity_pass"] and gate["completeness_pass"] and gate["real_minus_shuffled_ARI_pass"])
    return {
        "primitive_type": primitive_type,
        "status": status,
        "source": rel(source),
        "primitive_count": count,
        "primitive_purity_mean": purity,
        "primitive_purity_p10": None,
        "primitive_completeness_mean": completeness,
        "fragmentation_per_GT": None,
        "object_count_proxy": count,
        "ARI": ari,
        "scene0081_ARI": scene0081_ari,
        "scene0591_completeness": scene0591_completeness,
        "real_minus_shuffled_ARI": real_minus_shuffled,
        "real_minus_no_temporal_ARI": real_minus_no_temporal,
        "gate": gate,
        "gate_pass": gate["pass"],
        "note": note,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_primary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in rows if row.get("gate_pass")]
    if not passing:
        return None
    return max(passing, key=lambda row: (_float_or_none(row.get("ARI")) or -1.0, _float_or_none(row.get("primitive_completeness_mean")) or -1.0))


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    tracklet = load_optional_json(args.tracklet_root + "/tracklet_construction.json")
    tracklet_shuffled = load_optional_json(args.tracklet_shuffled_root + "/tracklet_construction.json")
    tracklet_no_temporal = load_optional_json(args.tracklet_no_temporal_root + "/tracklet_construction.json")
    carrier = load_optional_json(args.carrier_supertrack_root + "/carrier_supertrack_summary.json")
    final_decision = load_optional_json(args.v47_final_root + "/v47_final_decision.json")
    mask_atom_path = project_path(args.mask_atom_rows)

    p1_real_minus_shuffled = None
    if tracklet.get("tracklet_ARI") is not None and tracklet_shuffled.get("tracklet_ARI") is not None:
        p1_real_minus_shuffled = float(tracklet["tracklet_ARI"]) - float(tracklet_shuffled["tracklet_ARI"])
    p1_real_minus_no_temporal = None
    if tracklet.get("tracklet_ARI") is not None and tracklet_no_temporal.get("tracklet_ARI") is not None:
        p1_real_minus_no_temporal = float(tracklet["tracklet_ARI"]) - float(tracklet_no_temporal["tracklet_ARI"])

    carrier_component_row = first_present(nested(final_decision, "carrier_component_context", "row"), {})
    carrier_supertrack_row = first_present(nested(final_decision, "carrier_supertrack_summary"), carrier, {})

    rows = [
        _row(
            primitive_type="P1_short_tracklets",
            source=args.tracklet_root + "/tracklet_construction.json",
            count=tracklet.get("tracklet_count"),
            purity=tracklet.get("tracklet_purity"),
            completeness=tracklet.get("tracklet_completeness"),
            ari=tracklet.get("tracklet_ARI"),
            real_minus_shuffled=p1_real_minus_shuffled,
            real_minus_no_temporal=p1_real_minus_no_temporal,
            scene0081_ari=None,
            scene0591_completeness=None,
            note="Existing v47 strict-veto short-tracklet primitive; diagnostic GT only for metrics.",
        ),
        _row(
            primitive_type="P2_carrier_supertracks",
            source=args.carrier_supertrack_root + "/carrier_supertrack_summary.json",
            count=carrier.get("supertrack_count"),
            purity=carrier.get("object_from_supertrack_purity"),
            completeness=carrier.get("object_from_supertrack_completeness"),
            ari=carrier.get("object_from_supertrack_ARI"),
            real_minus_shuffled=carrier.get("real_minus_shuffled_supertrack_ARI"),
            real_minus_no_temporal=None,
            scene0081_ari=None,
            scene0591_completeness=None,
            note="Supertrack-as-object diagnostic from v47; strong purity but low object completeness.",
        ),
        _row(
            primitive_type="P3_carrier_components",
            source=args.carrier_supertrack_root + "/carrier_supertrack_summary.json",
            count=carrier.get("component_count"),
            purity=carrier.get("object_from_component_purity"),
            completeness=carrier.get("object_from_component_completeness"),
            ari=carrier.get("object_from_component_ARI"),
            real_minus_shuffled=carrier.get("real_minus_shuffled_component_ARI"),
            real_minus_no_temporal=carrier_component_row.get("real_minus_no_temporal_ARI"),
            scene0081_ari=carrier_component_row.get("scene0081_ARI"),
            scene0591_completeness=carrier_component_row.get("scene0591_completeness"),
            note="Carrier component primitive is the plan's v47 strongest partial baseline.",
        ),
    ]

    if mask_atom_path.exists():
        # Existing v47 file is intentionally allowed to report not_implemented.
        text = mask_atom_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        status = "available"
        note = "Mask atom rows exist, but metric extraction is unavailable."
        if len(text) >= 2 and "not_implemented" in text[1]:
            status = "not_implemented"
            note = "Existing v47 mask_atom_rows.csv reports atomization was not promoted into the method."
        rows.append(
            _row(
                primitive_type="P4_mask_atoms",
                source=args.mask_atom_rows,
                status=status,
                note=note,
            )
        )
    else:
        rows.append(
            _row(
                primitive_type="P4_mask_atoms",
                source=args.mask_atom_rows,
                status="missing",
                note="No mask atom metric artifact found.",
            )
        )

    primary = _select_primary(rows)
    stop1 = primary is None
    repair_recommendation = None
    if stop1:
        repair_recommendation = "No primitive satisfies purity>=0.88, completeness>=0.35, real_minus_shuffled>=0.20; plan says repair primitive generation before object completion."
    else:
        repair_recommendation = "Primary primitive satisfies Phase 1 gate; proceed to object completion and semantic/flow audits."

    payload = {
        "phase": "v48_primitive_audit",
        "created_at": utc_now(),
        "primitive_rows": rows,
        "primary_primitive": primary,
        "gate": {
            "pass": not stop1,
            "primary_primitive_type": None if primary is None else primary["primitive_type"],
            "stop1_triggered": stop1,
            "repair_recommendation": repair_recommendation,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": True,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v48 Phase 1 primitive audit from existing artifacts.")
    parser.add_argument("--tracklet-root", default="outputs/audit/v47_tracklets_strict_veto_A5")
    parser.add_argument("--tracklet-shuffled-root", default="outputs/audit/v47_tracklets_strict_veto_A7_shuffled")
    parser.add_argument("--tracklet-no-temporal-root", default="outputs/audit/v47_tracklets_strict_veto_A8_no_temporal")
    parser.add_argument("--carrier-supertrack-root", default="outputs/audit/v47_carrier_supertrack_union_32_fine_metricfix")
    parser.add_argument("--v47-final-root", default="outputs/audit/v47_final_decision_phase9_continued21_carrier_mdl_audit")
    parser.add_argument("--mask-atom-rows", default="outputs/audit/v47_failure_autopsy_continued21_carrier_mdl_audit/mask_atom_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v48_primitive_audit")
    args = parser.parse_args()

    payload = build_audit(args)
    out = project_path(args.output_root)
    write_json(out / "primitive_audit_summary.json", payload)
    write_csv(out / "primitive_audit_rows.csv", payload["primitive_rows"])
    print({"summary": str(out / "primitive_audit_summary.json"), "gate": payload["gate"]})


if __name__ == "__main__":
    main()

