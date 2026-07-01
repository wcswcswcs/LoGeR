from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
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


def main() -> None:
    output_root = rooted("outputs/audit/v73_final_decision")
    inputs = {
        "phase0": rooted("outputs/audit/v73_phase0_fact_lock/fact_lock_summary.json"),
        "phase1": rooted("outputs/audit/v73_phase1_source_signal_audit/source_signal_summary.json"),
        "phase2": rooted("outputs/audit/v73_phase2_semantic_extent_proposals/proposal_summary.json"),
        "phase3": rooted("outputs/audit/v73_phase3_d4rt_proposal_verification/d4rt_proposal_summary.json"),
        "phase4": rooted("outputs/audit/v73_phase4_local_slot_birth/local_slot_summary.json"),
        "phase5": rooted("outputs/audit/v73_phase5_local_controls/local_control_summary.json"),
    }
    missing = [{"name": name, "path": rel(path)} for name, path in inputs.items() if not path.exists()]
    if missing:
        write_csv(output_root / "missing_input_rows.csv", missing)
        payload = {"phase": "v73_final_decision", "decision": "NO_GO_FINAL_MISSING_INPUT", "missing_input_count": len(missing)}
        write_json(output_root / "final_decision.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    p0, p1, p2, p3, p4, p5 = (load(path) for path in inputs.values())
    phase2_pass = bool((p2.get("gate") or {}).get("pass"))
    phase4_pass = bool((p4.get("gate") or {}).get("pass"))
    phase5_pass = bool((p5.get("gate") or {}).get("pass"))
    d4rt_proven = bool(p3.get("D4RT_contribution_proven"))
    semantic_proven = bool(p5.get("can_claim_semantic_contribution"))
    boundary_proven = bool(p5.get("can_claim_boundary_contribution"))
    final_decision = "NO_GO_LOCAL_CONTROLS_AREA_LATTICE_BIAS"
    local_decision = "LOCAL_GATE_PASS_BUT_CONTROLS_FAIL" if phase4_pass and not phase5_pass else "LOCAL_NOT_PASSED"
    payload = {
        "phase": "v73_final_decision",
        "schema": "stream4d_v73_final_decision_v1",
        "final_decision": final_decision,
        "local_decision": local_decision,
        "local2history_decision": "BLOCKED_BY_LOCAL_CONTROLS",
        "primary_blocker": "AREA_LATTICE_CONTROL_BEATS_FINAL_P5",
        "secondary_blocker": "D4RT_FUSION_NOT_PROVEN",
        "method_uses_gt_anywhere": False,
        "local_gate_pass": phase4_pass,
        "local_controls_pass": phase5_pass,
        "local2history_gate_pass": False,
        "D4RT_contribution_proven": d4rt_proven,
        "semantic_contribution_proven": semantic_proven,
        "boundary_proposal_contribution_proven": boundary_proven,
        "can_claim_method_table": False,
        "can_claim_diagnostic_table_only": True,
        "can_enter_local2history": False,
        "phase_decisions": {
            "phase0": p0.get("decision"),
            "phase1": p1.get("decision"),
            "phase2": p2.get("decision"),
            "phase3": p3.get("decision"),
            "phase4": p4.get("decision"),
            "phase5": p5.get("decision"),
        },
        "key_metrics": {
            "phase2_best_method": (p2.get("gate") or {}).get("best_method_variant"),
            "phase2_best_SF50": (p2.get("gate") or {}).get("best_method_proposal_oracle_SF50"),
            "phase2_best_GT_best_IoU": (p2.get("gate") or {}).get("best_method_GT_best_IoU_mean"),
            "phase3_real_minus_shuffled_SF50": p3.get("real_minus_shuffled_SF50"),
            "phase3_real_minus_no_temporal_SF50": p3.get("real_minus_no_temporal_SF50"),
            "phase4_local_SF50": (p4.get("metric_values") or {}).get("local_SF50"),
            "phase4_same_frame_violation_count": (p4.get("metric_values") or {}).get("same_frame_violation_count"),
            "phase5_C3_SF50": (p5.get("gate") or {}).get("C3_semantic_plus_boundary_SF50"),
            "phase5_C6_area_only_SF50": (p5.get("gate") or {}).get("C6_area_only_control_SF50"),
            "phase5_area_only_gap": (p5.get("gate") or {}).get("area_only_gap"),
        },
        "notes": [
            "Phase2 and Phase4 pass only after area/lattice coverage rescue.",
            "Phase5 controls show final P5 does not beat P0/C6 area-lattice control, so semantic contribution is not proven.",
            "D4RT real-vs-control contribution is not proven; D4RT cannot be claimed as core method contribution.",
            "local2history is blocked because local method claim is not valid under controls.",
        ],
    }
    rows = [{"metric": key, "value": value} for key, value in payload["key_metrics"].items()]
    write_json(output_root / "final_decision.json", payload)
    write_json(output_root / "summary.json", payload)
    write_csv(output_root / "final_metric_rows.csv", rows)
    write_csv(output_root / "main_rows.csv", rows)
    write_csv(output_root / "missing_input_rows.csv", [])
    sha_rows = []
    for path in inputs.values():
        sha_rows.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path), "kind": "input"})
    for path in sorted(output_root.glob("*")):
        if path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path), "kind": "output"})
    write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
