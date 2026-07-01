#!/usr/bin/env python3
"""Build v85 decision matrix from current phase summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_ROOT = Path("results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT / "phase11_decision_matrix")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    phase1 = read_json(args.root / "phase1_anchor_pair_universe/anchor_pair_sufficiency_summary.json")
    phase2 = read_json(args.root / "phase2_qk_feature_bank/feature_sanity_summary.json")
    visual = read_json(args.root / "phase12_visual_rediscovery/visual_integrity_audit.json")
    rows = [
        {
            "decision_label": "D1_ANCHOR_PAIR_INSUFFICIENT",
            "active": not bool(phase1.get("phase1_gate_pass", False)),
            "evidence": f"phase1_fail_reasons={phase1.get('fail_reasons', [])}; strong_bad_pair_rows={phase1.get('strong_bad_pair_rows')}",
            "blocks_runtime": True,
        },
        {
            "decision_label": "D2_ALIGNMENT_NOT_SPECIFIC",
            "active": False,
            "evidence": "not evaluated because Phase1 gate did not pass",
            "blocks_runtime": False,
        },
        {
            "decision_label": "D3_ALIGNMENT_NOT_SCALE_RELEVANT",
            "active": False,
            "evidence": "not evaluated because Phase1 gate did not pass",
            "blocks_runtime": False,
        },
        {
            "decision_label": "D4_ROUTE_CARRIER_NOT_CONFIRMED",
            "active": False,
            "evidence": "not evaluated because Phase1/Phase3 gates did not pass",
            "blocks_runtime": False,
        },
        {
            "decision_label": "D10_VISUAL_AUDIT_INCOMPLETE",
            "active": not bool(visual.get("visual_audit_gate_pass", False)),
            "evidence": f"visual_audit_gate_pass={visual.get('visual_audit_gate_pass', False)}",
            "blocks_runtime": True,
        },
    ]
    active_labels = [row["decision_label"] for row in rows if row["active"]]
    notes = [
        "D1 is active because Phase1 strong bad anchor support failed after repair attempts.",
    ]
    if "D10_VISUAL_AUDIT_INCOMPLETE" in active_labels:
        notes.append("D10 remains active until Phase12 visual rediscovery produces real panels and passes integrity/review gates.")
    else:
        notes.append("D10 is closed because Phase12 visual rediscovery passed integrity/review gates.")
    payload = {
        "phase": "Phase11_decision_matrix",
        "active_decision_labels": active_labels,
        "phase1_gate_pass": bool(phase1.get("phase1_gate_pass", False)),
        "phase2_feature_gate_pass": bool(phase2.get("phase2_feature_gate_pass", False)),
        "can_enter_phase3_alignment": bool(phase2.get("can_enter_phase3_alignment", False)),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "final_no_go_allowed": bool(visual.get("visual_audit_gate_pass", False)),
        "no_go_blocker": "strong_bad_support_insufficient" if "D1_ANCHOR_PAIR_INSUFFICIENT" in active_labels else "",
        "notes": notes,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "decision_matrix.csv", rows)
    write_json(args.out_dir / "decision_matrix.json", payload)
    (args.out_dir / "next_route_recommendation.md").write_text(
        "\n".join(
            [
                "# v85 Next Route Recommendation",
                "",
                "- Do not run Phase3 latent C fitting, runtime SWA QK action, merge/gauge fallback, or TTT.",
                "- Current blocker: `strong_bad_support_insufficient`.",
                "- Required next work for final closure: complete Phase12 visual rediscovery and review.",
                "- If new audited labelled-bad high-confidence anchor pairs become available, rebuild Phase1 before any Phase3 attempt.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"active_decision_labels={','.join(active_labels)}")
    print(f"final_no_go_allowed={str(payload['final_no_go_allowed']).lower()}")
    print(f"runtime_action_allowed={str(payload['runtime_action_allowed']).lower()}")


if __name__ == "__main__":
    main()
