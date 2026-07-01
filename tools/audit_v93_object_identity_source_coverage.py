#!/usr/bin/env python3
"""Audit v93 Phase1 object/RADIO/component source coverage gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v86_soft_latent_utils import read_json, write_json  # noqa: E402
from tools.v93_semantic_object_identity_utils import ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=ROOT / "phase1_object_identity_row_join")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.phase1_dir / "object_identity_source_summary.json"
    summary: dict[str, Any] = read_json(summary_path)
    object_pass = (
        float(summary.get("object_identity_labelled_coverage") or 0.0) >= 0.50
        and int(summary.get("object_identity_seq_coverage") or 0) >= 3
    )
    radio_pass = (
        float(summary.get("radio_labelled_coverage") or 0.0) >= 0.30
        and int(summary.get("radio_seq_coverage") or 0) >= 2
        and bool(summary.get("radio_fields_include_interior_boundary_stability"))
    )
    diagnostic_component_proxy_pass = (
        float(summary.get("component_tracklet_available_ratio") or 0.0) >= 0.90
        and float(summary.get("object_identity_available_ratio") or 0.0) < 0.50
    )
    gate_pass = bool(object_pass or radio_pass or diagnostic_component_proxy_pass)
    audit = {
        "phase": "Phase1_object_identity_source_coverage_audit",
        "phase1_source_gate_pass": gate_pass,
        "object_identity_source_pass": object_pass,
        "radio_source_pass": radio_pass,
        "diagnostic_component_proxy_pass": diagnostic_component_proxy_pass,
        "object_identity_success_claim_allowed": object_pass or radio_pass,
        "no_object_identity_success_claim": not (object_pass or radio_pass),
        "allowed_next_scope": "object_or_radio_policy" if (object_pass or radio_pass) else "phase2_diagnostic_component_proxy_only",
        "blocker": "" if (object_pass or radio_pass) else "semantic_source_specificity_still_insufficient_object_identity_unavailable",
        "summary": summary,
        "runtime_action_allowed": False,
        "counterfactual_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.phase1_dir / "object_identity_join_audit.json", audit)
    print(f"phase1_source_gate_pass={gate_pass}")
    print(f"object_identity_source_pass={object_pass}")
    print(f"radio_source_pass={radio_pass}")
    print(f"diagnostic_component_proxy_pass={diagnostic_component_proxy_pass}")
    print(f"allowed_next_scope={audit['allowed_next_scope']}")


if __name__ == "__main__":
    main()
