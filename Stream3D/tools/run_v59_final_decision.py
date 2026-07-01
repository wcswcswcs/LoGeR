from __future__ import annotations

import argparse

from stream4d_native.v59_final_decision import build_v59_final_decision, write_v59_final_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Stream4D v59 final decision from completed gates.")
    parser.add_argument("--output-root", default="outputs/audit/v59_final_decision")
    parser.add_argument("--phase0", default="outputs/audit/v59_phase0_fact_lock/fact_lock.json")
    parser.add_argument("--phase1", default="outputs/audit/v59_phase1_graph/graph_summary.json")
    parser.add_argument("--phase2", default="outputs/audit/v59_phase2_paths_repair_margin070_noexcl_semcat/path_summary.json")
    args = parser.parse_args()
    decision = build_v59_final_decision(args.phase0, args.phase1, args.phase2)
    outputs = write_v59_final_decision(decision, args.output_root)
    print(
        {
            "final_decision": outputs["final_decision"],
            "goal_achieved": decision["goal_achieved"],
            "final_label": decision["final_label"],
            "partial_label": decision["partial_label"],
            "phase0_gate_pass": decision["phase0_gate_pass"],
            "phase1_gate_pass": decision["phase1_gate_pass"],
            "phase2_gate_pass": decision["phase2_gate_pass"],
            "stop_reason": decision["stop_reason"],
        }
    )


if __name__ == "__main__":
    main()
