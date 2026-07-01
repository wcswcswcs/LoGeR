from __future__ import annotations

import argparse

from stream4d_native.v64r2_final_eval import build_v64r2_final_decision, write_v64r2_final_decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v64-r2 final integrated decision.")
    parser.add_argument("--output-root", default="outputs/audit/v64r2_final")
    args = parser.parse_args()
    payload = build_v64r2_final_decision()
    write_v64r2_final_decision(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/final_decision.json",
            "decision_label": payload["decision_label"],
            "main_ownership_status": payload["main_ownership_status"],
            "scannet_ap_status": payload["scannet_ap_status"],
            "dynamic_status": payload["dynamic_status"],
            "active_query_status": payload["active_query_status"],
            "blocked_claims": payload["blocked_claims"],
        }
    )


if __name__ == "__main__":
    main()
