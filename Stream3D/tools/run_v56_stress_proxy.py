from __future__ import annotations

import argparse

from stream4d_native.v56_stress_proxy import build_v56_stress_proxy, write_v56_stress_proxy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v56 diagnostic stress proxy over component support artifacts.")
    parser.add_argument("--output-root", default="outputs/audit/v56_stress_proxy")
    args = parser.parse_args()
    payload = build_v56_stress_proxy()
    write_v56_stress_proxy(args.output_root, payload)
    summary = payload["summary"]
    print(
        {
            "summary": f"{args.output_root}/stress_proxy_summary.json",
            "gate": summary["gate"],
            "stress_real_minus_mask_only_ARI_pass_count": summary[
                "stress_real_minus_mask_only_ARI_pass_count"
            ],
            "best_real_minus_mask_only_ARI_proxy": summary["best_real_minus_mask_only_ARI_proxy"],
        }
    )


if __name__ == "__main__":
    main()

