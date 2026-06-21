from __future__ import annotations

import argparse

from stream4d_native.v49_mosaic_stage1 import build_component_proxy_feature_audit, write_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v49 component proxy feature audit.")
    parser.add_argument("--output-root", default="outputs/audit/v49_component_proxy_feature_audit")
    args = parser.parse_args()
    payload = build_component_proxy_feature_audit()
    write_bundle(
        args.output_root,
        "component_proxy_feature_audit_summary",
        payload,
        {
            "component_proxy_rows": payload["component_proxy_rows"],
            "hypothesis_proxy_rows": payload["hypothesis_proxy_rows"],
            "score_auc_rows": payload["score_auc_rows"],
        },
    )
    print(
        {
            "summary": f"{args.output_root}/component_proxy_feature_audit_summary.json",
            "gate": payload["gate"],
            "failure_label": payload["failure_label"],
        }
    )


if __name__ == "__main__":
    main()
