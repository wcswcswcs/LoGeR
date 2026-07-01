from __future__ import annotations

import argparse

from stream4d_native.v62_integrity import V62IntegrityConfig, build_v62_integrity, write_v62_integrity, write_v62_integrity_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v62 Phase 0 v61 package integrity audit.")
    parser.add_argument("--zip-path", default="code_audit_pack/stream4d_v61_code_audit_packet_20260621_205517.zip")
    parser.add_argument("--output-root", default="outputs/audit/v62_phase0_integrity")
    parser.add_argument("--visualization-root", default="outputs/audit/v62_visualizations/phase0")
    args = parser.parse_args()
    cfg = V62IntegrityConfig(zip_path=args.zip_path, output_root=args.output_root, visualization_root=args.visualization_root)
    result = build_v62_integrity(cfg)
    outputs = write_v62_integrity(result, args.output_root)
    visuals = write_v62_integrity_visualizations(result, args.visualization_root)
    print({"outputs": outputs, "visualization_status": visuals["visualization_status"], "gate": result["summary"]["gate"]})


if __name__ == "__main__":
    main()

