from __future__ import annotations

import argparse

from stream4d_native.v55_material_atoms import build_v55_material_atoms, write_v55_material_atoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v55 Phase 2 canonical material atom layer.")
    parser.add_argument("--output-root", default="outputs/audit/v55_atoms")
    parser.add_argument("--visualization-root", default="outputs/audit/v55_visualizations/atoms")
    parser.add_argument("--support-variant", default="R0_visible_tau0.05")
    args = parser.parse_args()
    payload = build_v55_material_atoms(support_variant=args.support_variant)
    write_v55_material_atoms(args.output_root, payload, visualization_root=args.visualization_root)
    summary = payload["summary"]
    print(
        {
            "atoms": f"{args.output_root}/atom_summary.json",
            "gate": summary["gate"],
            "component_count": summary["component_count"],
            "atom_count": summary["atom_count"],
            "atom_purity_diagnostic": summary["atom_purity_diagnostic"],
            "fragmentation_per_GT_object_decrease": summary["fragmentation_per_GT_object_decrease"],
        }
    )


if __name__ == "__main__":
    main()
