from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d_native.v44_typed_mask_assembly import descriptor_audit, load_scene_artifacts, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v44 mask descriptors from prepared mask artifacts.")
    parser.add_argument("--part-graph-root", default="outputs/audit/v43_2_current_v37_semantic_part_graph_dinov2_sample8")
    parser.add_argument("--scenes", default="scene0081_01,scene0591_00")
    parser.add_argument("--output-root", default="outputs/audit/v44_mask_descriptor")
    parser.add_argument(
        "--feature-smokes",
        default="outputs/audit/v42_feature_adapter_dinov2/feature_smoke.json,outputs/audit/v42_feature_adapter_radio/feature_smoke.json,outputs/audit/v42_feature_adapter_radio_h/feature_smoke.json,outputs/audit/v42_feature_adapter_cradio_v3b/feature_smoke.json",
    )
    args = parser.parse_args()
    artifacts = load_scene_artifacts(ROOT, part_graph_root=args.part_graph_root, scenes=args.scenes)
    feature_smokes = [ROOT / item.strip() for item in str(args.feature_smokes).split(",") if item.strip()]
    payload = descriptor_audit(artifacts, feature_smokes=feature_smokes)
    out = ROOT / args.output_root
    write_json(out / "mask_descriptor_audit.json", payload)
    write_csv(out / "mask_descriptor_scene_rows.csv", payload["scene_rows"])
    write_csv(out / "feature_backend_rows.csv", payload["backend_rows"])
    print(json.dumps({"summary": str(out / "mask_descriptor_audit.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
