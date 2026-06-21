from __future__ import annotations

import argparse

from stream4d_native.v53_fact_lock import build_v53_fact_lock, write_v53_fact_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v53 Phase 0 fact lock.")
    parser.add_argument("--scenes", default="scene0011_00,scene0030_00,scene0050_00,scene0081_01,scene0591_00")
    parser.add_argument("--output-root", default="outputs/audit/v53_fact_lock")
    args = parser.parse_args()
    scenes = [scene.strip() for scene in str(args.scenes).split(",") if scene.strip()]
    payload = build_v53_fact_lock(scenes=scenes)
    write_v53_fact_lock(args.output_root, payload)
    print(
        {
            "summary": f"{args.output_root}/fact_lock.json",
            "gate": payload["gate"],
            "cropformer_png_count": payload["summary"]["cropformer_png_count"],
            "sam2_png_count": payload["summary"]["sam2_png_count"],
            "sam_png_count": payload["summary"]["sam_png_count"],
        }
    )


if __name__ == "__main__":
    main()
