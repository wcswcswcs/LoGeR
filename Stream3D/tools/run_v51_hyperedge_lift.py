from __future__ import annotations

import argparse

from stream4d_native.v51_hyperedge_lift import build_v51_hyperedge_lift, write_v51_hyperedge_lift


def main() -> None:
    parser = argparse.ArgumentParser(description="Lift v51-r2 keymask component-set hyperedges with D4RT component support controls.")
    parser.add_argument("--keymask-root", required=True)
    parser.add_argument("--output-root", default="outputs/audit/v51_r2_hyperedge_lift")
    parser.add_argument("--max-frames-per-scene", type=int, default=4)
    args = parser.parse_args()
    payload = build_v51_hyperedge_lift(
        keymask_root=args.keymask_root,
        max_frames_per_scene=args.max_frames_per_scene,
    )
    write_v51_hyperedge_lift(args.output_root, payload)
    print({"summary": f"{args.output_root}/hyperedge_lift_summary.json", "gate": payload["gate"], "metrics": payload["summary"]})


if __name__ == "__main__":
    main()
