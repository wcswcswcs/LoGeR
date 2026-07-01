from __future__ import annotations

import argparse

from stream4d_native.v63_d4rt_query import V63D4RTQueryConfig, build_v63_d4rt_query, write_v63_d4rt_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stream4D v63 Phase 3 D4RT query preflight or real execution.")
    parser.add_argument("--selected-query-rows", default="outputs/audit/v63_query_policy/selected_query_rows.csv")
    parser.add_argument("--novelty-material-rows", default="outputs/audit/v62_increment_attribution/novelty_material_rows.csv")
    parser.add_argument("--output-root", default="outputs/audit/v63_d4rt_query")
    parser.add_argument("--scannet-root", default="data/scannet/processed")
    parser.add_argument("--d4rt-root", default="../Open-d4rt")
    parser.add_argument("--d4rt-config", default="../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml")
    parser.add_argument("--d4rt-ckpt", default="../Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-ids", default="R0_real_policy")
    parser.add_argument("--max-queries-per-policy", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-accepted-frames", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--real-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.real_run:
        raise SystemExit("Use only one of --dry-run or --real-run.")
    dry_run = not args.real_run
    if args.dry_run:
        dry_run = True
    max_queries = args.max_queries_per_policy
    if max_queries is not None and max_queries <= 0:
        max_queries = None
    cfg = V63D4RTQueryConfig(
        selected_query_rows=args.selected_query_rows,
        novelty_material_rows=args.novelty_material_rows,
        output_root=args.output_root,
        scannet_root=args.scannet_root,
        d4rt_root=args.d4rt_root,
        d4rt_config=args.d4rt_config,
        d4rt_ckpt=args.d4rt_ckpt,
        device=args.device,
        policy_ids=tuple(part for part in args.policy_ids.split(",") if part),
        max_queries_per_policy=max_queries,
        window_size=args.window_size,
        query_chunk_size=args.query_chunk_size,
        min_visibility=args.min_visibility,
        min_confidence=args.min_confidence,
        min_accepted_frames=args.min_accepted_frames,
        dry_run=dry_run,
    )
    result = build_v63_d4rt_query(cfg)
    outputs = write_v63_d4rt_query(result, cfg)
    print(
        {
            "outputs": outputs,
            "gate": result["summary"]["gate"],
            "method_status": result["summary"]["method_status"],
            "policy_query_counts": result["summary"]["policy_query_counts"],
            "skip_reason_counts": result["summary"]["skip_reason_counts"],
        }
    )


if __name__ == "__main__":
    main()
