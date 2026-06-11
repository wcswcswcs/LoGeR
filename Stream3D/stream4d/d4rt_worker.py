from __future__ import annotations

from pathlib import Path

from .d4rt_adapter import D4RTAdapter
from .diagnostics import write_json
from .run_scannet import _process_sequence, _seq_names, build_parser


def main() -> None:
    parser = build_parser()
    parser.description = "Persistent D4RT worker wrapper around stream4d.run_scannet."
    args = parser.parse_args()
    seqs = _seq_names(args)
    adapter = D4RTAdapter(
        d4rt_root=args.d4rt_root,
        model_config=args.d4rt_config,
        ckpt_path=args.d4rt_ckpt,
        device=args.device,
    )
    summaries = []
    errors = []
    for seq_name in seqs:
        args.seq_name = seq_name
        print(f"[d4rt-worker] start seq={seq_name}", flush=True)
        try:
            summaries.append(_process_sequence(args, adapter))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            message = f"{type(exc).__name__}: {exc}"
            print(f"[d4rt-worker][ERROR] seq={seq_name} {message}", flush=True)
            errors.append({"seq_name": seq_name, "error": message})
    write_json(Path(args.debug_root) / f"{args.output_config}_worker_summary.json", {"summaries": summaries, "errors": errors})


if __name__ == "__main__":
    main()
