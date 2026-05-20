#!/usr/bin/env python3
"""Create effective chunk input HMC states from saved pre-reset states.

Older runs saved ``chunk_XXX_before.pt`` before the external reset_every hook.
For sandbox replay, reset-boundary chunks need the state after that hook. This
tool derives ``chunk_XXX_input.pt`` with the same reset logic used by
``run_pipeline_abc_v2.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loger.pipeline.hybrid_memory_controller import hybrid_state_fingerprint  # noqa: E402
from run_pipeline_abc_v2 import (  # noqa: E402
    HybridMemoryState,
    _load_hmc_state,
    _reset_hybrid_state_if_needed,
    _save_hmc_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--reset-every", type=int, default=5)
    parser.add_argument("--chunk-count", type=int, default=38)
    parser.add_argument("--summary-jsonl", default="")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    rows = []
    for ci in range(int(args.chunk_count)):
        src = state_dir / f"chunk_{ci:03d}_before.pt"
        if not src.exists():
            raise SystemExit(f"Missing state snapshot: {src}")
        state = _load_hmc_state(str(src))
        reset_applied = bool(args.reset_every > 0 and ci > 0 and ci % args.reset_every == 0)
        if reset_applied:
            state = _reset_hybrid_state_if_needed(state) or HybridMemoryState()
        dst = state_dir / f"chunk_{ci:03d}_input.pt"
        _save_hmc_state(dst, state)
        rows.append(
            {
                "chunk_idx": ci,
                "source": str(src),
                "output": str(dst),
                "reset_applied": reset_applied,
                "input_hash": hybrid_state_fingerprint(state),
            }
        )

    summary = Path(args.summary_jsonl) if args.summary_jsonl else state_dir / "effective_input_state_summary.jsonl"
    with summary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} effective input states to {state_dir}")
    print(f"Wrote summary to {summary}")


if __name__ == "__main__":
    main()
