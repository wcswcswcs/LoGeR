from __future__ import annotations

import argparse
import random
from pathlib import Path


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tune-output", required=True)
    parser.add_argument("--final-output", required=True)
    parser.add_argument("--seed", type=int, default=20260607)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene_ids = _read_lines(Path(args.input))
    if not scene_ids:
        raise ValueError(f"empty split file: {args.input}")
    shuffled = list(scene_ids)
    rng = random.Random(args.seed)
    rng.shuffle(shuffled)
    pivot = len(shuffled) // 2
    tune = sorted(shuffled[:pivot])
    final = sorted(shuffled[pivot:])
    _write_lines(Path(args.tune_output), tune)
    _write_lines(Path(args.final_output), final)
    print(f"[split] input={args.input} scenes={len(scene_ids)} seed={args.seed}")
    print(f"[split] tune={args.tune_output} scenes={len(tune)}")
    print(f"[split] final={args.final_output} scenes={len(final)}")


if __name__ == "__main__":
    main()
