from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v68_final_decision import parse_args, run  # noqa: E402


if __name__ == "__main__":
    run(parse_args())
