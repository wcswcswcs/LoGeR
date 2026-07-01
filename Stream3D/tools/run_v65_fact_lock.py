from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stream4d_native.v65_fact_lock import build_v65_fact_lock, write_v65_fact_lock


def main() -> None:
    output_root = "outputs/audit/v65_phase0_fact_lock"
    payload = build_v65_fact_lock()
    write_v65_fact_lock(output_root, payload)
    print(
        {
            "summary": f"{output_root}/fact_lock_summary.json",
            "gate": payload["summary"]["gate"],
            "missing_artifact_count": payload["summary"]["missing_artifact_count"],
        }
    )


if __name__ == "__main__":
    main()
