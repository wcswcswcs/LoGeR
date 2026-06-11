from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream4d.measurement_bank import MeasurementBank, json_safe, read_seq_list, summarize_bank, write_summary_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/v12_measurement_bank")
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-prefix", default="outputs/audit/v12_measurement_bank/measurement_bank_probe5")
    parser.add_argument("--boundary-safe-px", type=float, default=3.0)
    args = parser.parse_args()
    rows = []
    for scene in read_seq_list(Path(args.seq_list)):
        bank_path = Path(args.bank_root) / scene / "measurement_bank.npz"
        bank = MeasurementBank.load(bank_path)
        row = summarize_bank(bank, boundary_safe_px=float(args.boundary_safe_px))
        row["bank_path"] = str(bank_path)
        rows.append(row)
    aggregate = write_summary_bundle(Path(args.output_prefix), rows)
    print(json.dumps(json_safe(aggregate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
