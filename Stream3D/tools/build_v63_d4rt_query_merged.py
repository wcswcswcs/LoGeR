from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key, "") for key in keys})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge split v63 D4RT query runs into the canonical Phase 3 audit directory.")
    parser.add_argument("--source-root", action="append", required=True)
    parser.add_argument("--output-root", default="Stream3D/outputs/audit/v63_d4rt_query")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_query_rows: list[dict[str, Any]] = []
    all_control_rows: list[dict[str, Any]] = []
    source_run_rows: list[dict[str, Any]] = []
    copied_carriers: list[str] = []
    all_group_summaries: list[dict[str, Any]] = []
    for source in args.source_root:
        root = Path(source)
        summary = _read_json(root / "query_execution_summary.json")
        query_rows = _read_csv(root / "query_result_rows.csv")
        control_rows = _read_csv(root / "query_control_rows.csv")
        for carrier_path in sorted(root.glob("carrier_batch_*.npz")):
            target = output_root / carrier_path.name
            shutil.copy2(carrier_path, target)
            copied_carriers.append(str(target))
        for row in query_rows:
            carrier = row.get("carrier_batch_npz", "")
            if carrier:
                name = Path(carrier).name
                row["carrier_batch_npz"] = str(Path("outputs/audit/v63_d4rt_query") / name)
            row["source_run_root"] = str(root)
            all_query_rows.append(row)
        for row in control_rows:
            row["source_run_root"] = str(root)
            all_control_rows.append(row)
        all_group_summaries.extend(summary.get("group_summaries", []))
        source_run_rows.append(
            {
                "source_run_root": str(root),
                "gate_pass": (summary.get("gate") or {}).get("pass"),
                "policy_query_counts": summary.get("policy_query_counts", {}),
                "skip_reason_counts": summary.get("skip_reason_counts", {}),
                "group_count": summary.get("group_count"),
                "method_status": summary.get("method_status"),
            }
        )

    policy_counts: dict[str, int] = {}
    status_counts: dict[str, dict[str, int]] = {}
    for row in all_query_rows:
        policy = row.get("policy_id", "")
        policy_counts[policy] = policy_counts.get(policy, 0) + 1
        status = row.get("d4rt_status", "")
        status_counts.setdefault(policy, {})
        status_counts[policy][status] = status_counts[policy].get(status, 0) + 1
    expected_real_policies = [
        "R0_real_policy",
        "C0_v62_original",
        "C1_random_matched",
        "C2_mask_boundary",
        "C3_semantic_only",
        "C4_K_mask_only_ablation",
    ]
    missing_real = [policy for policy in expected_real_policies if policy_counts.get(policy) != 64]
    pending_controls = ["R1_shuffled_history_association", "R2_no_temporal_source_frame_only"]
    summary = {
        "phase": "v63_d4rt_query",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method_status": "merged_real_D4RT_query_runs_with_pending_association_controls",
        "source_run_roots": [str(Path(source)) for source in args.source_root],
        "query_result_count": len(all_query_rows),
        "query_control_count": len(all_control_rows),
        "carrier_batch_count": len(copied_carriers),
        "policy_query_counts": policy_counts,
        "policy_status_counts": status_counts,
        "group_count": len(all_group_summaries),
        "group_summaries": all_group_summaries,
        "pending_controls": pending_controls,
        "gate": {
            "real_D4RT_R0_C0_C1_C2_C3_C4_q64_executed": not missing_real,
            "missing_or_incomplete_real_policies": missing_real,
            "association_controls_R1_R2_pending": True,
            "pass": False,
        },
        "uses_gt_for_prediction": False,
        "uses_gt_for_diagnostic_labels": False,
    }
    _write_json(output_root / "query_execution_summary.json", summary)
    _write_csv(output_root / "query_result_rows.csv", all_query_rows)
    _write_csv(output_root / "query_control_rows.csv", all_control_rows)
    _write_csv(output_root / "source_run_rows.csv", source_run_rows)
    print(
        {
            "output_root": str(output_root),
            "query_result_count": len(all_query_rows),
            "carrier_batch_count": len(copied_carriers),
            "gate": summary["gate"],
        }
    )


if __name__ == "__main__":
    main()
