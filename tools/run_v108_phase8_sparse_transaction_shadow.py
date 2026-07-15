#!/usr/bin/env python3
"""Shadow sparse-transaction audit for v108 Phase8.

The real path only queues durable memory requests from rows that already passed
visual review and physical/watcher checks. It never applies SAM2 mutations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "Stream3D"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from Stream3D.stream4d_v108.transaction_manager import (  # noqa: E402
    Plane,
    Sam2MemoryMutationRequest,
    SparseTransactionScheduler,
    TransactionManager,
)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_path(text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    return ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key)) for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def requests_from_audit_rows(rows: list[dict[str, str]]) -> tuple[list[Sam2MemoryMutationRequest], list[dict[str, Any]]]:
    requests: list[Sam2MemoryMutationRequest] = []
    request_rows: list[dict[str, Any]] = []
    for row in rows:
        accepted = parse_bool(row.get("durable_memory_allowed", "false"))
        scene_id = str(row.get("scene_id", ""))
        object_id = parse_int(row.get("object_id", -1), -1)
        frame_id = parse_int(row.get("frame_id", -1), -1)
        if not accepted:
            request_rows.append(
                {
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "frame_id": frame_id,
                    "queued": False,
                    "skip_reason": "durable_memory_allowed_false",
                    "visual_review_status": row.get("visual_review_status", ""),
                    "durable_reject_reasons": row.get("durable_reject_reasons", ""),
                }
            )
            continue
        request = Sam2MemoryMutationRequest(
            frame_id=frame_id,
            global_object_id=object_id,
            sam2_runtime_object_id=None,
            mutation="ADMIT_OR_REFRESH_DURABLE_OBJECT",
            prompt_count=parse_int(row.get("projected_positive_count", 0), 0),
            evidence_status=str(row.get("visual_review_status", "")),
        )
        requests.append(request)
        request_rows.append(
            {
                "scene_id": scene_id,
                "object_id": object_id,
                "frame_id": frame_id,
                "queued": True,
                "skip_reason": "",
                "visual_review_status": row.get("visual_review_status", ""),
                "durable_reject_reasons": row.get("durable_reject_reasons", ""),
            }
        )
    return requests, request_rows


def synthetic_requests(count: int) -> list[Sam2MemoryMutationRequest]:
    out: list[Sam2MemoryMutationRequest] = []
    for idx in range(int(count)):
        out.append(
            Sam2MemoryMutationRequest(
                frame_id=100 + idx,
                global_object_id=1000 + idx,
                sam2_runtime_object_id=None,
                mutation="SYNTHETIC_ADMIT_OR_REFRESH",
                prompt_count=4,
                evidence_status="SYNTHETIC_CODE_SMOKE_NOT_EXPERIMENT",
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-audit-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", default="event_driven")
    parser.add_argument("--max-requests-per-batch", type=int, default=4)
    parser.add_argument("--synthetic-smoke-count", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    output_root = resolve_path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "last_command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n",
        encoding="utf-8",
    )
    audit_csv = resolve_path(str(args.phase7_audit_csv))
    audit_rows = read_csv_rows(audit_csv)
    real_requests, request_rows = requests_from_audit_rows(audit_rows)
    synthetic = synthetic_requests(int(args.synthetic_smoke_count))
    scheduler = SparseTransactionScheduler(
        mode=str(args.mode),
        max_requests_per_batch=int(args.max_requests_per_batch),
    )
    real_batches = scheduler.build_batches(real_requests)
    synthetic_batches = scheduler.build_batches(synthetic)

    manager = TransactionManager()
    transaction_rows: list[dict[str, Any]] = []
    for batch in real_batches:
        for request in batch.requests:
            tx = manager.propose(
                frame_id=int(request.frame_id),
                global_object_id=int(request.global_object_id),
                plane=Plane.SAM2_MEMORY,
                action=str(request.mutation),
                evidence=request.as_transaction_evidence(),
            )
            queued = manager.shadow_queue(tx, reason="shadow Phase8 audit; no SAM2 memory mutation applied")
            transaction_rows.append(
                {
                    "transaction_id": queued.transaction_id,
                    "frame_id": queued.frame_id,
                    "global_object_id": queued.global_object_id,
                    "plane": queued.plane.value,
                    "action": queued.action,
                    "status": queued.status.value,
                    "reason": queued.reason,
                    "synthetic": False,
                }
            )

    batch_rows = [row for batch in real_batches for row in batch.as_rows()]
    synthetic_batch_rows = [row for batch in synthetic_batches for row in batch.as_rows()]

    request_csv = output_root / "request_rows.csv"
    batch_csv = output_root / "transaction_batch_rows.csv"
    synthetic_csv = output_root / "synthetic_smoke_batch_rows.csv"
    transaction_csv = output_root / "shadow_transaction_rows.csv"
    write_csv(request_csv, request_rows)
    write_csv(batch_csv, batch_rows)
    write_csv(synthetic_csv, synthetic_batch_rows)
    write_csv(transaction_csv, transaction_rows)
    summary_path = output_root / "phase8_sparse_transaction_shadow_summary.json"
    summary = {
        "schema_version": "stream4d_v108_phase8_sparse_transaction_shadow_v1",
        "created_unix_time": time.time(),
        "runtime_sec": float(time.time() - started),
        "phase7_audit_csv": rel(audit_csv),
        "phase7_audit_csv_sha256": sha256_file(audit_csv),
        "input_row_count": int(len(audit_rows)),
        "real_pending_request_count": int(len(real_requests)),
        "real_batch_count": int(len(real_batches)),
        "real_shadow_transaction_count": int(len(transaction_rows)),
        "synthetic_smoke_request_count": int(len(synthetic)),
        "synthetic_smoke_batch_count": int(len(synthetic_batches)),
        "mode": str(args.mode),
        "max_requests_per_batch": int(args.max_requests_per_batch),
        "request_rows_csv": rel(request_csv),
        "request_rows_csv_sha256": sha256_file(request_csv),
        "transaction_batch_rows_csv": rel(batch_csv),
        "transaction_batch_rows_csv_sha256": sha256_file(batch_csv),
        "synthetic_smoke_batch_rows_csv": rel(synthetic_csv),
        "synthetic_smoke_batch_rows_csv_sha256": sha256_file(synthetic_csv),
        "shadow_transaction_rows_csv": rel(transaction_csv),
        "shadow_transaction_rows_csv_sha256": sha256_file(transaction_csv),
        "collateral_drift_measurement": "not_measured_no_real_sam2_mutation",
        "visual_acceptance_boundary": "real requests are queued only when durable_memory_allowed is true in Phase7 audit",
        "metrics_are_diagnostic_only": True,
        "shadow_only": True,
    }
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": rel(summary_path),
                "real_pending_request_count": len(real_requests),
                "real_batch_count": len(real_batches),
                "synthetic_smoke_batch_count": len(synthetic_batches),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
