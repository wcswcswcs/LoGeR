from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Plane(str, Enum):
    OUTPUT = "output_plane"
    SAM2_MEMORY = "sam2_memory_plane"


class TransactionStatus(str, Enum):
    PROPOSED = "PROPOSED"
    APPLIED_OUTPUT_ONLY = "APPLIED_OUTPUT_ONLY"
    APPLIED_MEMORY = "APPLIED_MEMORY"
    REJECTED = "REJECTED"
    SHADOW_QUEUED = "SHADOW_QUEUED"


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    frame_id: int
    global_object_id: int | None
    plane: Plane
    action: str
    evidence: dict[str, Any] = field(default_factory=dict)
    status: TransactionStatus = TransactionStatus.PROPOSED
    reason: str = ""


class TransactionManager:
    """Keeps output-plane decisions separate from durable SAM2 mutations."""

    def __init__(self) -> None:
        self._transactions: list[Transaction] = []

    @property
    def transactions(self) -> tuple[Transaction, ...]:
        return tuple(self._transactions)

    def propose(
        self,
        frame_id: int,
        global_object_id: int | None,
        plane: Plane,
        action: str,
        evidence: dict[str, Any] | None = None,
    ) -> Transaction:
        tx = Transaction(
            transaction_id=f"tx_{len(self._transactions):06d}",
            frame_id=frame_id,
            global_object_id=global_object_id,
            plane=plane,
            action=action,
            evidence=evidence or {},
        )
        self._transactions.append(tx)
        return tx

    def apply_output_only(self, tx: Transaction, reason: str) -> Transaction:
        if tx.plane is not Plane.OUTPUT:
            raise ValueError("output-only apply requires Plane.OUTPUT")
        return self._replace(tx, status=TransactionStatus.APPLIED_OUTPUT_ONLY, reason=reason)

    def apply_memory(self, tx: Transaction, reason: str) -> Transaction:
        if tx.plane is not Plane.SAM2_MEMORY:
            raise ValueError("durable memory apply requires Plane.SAM2_MEMORY")
        return self._replace(tx, status=TransactionStatus.APPLIED_MEMORY, reason=reason)

    def reject(self, tx: Transaction, reason: str) -> Transaction:
        return self._replace(tx, status=TransactionStatus.REJECTED, reason=reason)

    def shadow_queue(self, tx: Transaction, reason: str) -> Transaction:
        return self._replace(tx, status=TransactionStatus.SHADOW_QUEUED, reason=reason)

    def _replace(self, tx: Transaction, status: TransactionStatus, reason: str) -> Transaction:
        updated = Transaction(
            transaction_id=tx.transaction_id,
            frame_id=tx.frame_id,
            global_object_id=tx.global_object_id,
            plane=tx.plane,
            action=tx.action,
            evidence=tx.evidence,
            status=status,
            reason=reason,
        )
        for idx, existing in enumerate(self._transactions):
            if existing.transaction_id == tx.transaction_id:
                self._transactions[idx] = updated
                return updated
        raise KeyError(tx.transaction_id)


@dataclass(frozen=True)
class Sam2MemoryMutationRequest:
    frame_id: int
    global_object_id: int
    sam2_runtime_object_id: int | None
    mutation: str
    prompt_count: int = 0
    evidence_status: str = "RECORDED_NOT_ACCEPTED"

    def as_transaction_evidence(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "global_object_id": self.global_object_id,
            "sam2_runtime_object_id": self.sam2_runtime_object_id,
            "mutation": self.mutation,
            "prompt_count": self.prompt_count,
            "evidence_status": self.evidence_status,
        }


@dataclass(frozen=True)
class SparseTransactionBatch:
    batch_id: str
    mode: str
    request_count: int
    frame_id_min: int | None
    frame_id_max: int | None
    global_object_ids: tuple[int, ...]
    requests: tuple[Sam2MemoryMutationRequest, ...]
    shadow_only: bool = True
    reason: str = ""

    def as_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, request in enumerate(self.requests):
            rows.append(
                {
                    "batch_id": self.batch_id,
                    "batch_request_index": int(idx),
                    "mode": self.mode,
                    "request_count": int(self.request_count),
                    "frame_id_min": self.frame_id_min,
                    "frame_id_max": self.frame_id_max,
                    "global_object_ids": list(self.global_object_ids),
                    "frame_id": int(request.frame_id),
                    "global_object_id": int(request.global_object_id),
                    "sam2_runtime_object_id": request.sam2_runtime_object_id,
                    "mutation": request.mutation,
                    "prompt_count": int(request.prompt_count),
                    "evidence_status": request.evidence_status,
                    "shadow_only": bool(self.shadow_only),
                    "reason": self.reason,
                }
            )
        if not self.requests:
            rows.append(
                {
                    "batch_id": self.batch_id,
                    "batch_request_index": -1,
                    "mode": self.mode,
                    "request_count": 0,
                    "frame_id_min": self.frame_id_min,
                    "frame_id_max": self.frame_id_max,
                    "global_object_ids": [],
                    "frame_id": -1,
                    "global_object_id": -1,
                    "sam2_runtime_object_id": None,
                    "mutation": "",
                    "prompt_count": 0,
                    "evidence_status": "",
                    "shadow_only": bool(self.shadow_only),
                    "reason": self.reason,
                }
            )
        return rows


class SparseTransactionScheduler:
    """Groups durable memory mutation requests without applying them."""

    def __init__(self, *, mode: str = "event_driven", max_requests_per_batch: int = 4) -> None:
        if int(max_requests_per_batch) <= 0:
            raise ValueError("max_requests_per_batch must be positive")
        self.mode = str(mode)
        self.max_requests_per_batch = int(max_requests_per_batch)

    def build_batches(self, requests: list[Sam2MemoryMutationRequest]) -> list[SparseTransactionBatch]:
        ordered = sorted(requests, key=lambda req: (int(req.frame_id), int(req.global_object_id), str(req.mutation)))
        batches: list[SparseTransactionBatch] = []
        if not ordered:
            return batches
        for start in range(0, len(ordered), self.max_requests_per_batch):
            chunk = tuple(ordered[start : start + self.max_requests_per_batch])
            frame_ids = [int(req.frame_id) for req in chunk]
            object_ids = tuple(sorted({int(req.global_object_id) for req in chunk}))
            batches.append(
                SparseTransactionBatch(
                    batch_id=f"batch_{len(batches):06d}",
                    mode=self.mode,
                    request_count=int(len(chunk)),
                    frame_id_min=min(frame_ids) if frame_ids else None,
                    frame_id_max=max(frame_ids) if frame_ids else None,
                    global_object_ids=object_ids,
                    requests=chunk,
                    reason="shadow sparse transaction grouping; no SAM2 memory mutation applied",
                )
            )
        return batches
