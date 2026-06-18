from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import json
import numpy as np


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@dataclass
class TubeRecord:
    tube_id: int
    persistent_tube_id: int
    chunk_id: int
    submap_id: int
    source_frame_global: int
    source_xy: tuple[int, int]
    source_uv: tuple[float, float]
    target_frames_global: np.ndarray
    uv: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    xyz_local: np.ndarray
    xyz_ref0: np.ndarray
    xyz_canonical: np.ndarray | None = None
    T_chunk_to_canonical: dict[str, Any] | None = None
    alignment_quality: dict[str, Any] = field(default_factory=dict)
    coordinate_frame: str = "unknown"
    scale_status: str = "unknown"
    allow_metric_merge: bool = False
    alignment_source: str = "unknown"
    transform_id: str | None = None

    @property
    def source_pixel_key(self) -> str:
        x, y = self.source_xy
        return f"{int(self.source_frame_global)}:{int(x)}:{int(y)}"

    def validate(self) -> None:
        if self.coordinate_frame == "unknown":
            raise ValueError("TubeRecord.coordinate_frame must not be unknown")
        if self.coordinate_frame == "d4rt_canonical" and self.xyz_canonical is None:
            raise ValueError("d4rt_canonical TubeRecord requires xyz_canonical")
        if self.alignment_source == "eval_gt_sim3" and self.allow_metric_merge:
            raise ValueError("eval_gt_sim3 TubeRecord cannot allow method metric merge")

    def to_jsonable(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("target_frames_global", "uv", "visibility", "confidence", "xyz_local", "xyz_ref0", "xyz_canonical"):
            value = out.get(key)
            if isinstance(value, np.ndarray):
                out[key] = value.tolist()
        out["T_chunk_to_canonical"] = _jsonable_value(out.get("T_chunk_to_canonical"))
        out["alignment_quality"] = _jsonable_value(out.get("alignment_quality", {}))
        return out

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> "TubeRecord":
        data = dict(payload)
        for key in ("target_frames_global", "uv", "visibility", "confidence", "xyz_local", "xyz_ref0", "xyz_canonical"):
            if data.get(key) is not None:
                dtype = np.int64 if key == "target_frames_global" else np.float32
                data[key] = np.asarray(data[key], dtype=dtype)
        data["source_xy"] = tuple(int(v) for v in data["source_xy"])
        data["source_uv"] = tuple(float(v) for v in data["source_uv"])
        return cls(**data)

    def get_geometry_for_measurement(self, *, field: str = "uv") -> np.ndarray:
        """Return only image-space fields for measurement construction."""

        if field == "uv":
            return np.asarray(self.uv, dtype=np.float32)
        if field == "visibility":
            return np.asarray(self.visibility, dtype=np.float32)
        if field == "confidence":
            return np.asarray(self.confidence, dtype=np.float32)
        raise ValueError(f"unsupported measurement geometry field: {field}")

    def _merge_geometry_array(self, field: str) -> np.ndarray:
        if field == "xyz_canonical":
            if self.xyz_canonical is None:
                raise MergeGeometryError(f"tube {self.tube_id} has no xyz_canonical")
            return np.asarray(self.xyz_canonical, dtype=np.float32)
        if field == "xyz_local":
            return np.asarray(self.xyz_local, dtype=np.float32)
        if field == "xyz_ref0":
            return np.asarray(self.xyz_ref0, dtype=np.float32)
        raise MergeGeometryError(f"unsupported merge geometry field: {field}")

    def get_geometry_for_merge(
        self,
        other: "TubeRecord",
        context: str,
        *,
        merge_type: str = "metric_edge",
        event_logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return guarded geometry for a method-path metric merge."""

        event = assert_merge_geometry_valid(self, other, context, merge_type=merge_type)
        same_chunk = int(self.chunk_id) == int(other.chunk_id)
        if not same_chunk:
            field = "xyz_canonical"
        elif self.coordinate_frame == "d4rt_canonical" and other.coordinate_frame == "d4rt_canonical":
            field = "xyz_canonical"
        elif self.coordinate_frame == "ref0_local" and other.coordinate_frame == "ref0_local":
            field = "xyz_ref0"
        else:
            field = "xyz_local"
        event["geometry_field_used"] = field
        if event_logger is not None:
            event_logger(dict(event))
        return self._merge_geometry_array(field), other._merge_geometry_array(field), event


class MergeGeometryError(ValueError):
    pass


def _tube_id(tube: TubeRecord | Any) -> int:
    return int(getattr(tube, "tube_id", -1))


def assert_merge_geometry_valid(
    tube_i: TubeRecord,
    tube_j: TubeRecord,
    context: str,
    *,
    merge_type: str = "metric_edge",
) -> dict[str, Any]:
    """Validate geometry provenance before a method-path metric merge.

    Same-chunk merges may use per-chunk local coordinates, but every cross-chunk
    metric merge must use D4RT-only canonical coordinates with a passing
    alignment gate. Evaluation-aligned geometry is forbidden.
    """

    tube_i.validate()
    tube_j.validate()
    same_chunk = int(tube_i.chunk_id) == int(tube_j.chunk_id)
    same_submap = int(tube_i.submap_id) == int(tube_j.submap_id)
    event = {
        "merge_type": merge_type,
        "context": str(context),
        "tube_i": _tube_id(tube_i),
        "tube_j": _tube_id(tube_j),
        "chunk_i": int(tube_i.chunk_id),
        "chunk_j": int(tube_j.chunk_id),
        "submap_i": int(tube_i.submap_id),
        "submap_j": int(tube_j.submap_id),
        "coordinate_frame_used": tube_i.coordinate_frame if tube_i.coordinate_frame == tube_j.coordinate_frame else "mixed",
        "alignment_source": tube_i.alignment_source if tube_i.alignment_source == tube_j.alignment_source else "mixed",
        "transform_i": tube_i.transform_id,
        "transform_j": tube_j.transform_id,
        "allow_metric_merge": bool(tube_i.allow_metric_merge and tube_j.allow_metric_merge),
        "guard_pass": False,
        "guard_reason": "",
    }
    if tube_i.alignment_source == "eval_gt_sim3" or tube_j.alignment_source == "eval_gt_sim3":
        event["guard_reason"] = "eval_aligned_geometry_forbidden"
        raise MergeGeometryError(json.dumps(event, sort_keys=True))
    if same_chunk:
        if tube_i.coordinate_frame not in {"chunk_local", "ref0_local", "d4rt_canonical"}:
            event["guard_reason"] = "invalid_same_chunk_coordinate_frame"
            raise MergeGeometryError(json.dumps(event, sort_keys=True))
        event["guard_pass"] = True
        event["guard_reason"] = "same_chunk_identity"
        return event
    if not same_submap:
        event["guard_reason"] = "cross_submap_metric_merge_forbidden"
        raise MergeGeometryError(json.dumps(event, sort_keys=True))
    if tube_i.coordinate_frame != "d4rt_canonical" or tube_j.coordinate_frame != "d4rt_canonical":
        event["guard_reason"] = "cross_chunk_requires_xyz_canonical"
        raise MergeGeometryError(json.dumps(event, sort_keys=True))
    allowed_cross_chunk_sources = {"same_chunk_identity", "d4rt_self_sim3"}
    if tube_i.alignment_source not in allowed_cross_chunk_sources or tube_j.alignment_source not in allowed_cross_chunk_sources:
        event["guard_reason"] = "cross_chunk_requires_d4rt_self_sim3"
        raise MergeGeometryError(json.dumps(event, sort_keys=True))
    if not (tube_i.allow_metric_merge and tube_j.allow_metric_merge):
        event["guard_reason"] = "metric_merge_disabled_by_alignment"
        raise MergeGeometryError(json.dumps(event, sort_keys=True))
    if not bool(tube_i.alignment_quality.get("pass_gate", False)) or not bool(tube_j.alignment_quality.get("pass_gate", False)):
        event["guard_reason"] = "alignment_quality_gate_failed"
        raise MergeGeometryError(json.dumps(event, sort_keys=True))
    event["guard_pass"] = True
    event["guard_reason"] = "cross_chunk_canonical_self_sim3"
    return event


def write_tube_records_jsonl(path: str | Path, tubes: list[TubeRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for tube in tubes:
            tube.validate()
            fh.write(json.dumps(tube.to_jsonable(), sort_keys=True) + "\n")


def read_tube_records_jsonl(path: str | Path) -> list[TubeRecord]:
    out: list[TubeRecord] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(TubeRecord.from_jsonable(json.loads(line)))
    return out
