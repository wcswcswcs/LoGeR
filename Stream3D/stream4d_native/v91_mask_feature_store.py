from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class MaskFeatureStore:
    features: np.ndarray
    scene_ids: np.ndarray
    frame_ids: np.ndarray
    mask_ids: np.ndarray
    observation_ids: np.ndarray
    feature_sha256: np.ndarray
    backend: str
    layer: str

    def as_keyed_dict(self) -> dict[tuple[str, int, int], np.ndarray]:
        out: dict[tuple[str, int, int], np.ndarray] = {}
        for i in range(int(self.features.shape[0])):
            key = (str(self.scene_ids[i]), int(self.frame_ids[i]), int(self.mask_ids[i]))
            vec = np.asarray(self.features[i], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8 and np.all(np.isfinite(vec)):
                out[key] = vec / norm
        return out


def _sha256_array(vec: np.ndarray) -> str:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_mask_feature_store(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    features: np.ndarray,
    backend: str,
    layer: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"features must be N x D, got shape={arr.shape}")
    if arr.shape[0] != len(rows):
        raise ValueError(f"feature row mismatch: features={arr.shape[0]} rows={len(rows)}")
    if arr.shape[0] == 0:
        raise ValueError("feature store cannot be empty")

    scene_ids = np.asarray([str(row.get("scene_id", "")) for row in rows])
    frame_ids = np.asarray([int(row.get("frame_id", -1)) for row in rows], dtype=np.int32)
    mask_ids = np.asarray([int(row.get("mask_id", -1)) for row in rows], dtype=np.int32)
    observation_ids = np.asarray([str(row.get("mask_observation_id", "")) for row in rows])
    feature_sha = np.asarray([_sha256_array(arr[i]) for i in range(arr.shape[0])])
    store_path = output_dir / "mask_features.npz"
    np.savez_compressed(
        store_path,
        features=arr,
        scene_id=scene_ids,
        frame_id=frame_ids,
        mask_id=mask_ids,
        mask_observation_id=observation_ids,
        feature_sha256=feature_sha,
        backend=np.asarray(str(backend)),
        layer=np.asarray(str(layer)),
    )

    index_path = output_dir / "mask_feature_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["row_index", "scene_id", "frame_id", "mask_id", "mask_observation_id", "feature_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow(
                {
                    "row_index": i,
                    "scene_id": row.get("scene_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "mask_id": row.get("mask_id", ""),
                    "mask_observation_id": row.get("mask_observation_id", ""),
                    "feature_sha256": str(feature_sha[i]),
                }
            )

    manifest = {
        "schema": "stream4d_mask_feature_store_v1",
        "backend": str(backend),
        "layer": str(layer),
        "row_count": int(arr.shape[0]),
        "feature_dim": int(arr.shape[1]),
        "feature_dtype": str(arr.dtype),
        "store_path": str(store_path),
        "index_path": str(index_path),
        "store_sha256": _sha256_file(store_path),
        "index_sha256": _sha256_file(index_path),
        "metadata": metadata or {},
    }
    manifest_path = output_dir / "feature_store_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_mask_feature_store(path: Path) -> MaskFeatureStore:
    store_path = path / "mask_features.npz" if path.is_dir() else path
    with np.load(store_path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
        scene_ids = np.asarray(payload["scene_id"])
        frame_ids = np.asarray(payload["frame_id"], dtype=np.int32)
        mask_ids = np.asarray(payload["mask_id"], dtype=np.int32)
        observation_ids = np.asarray(payload["mask_observation_id"])
        feature_sha = np.asarray(payload["feature_sha256"])
        backend = str(np.asarray(payload["backend"]).item())
        layer = str(np.asarray(payload["layer"]).item())
    return MaskFeatureStore(
        features=features,
        scene_ids=scene_ids,
        frame_ids=frame_ids,
        mask_ids=mask_ids,
        observation_ids=observation_ids,
        feature_sha256=feature_sha,
        backend=backend,
        layer=layer,
    )


def merge_mask_feature_stores(paths: Iterable[Path], output_dir: Path, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    stores = [load_mask_feature_store(path) for path in paths]
    if not stores:
        raise ValueError("no feature stores to merge")
    backend = stores[0].backend
    layer = stores[0].layer
    if any(store.backend != backend or store.layer != layer for store in stores):
        raise ValueError("cannot merge stores with different backend/layer")
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    seen: set[tuple[str, int, int]] = set()
    for store in stores:
        for i in range(int(store.features.shape[0])):
            key = (str(store.scene_ids[i]), int(store.frame_ids[i]), int(store.mask_ids[i]))
            if key in seen:
                raise ValueError(f"duplicate mask feature key: {key}")
            seen.add(key)
            rows.append(
                {
                    "scene_id": key[0],
                    "frame_id": key[1],
                    "mask_id": key[2],
                    "mask_observation_id": str(store.observation_ids[i]),
                }
            )
            features.append(np.asarray(store.features[i], dtype=np.float32))
    return save_mask_feature_store(
        output_dir=output_dir,
        rows=rows,
        features=np.stack(features, axis=0),
        backend=backend,
        layer=layer,
        metadata=metadata or {},
    )
