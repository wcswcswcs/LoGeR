from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stream4d_native.frozen_feature_adapter import (
    FrozenFeatureAdapter,
    locate_default_cradio_checkpoint,
    locate_default_dinov2_checkpoint,
    locate_default_radio_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _import_status(module: str, extra_path: Path | None = None) -> dict[str, Any]:
    old_path = list(sys.path)
    if extra_path is not None:
        sys.path.insert(0, str(extra_path))
    try:
        importlib.import_module(module)
        return {"module": module, "ok": True, "failure_stage": "", "error_type": "", "error_message": ""}
    except Exception as exc:
        return {
            "module": module,
            "ok": False,
            "failure_stage": "import",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    finally:
        sys.path[:] = old_path


def _checkpoint_rows() -> list[dict[str, Any]]:
    checkpoint_dir = Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints")
    rows: list[dict[str, Any]] = []
    for pattern, family in [
        ("*radio*", "RADIO"),
        ("*RADIO*", "RADIO"),
        ("*radseg*", "RADSeg"),
        ("*dinov2*", "DINOv2"),
        ("*sam*", "SAM"),
    ]:
        for path in sorted(checkpoint_dir.glob(pattern)):
            rows.append({"family": family, "path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0})
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radio-root", default=str(REPO_ROOT / "third_party/RADIO-ViPE"))
    parser.add_argument("--output-root", default="outputs/audit/v42_source_audit")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ.setdefault("CONDA_PREFIX", sys.prefix)
    radio_root = Path(args.radio_root)
    imports = [
        _import_status("semseg.image_encoders.radseg", radio_root),
        _import_status("vipe.priors.embedding.radseg_encoder", radio_root),
        _import_status("vipe.priors.embedding.dinov2", radio_root),
    ]
    checkpoint_rows = _checkpoint_rows()
    dino_checkpoint = locate_default_dinov2_checkpoint()
    radio_checkpoint = locate_default_radio_checkpoint()
    cradio_checkpoint = locate_default_cradio_checkpoint()

    smoke: dict[str, Any] = {"backend": "rgb_stats", "ok": False}
    try:
        frame = np.zeros((8, 10, 3), dtype=np.uint8)
        frame[:, :5, 0] = 255
        mask = np.zeros((8, 10), dtype=bool)
        mask[:, :5] = True
        adapter = FrozenFeatureAdapter(backend="rgb_stats", device=str(args.device))
        fmap = adapter.extract_dense_features(frame)
        pooled = adapter.pool_mask_feature(fmap, mask)
        smoke.update(
            {
                "ok": True,
                "feature_shape": list(fmap.features.shape),
                "pooled_shape": list(pooled.shape),
                "boundary_contrast": adapter.compute_boundary_contrast(fmap, mask),
            }
        )
    except Exception as exc:
        smoke.update({"ok": False, "failure_stage": "feature_smoke", "error_type": type(exc).__name__, "error_message": str(exc)})

    radio_smoke: dict[str, Any] = {"backend": "radio_radseg", "ok": False, "checkpoint": radio_checkpoint}
    if radio_checkpoint is not None:
        try:
            frame = np.zeros((224, 320, 3), dtype=np.uint8)
            frame[:, :160, 0] = 255
            frame[:, 160:, 1] = 255
            mask = np.zeros((224, 320), dtype=bool)
            mask[:, :160] = True
            adapter = FrozenFeatureAdapter(backend="radio_radseg", device=str(args.device), checkpoint=radio_checkpoint)
            fmap = adapter.extract_dense_features(frame)
            pooled = adapter.pool_mask_feature(fmap, mask)
            radio_smoke.update(
                {
                    "ok": True,
                    "feature_shape": list(fmap.features.shape),
                    "pooled_shape": list(pooled.shape),
                    "patch_size": fmap.patch_size,
                    "radio_lang_model": adapter.radio_lang_model,
                    "radio_lang_align": adapter.radio_lang_align,
                }
            )
        except Exception as exc:
            radio_smoke.update({"ok": False, "failure_stage": "radio_feature_smoke", "error_type": type(exc).__name__, "error_message": str(exc)})

    radio_import_ok = any(row["ok"] for row in imports[:2])
    radio_available = bool(radio_import_ok and radio_smoke.get("ok"))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v42_radio_vipe_availability",
        "radio_vipe_path": str(radio_root),
        "radio_vipe_path_exists": radio_root.exists(),
        "imports": imports,
        "checkpoint_rows": checkpoint_rows,
        "radio_or_radseg_import_available": bool(radio_import_ok),
        "radio_available": radio_available,
        "radio_checkpoint": radio_checkpoint,
        "cradio_checkpoint": cradio_checkpoint,
        "radio_feature_smoke": radio_smoke,
        "dinov2_checkpoint": dino_checkpoint,
        "dinov2_fallback_available": dino_checkpoint is not None,
        "feature_adapter_contract_smoke": smoke,
        "failure_stage": "" if radio_available else ("radio_feature_smoke" if radio_import_ok else "import"),
        "note": "RADIO/RADSeg availability requires import, local checkpoint, and a lightweight frozen-feature smoke.",
    }
    source_availability = {
        "sources": [
            {
                "source": "RADIO/RADSeg via third_party/RADIO-ViPE",
                "source_available": payload["radio_available"],
                "checkpoint": radio_checkpoint,
                "failure_stage": payload["failure_stage"],
                "counts_as_frozen_semantic_model": bool(payload["radio_available"]),
            },
            {
                "source": "DINOv2 local checkpoint fallback",
                "source_available": dino_checkpoint is not None,
                "checkpoint": dino_checkpoint,
                "failure_stage": "" if dino_checkpoint is not None else "checkpoint",
                "counts_as_frozen_semantic_model": dino_checkpoint is not None,
            },
            {
                "source": "rgb_stats adapter smoke",
                "source_available": bool(smoke.get("ok")),
                "failure_stage": "" if smoke.get("ok") else smoke.get("failure_stage", "feature_smoke"),
                "counts_as_frozen_semantic_model": False,
            },
        ]
    }
    out = ROOT / args.output_root
    _write_json(out / "radio_vipe_availability.json", payload)
    _write_json(out / "source_availability.json", source_availability)
    print(json.dumps({"availability": str(out / "radio_vipe_availability.json"), "radio_available": payload["radio_available"], "dinov2_fallback_available": payload["dinov2_fallback_available"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
