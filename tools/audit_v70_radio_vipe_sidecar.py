#!/usr/bin/env python3
"""Audit RADIO-ViPE/RADSeg availability for ACL2 v70-v2."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from v70_radio_sidecar_common import (
    import_status,
    load_radseg_encoder,
    locate_default_radio_checkpoint,
    utc_now,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio-vipe-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lang-model", default="siglip2")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def _git_status(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {"github_repo_detected": False, "head": "", "remote": ""}
    head = ""
    remote = ""
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        head = ""
    try:
        remote = subprocess.check_output(["git", "-C", str(root), "remote", "-v"], text=True).strip()
    except Exception:
        remote = ""
    return {"github_repo_detected": True, "head": head, "remote": remote}


def _checkpoint_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in [
        Path("/mnt/data/users/chengshun.wang/.cache/torch/hub/checkpoints"),
        Path.home() / ".cache/torch/hub/checkpoints",
    ]:
        if not base.exists():
            continue
        for pattern in ("*radio*", "*RADIO*", "*radseg*", "*RADSeg*", "*dinov2*", "*dinov3*"):
            for path in sorted(base.glob(pattern)):
                if not path.is_file():
                    continue
                rows.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "partial": path.name.endswith(".partial"),
                        "family": "RADIO" if "radio" in path.name.lower() else ("RADSeg" if "radseg" in path.name.lower() else "other"),
                    }
                )
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["path"] in seen:
            continue
        seen.add(row["path"])
        out.append(row)
    return out


def _package_status() -> list[dict[str, Any]]:
    modules = ["torch", "torchvision", "timm", "segment_anything", "hydra", "omegaconf", "cv2", "sklearn", "PIL"]
    rows: list[dict[str, Any]] = []
    for module in modules:
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "")
            rows.append({"module": module, "ok": True, "version": str(version), "error_type": "", "error_message": ""})
        except Exception as exc:
            rows.append({"module": module, "ok": False, "version": "", "error_type": type(exc).__name__, "error_message": str(exc)})
    try:
        import torch

        rows.append(
            {
                "module": "torch.cuda",
                "ok": bool(torch.cuda.is_available()),
                "version": str(torch.version.cuda),
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "error_type": "",
                "error_message": "",
            }
        )
    except Exception as exc:
        rows.append({"module": "torch.cuda", "ok": False, "version": "", "error_type": type(exc).__name__, "error_message": str(exc)})
    return rows


def main() -> None:
    args = parse_args()
    os.environ["CONDA_PREFIX"] = sys.prefix
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    radio_root = args.radio_vipe_root
    checkpoint = args.checkpoint or locate_default_radio_checkpoint()

    imports = [
        import_status("semseg.image_encoders.radseg", radio_root),
        import_status("vipe.priors.embedding.radseg_encoder", radio_root),
        import_status("vipe.priors.embedding.radseg", radio_root),
        import_status("vipe.slam.semantic_flow", radio_root),
    ]

    entrypoint_candidates = [
        radio_root / "semseg/image_encoders/radseg.py",
        radio_root / "vipe/priors/embedding/radseg_encoder.py",
        radio_root / "run.py",
        radio_root / "scripts/semseg_eval.py",
        radio_root / "pca_component_calc.py",
    ]
    entrypoints = [
        {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        for path in entrypoint_candidates
    ]

    smoke: dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "device": args.device,
        "checkpoint": checkpoint,
        "feature_shape": None,
        "error_type": "",
        "error_message": "",
    }
    if not args.skip_smoke and radio_root.exists() and checkpoint is not None:
        smoke["attempted"] = True
        try:
            import numpy as np
            import torch
            from PIL import Image

            model = load_radseg_encoder(radio_root, checkpoint, args.device, args.lang_model, amp=False)
            frame = np.zeros((224, 448, 3), dtype=np.uint8)
            frame[:, :224, 0] = 255
            frame[:, 224:, 1] = 255
            tmp = out_dir / "_radio_smoke_input.png"
            Image.fromarray(frame).save(tmp)
            arr = np.asarray(Image.open(tmp).convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()[None].to(args.device)
            with torch.inference_mode():
                feat = model.encode_image_to_feat_map(tensor)
            smoke.update(
                {
                    "ok": True,
                    "feature_shape": list(feat.shape),
                    "feature_finite": bool(torch.isfinite(feat).all().item()),
                    "feature_abs_mean": float(feat.float().abs().mean().item()),
                    "patch_size": int(getattr(model.model, "patch_size", -1)),
                }
            )
        except Exception as exc:
            smoke.update({"ok": False, "error_type": type(exc).__name__, "error_message": str(exc)})

    required_packages = _package_status()
    checkpoint_rows = _checkpoint_rows()
    package_ok = {row["module"]: row["ok"] for row in required_packages}
    callable_feature_path_found = any(row["ok"] for row in imports[:2])
    checkpoint_available = checkpoint is not None and Path(str(checkpoint)).exists()
    pass_gate = bool(radio_root.exists() and callable_feature_path_found and checkpoint_available and (args.skip_smoke or smoke["ok"]))
    if not radio_root.exists():
        blocker = "blocked_missing_radio_vipe_code"
    elif not callable_feature_path_found:
        blocker = "blocked_missing_radseg_import"
    elif not checkpoint_available:
        blocker = "blocked_missing_checkpoint"
    elif not args.skip_smoke and not smoke["ok"]:
        blocker = "blocked_radio_single_image_smoke_failed"
    else:
        blocker = ""

    audit = {
        "created_at": utc_now(),
        "phase": "R0_radio_code_audit",
        "python": sys.executable,
        "cwd": os.getcwd(),
        "radio_vipe_root": str(radio_root),
        "radio_vipe_root_exists": radio_root.exists(),
        **_git_status(radio_root),
        "imports": imports,
        "entrypoints": entrypoints,
        "checkpoint": checkpoint,
        "checkpoint_available": checkpoint_available,
        "checkpoint_rows": checkpoint_rows,
        "required_packages": required_packages,
        "required_package_ok": package_ok,
        "license_note": "RADIO-ViPE source is Apache-2.0 per README/LICENSE; downloaded third-party models may have separate terms.",
        "gpu_memory_estimate": "c-radio_v3-b single-image smoke expected to fit on one 23GB RTX A5000; target chunks should run chunkwise.",
        "download_or_prepare_command_identified": checkpoint_available,
        "checkpoint_download_note": "Local checkpoint found; no download command executed. README suggests Docker/pip setup and torch.hub will use NVlabs/RADIO if missing.",
        "single_image_smoke": smoke,
        "gate_pass": pass_gate,
        "blocker": blocker,
    }
    write_json(out_dir / "radio_code_audit.json", audit)

    entry_md = [
        "# RADIO/RADSeg Entrypoints",
        "",
        f"- root: `{radio_root}`",
        f"- checkpoint: `{checkpoint}`",
        f"- gate_pass: `{pass_gate}`",
        f"- blocker: `{blocker or 'none'}`",
        "",
        "## Import Status",
        "",
    ]
    for row in imports:
        entry_md.append(f"- `{row['module']}`: `{row['ok']}` {row.get('error_type') or ''} {row.get('error_message') or ''}".rstrip())
    entry_md.extend(["", "## Candidate Files", ""])
    for row in entrypoints:
        entry_md.append(f"- `{row['path']}` exists=`{row['exists']}` bytes=`{row['bytes']}`")
    write_text(out_dir / "radio_entrypoints.md", "\n".join(entry_md) + "\n")

    dep_lines = [
        f"python={sys.executable}",
        f"radio_root_exists={radio_root.exists()}",
        f"checkpoint={checkpoint}",
        f"checkpoint_available={checkpoint_available}",
        f"single_image_smoke_ok={smoke['ok']}",
        f"gate_pass={pass_gate}",
        f"blocker={blocker}",
        "",
        json.dumps(required_packages, indent=2, sort_keys=True),
    ]
    write_text(out_dir / "radio_dependency_status.txt", "\n".join(dep_lines) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "gate_pass": pass_gate, "blocker": blocker}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
