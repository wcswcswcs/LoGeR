from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _maybe_copy_ckpt(src: Path, dst_arg: str) -> tuple[Path, dict[str, Any]]:
    if not dst_arg:
        return src, {"local_copy_requested": False}
    dst = Path(dst_arg)
    if dst.exists() and dst.is_dir():
        dst = dst / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    copied = False
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
        copied = True
    return dst, {
        "local_copy_requested": True,
        "local_ckpt_path": str(dst),
        "local_copy_performed": copied,
        "seconds_local_copy": float(time.time() - t0),
    }


def _gpu_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        payload.update(
            {
                "cuda_current_device": int(device),
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_memory_allocated": int(torch.cuda.memory_allocated(device)),
                "cuda_memory_reserved": int(torch.cuda.memory_reserved(device)),
            }
        )
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    d4rt_root = Path(args.d4rt_root).resolve()
    model_config = Path(args.d4rt_config).resolve()
    ckpt_path = Path(args.d4rt_ckpt).resolve()
    payload: dict[str, Any] = {
        "d4rt_root": str(d4rt_root),
        "model_config": str(model_config),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_exists": ckpt_path.exists(),
        "checkpoint_size_bytes": int(ckpt_path.stat().st_size) if ckpt_path.exists() else 0,
        "device": args.device,
        "ok": False,
        "error": None,
    }
    try:
        if not d4rt_root.exists():
            raise FileNotFoundError(d4rt_root)
        if not model_config.exists():
            raise FileNotFoundError(model_config)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        if args.sha256:
            sha_t0 = time.time()
            digest = _sha256(ckpt_path)
            payload["checkpoint_sha256"] = digest
            payload["checkpoint_sha256_prefix8"] = digest[:8]
            payload["seconds_sha256"] = float(time.time() - sha_t0)

        load_ckpt_path, copy_payload = _maybe_copy_ckpt(ckpt_path, args.local_ckpt_copy)
        payload.update(copy_payload)
        sys.path.insert(0, str(d4rt_root))
        from src.core.config import load_yaml_config
        from src.model.builder import build_model

        cfg_t0 = time.time()
        cfg = load_yaml_config(model_config)
        payload["seconds_load_config"] = float(time.time() - cfg_t0)
        image_hw = tuple(int(v) for v in cfg.get_path("model.input.image_size", [256, 256]))
        clip_frames = int(cfg.get_path("model.input.clip_frames", 48))
        payload["image_hw"] = list(image_hw)
        payload["clip_frames"] = clip_frames

        build_t0 = time.time()
        model = build_model(cfg["model"])
        payload["seconds_build_model"] = float(time.time() - build_t0)

        torch_load_t0 = time.time()
        ckpt = torch.load(load_ckpt_path, map_location="cpu")
        payload["seconds_torch_load"] = float(time.time() - torch_load_t0)
        state = None
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model", "module", "network", "net"):
                if isinstance(ckpt.get(key), dict):
                    state = ckpt[key]
                    payload["state_dict_key"] = key
                    break
            if state is None and ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
                state = ckpt
                payload["state_dict_key"] = "root"
        if state is None:
            raise ValueError("No usable state_dict in checkpoint")

        load_state_t0 = time.time()
        missing, unexpected = model.load_state_dict(state, strict=False)
        payload["seconds_load_state_dict"] = float(time.time() - load_state_t0)
        payload["missing_state_keys"] = len(missing)
        payload["unexpected_state_keys"] = len(unexpected)
        if missing or unexpected:
            payload["first_missing_state_keys"] = list(missing[:5])
            payload["first_unexpected_state_keys"] = list(unexpected[:5])

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        device = torch.device(args.device)
        to_device_t0 = time.time()
        model.to(device)
        model.eval()
        payload["seconds_model_to_device"] = float(time.time() - to_device_t0)

        frames = min(max(2, int(args.fake_frames)), clip_frames)
        h, w = image_hw
        fake = torch.zeros((1, frames, 3, h, w), dtype=torch.float32, device=device)
        aspect = torch.ones((1, 1), dtype=torch.float32, device=device)
        n = max(1, int(args.fake_queries))
        query = {
            "u": torch.linspace(0.25, 0.75, n, device=device).view(1, -1),
            "v": torch.linspace(0.25, 0.75, n, device=device).view(1, -1),
            "t_src": torch.zeros((1, n), dtype=torch.long, device=device),
            "t_tgt": torch.full((1, n), frames - 1, dtype=torch.long, device=device),
            "t_cam": torch.full((1, n), frames - 1, dtype=torch.long, device=device),
        }
        with torch.inference_mode():
            encode_t0 = time.time()
            memory = model.encode_video(video=fake, aspect_ratio=aspect)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            payload["seconds_fake_encode"] = float(time.time() - encode_t0)
            decode_t0 = time.time()
            pred = model.decode_queries(video=fake, query=query, memory=memory)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            payload["seconds_fake_decode"] = float(time.time() - decode_t0)
        payload["fake_decode_keys"] = sorted(str(k) for k in pred.keys())
        payload.update(_gpu_payload())
        payload["ok"] = True
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload.update(_gpu_payload())
    return _json_safe(payload)


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    lines = ["# D4RT Preflight", "", "## Summary", ""]
    for key in (
        "ok",
        "error",
        "checkpoint_path",
        "checkpoint_size_bytes",
        "checkpoint_sha256_prefix8",
        "local_ckpt_path",
        "clip_frames",
        "image_hw",
        "seconds_sha256",
        "seconds_local_copy",
        "seconds_torch_load",
        "seconds_build_model",
        "seconds_load_state_dict",
        "seconds_fake_encode",
        "seconds_fake_decode",
        "cuda_available",
        "cuda_device_name",
        "cuda_memory_reserved",
    ):
        if key in payload:
            lines.append(f"- {key}: `{payload[key]}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--d4rt-config", required=True)
    parser.add_argument("--d4rt-ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-ckpt-copy", default="")
    parser.add_argument("--fake-frames", type=int, default=2)
    parser.add_argument("--fake-queries", type=int, default=16)
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--output", default="outputs/audit/d4rt_preflight_v5.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(output, payload)
    print(f"[d4rt-preflight] wrote {output}")
    print(f"[d4rt-preflight] wrote {output.with_suffix('.json')}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
