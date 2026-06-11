from __future__ import annotations

import argparse
from pathlib import Path


def _status(label: str, ok: bool, detail: str) -> bool:
    state = "OK" if ok else "FAIL"
    print(f"[{state}] {label}: {detail}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d4rt-root", required=True)
    parser.add_argument("--stream3d-root", default=".")
    parser.add_argument("--d4rt-config", default="")
    parser.add_argument("--d4rt-ckpt", default="")
    parser.add_argument("--seq-name", default="scene0050_00")
    parser.add_argument("--backbone", default="SAM2")
    args = parser.parse_args()

    stream3d_root = Path(args.stream3d_root).resolve()
    d4rt_root = Path(args.d4rt_root).resolve()
    if args.d4rt_config:
        d4rt_config = Path(args.d4rt_config)
    else:
        d4rt_config = d4rt_root / "checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml"
    if args.d4rt_ckpt:
        d4rt_ckpt = Path(args.d4rt_ckpt)
    else:
        d4rt_ckpt = d4rt_root / "checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt"

    checks = []
    checks.append(_status("OpenD4RT root", d4rt_root.exists(), str(d4rt_root)))
    checks.append(_status("D4RT model config", d4rt_config.exists(), str(d4rt_config)))
    checks.append(_status("D4RT checkpoint", d4rt_ckpt.exists(), str(d4rt_ckpt)))

    processed = stream3d_root / "data/scannet/processed"
    scene = processed / args.seq_name
    checks.append(_status("ScanNet processed root", processed.exists(), str(processed)))
    checks.append(_status("ScanNet scene", scene.exists(), str(scene)))
    for name in ["color", "depth", "pose", "intrinsic"]:
        checks.append(_status(f"ScanNet {name}", (scene / name).exists(), str(scene / name)))
    checks.append(_status("ScanNet mesh", (scene / f"{args.seq_name}_vh_clean_2.ply").exists(), str(scene / f"{args.seq_name}_vh_clean_2.ply")))
    mask_dir = scene / f"output_{args.backbone}/mask"
    checks.append(_status(f"2D mask dir ({args.backbone})", mask_dir.exists(), str(mask_dir)))
    cropformer_dir = scene / "output_Cropformer/mask"
    if args.backbone != "Cropformer":
        _status("2D mask dir (Cropformer fallback visibility)", cropformer_dir.exists(), str(cropformer_dir))

    clip_cache = Path.home() / ".cache/huggingface"
    checks.append(_status("HuggingFace cache root", clip_cache.exists(), str(clip_cache)))

    if not all(checks):
        raise SystemExit(2)
    print("Stream4D environment check passed.")


if __name__ == "__main__":
    main()
