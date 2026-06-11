from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tools.prediction_manifest import build_prediction_manifest, write_prediction_manifest


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _pred_dir(root: Path, config: str) -> Path:
    suffix = config if config.endswith("_class_agnostic") else f"{config}_class_agnostic"
    return root / "data" / "prediction" / suffix


def _tmp_file(root: Path, config: str, scene_id: str) -> Path:
    candidates = [
        root / "data" / "TMP" / config / f"{scene_id}_pre_points.npy",
        root / "TMP" / config / f"{scene_id}_pre_points.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--copy", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    scene_ids = _read_lines(Path(args.seq_list))
    src_pred_dir = _pred_dir(root, args.config)
    dst_pred_dir = _pred_dir(root, args.output_config)
    dst_tmp_dir = root / "data" / "TMP" / args.output_config
    dst_pred_dir.mkdir(parents=True, exist_ok=True)
    dst_tmp_dir.mkdir(parents=True, exist_ok=True)
    for scene_id in scene_ids:
        _link_or_copy(src_pred_dir / f"{scene_id}.npz", dst_pred_dir / f"{scene_id}.npz", args.copy)
        _link_or_copy(_tmp_file(root, args.config, scene_id), dst_tmp_dir / f"{scene_id}_pre_points.npy", args.copy)
    manifest = build_prediction_manifest(
        root=root,
        output_config=args.output_config,
        is_method_result=True,
        is_diagnostic_only=False,
        uses_gt=False,
        gt_usage="none",
        source_configs=[args.config],
        pre_points_policy="inherit",
        support_policy="subset_materialization",
        notes="Materialized evaluation subset by linking/copying existing predictions and TMP pre_points.",
    )
    write_prediction_manifest(args.output_config, manifest, root=root)
    print(f"[subset] config={args.config} output_config={args.output_config}")
    print(f"[subset] seq_list={args.seq_list} scenes={len(scene_ids)}")
    print(f"[subset] pred_dir={dst_pred_dir}")
    print(f"[subset] tmp_dir={dst_tmp_dir}")


if __name__ == "__main__":
    main()
