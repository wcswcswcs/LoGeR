from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _annotation_name(split: str) -> str:
    return f"frame_annotations_{split}.json"


def _has_any(scene_dir: Path, names: list[str], pattern: str = "*") -> bool:
    for name in names:
        path = scene_dir / name
        if path.exists() and any(path.glob(pattern)):
            return True
    return False


def _load_annotations(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], "annotation file missing"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if not isinstance(raw, list):
        return [], f"annotation root is {type(raw).__name__}, expected list"
    return raw, None


def _camera_fields_summary(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    required = ["R", "T", "focal_length", "principal_point"]
    present_counts = {key: 0 for key in required}
    viewpoint_count = 0
    for item in annotations:
        vp = item.get("viewpoint")
        if not isinstance(vp, dict):
            continue
        viewpoint_count += 1
        for key in required:
            if key in vp:
                present_counts[key] += 1
    return {
        "annotation_items": len(annotations),
        "viewpoint_items": viewpoint_count,
        "camera_fields_present_counts": present_counts,
        "all_required_camera_fields_present": bool(
            viewpoint_count > 0 and all(present_counts[key] == viewpoint_count for key in required)
        ),
    }


def _scene_summary(scene_dir: Path) -> dict[str, Any]:
    images = sorted((scene_dir / "images").glob("*.png"))
    depths = sorted((scene_dir / "depths").glob("*.geometric.png"))
    trajectories = sorted((scene_dir / "trajectories").glob("*.pth"))
    return {
        "scene": scene_dir.name,
        "images_count": len(images),
        "depths_count": len(depths),
        "trajectories_count": len(trajectories),
        "aligned_frame_count": min(len(images), len(depths), len(trajectories)),
        "has_gt_semantic_labels": _has_any(
            scene_dir,
            ["semantic", "semantics", "semantic_masks", "labels", "label"],
            "*",
        ),
        "has_gt_instance_masks": _has_any(
            scene_dir,
            ["instances", "instance", "instance_masks", "masks", "segmentation"],
            "*",
        ),
        "has_object_ids": _has_any(scene_dir, ["object_ids", "objects", "tracks"], "*"),
    }


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Dynamic Replica Environment Check")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- data_root_exists: `{payload['data_root_exists']}`")
    lines.append(f"- split_dir_exists: `{payload['split_dir_exists']}`")
    lines.append(f"- annotation_exists: `{payload['annotation_exists']}`")
    lines.append(f"- all_required_camera_fields_present: `{payload['camera_summary']['all_required_camera_fields_present']}`")
    lines.append(f"- usable_scene_count: `{payload['usable_scene_count']}`")
    lines.append(f"- can_report_official_instance_tracking: `{payload['can_report_official_instance_tracking']}`")
    lines.append(f"- can_report_d4rt_trajectory_metrics: `{payload['can_report_d4rt_trajectory_metrics']}`")
    lines.append(f"- can_report_only_qualitative_consistency: `{payload['can_report_only_qualitative_consistency']}`")
    if payload.get("annotation_error"):
        lines.append(f"- annotation_error: `{payload['annotation_error']}`")
    lines.append("")
    lines.append("## Scene Sample")
    lines.append("")
    lines.append("| Scene | images | depths | trajectories | aligned | semantic GT | instance GT | object IDs |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for row in payload["scenes"][:30]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scene"],
                    str(row["images_count"]),
                    str(row["depths_count"]),
                    str(row["trajectories_count"]),
                    str(row["aligned_frame_count"]),
                    str(row["has_gt_semantic_labels"]),
                    str(row["has_gt_instance_masks"]),
                    str(row["has_object_ids"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Evidence Boundary")
    lines.append("")
    lines.append(
        "如果没有 GT instance masks / object IDs，本工具只允许后续报告 qualitative 或 pseudo consistency，不能写 official IDF1 / semantic AP。"
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/dynamic-replica/v2")
    parser.add_argument("--root", dest="data_root", help="alias for --data-root")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--output", default="outputs/audit/dynamic_replica_env_check.md")
    parser.add_argument("--max-scenes", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.data_root).resolve()
    split_dir = root / args.split
    annotation_path = split_dir / _annotation_name(args.split)
    annotations, annotation_error = _load_annotations(annotation_path)
    scene_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()]) if split_dir.exists() else []
    if args.max_scenes > 0:
        scene_dirs = scene_dirs[: args.max_scenes]
    scenes = [_scene_summary(path) for path in scene_dirs]
    usable_scene_count = int(sum(1 for row in scenes if row["aligned_frame_count"] > 0))
    has_instance_gt = any(row["has_gt_instance_masks"] or row["has_object_ids"] for row in scenes)
    has_trajectory = any(row["trajectories_count"] > 0 for row in scenes)
    has_rgbd_camera = bool(
        usable_scene_count > 0
        and any(row["images_count"] > 0 and row["depths_count"] > 0 for row in scenes)
        and _camera_fields_summary(annotations)["all_required_camera_fields_present"]
    )
    payload = {
        "data_root": str(root),
        "split": args.split,
        "split_dir": str(split_dir),
        "annotation_path": str(annotation_path),
        "data_root_exists": root.exists(),
        "split_dir_exists": split_dir.exists(),
        "annotation_exists": annotation_path.exists(),
        "annotation_error": annotation_error,
        "camera_summary": _camera_fields_summary(annotations),
        "scene_count": len(scenes),
        "usable_scene_count": usable_scene_count,
        "mean_aligned_frame_count": float(mean([row["aligned_frame_count"] for row in scenes])) if scenes else 0.0,
        "scenes": scenes,
        "can_report_official_instance_tracking": bool(has_instance_gt and has_trajectory),
        "can_report_d4rt_trajectory_metrics": bool(has_trajectory and has_rgbd_camera),
        "can_report_only_qualitative_consistency": bool(has_rgbd_camera and not has_instance_gt),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output, payload)
    print(f"[dynamic-replica-env] wrote {output}")
    print(f"[dynamic-replica-env] wrote {output.with_suffix('.json')}")
    print(
        "[dynamic-replica-env] "
        f"root_exists={payload['data_root_exists']} split_exists={payload['split_dir_exists']} "
        f"usable_scenes={payload['usable_scene_count']}"
    )


if __name__ == "__main__":
    main()
