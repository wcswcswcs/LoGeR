from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stream4d_native.frame_index_map import FrameIndexMap


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_ids_from_dir(path: Path, suffixes: tuple[str, ...]) -> list[int]:
    if not path.exists():
        return []
    ids: list[int] = []
    for child in path.iterdir():
        if child.suffix.lower() not in suffixes:
            continue
        try:
            ids.append(int(child.stem))
        except ValueError:
            continue
    return sorted(ids)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def build_frame_stride(scene_dir: Path, max_rgb_frames: int) -> dict[str, Any]:
    rgb_ids = _frame_ids_from_dir(scene_dir / "color", (".jpg", ".png"))[: int(max_rgb_frames)]
    mask_ids = _frame_ids_from_dir(scene_dir / "output_Cropformer" / "mask", (".png",))[: int(max_rgb_frames)]
    if not rgb_ids:
        rgb_ids = list(range(int(max_rgb_frames)))
    if not mask_ids:
        mask_ids = rgb_ids[::10]
    mask_set = set(mask_ids)
    mask_ids = [raw_id for raw_id in rgb_ids if raw_id in mask_set]
    fmap = FrameIndexMap.from_frame_ids(rgb_ids, mask_ids)
    return fmap.audit_summary()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/audit/v42_fact_lock")
    parser.add_argument("--scene-dir", default="/home/tmp_datasets/scannet_v2/scans/scene0050_00")
    parser.add_argument("--max-rgb-frames", type=int, default=120)
    args = parser.parse_args()

    v41_cmp_path = ROOT / "outputs/audit/v41_1_stream3d_first_comparison/stream3d_first_comparison_summary.json"
    v41_native_path = (
        ROOT
        / "outputs/audit/v41_1_native_support_metrics_probe5_sweep/offsetfix2_closure_rgb090_t035_m010_birthgate/native_support_metrics_summary.json"
    )
    v41_bridge_path = ROOT / "outputs/audit/v41_1_method_ap_bridge_feasibility/method_ap_bridge_feasibility_summary.json"
    v40_decision_path = ROOT / "outputs/audit/v40R_final_decision/decision_summary.json"
    v39_decision_path = ROOT / "outputs/audit/v39_final_decision/decision_summary.json"
    v37_decision_path = ROOT / "outputs/audit/v37_final_decision/decision_summary.json"

    v41_cmp = _load_json(v41_cmp_path)
    v41_native = _load_json(v41_native_path)
    v41_bridge = _load_json(v41_bridge_path)
    v40_decision = _load_json(v40_decision_path)
    v39_decision = _load_json(v39_decision_path)
    v37_decision = _load_json(v37_decision_path)
    stride = build_frame_stride(Path(args.scene_dir), int(args.max_rgb_frames))

    rows = [
        {
            "fact": "v41_1_native_support_4D_ARI",
            "value": (v41_native or {}).get("aggregate_metrics", {}).get("4D_ARI"),
            "status": "imported_prior" if v41_native else "missing",
            "source": str(v41_native_path),
        },
        {
            "fact": "v41_1_native_support_purity",
            "value": (v41_native or {}).get("aggregate_metrics", {}).get("4D_purity"),
            "status": "imported_prior" if v41_native else "missing",
            "source": str(v41_native_path),
        },
        {
            "fact": "v41_1_native_support_completeness",
            "value": (v41_native or {}).get("aggregate_metrics", {}).get("4D_completeness"),
            "status": "imported_prior" if v41_native else "missing",
            "source": str(v41_native_path),
        },
        {
            "fact": "v41_1_birth_from_d4rt_tube_count_sum",
            "value": (v41_native or {}).get("aggregate_metrics", {}).get("birth_from_d4rt_tube_count_sum"),
            "status": "imported_prior" if v41_native else "missing",
            "source": str(v41_native_path),
        },
        {
            "fact": "v41_1_AP_bridge_status",
            "value": (v41_bridge or {}).get("status") or (v41_native or {}).get("AP_bridge_status"),
            "status": "imported_prior" if v41_bridge else "missing",
            "source": str(v41_bridge_path),
        },
        {
            "fact": "v40R_final_status",
            "value": (v40_decision or {}).get("final_status"),
            "status": "imported_prior" if v40_decision else "missing",
            "source": str(v40_decision_path),
        },
        {
            "fact": "v39_final_status",
            "value": (v39_decision or {}).get("final_status"),
            "status": "imported_prior" if v39_decision else "missing",
            "source": str(v39_decision_path),
        },
        {
            "fact": "v37_final_status",
            "value": (v37_decision or {}).get("final_status"),
            "status": "imported_prior" if v37_decision else "missing",
            "source": str(v37_decision_path),
        },
        {
            "fact": "d4rt_encoder_stride",
            "value": stride.get("d4rt_encoder_stride"),
            "status": "current_frame_index_audit",
            "source": str(args.scene_dir),
        },
        {
            "fact": "uses_contiguous_rgb_for_d4rt",
            "value": stride.get("uses_contiguous_rgb_for_d4rt"),
            "status": "current_frame_index_audit",
            "source": str(args.scene_dir),
        },
    ]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v42_fact_lock",
        "policy": {
            "training_free": True,
            "imported_prior_evidence_is_not_current_v42_method_result": True,
            "gt_allowed_only_for_diagnostic_scoring": True,
        },
        "artifacts": {
            "v41_stream3d_first_comparison": str(v41_cmp_path),
            "v41_native_support": str(v41_native_path),
            "v41_ap_bridge_feasibility": str(v41_bridge_path),
            "v40R_decision": str(v40_decision_path),
            "v39_decision": str(v39_decision_path),
            "v37_decision": str(v37_decision_path),
        },
        "v41_1_key_numbers": (v41_native or {}).get("aggregate_metrics", {}),
        "v41_1_hard_scene_rows": [
            row
            for row in (v41_native or {}).get("scene_rows", [])
            if row.get("scene") in {"scene0081_01", "scene0011_00", "scene0591_00"}
        ],
        "v41_1_stream3d_first_key_numbers": (v41_cmp or {}).get("key_numbers", {}),
        "v41_1_ap_bridge": {
            "status": (v41_bridge or {}).get("status"),
            "method_ap_goal_reached": (v41_bridge or {}).get("method_ap_goal_reached"),
            "blocker": (v41_bridge or {}).get("blocker"),
        },
        "frame_stride_audit": stride,
        "gate": {
            "v41_1_facts_loaded": v41_native is not None and v41_cmp is not None,
            "d4rt_encoder_stride_is_1": stride.get("d4rt_encoder_stride") == 1,
            "uses_contiguous_rgb_for_d4rt": bool(stride.get("uses_contiguous_rgb_for_d4rt")),
        },
        "rows": rows,
    }
    out = ROOT / args.output_root
    _write_json(out / "fact_lock.json", payload)
    _write_csv(out / "fact_lock_rows.csv", rows)
    print(json.dumps({"fact_lock": str(out / "fact_lock.json"), "gate": payload["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

