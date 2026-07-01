from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _load_dict(path: str | Path) -> dict[str, Any]:
    path_obj = _project(path)
    if not path_obj.exists():
        return {}
    payload = read_json(path_obj)
    return payload if isinstance(payload, dict) else {}


def _cell(value: Any) -> str:
    return html.escape(str(value))


def build_v64r2_dashboard_html(
    *,
    final_decision_path: str | Path = "outputs/audit/v64r2_final/final_decision.json",
) -> str:
    final = _load_dict(final_decision_path)
    rows = final.get("metric_rows", []) if isinstance(final.get("metric_rows"), list) else []
    row_html = "\n".join(
        "<tr>"
        f"<td>{_cell(row.get('track'))}</td>"
        f"<td>{_cell(row.get('status'))}</td>"
        f"<td>{_cell(row.get('key_metric'))}</td>"
        f"<td>{_cell(row.get('value'))}</td>"
        f"<td>{_cell(row.get('claim_allowed'))}</td>"
        "</tr>"
        for row in rows
        if isinstance(row, dict)
    )
    blocked = "".join(f"<li>{_cell(item)}</li>" for item in final.get("blocked_claims", []))
    allowed = "".join(f"<li>{_cell(item)}</li>" for item in final.get("final_claim_allowed", []))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stream4D v64-r2 Dashboard</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ccd4dd; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eef3f7; }}
    code {{ background: #f3f5f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Stream4D v64-r2 Evaluation-First Dashboard</h1>
  <p><strong>Decision:</strong> <code>{_cell(final.get('decision_label'))}</code></p>
  <p><strong>Main:</strong> {_cell(final.get('main_ownership_status'))}</p>
  <p><strong>ScanNet AP:</strong> {_cell(final.get('scannet_ap_status'))}</p>
  <p><strong>Dynamic:</strong> {_cell(final.get('dynamic_status'))}</p>
  <p><strong>Active query:</strong> {_cell(final.get('active_query_status'))}</p>
  <h2>Metric Rows</h2>
  <table>
    <thead><tr><th>Track</th><th>Status</th><th>Metric</th><th>Value</th><th>Claim allowed</th></tr></thead>
    <tbody>{row_html}</tbody>
  </table>
  <h2>Allowed Claims</h2>
  <ul>{allowed}</ul>
  <h2>Blocked Claims</h2>
  <ul>{blocked}</ul>
  <h2>Evidence</h2>
  <p>Final decision JSON: <code>{_cell(final_decision_path)}</code></p>
</body>
</html>
"""


def write_v64r2_dashboard(output_path: str | Path, html_text: str) -> None:
    out = _project(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")


def _draw_text_png(path: Path, title: str, lines: list[str]) -> None:
    import cv2
    import numpy as np

    width = 1280
    height = max(360, 90 + 34 * (len(lines) + 1))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, title, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 45, 70), 2, cv2.LINE_AA)
    y = 96
    for line in lines:
        cv2.putText(canvas, line[:150], (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (30, 30, 30), 2, cv2.LINE_AA)
        y += 34
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def write_v64r2_summary_visualizations(
    *,
    output_root: str | Path = "outputs/audit/v64r2_visualizations",
    ap_summary_path: str | Path = "outputs/audit/v64r2_scannet_ap_probe5/ap_smoke_summary.json",
    failure_summary_path: str | Path = "outputs/audit/v64r2_ap_failure_attribution/failure_summary.json",
    dynamic_summary_path: str | Path = "outputs/audit/v64r2_dynamic_env/dynamic_env_summary.json",
) -> list[str]:
    root = _project(output_root)
    ap = _load_dict(ap_summary_path)
    failure = _load_dict(failure_summary_path)
    dynamic = _load_dict(dynamic_summary_path)
    outputs = [
        root / "scannet_ap_probe5" / "ap_metrics.png",
        root / "ap_failure" / "failure_counts.png",
        root / "dynamic" / "dynamic_env.png",
    ]
    _draw_text_png(
        outputs[0],
        "v64-r2 ScanNet AP Probe5",
        [
            f"status: {ap.get('scannet_ap_status')}",
            f"best diagnostic AP/AP50/AP25: {ap.get('best_diagnostic_AP')} / {ap.get('best_diagnostic_AP50')} / {ap.get('best_diagnostic_AP25')}",
            f"method_safe_AP_available: {ap.get('method_safe_AP_available')}",
            f"diagnostic_AP_available: {ap.get('diagnostic_AP_available')}",
            f"bridge_config: {ap.get('bridge_config')}",
        ],
    )
    _draw_text_png(
        outputs[1],
        "v64-r2 AP Failure Attribution",
        [
            f"top_failure_category: {failure.get('top_failure_category')}",
            f"attribution_coverage: {failure.get('attribution_coverage')}",
            f"AP-scope failed_gt_count/counts: {failure.get('failed_gt_count')} / {failure.get('failure_category_counts')}",
            f"full-scene top/counts: {failure.get('top_full_scene_failure_category')} / {failure.get('full_scene_failure_category_counts')}",
            f"method_safe_AP_available: {failure.get('method_safe_AP_available')}",
        ],
    )
    _draw_text_png(
        outputs[2],
        "v64-r2 Dynamic Replica Environment",
        [
            f"dyn_level: {dynamic.get('dyn_level_label')}",
            f"rgb/depth/masks/object_ids: {dynamic.get('rgb_frames_exist')} / {dynamic.get('depth_frames_exist')} / {dynamic.get('instance_masks_exist')} / {dynamic.get('object_ids_exist')}",
            f"trajectories_exist: {dynamic.get('trajectories_exist')}",
            f"can_report_official_object_tracking: {dynamic.get('can_report_official_object_tracking')}",
            f"blocked_official_metric_reasons: {dynamic.get('blocked_official_metric_reasons')}",
        ],
    )
    return [str(path.relative_to(ROOT)) for path in outputs]
