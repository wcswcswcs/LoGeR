from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v47_common import ROOT, read_json


@dataclass(frozen=True)
class V62VisualizationConfig:
    final_decision_path: str | Path = "outputs/audit/v62_final/final_decision.json"
    output_path: str | Path = "outputs/audit/v62_visualizations/v62_dashboard.html"


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)


def build_v62_visual_dashboard(config: V62VisualizationConfig | None = None) -> dict[str, Any]:
    cfg = config or V62VisualizationConfig()
    final_path = _project(cfg.final_decision_path)
    final = read_json(final_path)
    output_path = _project(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    claim_rows = []
    for name, claim in final.get("claim_table", {}).items():
        status = "PASS" if claim.get("pass") else "PARTIAL/NO-GO"
        claim_rows.append(f"<tr><td>{name}</td><td>{claim.get('label')}</td><td>{status}</td><td>{claim.get('evidence')}</td></tr>")
    metric_rows = []
    for key, value in final.get("key_metrics", {}).items():
        metric_rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stream4D v62 Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #20242a; }}
    h1 {{ font-size: 26px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d7dce2; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f6; }}
    code {{ background: #f4f6f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Stream4D v62 Verified SOMA-Manifold</h1>
  <p>Decision: <code>{final.get('decision_label')}</code></p>
  <h2>Claims</h2>
  <table><thead><tr><th>Claim</th><th>Label</th><th>Status</th><th>Evidence</th></tr></thead><tbody>{''.join(claim_rows)}</tbody></table>
  <h2>Key Metrics</h2>
  <table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table>
  <h2>Blocked Claims</h2>
  <p><code>{', '.join(final.get('blocked_claims', []))}</code></p>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return {"dashboard": _rel(output_path), "visualization_status": "created"}


