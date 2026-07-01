from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .v47_common import ROOT


@dataclass(frozen=True)
class V61DashboardConfig:
    final_decision_path: str | Path = "outputs/audit/v61_final_decision/final_decision.json"
    output_path: str | Path = "outputs/audit/v61_visualizations/v61_dashboard.html"


def build_v61_dashboard(config: V61DashboardConfig | None = None) -> dict[str, str]:
    cfg = config or V61DashboardConfig()
    final_path = _project(cfg.final_decision_path)
    decision = json.loads(final_path.read_text(encoding="utf-8"))
    output = _project(cfg.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_html(decision), encoding="utf-8")
    return {"dashboard": _rel(output)}


def _html(decision: dict[str, Any]) -> str:
    metrics = decision.get("key_metrics") or {}
    answers = decision.get("required_answers") or {}
    blocked = decision.get("blocked_claims") or []
    image_paths = [
        "phase0/v61_phase0_unit_mismatch_dashboard.png",
        "graph_v3/material_candidate_coverage.png",
        "global_embedding/global_manifold_embedding_state_counts.png",
        "refinement/refinement_quarantine_counts.png",
        "query/query_control_comparison.png",
        "stress/stress_real_minus_mask_only_ari.png",
        "native_field/native_carrier_state_counts.png",
    ]
    rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metrics.items()
    )
    answer_items = "\n".join(
        f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>"
        for key, value in answers.items()
    )
    blocked_items = "\n".join(f"<li>{html.escape(str(item))}</li>" for item in blocked) or "<li>none</li>"
    figures = "\n".join(
        f'<figure><img src="{html.escape(path)}" alt="{html.escape(path)}"><figcaption>{html.escape(path)}</figcaption></figure>'
        for path in image_paths
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stream4D v61 SOMA-Manifold Dashboard</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1f2933; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1100px; }}
    th, td {{ border: 1px solid #d6dce1; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ width: 320px; background: #f5f7f9; }}
    img {{ max-width: 980px; width: 100%; border: 1px solid #d6dce1; }}
    figure {{ margin: 18px 0; }}
    figcaption {{ font-size: 13px; color: #52616b; margin-top: 6px; }}
    .label {{ display: inline-block; padding: 4px 8px; border: 1px solid #9fb3c8; background: #f0f4f8; }}
  </style>
</head>
<body>
  <h1>Stream4D v61 SOMA-Manifold Dashboard</h1>
  <p class="label">{html.escape(str(decision.get("decision_label")))}</p>
  <section>
    <h2>Decision Metrics</h2>
    <table>{rows}</table>
  </section>
  <section>
    <h2>Blocked Claims</h2>
    <ul>{blocked_items}</ul>
  </section>
  <section>
    <h2>Required Answers</h2>
    <ol>{answer_items}</ol>
  </section>
  <section>
    <h2>Evidence Figures</h2>
    {figures}
  </section>
</body>
</html>
"""


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj


def _rel(path: str | Path) -> str:
    path_obj = _project(path)
    try:
        return str(path_obj.relative_to(ROOT))
    except ValueError:
        return str(path_obj)
