#!/usr/bin/env python3
"""Build a readable layerwise PCA visual report from contact sheets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


FAMILIES = [
    {
        "section": "Frame Attention",
        "anchor": "frame-attention",
        "kind": "frame_attn",
        "what": (
            "Frame attention 的 q/k/v 特征。这里看的是偶数 decoder layer 的 "
            "frame-attention 内部表示：q 是当前 token 发起 attention 的 query 表征，"
            "k 是被匹配的 key 表征，v 是被传播的 value 表征。"
        ),
        "layer_note": "偶数 decoder layer，来自 frame-attention path。",
        "components": [
            ("q", "pca_attn_frame_q_layers", "query features used by frame attention"),
            ("k", "pca_attn_frame_k_layers", "key features matched by frame attention"),
            ("v", "pca_attn_frame_v_layers", "value features propagated by frame attention"),
        ],
    },
    {
        "section": "Global Attention",
        "anchor": "global-attention",
        "kind": "global_attn",
        "what": (
            "Global/chunk attention 的 q/k/v 特征。这里看的是奇数 decoder layer 的 "
            "global attention 内部表示，用来观察全局或跨 chunk source token 是否已经形成"
            "道路、背景、物体边界等结构。"
        ),
        "layer_note": "奇数 decoder layer，来自 global/chunk attention path。",
        "components": [
            ("q", "pca_attn_global_q_layers", "query features used by global attention"),
            ("k", "pca_attn_global_k_layers", "key/source features matched by global attention"),
            ("v", "pca_attn_global_v_layers", "value/source features propagated by global attention"),
        ],
    },
    {
        "section": "SWA Current",
        "anchor": "swa-current",
        "kind": "swa_current",
        "what": (
            "SWA adapter 当前输入侧 q/k/v 特征。这里只覆盖当前 H35 配置实际存在的 "
            "SWA adapter layer，不是 36 个 decoder layer 全部都有 SWA。"
        ),
        "layer_note": "当前配置的 SWA adapter layer。",
        "components": [
            ("q", "pca_swa_current_q_layers", "current SWA query features"),
            ("k", "pca_swa_current_k_layers", "current SWA key features"),
            ("v", "pca_swa_current_v_layers", "current SWA value features"),
        ],
    },
    {
        "section": "SWA Cache",
        "anchor": "swa-cache",
        "kind": "swa_cache",
        "what": (
            "SWA cache 侧 k/v 特征。这里看历史或缓存 source token 被 SWA 读取时的 "
            "key/value 表征。SWA cache 没有 q，q 在 SWA current 里。"
        ),
        "layer_note": "当前配置的 SWA cache layer，与 SWA current 层号一致。",
        "components": [
            ("k", "pca_swa_cache_k_layers", "cached/source SWA key features"),
            ("v", "pca_swa_cache_v_layers", "cached/source SWA value features"),
        ],
    },
    {
        "section": "TTT Internals",
        "anchor": "ttt-internals",
        "kind": "ttt",
        "what": (
            "TTT 模块内部特征。q/k/v 是 TTT primitive；input 是进入 TTT 的输入表征；"
            "apply_raw 是 raw apply 输出；operator_output、update_term、final_output "
            "是不同阶段输出，必须分开看，不能混成一个 TTT 图。"
        ),
        "layer_note": "偶数 decoder layer，TTT 插入层。",
        "components": [
            ("q", "pca_ttt_q_layers", "TTT query primitive"),
            ("k", "pca_ttt_k_layers", "TTT key primitive"),
            ("v", "pca_ttt_v_layers", "TTT value primitive"),
            ("input", "pca_ttt_input_layers", "input feature before TTT application"),
            ("apply_raw", "pca_ttt_apply_raw_layers", "raw output from TTT apply path"),
            (
                "operator_output",
                "pca_ttt_operator_output_layers",
                "operator output before final residual/output interpretation",
            ),
            ("update_term", "pca_ttt_update_term_layers", "TTT update term"),
            ("final_output", "pca_ttt_final_output_layers", "final TTT output feature"),
        ],
    },
]


def _layer_ids(available_taps: dict, components: list[tuple[str, str, str]]) -> list[int]:
    return sorted({layer for _, tap, _ in components for layer in available_taps[tap]["layer_ids"]})


def _copy_images(source_root: Path, out_root: Path, summary: dict) -> list[dict]:
    contact_root = source_root / "contact_sheets"
    img_root = out_root / "imgs"
    img_root.mkdir(parents=True, exist_ok=True)
    rows = []
    available_taps = summary["available_taps"]

    for family in FAMILIES:
        family["layers"] = _layer_ids(available_taps, family["components"])
        for layer in family["layers"]:
            for component, tap, description in family["components"]:
                if layer not in available_taps[tap]["layer_ids"]:
                    continue
                src = contact_root / f"{tap}_L{layer:02d}_contact.png"
                if not src.exists():
                    raise FileNotFoundError(src)
                dst_name = f"{family['kind']}_L{layer:02d}_{component}.png"
                dst = img_root / dst_name
                shutil.copy2(src, dst)
                rows.append(
                    {
                        "section": family["section"],
                        "kind": family["kind"],
                        "layer_id": layer,
                        "component": component,
                        "tap": tap,
                        "description": description,
                        "image": f"imgs/{dst_name}",
                        "source": str(src),
                    }
                )
    return rows


def _write_manifest(out_root: Path, rows: list[dict]) -> Path:
    manifest_path = out_root / "pca_layerwise_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "section",
                "kind",
                "layer_id",
                "component",
                "tap",
                "description",
                "image",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_report(out_root: Path, source_root: Path, summary: dict, rows: list[dict]) -> Path:
    lines: list[str] = [
        "# PCA Layerwise Visual Report",
        "",
        "Date: 2026-06-20",
        "",
        (
            "这个报告是重新整理后的 PCA 视觉确认页。它不再把所有图堆成一个大图，而是按类别、"
            "真实 decoder layer id、q/k/v 或 TTT 输出类型组织。每张图都是一个单独的 contact sheet，"
            "图片文件都在 `results/pca/imgs`。"
        ),
        "",
        "## How To Read The Images",
        "",
        "- 每个图片文件对应一个 `类别 + layer id + component/output`，例如 `frame_attn_L10_q.png`。",
        "- 每张 contact sheet 的列固定为：原始 RGB、semantic、semantic trust、PCA。",
        (
            "- 每张图包含 4 个 chunk，每个 chunk 采样 4 个 view，总共 16 行。采样帧是 "
            "chunk0: 0/10/20/31，chunk1: 29/39/49/60，chunk2: 58/68/78/89，"
            "chunk3: 87/89/92/95。"
        ),
        (
            "- PCA 是把该层该 feature tap 的 token feature 做 3 个主成分可视化，用来判断是否出现"
            "道路、背景、物体轮廓、动态区域或 chunk/frame 结构线索。它不是 ATE 成功证据。"
        ),
        "- 图片默认放在折叠块里，按 layer 打开看；这样不会一进文档就被一整页大图淹没。",
        "",
        "## Source And Coverage",
        "",
        f"- Source PCA root: `{source_root}`",
        "- Feature window: KITTI01 frames 0-95, chunks 0-32 / 29-61 / 58-90 / 87-96.",
        f"- PCA units copied: `{len(rows)}` images.",
        "- Manifest: `pca_layerwise_manifest.csv`.",
        (
            f"- RGB available: `{summary.get('rgb_available')}`; semantic available: "
            f"`{summary.get('semantic_available')}`; unavailable taps: `{summary.get('unavailable_taps')}`."
        ),
        "",
        "## Category Summary",
        "",
        "| Category | Layer ids | Components | What this visualizes |",
        "|---|---|---|---|",
    ]

    for family in FAMILIES:
        components = ", ".join(component for component, _, _ in family["components"])
        layers = ", ".join(f"L{layer:02d}" for layer in family["layers"])
        lines.append(
            f"| [{family['section']}](#{family['anchor']}) | {layers} | "
            f"{components} | {family['what']} |"
        )

    lines.extend(["", "## Quick Examples", ""])
    examples = [
        ("Frame attention L10 q", "imgs/frame_attn_L10_q.png"),
        ("Global attention L17 k", "imgs/global_attn_L17_k.png"),
        ("SWA current L10 q", "imgs/swa_current_L10_q.png"),
        ("TTT operator output L18", "imgs/ttt_L18_operator_output.png"),
        ("TTT final output L18", "imgs/ttt_L18_final_output.png"),
    ]
    for title, image in examples:
        lines.extend([f"### {title}", "", f'<img src="{image}" alt="{title}" width="760">', ""])

    lines.extend(["## Full Layer Index", ""])
    available_taps = summary["available_taps"]
    for family in FAMILIES:
        lines.extend(
            [
                f"## {family['section']}",
                f'<a id="{family["anchor"]}"></a>',
                "",
                family["what"],
                "",
                f"Layer note: {family['layer_note']}",
                "",
                "| Layer | Image links |",
                "|---|---|",
            ]
        )
        for layer in family["layers"]:
            links = []
            for component, tap, _ in family["components"]:
                if layer in available_taps[tap]["layer_ids"]:
                    links.append(f"[{component}](imgs/{family['kind']}_L{layer:02d}_{component}.png)")
            lines.append(f"| L{layer:02d} | " + " / ".join(links) + " |")
        lines.append("")

        for layer in family["layers"]:
            components = [
                component
                for component, tap, _ in family["components"]
                if layer in available_taps[tap]["layer_ids"]
            ]
            lines.extend(
                [
                    "<details>",
                    f"<summary>{family['section']} L{layer:02d}: " + ", ".join(components) + "</summary>",
                    "",
                    f"Layer L{layer:02d}. {family['layer_note']} 每个子图都显示同一层的一个 component/output。",
                    "",
                ]
            )
            for component, tap, description in family["components"]:
                if layer not in available_taps[tap]["layer_ids"]:
                    continue
                image = f"imgs/{family['kind']}_L{layer:02d}_{component}.png"
                lines.extend(
                    [
                        f"**{component}** - `{tap}`: {description}.",
                        "",
                        f'<img src="{image}" alt="{family["section"]} L{layer:02d} {component}" width="760">',
                        "",
                    ]
                )
            lines.extend(["</details>", ""])

    lines.extend(
        [
            "## Audit Notes",
            "",
            (
                "- 该报告只是重排已生成的 `full_qkv_smoke96_pca_rgb4views/contact_sheets`，"
                "没有重新跑 ATE，也没有生成新的 success claim。"
            ),
            "- 所有 layer id 都来自 `pca_summary.json` / `layer_ids::<tap>`，不是按文件排序猜出来的。",
            "- SWA 只有当前配置实际存在的 adapter/cache layer: L10, L18, L26, L34。",
            (
                "- TTT 的 `apply_raw`、`operator_output`、`update_term`、`final_output` 分开列出，"
                "避免把不同内部输出混成一个结论。"
            ),
            "",
        ]
    )

    report_path = out_root / "pca_layerwise_visual_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing pca_summary.json and contact_sheets/.",
    )
    parser.add_argument("--out-root", type=Path, default=Path("results/pca"))
    args = parser.parse_args()

    summary = json.loads((args.source_root / "pca_summary.json").read_text())
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = _copy_images(args.source_root, args.out_root, summary)
    manifest_path = _write_manifest(args.out_root, rows)
    report_path = _write_report(args.out_root, args.source_root, summary, rows)

    print(f"report={report_path}")
    print(f"manifest={manifest_path}")
    print(f"images={len(rows)}")


if __name__ == "__main__":
    main()
