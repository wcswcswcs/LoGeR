#!/usr/bin/env python3
"""Write the manual review for v78 Phase8-after-head-selective visuals.

This script records the human/visual inspection outcome for
HYP-PCA-REDISC-017..020. It does not compute metrics or claim method success.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REVIEW_BY_HYPOTHESIS = {
    "HYP-PCA-REDISC-017": {
        "review_status": "ambiguous",
        "visual_pattern_observed": (
            "SWA current-Q L10 shows road/corridor and vanishing-point-like structure; "
            "actual mask is coherent and random controls are scattered, but the pattern "
            "is broad and not yet boundary-only."
        ),
        "semantic_alignment": "mostly road/static corridor; not a narrow semantic class split",
        "geometry_alignment": "partly aligned with D_geo corridor, but no clean boundary-position isolation",
        "failure_alignment": "chunk-level failure overlay present; boundary-local causality not visually established",
        "action_mask_alignment": "actual mask follows broad road/corridor rather than a sparse boundary gate",
        "random_mask_difference": (
            "actual is more coherent than same-mass random and group-stratified random, "
            "but coherence alone is not a method claim"
        ),
        "reviewer_note": (
            "manual review 2026-06-21: keep as boundary-position hypothesis only; "
            "needs runtime position/quality audit before action."
        ),
    },
    "HYP-PCA-REDISC-018": {
        "review_status": "rejected",
        "visual_pattern_observed": (
            "SWA cache-K L34 mostly shows broad horizontal/layered bands and does not "
            "isolate a stricter high-quality source subset beyond road/static masks."
        ),
        "semantic_alignment": "broad road/static context; no clear useful-vs-random source-quality split",
        "geometry_alignment": "no visibly sharper high-quality source boundary than existing stable-topq80 route maps",
        "failure_alignment": "does not explain P9_36 weak amplification failure with a new boundary-local cue",
        "action_mask_alignment": "actual mask remains broad and random-like at the selector-quality level",
        "random_mask_difference": "random controls are scattered, but actual mask is not uniquely source-quality selective",
        "reviewer_note": (
            "manual review 2026-06-21: do not promote cache-K L34 source-quality selector "
            "from this visual evidence."
        ),
    },
    "HYP-PCA-REDISC-019": {
        "review_status": "rejected",
        "visual_pattern_observed": (
            "SWA current-K L34 resembles broad horizontal/road-like structure; no clear "
            "boundary-local key mismatch explaining heads0/6/8 negative transfer."
        ),
        "semantic_alignment": "broad road/static context; not a clean mismatch map",
        "geometry_alignment": "no sharper boundary residual alignment than cache-K L34 or current-Q views",
        "failure_alignment": "does not visually account for overlap-to-future degradation in P9_38",
        "action_mask_alignment": "actual mask remains broad, not a targeted boundary mismatch gate",
        "random_mask_difference": "actual is spatially coherent but not enough to beat the group/random concern as a new selector",
        "reviewer_note": (
            "manual review 2026-06-21: stop current-K L34 as a new route-bias clue "
            "unless new per-position metrics support it."
        ),
    },
    "HYP-PCA-REDISC-020": {
        "review_status": "ambiguous",
        "visual_pattern_observed": (
            "SWA cache-V L18 retains a visible road/corridor and vanishing-point-like "
            "structure, consistent with prior cache-V evidence, but it is broad and "
            "does not isolate a narrow boundary-local carry-over body."
        ),
        "semantic_alignment": "road/static corridor visible; dynamic/low-confidence exclusion is not the key signal",
        "geometry_alignment": "corridor overlaps D_geo layout but boundary-local body remains unconfirmed",
        "failure_alignment": "could motivate position/quality gating, not another full selected-mass boost",
        "action_mask_alignment": "actual mask is coherent but too broad for direct all-selected-token amplification",
        "random_mask_difference": (
            "actual differs from scattered random controls, but this repeats prior cache-V clue "
            "rather than confirming a new boundary gate"
        ),
        "reviewer_note": (
            "manual review 2026-06-21: retain cache-V L18 as ambiguous; next action "
            "must be boundary-position or selected-source quality audited."
        ),
    },
}


INSIGHT_APPENDIX = """

## Manual Review 2026-06-21 After Head-Selective Route-Bias

Scope: HYP-PCA-REDISC-017..020 generated after P9_36/P9_38 showed head-selective SWA route-bias is a real actuator but does not amplify weak positives into stable boundary/geometry gains.

- HYP-PCA-REDISC-017 / SWA current-Q L10: ambiguous. The PCA panels show coherent road/corridor and vanishing-point-like structure, and actual masks are less scattered than random controls, but the evidence is still broad road/corridor structure rather than a narrow boundary-only gate.
- HYP-PCA-REDISC-018 / SWA cache-K L34: rejected for now. The view is mostly broad horizontal/layered structure and does not isolate high-quality source tokens beyond road/static masks.
- HYP-PCA-REDISC-019 / SWA current-K L34: rejected for now. It does not reveal a clear boundary-local key mismatch explaining why heads0/6/8 harmed overlap-to-future metrics.
- HYP-PCA-REDISC-020 / SWA cache-V L18: ambiguous. The prior cache-V corridor signal persists, but no narrow boundary-local carry-over body is confirmed.

Conclusion: no new confirmed visual clue was found in this Phase8 pass. The honest next direction is not more head subset or beta sweep; it is a boundary-position / selected-source-quality audit that can prove which source positions within the broad cache-V/current-Q corridor actually affect SWA boundary residuals.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-root", type=Path, required=True)
    return parser.parse_args()


def _hypothesis_from_path(path: str) -> str:
    for hyp in REVIEW_BY_HYPOTHESIS:
        if hyp in path:
            return hyp
    return ""


def main() -> None:
    args = parse_args()
    review_path = args.visual_root / "visual_review.csv"
    insight_path = args.visual_root / "visual_insight.md"

    with review_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    updated = 0
    for row in rows:
        hyp = _hypothesis_from_path(row.get("visual_file", ""))
        if not hyp:
            continue
        row.update(REVIEW_BY_HYPOTHESIS[hyp])
        row["new_hypothesis_id"] = hyp
        updated += 1

    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    text = insight_path.read_text(encoding="utf-8") if insight_path.exists() else "# v78 Generated Visual Panels\n"
    marker = "Manual Review 2026-06-21 After Head-Selective Route-Bias"
    if marker not in text:
        insight_path.write_text(text.rstrip() + INSIGHT_APPENDIX + "\n", encoding="utf-8")

    print(f"updated_review_rows={updated}")


if __name__ == "__main__":
    main()
