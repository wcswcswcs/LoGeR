#!/usr/bin/env python3
"""Build v82 confidence-stratified overlap-set manifests.

This is the plan-named entrypoint for Phase1. It delegates to the audit tool,
which writes both quality summaries and stratified manifests.
"""

from __future__ import annotations

from audit_v82_overlap_artifact_quality_stratification import main


if __name__ == "__main__":
    main()
