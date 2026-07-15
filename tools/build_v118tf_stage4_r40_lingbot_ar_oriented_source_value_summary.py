#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R40 LingBot AR oriented source-value run."""

from __future__ import annotations

import os


os.environ.setdefault("ACL2_V118_AR_STAGE_TAG", "r40")
os.environ.setdefault("ACL2_V118_AR_STAGE_SLUG", "stage4_r40_lingbot_ar_oriented_source_value")

import build_v118tf_stage4_r28_lingbot_ar_anchor_read_summary as base_summary


if __name__ == "__main__":
    base_summary.main()
