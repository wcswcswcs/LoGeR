#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R32 LingBot AR component-shuffle controls."""

from __future__ import annotations

import os


os.environ.setdefault("ACL2_V118_AR_STAGE_TAG", "r32")
os.environ.setdefault("ACL2_V118_AR_STAGE_SLUG", "stage4_r32_lingbot_ar_component_shuffle")

import build_v118tf_stage4_r31_lingbot_ar_source_value_cue_ablation_summary as r31_summary


if __name__ == "__main__":
    r31_summary.main()
