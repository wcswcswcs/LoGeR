#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R42 fresh 01/05 token-gated source-value run."""

from __future__ import annotations

import os


os.environ.setdefault("ACL2_V118_AR_STAGE_TAG", "r42")
os.environ.setdefault("ACL2_V118_AR_STAGE_SLUG", "stage4_r42_lingbot_ar_token_gated_oriented_source_value_holdout")
os.environ.setdefault("ACL2_V118_AR_SEQS", "01,05")

import build_v118tf_stage4_r28_lingbot_ar_anchor_read_summary as base


if __name__ == "__main__":
    base.main()
