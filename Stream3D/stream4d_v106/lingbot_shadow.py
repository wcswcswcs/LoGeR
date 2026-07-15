from __future__ import annotations

from dataclasses import dataclass

from .config import V106Config


@dataclass(frozen=True)
class LingBotShadowRecord:
    enabled: bool
    mode: str
    affects_core_method: bool


def audit_lingbot_shadow(config: V106Config) -> LingBotShadowRecord:
    affects_core = (
        config.lingbot.affects_mask
        or config.lingbot.affects_gap
        or config.lingbot.affects_identity
        or config.lingbot.affects_gate
    )
    if config.lingbot.enabled and (config.lingbot.mode != "shadow" or affects_core):
        raise RuntimeError("LingBot is only allowed as a no-effect shadow provider in v106 core")
    return LingBotShadowRecord(config.lingbot.enabled, config.lingbot.mode, affects_core)

