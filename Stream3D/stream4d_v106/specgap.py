from __future__ import annotations

from dataclasses import dataclass

from .config import V106Config


@dataclass(frozen=True)
class SpecGapPolicy:
    enabled: bool
    inherited_only_proxy: bool
    final_exact_residual_rounds: int


def build_specgap_policy(config: V106Config) -> SpecGapPolicy:
    if config.specgap.enabled and not config.specgap.inherited_only_proxy:
        raise RuntimeError("v106 speculative gap is only allowed as inherited-only proxy before exact residual pass")
    return SpecGapPolicy(
        enabled=config.specgap.enabled,
        inherited_only_proxy=config.specgap.inherited_only_proxy,
        final_exact_residual_rounds=config.specgap.final_exact_residual_rounds,
    )

