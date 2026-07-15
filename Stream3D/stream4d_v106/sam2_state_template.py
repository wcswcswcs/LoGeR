from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SAM2StateTemplateContract:
    model_reuse_across_chunks: bool
    template_reuse: bool
    image_autocast_dtype: str
    video_autocast_dtype: str

    def validate(self) -> None:
        if self.image_autocast_dtype != "bfloat16":
            raise ValueError("v106 default expects image autocast bfloat16")
        if self.video_autocast_dtype != "bfloat16":
            raise ValueError("v106 default expects video autocast bfloat16")

