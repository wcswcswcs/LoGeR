from __future__ import annotations

import os
import unittest


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@unittest.skipIf(not _cuda_available(), "CUDA is not available to this Python environment")
class D4RTGpuOptionalTests(unittest.TestCase):
    def test_gpu_visibility_includes_requested_devices_when_set(self) -> None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.assertTrue(visible == "" or any(item.strip() in {"6", "7"} for item in visible.split(",")))


if __name__ == "__main__":
    unittest.main()
