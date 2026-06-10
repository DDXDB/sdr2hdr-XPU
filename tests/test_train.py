from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from scripts.train import compute_loss, resolve_training_device


class TrainScriptTests(unittest.TestCase):
    def test_resolve_training_device_prefers_cuda(self) -> None:
        with patch("scripts.train.torch.cuda.is_available", return_value=True):
            device = resolve_training_device("auto")
        self.assertEqual(device.type, "cuda")

    def test_resolve_training_device_uses_mps_when_cuda_is_unavailable(self) -> None:
        with patch("scripts.train.torch.cuda.is_available", return_value=False):
            with patch("scripts.train.torch.backends.mps.is_available", return_value=True):
                device = resolve_training_device("auto")
        self.assertEqual(device.type, "mps")

    def test_resolve_training_device_falls_back_to_cpu(self) -> None:
        with patch("scripts.train.torch.cuda.is_available", return_value=False):
            with patch("scripts.train.torch.backends.mps.is_available", return_value=False):
                device = resolve_training_device("auto")
        self.assertEqual(device.type, "cpu")

    def test_resolve_training_device_respects_explicit_request(self) -> None:
        device = resolve_training_device("cpu")
        self.assertEqual(device.type, "cpu")

    def test_compute_loss_accepts_extended_training_batch(self) -> None:
        pred = torch.zeros(1, 3, 8, 8)
        batch = {
            "sdr_linear": torch.full((1, 3, 8, 8), 0.2),
            "target_maps": torch.zeros(1, 3, 8, 8),
            "clip_mask": torch.zeros(1, 1, 8, 8),
            "near_white_mask": torch.zeros(1, 1, 8, 8),
            "shadow_mask": torch.ones(1, 1, 8, 8),
            "memory_color_mask": torch.zeros(1, 1, 8, 8),
            "region_weight": torch.ones(1, 1, 8, 8),
        }
        loss, components = compute_loss(pred, batch)
        self.assertGreaterEqual(float(loss), 0.0)
        self.assertIn("tone", components)


if __name__ == "__main__":
    unittest.main()
