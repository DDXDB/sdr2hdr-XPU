from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.train import compute_loss, resolve_training_device, split_train_val_indices


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

    def test_split_train_val_indices_groups_by_source_video(self) -> None:
        paths = [
            Path("videoA_frame_000001_natural_v00.npz"),
            Path("videoA_frame_000002_natural_v00.npz"),
            Path("videoB_frame_000001_natural_v00.npz"),
            Path("videoB_frame_000002_natural_v00.npz"),
            Path("videoC_frame_000001_natural_v00.npz"),
        ]
        train_indices, val_indices = split_train_val_indices(paths, val_fraction=0.34)
        train_keys = {paths[index].name.split("_frame_")[0] for index in train_indices}
        val_keys = {paths[index].name.split("_frame_")[0] for index in val_indices}
        self.assertTrue(train_keys)
        self.assertTrue(val_keys)
        self.assertTrue(train_keys.isdisjoint(val_keys))
        self.assertEqual(sorted(train_indices + val_indices), list(range(len(paths))))

    def test_split_train_val_indices_falls_back_for_single_video(self) -> None:
        paths = [Path(f"only_frame_{index:06d}_natural_v00.npz") for index in range(10)]
        train_indices, val_indices = split_train_val_indices(paths)
        self.assertTrue(val_indices)
        self.assertEqual(sorted(train_indices + val_indices), list(range(len(paths))))


if __name__ == "__main__":
    unittest.main()
