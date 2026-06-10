from __future__ import annotations

import argparse
from dataclasses import dataclass
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from sdr2hdr.review import REC2020_TO_REC709, linear_to_srgb, pq_to_relative_linear


@dataclass(frozen=True)
class SDRDegradationProfile:
    exposure_bias: float
    shoulder_strength: float
    toe_strength: float
    saturation_scale: float
    warmth_shift: float


DEGRADATION_PROFILES = {
    "natural": SDRDegradationProfile(1.0, 1.0, 1.0, 1.0, 0.0),
    "clipped": SDRDegradationProfile(1.08, 1.28, 1.05, 0.94, 0.02),
    "compressed": SDRDegradationProfile(0.94, 1.15, 1.12, 0.88, -0.01),
    "night": SDRDegradationProfile(0.88, 1.08, 1.35, 0.82, -0.03),
}


def extract_raw_frames(input_path: str, output_dir: Path, pix_fmt: str, sample_every: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        input_path,
        "-vf",
        f"select='not(mod(n\\,{sample_every}))'",
        "-vsync",
        "0",
        "-pix_fmt",
        pix_fmt,
        str(output_dir / "frame_%06d.png"),
    ]
    subprocess.run(cmd, check=True)


def srgb_to_linear(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame, 0.0, 1.0)
    return np.where(frame <= 0.04045, frame / 12.92, ((frame + 0.055) / 1.055) ** 2.4)


def _apply_color_temperature(frame_709_linear: np.ndarray, warmth_shift: float) -> np.ndarray:
    if abs(warmth_shift) < 1e-6:
        return frame_709_linear
    gains = np.array([1.0 + warmth_shift, 1.0, 1.0 - warmth_shift], dtype=np.float32)
    return np.clip(frame_709_linear * gains[None, None, :], 0.0, None)


def _apply_sdr_degradation(
    frame_709_linear: np.ndarray,
    profile: SDRDegradationProfile,
    variant_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(variant_seed)
    frame = _apply_color_temperature(frame_709_linear, profile.warmth_shift + rng.uniform(-0.01, 0.01))
    luma = np.clip(
        0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2],
        0.0,
        None,
    )
    white = max(float(np.percentile(luma, 99.7)), 0.5)
    exposure = frame / white * profile.exposure_bias
    shoulder = np.power(np.clip(exposure, 0.0, None), np.clip(profile.shoulder_strength, 0.8, 1.6))
    mapped = shoulder / (1.0 + shoulder)
    if profile.toe_strength != 1.0:
        mapped = np.power(np.clip(mapped, 0.0, 1.0), np.clip(profile.toe_strength, 0.7, 1.6))
    mapped = np.clip(mapped, 0.0, 1.0)
    mapped_luma = np.clip(
        0.2126 * mapped[..., 0] + 0.7152 * mapped[..., 1] + 0.0722 * mapped[..., 2],
        1e-5,
        1.0,
    )
    saturation_scale = np.clip(profile.saturation_scale + rng.uniform(-0.06, 0.06), 0.72, 1.08)
    mapped = np.clip(mapped_luma[..., None] + (mapped - mapped_luma[..., None]) * saturation_scale, 0.0, 1.0)
    return mapped.astype(np.float32)


def tone_map_hdr_linear_to_sdr_linear(
    frame_709_linear: np.ndarray,
    profile_name: str = "natural",
    variant_seed: int = 0,
) -> np.ndarray:
    luma = np.clip(
        0.2126 * frame_709_linear[..., 0]
        + 0.7152 * frame_709_linear[..., 1]
        + 0.0722 * frame_709_linear[..., 2],
        0.0,
        None,
    )
    white = max(float(np.percentile(luma, 99.7)), 0.5)
    exposed = frame_709_linear / white
    mapped = exposed / (1.0 + exposed)
    degraded = _apply_sdr_degradation(mapped, DEGRADATION_PROFILES[profile_name], variant_seed)
    srgb = linear_to_srgb(np.clip(degraded, 0.0, 1.0))
    return srgb_to_linear(srgb).astype(np.float32)


def convert_frame_to_npz(
    hdr_frame_path: Path,
    output_path: Path,
    peak_nits: float,
    profile_name: str = "natural",
    variant_seed: int = 0,
) -> None:
    hdr_bgr16 = cv2.imread(str(hdr_frame_path), cv2.IMREAD_UNCHANGED)
    if hdr_bgr16 is None:
        raise RuntimeError(f"failed to read HDR frame: {hdr_frame_path}")
    hdr_rgb16 = cv2.cvtColor(hdr_bgr16, cv2.COLOR_BGR2RGB)
    hdr_2020_linear = pq_to_relative_linear(hdr_rgb16.astype(np.float32) / 65535.0, peak_nits=peak_nits)
    hdr_709_linear = np.clip(np.tensordot(hdr_2020_linear, REC2020_TO_REC709.T, axes=1), 0.0, 1.5)
    sdr_linear = tone_map_hdr_linear_to_sdr_linear(
        hdr_709_linear,
        profile_name=profile_name,
        variant_seed=variant_seed,
    )
    np.savez_compressed(
        output_path,
        sdr_linear=sdr_linear.astype(np.float16),
        hdr_linear=hdr_709_linear.astype(np.float16),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare SDR/HDR training pairs from HDR videos.")
    parser.add_argument("--input-dir", required=True, help="Directory containing HDR videos")
    parser.add_argument("--out-dir", required=True, help="Directory to save .npz training samples")
    parser.add_argument("--sample-every", type=int, default=24, help="Sample every N frames")
    parser.add_argument("--peak-nits", type=float, default=1000.0, help="Peak nits used for relative linear decode")
    parser.add_argument(
        "--degradation-profiles",
        default="natural,clipped,compressed",
        help=f"Comma-separated SDR degradation profiles: {', '.join(sorted(DEGRADATION_PROFILES))}",
    )
    parser.add_argument("--variants-per-profile", type=int, default=1, help="Number of randomized SDR variants per profile")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_names = [name.strip() for name in args.degradation_profiles.split(",") if name.strip()]
    unknown_profiles = [name for name in profile_names if name not in DEGRADATION_PROFILES]
    if unknown_profiles:
        raise ValueError(f"unknown degradation profile(s): {', '.join(unknown_profiles)}")

    with tempfile.TemporaryDirectory(prefix="sdr2hdr-train-") as temp_dir:
        temp_root = Path(temp_dir)
        for video_path in sorted(input_dir.iterdir()):
            if not video_path.is_file():
                continue
            hdr_frames_dir = temp_root / f"{video_path.stem}_hdr_frames"
            extract_raw_frames(str(video_path), hdr_frames_dir, "rgb48le", args.sample_every)
            for hdr_frame_path in sorted(hdr_frames_dir.glob("frame_*.png")):
                for profile_name in profile_names:
                    for variant_index in range(max(args.variants_per_profile, 1)):
                        output_path = out_dir / f"{video_path.stem}_{hdr_frame_path.stem}_{profile_name}_v{variant_index:02d}.npz"
                        seed_source = f"{video_path.stem}|{hdr_frame_path.stem}|{profile_name}|{variant_index}"
                        seed = sum(seed_source.encode("utf-8")) & 0xFFFFFFFF
                        convert_frame_to_npz(
                            hdr_frame_path,
                            output_path,
                            peak_nits=args.peak_nits,
                            profile_name=profile_name,
                            variant_seed=seed,
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
