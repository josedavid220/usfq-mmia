"""Dataset and DataModule utilities for Taller 2 semantic segmentation."""

from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from settings import DATA_DIR
from settings.optimizations import NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR

NUM_CLASSES = 7
CLASS_NAMES = [
    "background",
    "track",
    "kart",
    "pickup",
    "nitro",
    "bomb",
    "projectile",
]
FRAME_PATTERN = re.compile(r"frame_(\d{4})\.png$")
MASK_TEMPLATE = "mask_combined_{frame_id:04d}.png"
DEFAULT_COLORS = {
    0: (15, 20, 32),  # dark background
    1: (41, 127, 185),  # blue track
    2: (230, 155, 57),  # orange karts
    3: (30, 160, 75),  # green pickups
    4: (115, 215, 255),  # cyan nitro
    5: (220, 66, 66),  # red bombs
    6: (182, 97, 196),  # purple projectiles
}


@dataclass(frozen=True)
class SegmentationSample:
    """Container with paired paths and metadata for one frame.

    Args:
        track: Track identifier.
        frame_id: Frame identifier from filename.
        image_path: Path to RGB frame file.
        mask_path: Path to combined class-index mask.
    """

    track: str
    frame_id: int
    image_path: Path
    mask_path: Path


def collect_segmentation_samples(
    data_root: Path,
    min_frame_id: int = 0,
    max_frame_id: int = 249,
) -> tuple[list[SegmentationSample], list[tuple[str, int]]]:
    """Collect valid frame/mask pairs from all tracks.

    Args:
        data_root: Base directory containing all track folders.
        min_frame_id: Minimum accepted frame index.
        max_frame_id: Maximum accepted frame index.

    Returns:
        A tuple containing:
        - List of valid samples with existing combined masks.
        - List of discarded (track, frame_id) entries missing masks.
    """

    samples: list[SegmentationSample] = []
    missing_masks: list[tuple[str, int]] = []
    for track_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        frame_dir = track_dir / "frame"
        combined_dir = track_dir / "combined"
        if not frame_dir.exists() or not combined_dir.exists():
            continue

        for image_path in sorted(frame_dir.glob("frame_*.png")):
            match = FRAME_PATTERN.search(image_path.name)
            if not match:
                continue
            frame_id = int(match.group(1))
            if frame_id < min_frame_id or frame_id > max_frame_id:
                continue

            mask_path = combined_dir / MASK_TEMPLATE.format(frame_id=frame_id)
            if not mask_path.exists():
                missing_masks.append((track_dir.name, frame_id))
                continue

            samples.append(
                SegmentationSample(
                    track=track_dir.name,
                    frame_id=frame_id,
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )
    return samples, missing_masks


def split_by_minority_presence(
    samples: Iterable[SegmentationSample],
    val_fraction: float,
    val_minority_fraction: float,
    seed: int,
) -> tuple[list[SegmentationSample], list[SegmentationSample]]:
    """Split samples into train/validation with minority-presence control.

    Args:
        samples: Full sample iterable.
        val_fraction: Fraction of samples to allocate to validation.
        val_minority_fraction: Desired ratio of minority-present samples in validation.
        seed: Random seed for reproducible split.

    Returns:
        Tuple with train and validation sample lists.

    Raises:
        ValueError: If split configuration is invalid or any split is empty.
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "val_fraction must be in (0, 1). "
            f"Got {val_fraction}."
        )
    if not 0.0 <= val_minority_fraction <= 1.0:
        raise ValueError(
            "val_minority_fraction must be in [0, 1]. "
            f"Got {val_minority_fraction}."
        )

    all_samples = list(samples)
    if len(all_samples) < 2:
        raise ValueError("Need at least 2 samples to create train/validation split.")

    minority_present: list[SegmentationSample] = []
    majority_only: list[SegmentationSample] = []
    for sample in all_samples:
        mask = np.array(Image.open(sample.mask_path), dtype=np.int64)
        counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)
        if counts[3:].sum() > 0:
            minority_present.append(sample)
        else:
            majority_only.append(sample)

    rng = random.Random(seed)
    rng.shuffle(minority_present)
    rng.shuffle(majority_only)

    n_total = len(all_samples)
    n_val = int(round(n_total * val_fraction))
    n_val = max(1, min(n_total - 1, n_val))
    n_val_minority = min(len(minority_present), int(round(n_val * val_minority_fraction)))
    n_val_majority = n_val - n_val_minority
    if n_val_majority > len(majority_only):
        overflow = n_val_majority - len(majority_only)
        n_val_majority = len(majority_only)
        n_val_minority = min(len(minority_present), n_val_minority + overflow)

    val = minority_present[:n_val_minority] + majority_only[:n_val_majority]
    rng.shuffle(val)

    val_set = set(val)
    train = [sample for sample in all_samples if sample not in val_set]
    if not train or not val:
        raise ValueError(
            "Split produced an empty set. "
            f"train={len(train)}, val={len(val)}"
        )

    return train, val


def compute_class_statistics(
    samples: Iterable[SegmentationSample],
    num_classes: int = NUM_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute class-pixel frequencies and per-sample sampler weights.

    Args:
        samples: Dataset sample iterable.
        num_classes: Number of segmentation classes.

    Returns:
        Tuple with class pixel counts and sample-level weights.
    """

    class_counts = np.zeros(num_classes, dtype=np.int64)
    weights: list[float] = []
    for sample in samples:
        mask = np.array(Image.open(sample.mask_path), dtype=np.int64)
        counts = np.bincount(mask.reshape(-1), minlength=num_classes)
        class_counts += counts
        minority_pixels = counts[3:].sum()
        total_pixels = counts.sum()
        minority_ratio = float(minority_pixels) / float(max(total_pixels, 1))
        weights.append(1.0 + 8.0 * minority_ratio)
    return class_counts, np.array(weights, dtype=np.float32)


def build_class_weights(
    class_counts: np.ndarray, smoothing: float = 1e-6
) -> torch.Tensor:
    """Create class weights from inverse frequency.

    Args:
        class_counts: Pixel counts per class.
        smoothing: Small value to avoid division by zero.

    Returns:
        Normalized class weights as float tensor.
    """

    frequencies = class_counts.astype(np.float64)
    frequencies = frequencies / max(float(frequencies.sum()), 1.0)
    inverse = 1.0 / np.sqrt(frequencies + smoothing)
    normalized = inverse / inverse.mean()
    return torch.tensor(normalized, dtype=torch.float32)


class DenseSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for dense_data semantic segmentation with paired augmentations.

    Args:
        samples: List of paired frame/mask entries.
        crop_size: Output crop size.
        split: Dataset split name.
        normalize: Whether to apply ImageNet normalization.
        minority_focus_prob: Probability of minority-aware crop selection.
    """

    def __init__(
        self,
        samples: list[SegmentationSample],
        crop_size: int = 256,
        split: str = "train",
        normalize: bool = True,
        minority_focus_prob: float = 0.7,
    ) -> None:
        self.samples = samples
        self.crop_size = crop_size
        self.split = split
        self.normalize = normalize
        self.minority_focus_prob = minority_focus_prob
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
            3, 1, 1
        )
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
            3, 1, 1
        )

    def __len__(self) -> int:
        """Return dataset length."""

        return len(self.samples)

    def _pick_crop_origin(self, mask: np.ndarray) -> tuple[int, int]:
        """Select crop top-left location using minority-aware sampling.

        Args:
            mask: Full-resolution class-index mask.

        Returns:
            Tuple of top and left crop coordinates.
        """

        height, width = mask.shape
        crop_h = min(self.crop_size, height)
        crop_w = min(self.crop_size, width)
        max_top = max(height - crop_h, 0)
        max_left = max(width - crop_w, 0)

        minority_coords = np.argwhere(mask >= 3)
        if (
            self.split == "train"
            and minority_coords.size > 0
            and random.random() < self.minority_focus_prob
        ):
            center_y, center_x = minority_coords[random.randrange(len(minority_coords))]
            top = int(np.clip(center_y - crop_h // 2, 0, max_top))
            left = int(np.clip(center_x - crop_w // 2, 0, max_left))
            return top, left

        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
        return top, left

    def _apply_train_augmentations(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply lightweight paired augmentations for train samples.

        Args:
            image: Image tensor in CHW format.
            mask: Mask tensor in HW format.

        Returns:
            Augmented image and mask tensors.
        """

        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if random.random() < 0.2:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])

        brightness = 1.0 + random.uniform(-0.15, 0.15)
        contrast = 1.0 + random.uniform(-0.15, 0.15)
        image = torch.clamp(image * brightness, 0.0, 1.0)
        mean = image.mean(dim=(1, 2), keepdim=True)
        image = torch.clamp((image - mean) * contrast + mean, 0.0, 1.0)
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load image-mask pair with split-specific preprocessing.

        Args:
            index: Sample index.

        Returns:
            Tuple of image tensor (float32) and mask tensor (int64).
        """

        sample = self.samples[index]
        image_np = np.array(
            Image.open(sample.image_path).convert("RGB"), dtype=np.uint8
        )
        mask_np = np.array(Image.open(sample.mask_path), dtype=np.int64)

        top, left = self._pick_crop_origin(mask_np)
        crop_h = min(self.crop_size, mask_np.shape[0])
        crop_w = min(self.crop_size, mask_np.shape[1])
        image_np = image_np[top : top + crop_h, left : left + crop_w]
        mask_np = mask_np[top : top + crop_h, left : left + crop_w]

        image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask_np).long()

        if self.split == "train":
            image, mask = self._apply_train_augmentations(image, mask)

        if self.normalize:
            image = (image - self._mean) / self._std

        return image, mask


class DenseSegmentationDataModule(L.LightningDataModule):
    """Lightning DataModule for the dense_data segmentation dataset.

    Args:
        data_root: Dataset root path.
        batch_size: Batch size.
        crop_size: Crop size for model inputs.
        val_fraction: Fraction of samples used for validation.
        val_minority_fraction: Desired minority-present ratio in validation split.
        num_workers: Number of workers for DataLoader.
        pin_memory: Whether DataLoader uses pinned memory.
        seed: Random seed.
    """

    def __init__(
        self,
        data_root: Path | None = None,
        batch_size: int = 12,
        crop_size: int = 256,
        val_fraction: float = 0.2,
        val_minority_fraction: float = 0.5,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = PIN_MEMORY,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_root = data_root or (DATA_DIR / "dense_data")
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.val_fraction = val_fraction
        self.val_minority_fraction = val_minority_fraction
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self.train_dataset: DenseSegmentationDataset | None = None
        self.val_dataset: DenseSegmentationDataset | None = None

        self.missing_masks: list[tuple[str, int]] = []
        self.class_counts: np.ndarray = np.zeros(NUM_CLASSES, dtype=np.int64)
        self.class_weights: torch.Tensor = torch.ones(NUM_CLASSES, dtype=torch.float32)
        self.sample_weights: np.ndarray = np.array([], dtype=np.float32)
        self.train_samples: list[SegmentationSample] = []
        self.val_samples: list[SegmentationSample] = []

    def prepare_data(self) -> None:
        """Validate expected data root exists.

        Raises:
            FileNotFoundError: If dense_data root is not available.
        """

        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")

    def setup(self, stage: str | None = None) -> None:
        """Build sample lists, statistics, and split datasets.

        Args:
            stage: Optional stage used by Lightning.
        """

        random.seed(self.seed)
        np.random.seed(self.seed)
        samples, missing_masks = collect_segmentation_samples(self.data_root)
        self.missing_masks = missing_masks

        train, val = split_by_minority_presence(
            samples=samples,
            val_fraction=self.val_fraction,
            val_minority_fraction=self.val_minority_fraction,
            seed=self.seed,
        )
        self.train_samples = train
        self.val_samples = val
        self.class_counts, self.sample_weights = compute_class_statistics(train)
        self.class_weights = build_class_weights(self.class_counts)

        self.train_dataset = DenseSegmentationDataset(
            samples=train,
            crop_size=self.crop_size,
            split="train",
        )
        self.val_dataset = DenseSegmentationDataset(
            samples=val,
            crop_size=self.crop_size,
            split="val",
        )

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create train DataLoader with weighted sampling."""

        if self.train_dataset is None:
            raise RuntimeError(
                "DataModule.setup must be called before creating dataloaders."
            )
        sampler = WeightedRandomSampler(
            weights=self.sample_weights.tolist(),
            num_samples=len(self.sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create validation DataLoader."""

        if self.val_dataset is None:
            raise RuntimeError(
                "DataModule.setup must be called before creating dataloaders."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=self.num_workers > 0,
        )

    def fixed_validation_subset(
        self,
        n_samples: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor, list[SegmentationSample]]:
        """Return a deterministic subset of validation samples for visual logs.

        Args:
            n_samples: Number of examples to return.

        Returns:
            Image and mask tensors plus source sample metadata.

        Raises:
            RuntimeError: If setup has not been called.
        """

        if self.val_dataset is None:
            raise RuntimeError("Call setup before requesting fixed validation subset.")
        total = min(n_samples, len(self.val_dataset))
        indices = list(range(total))
        images: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        subset_meta: list[SegmentationSample] = []
        for idx in indices:
            image, mask = self.val_dataset[idx]
            images.append(image)
            masks.append(mask)
            subset_meta.append(self.val_samples[idx])
        return torch.stack(images), torch.stack(masks), subset_meta
