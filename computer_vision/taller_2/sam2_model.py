"""Simple SAM2 transfer-learning Lightning module for 7-class segmentation."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from computer_vision.taller_2.dataset import CLASS_NAMES, DEFAULT_COLORS
except ModuleNotFoundError:
    # Notebook-local fallback when importing as `from sam2_model import ...`.
    from dataset import CLASS_NAMES, DEFAULT_COLORS


def _import_sam2_components() -> tuple[Any, Any]:
    """Import SAM2 builder and predictor with a safe cwd workaround.

    SAM2 raises at import time when Python runs from the parent folder of the
    cloned repo. Temporarily switching cwd avoids that guard in notebook usage.
    """

    original_cwd = Path.cwd()
    sam2_repo_dir = PROJECT_ROOT / "sam2"
    try:
        os.chdir(sam2_repo_dir)
        build_module = import_module("sam2.build_sam")
        predictor_module = import_module("sam2.sam2_image_predictor")
    finally:
        os.chdir(original_cwd)

    return build_module.build_sam2, predictor_module.SAM2ImagePredictor


def _resolve_sam2_config_path(config_name: str) -> str:
    """Resolve SAM2 config path as Hydra package-relative path.

    SAM2's `build_sam2` expects a config path relative to the `sam2` package,
    e.g. `configs/sam2.1/sam2.1_hiera_t.yaml`.
    """

    candidate = Path(config_name)
    if candidate.exists() and "configs" in candidate.parts:
        configs_index = candidate.parts.index("configs")
        return "/".join(candidate.parts[configs_index:])

    search_roots = [
        PROJECT_ROOT / "sam2" / "configs" / "sam2.1",
        PROJECT_ROOT / "sam2" / "sam2" / "configs" / "sam2.1",
    ]
    for root in search_roots:
        path = root / config_name
        if path.exists():
            return f"configs/sam2.1/{config_name}"

    raise FileNotFoundError(f"SAM2 config not found: {config_name}")


def _resolve_sam2_checkpoint_path(checkpoint_name: str) -> Path:
    """Resolve SAM2 checkpoint path using known local layouts."""

    candidate = Path(checkpoint_name)
    if candidate.exists():
        return candidate

    search_roots = [
        PROJECT_ROOT / "sam2" / "checkpoints",
    ]
    for root in search_roots:
        path = root / checkpoint_name
        if path.exists():
            return path

    raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_name}")


class SimpleSAM2Decoder(nn.Module):
    """Small convolutional decoder on top of SAM2 image embeddings."""

    def __init__(self, in_channels: int = 256, num_classes: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class logits on embedding feature maps."""

        return self.net(x)


class SAM2TransferLightningModule(L.LightningModule):
    """LightningModule that trains a lightweight decoder on frozen SAM2 embeddings.

    Args:
        sam2_config_name: SAM2 model config filename.
        sam2_checkpoint_name: SAM2 checkpoint filename.
        num_classes: Number of output classes.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        class_weights: Optional class weights for weighted CrossEntropy.
        visual_log_every_n_epochs: Qualitative TensorBoard logging cadence.
    """

    def __init__(
        self,
        sam2_config_name: str = "sam2.1_hiera_t.yaml",
        sam2_checkpoint_name: str = "sam2.1_hiera_tiny.pt",
        sam2_device: str = "cpu",
        num_classes: int = 7,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
        visual_log_every_n_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.sam2_config_path = _resolve_sam2_config_path(sam2_config_name)
        self.sam2_checkpoint_path = _resolve_sam2_checkpoint_path(sam2_checkpoint_name)
        self.sam2_device = sam2_device
        build_sam2, sam2_predictor_class = _import_sam2_components()
        original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            self.sam2_model = build_sam2(
                config_file=self.sam2_config_path,
                ckpt_path=str(self.sam2_checkpoint_path),
                device=self.sam2_device,
                mode="eval",
                hydra_overrides_extra=[
                    "++model.image_encoder.neck.position_encoding.warmup_cache=false"
                ],
            )
        finally:
            torch.cuda.is_available = original_is_available
        self.predictor = sam2_predictor_class(self.sam2_model)

        for parameter in self.sam2_model.parameters():
            parameter.requires_grad = False

        self.decoder = SimpleSAM2Decoder(in_channels=256, num_classes=num_classes)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.visual_log_every_n_epochs = visual_log_every_n_epochs
        self.class_weights = class_weights

        self.train_iou_macro = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.val_iou_macro = MulticlassJaccardIndex(num_classes=num_classes, average="macro")

        self.train_iou_per_class = MulticlassJaccardIndex(num_classes=num_classes, average=None)
        self.val_iou_per_class = MulticlassJaccardIndex(num_classes=num_classes, average=None)

        self.train_acc_macro = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc_macro = MulticlassAccuracy(num_classes=num_classes, average="macro")

        self.train_acc_per_class = MulticlassAccuracy(num_classes=num_classes, average=None)
        self.val_acc_per_class = MulticlassAccuracy(num_classes=num_classes, average=None)

        self.train_acc_micro = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc_micro = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self._fixed_val_images: torch.Tensor | None = None
        self._fixed_val_masks: torch.Tensor | None = None

    def set_fixed_validation_samples(self, images: torch.Tensor, masks: torch.Tensor) -> None:
        """Store fixed validation samples for qualitative TensorBoard tracking."""

        self._fixed_val_images = images.clone()
        self._fixed_val_masks = masks.clone()

    def _denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """Convert normalized CHW tensor back to [0, 1] range."""

        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        output = image * std + mean
        return torch.clamp(output, 0.0, 1.0)

    def _to_uint8_rgb(self, image: torch.Tensor) -> np.ndarray:
        """Convert normalized CHW tensor to uint8 HWC RGB array for SAM2."""

        denorm = self._denormalize(image)
        hwc = denorm.permute(1, 2, 0).detach().cpu().numpy()
        return (hwc * 255.0).clip(0.0, 255.0).astype(np.uint8)

    def _encode_with_sam2(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch into SAM2 image embeddings.

        This intentionally loops over the batch to keep the implementation easy to read.
        """

        embeddings: list[torch.Tensor] = []
        for idx in range(images.shape[0]):
            image_np = self._to_uint8_rgb(images[idx])
            self.predictor.set_image(image_np)
            image_embedding = self.predictor.get_image_embedding().squeeze(0)
            embeddings.append(image_embedding)

        stacked = torch.stack(embeddings, dim=0).to(self.device)
        return stacked

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run decoder logits over frozen SAM2 embeddings."""

        with torch.no_grad():
            embeddings = self._encode_with_sam2(images)
        return self.decoder(embeddings)

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted CrossEntropy loss only."""

        class_weights = None
        if self.class_weights is not None:
            class_weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        return nn.functional.cross_entropy(logits, target, weight=class_weights)

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass and return loss/predictions/targets."""

        images, masks = batch
        logits = self(images)
        logits = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = self._compute_loss(logits, masks)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, masks

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Execute one training step and update train metrics."""

        loss, preds, target = self._shared_step(batch)
        self.train_iou_macro.update(preds, target)
        self.train_iou_per_class.update(preds, target)
        self.train_acc_macro.update(preds, target)
        self.train_acc_per_class.update(preds, target)
        self.train_acc_micro.update(preds, target)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Execute one validation step and update validation metrics."""

        loss, preds, target = self._shared_step(batch)
        self.val_iou_macro.update(preds, target)
        self.val_iou_per_class.update(preds, target)
        self.val_acc_macro.update(preds, target)
        self.val_acc_per_class.update(preds, target)
        self.val_acc_micro.update(preds, target)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def _log_epoch_metrics(
        self,
        stage: str,
        iou_macro: MulticlassJaccardIndex,
        iou_per_class: MulticlassJaccardIndex,
        acc_macro: MulticlassAccuracy,
        acc_per_class: MulticlassAccuracy,
        acc_micro: MulticlassAccuracy,
    ) -> None:
        """Log aggregate and per-class metrics for one stage."""

        macro_iou = iou_macro.compute()
        iou_pc = iou_per_class.compute()
        macro_acc = acc_macro.compute()
        acc_pc = acc_per_class.compute()
        micro_acc = acc_micro.compute()

        self.log(f"{stage}_mean_iou", macro_iou, prog_bar=True)
        self.log(f"{stage}_mean_acc", macro_acc, prog_bar=False)
        self.log(f"{stage}_pixel_acc", micro_acc, prog_bar=False)

        for idx in range(self.num_classes):
            class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
            self.log(f"{stage}_iou_{class_name}", iou_pc[idx], prog_bar=False)
            self.log(f"{stage}_acc_{class_name}", acc_pc[idx], prog_bar=False)

        iou_macro.reset()
        iou_per_class.reset()
        acc_macro.reset()
        acc_per_class.reset()
        acc_micro.reset()

    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""

        self._log_epoch_metrics(
            stage="train",
            iou_macro=self.train_iou_macro,
            iou_per_class=self.train_iou_per_class,
            acc_macro=self.train_acc_macro,
            acc_per_class=self.train_acc_per_class,
            acc_micro=self.train_acc_micro,
        )

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics and fixed qualitative examples."""

        self._log_epoch_metrics(
            stage="val",
            iou_macro=self.val_iou_macro,
            iou_per_class=self.val_iou_per_class,
            acc_macro=self.val_acc_macro,
            acc_per_class=self.val_acc_per_class,
            acc_micro=self.val_acc_micro,
        )
        self._log_fixed_validation_examples()

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Map class-index mask to RGB colors."""

        color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for class_id, rgb in DEFAULT_COLORS.items():
            color[mask == class_id] = rgb
        return color

    def _log_fixed_validation_examples(self) -> None:
        """Log fixed validation examples to TensorBoard for evolution tracking."""

        if self.current_epoch % self.visual_log_every_n_epochs != 0:
            return
        if self._fixed_val_images is None or self._fixed_val_masks is None:
            return
        if not isinstance(self.logger, TensorBoardLogger):
            return

        self.decoder.eval()
        with torch.no_grad():
            images = self._fixed_val_images.to(self.device)
            masks = self._fixed_val_masks.to(self.device)
            logits = self(images)
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            preds = torch.argmax(logits, dim=1)

        count = min(images.shape[0], 5)
        fig, axes = plt.subplots(count, 3, figsize=(12, max(3 * count, 4)))
        if count == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx in range(count):
            display_image = self._denormalize(images[idx]).permute(1, 2, 0).cpu().numpy()
            gt_color = self._colorize_mask(masks[idx].cpu().numpy())
            pred_color = self._colorize_mask(preds[idx].cpu().numpy())

            axes[idx, 0].imshow(display_image)
            axes[idx, 0].set_title("Frame")
            axes[idx, 1].imshow(gt_color)
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 2].imshow(pred_color)
            axes[idx, 2].set_title("Prediction")

            for col in range(3):
                axes[idx, col].axis("off")

        fig.tight_layout()
        self.logger.experiment.add_figure(
            "val/fixed_examples",
            fig,
            global_step=self.current_epoch,
        )
        plt.close(fig)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create optimizer for trainable decoder parameters."""

        return torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
