"""PyTorch Lightning model for multiclass semantic segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from computer_vision.taller_2.dataset import CLASS_NAMES, DEFAULT_COLORS


class SegmentationLightningModule(L.LightningModule):
    """LightningModule wrapping SMP architectures for seven-class segmentation.

    Args:
        architecture: SMP architecture name.
        encoder_name: Backbone encoder name.
        encoder_weights: Pretrained encoder weights.
        num_classes: Number of classes.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        loss_name: Training loss identifier.
        class_weights: Optional class weights tensor.
        visual_log_every_n_epochs: Validation image logging cadence.
    """

    def __init__(
        self,
        architecture: str = "Unet",
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        num_classes: int = 7,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_name: str = "ce_dice",
        class_weights: torch.Tensor | None = None,
        visual_log_every_n_epochs: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.model = smp.create_model(
            arch=architecture,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.loss_name = loss_name.lower()
        self.visual_log_every_n_epochs = visual_log_every_n_epochs
        self.class_weights = class_weights

        self.train_iou_macro = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.val_iou_macro = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.test_iou_macro = MulticlassJaccardIndex(num_classes=num_classes, average="macro")

        self.train_iou_per_class = MulticlassJaccardIndex(num_classes=num_classes, average=None)
        self.val_iou_per_class = MulticlassJaccardIndex(num_classes=num_classes, average=None)
        self.test_iou_per_class = MulticlassJaccardIndex(num_classes=num_classes, average=None)

        self.train_acc_macro = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc_macro = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.test_acc_macro = MulticlassAccuracy(num_classes=num_classes, average="macro")

        self.train_acc_per_class = MulticlassAccuracy(num_classes=num_classes, average=None)
        self.val_acc_per_class = MulticlassAccuracy(num_classes=num_classes, average=None)
        self.test_acc_per_class = MulticlassAccuracy(num_classes=num_classes, average=None)

        self.train_acc_micro = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc_micro = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_acc_micro = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self._dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self._focal_loss = smp.losses.FocalLoss(mode="multiclass")
        self._fixed_val_images: torch.Tensor | None = None
        self._fixed_val_masks: torch.Tensor | None = None

    def set_fixed_validation_samples(self, images: torch.Tensor, masks: torch.Tensor) -> None:
        """Store fixed validation samples for qualitative TensorBoard tracking.

        Args:
            images: Validation images tensor in BCHW format.
            masks: Validation class-index masks in BHW format.
        """

        self._fixed_val_images = images.clone()
        self._fixed_val_masks = masks.clone()

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute configured segmentation loss.

        Args:
            logits: Model output logits of shape BxCxHxW.
            target: Class-index masks of shape BxHxW.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If loss_name is unsupported.
        """

        class_weights = None
        if self.class_weights is not None:
            class_weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)

        if self.loss_name == "cross_entropy":
            return nn.functional.cross_entropy(logits, target, weight=class_weights)
        if self.loss_name == "focal":
            focal = self._focal_loss(logits, target)
            if class_weights is None:
                return focal
            ce = nn.functional.cross_entropy(logits, target, weight=class_weights)
            return 0.5 * focal + 0.5 * ce
        if self.loss_name == "dice":
            return self._dice_loss(logits, target)
        if self.loss_name == "ce_dice":
            ce = nn.functional.cross_entropy(logits, target, weight=class_weights)
            dice = self._dice_loss(logits, target)
            return 0.5 * ce + 0.5 * dice

        raise ValueError(
            "Unsupported loss_name. Use one of: "
            "cross_entropy, focal, dice, ce_dice."
        )

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass and return loss/predictions/targets.

        Args:
            batch: Batch tuple with image and mask tensors.

        Returns:
            Loss scalar, predicted labels, and ground truth labels.
        """

        images, masks = batch
        logits = self.model(images)
        loss = self._compute_loss(logits, masks)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, masks

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute one training step and update train metrics."""

        loss, preds, target = self._shared_step(batch)
        self.train_iou_macro.update(preds, target)
        self.train_iou_per_class.update(preds, target)
        self.train_acc_macro.update(preds, target)
        self.train_acc_per_class.update(preds, target)
        self.train_acc_micro.update(preds, target)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Execute one validation step and update validation metrics."""

        loss, preds, target = self._shared_step(batch)
        self.val_iou_macro.update(preds, target)
        self.val_iou_per_class.update(preds, target)
        self.val_acc_macro.update(preds, target)
        self.val_acc_per_class.update(preds, target)
        self.val_acc_micro.update(preds, target)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Execute one test step and update test metrics."""

        loss, preds, target = self._shared_step(batch)
        self.test_iou_macro.update(preds, target)
        self.test_iou_per_class.update(preds, target)
        self.test_acc_macro.update(preds, target)
        self.test_acc_per_class.update(preds, target)
        self.test_acc_micro.update(preds, target)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def _log_epoch_metrics(
        self,
        stage: str,
        iou_macro: MulticlassJaccardIndex,
        iou_per_class: MulticlassJaccardIndex,
        acc_macro: MulticlassAccuracy,
        acc_per_class: MulticlassAccuracy,
        acc_micro: MulticlassAccuracy,
    ) -> None:
        """Log aggregate and per-class metrics for one stage.

        Args:
            stage: Stage prefix for metric names.
            iou_macro: Macro IoU metric instance.
            iou_per_class: Per-class IoU metric instance.
            acc_macro: Macro accuracy metric instance.
            acc_per_class: Per-class accuracy metric instance.
            acc_micro: Micro accuracy metric instance.
        """

        macro_iou = iou_macro.compute()
        iou_pc = iou_per_class.compute()
        macro_acc = acc_macro.compute()
        acc_pc = acc_per_class.compute()
        micro_acc = acc_micro.compute()

        self.log(f"{stage}_mean_iou", macro_iou, prog_bar=stage != "test")
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

    def on_test_epoch_end(self) -> None:
        """Log test metrics at epoch end."""

        self._log_epoch_metrics(
            stage="test",
            iou_macro=self.test_iou_macro,
            iou_per_class=self.test_iou_per_class,
            acc_macro=self.test_acc_macro,
            acc_per_class=self.test_acc_per_class,
            acc_micro=self.test_acc_micro,
        )

    def _denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """Convert normalized image tensor back to display range.

        Args:
            image: Normalized CHW tensor.

        Returns:
            Tensor clipped to [0, 1] in CHW format.
        """

        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        image = image * std + mean
        return torch.clamp(image, 0.0, 1.0)

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Map class-index mask to RGB colors.

        Args:
            mask: HxW class-index mask.

        Returns:
            HxWx3 uint8 color mask.
        """

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

        self.model.eval()
        with torch.no_grad():
            images = self._fixed_val_images.to(self.device)
            masks = self._fixed_val_masks.to(self.device)
            logits = self.model(images)
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run model forward pass.

        Args:
            images: Input image tensor in BCHW format.

        Returns:
            Raw logits in BCHW format.
        """

        return self.model(images)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create optimizer.

        Returns:
            AdamW optimizer instance.
        """

        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
