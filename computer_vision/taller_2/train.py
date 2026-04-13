"""Training script for Taller 2 semantic segmentation experiments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from settings import LOGS_DIR
from settings.optimizations import ENABLE_PROGRESS_BAR

from computer_vision.taller_2.dataset import CLASS_NAMES, DenseSegmentationDataModule
from computer_vision.taller_2.model import SegmentationLightningModule


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for segmentation experiments.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(description="Train semantic segmentation model for Taller 2.")
    parser.add_argument("--architecture", type=str, default="Unet")
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--loss-name", type=str, default="ce_dice")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="taller2-segmentation")
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--val-tracks", nargs="+", default=["olivermath"])
    parser.add_argument("--test-tracks", nargs="+", default=["volcano_island"])
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--fixed-val-samples", type=int, default=5)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--num-sanity-val-steps", type=int, default=1)
    return parser.parse_args()


def _normalize_encoder_weights(value: str) -> str | None:
    """Map CLI encoder weight values to SMP expectations.

    Args:
        value: Raw encoder weights argument.

    Returns:
        None for disabled pretraining, otherwise original value.
    """

    if value.lower() in {"none", "null"}:
        return None
    return value


def _sanitize_name(value: str) -> str:
    """Sanitize a value for use in checkpoint file names.

    Args:
        value: Raw name.

    Returns:
        Safe lowercase name without spaces or path separators.
    """

    cleaned = value.lower().strip()
    cleaned = re.sub(r"[^a-z0-9_.-]+", "-", cleaned)
    return cleaned.strip("-")


def _build_callbacks(patience: int, checkpoint_prefix: str) -> list:
    """Build trainer callbacks used across runs.

    Args:
        patience: Early stopping patience.
        checkpoint_prefix: Prefix identifying architecture/backbone/loss.

    Returns:
        Callback list.
    """

    checkpoint = ModelCheckpoint(
        monitor="val_mean_iou",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename=f"{checkpoint_prefix}-e{{epoch:02d}}-miou{{val_mean_iou:.4f}}",
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(
        monitor="val_mean_iou",
        mode="max",
        patience=patience,
        verbose=True,
    )
    callbacks = [checkpoint, early_stop, LearningRateMonitor(logging_interval="epoch")]
    if ENABLE_PROGRESS_BAR:
        callbacks.append(RichProgressBar(leave=True))
    return callbacks


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one training experiment from config dictionary.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        Summary dictionary with important output paths and metrics.
    """

    L.seed_everything(int(config["seed"]), workers=True)

    data_root = Path(config["data_root"]) if config["data_root"] else None
    datamodule = DenseSegmentationDataModule(
        data_root=data_root,
        batch_size=int(config["batch_size"]),
        crop_size=int(config["crop_size"]),
        val_tracks=list(config["val_tracks"]),
        test_tracks=list(config["test_tracks"]),
        seed=int(config["seed"]),
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    model = SegmentationLightningModule(
        architecture=str(config["architecture"]),
        encoder_name=str(config["encoder_name"]),
        encoder_weights=_normalize_encoder_weights(str(config["encoder_weights"])),
        num_classes=len(CLASS_NAMES),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        loss_name=str(config["loss_name"]),
        class_weights=datamodule.class_weights,
    )
    fixed_images, fixed_masks, _ = datamodule.fixed_validation_subset(
        n_samples=int(config["fixed_val_samples"])
    )
    model.set_fixed_validation_samples(fixed_images, fixed_masks)

    logger = TensorBoardLogger(
        save_dir=str(LOGS_DIR),
        name=str(config["experiment_name"]),
        default_hp_metric=False,
    )
    logger.log_hyperparams(config)

    ckpt_prefix = "-".join(
        [
            _sanitize_name(str(config["architecture"])),
            _sanitize_name(str(config["encoder_name"])),
            _sanitize_name(str(config["loss_name"])),
        ]
    )

    trainer = L.Trainer(
        max_epochs=int(config["max_epochs"]),
        logger=logger,
        callbacks=_build_callbacks(int(config["patience"]), checkpoint_prefix=ckpt_prefix),
        accelerator=str(config["accelerator"]),
        devices=config["devices"],
        precision=str(config["precision"]),
        strategy=str(config["strategy"]),
        accumulate_grad_batches=int(config["accumulate_grad_batches"]),
        log_every_n_steps=10,
        enable_progress_bar=ENABLE_PROGRESS_BAR,
        num_sanity_val_steps=int(config["num_sanity_val_steps"]),
    )

    trainer.fit(model=model, datamodule=datamodule)
    test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    checkpoint_callback = None
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            checkpoint_callback = callback
            break

    results = {
        "log_dir": logger.log_dir,
        "best_model_path": checkpoint_callback.best_model_path if checkpoint_callback else "",
        "best_val_mean_iou": float(checkpoint_callback.best_model_score)
        if checkpoint_callback and checkpoint_callback.best_model_score is not None
        else None,
        "missing_masks": len(datamodule.missing_masks),
        "class_counts": datamodule.class_counts.tolist(),
        "train_samples": len(datamodule.train_samples),
        "val_samples": len(datamodule.val_samples),
        "test_samples": len(datamodule.test_samples),
        "test_metrics": test_results[0] if test_results else {},
    }
    return results


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config = vars(args)
    results = run_experiment(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
