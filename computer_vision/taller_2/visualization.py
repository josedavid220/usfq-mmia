"""Visualization utilities and Gradio app for segmentation results."""

from __future__ import annotations

import argparse
import sys
import csv
import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml

from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from settings import DATA_DIR, LOGS_DIR

from computer_vision.taller_2.dataset import (
    CLASS_NAMES,
    DEFAULT_COLORS,
    DenseSegmentationDataModule,
)
from computer_vision.taller_2.model import SegmentationLightningModule

load_dotenv()


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor to displayable image.

    Args:
        image: CHW float tensor.

    Returns:
        HWC image in [0, 1].
    """

    mean = torch.tensor(
        [0.485, 0.456, 0.406], dtype=image.dtype, device=image.device
    ).view(3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], dtype=image.dtype, device=image.device
    ).view(3, 1, 1)
    output = image * std + mean
    output = torch.clamp(output, 0.0, 1.0)
    return output.permute(1, 2, 0).cpu().numpy()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
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


def load_sample(track: str, frame_id: int) -> tuple[torch.Tensor, np.ndarray]:
    """Load one image/mask pair from dense_data.

    Args:
        track: Track name.
        frame_id: Frame identifier.

    Returns:
        Tuple with normalized image tensor and ground-truth mask array.

    Raises:
        FileNotFoundError: If image or mask does not exist.
    """

    image_path = DATA_DIR / "dense_data" / track / "frame" / f"frame_{frame_id:04d}.png"
    mask_path = (
        DATA_DIR
        / "dense_data"
        / track
        / "combined"
        / f"mask_combined_{frame_id:04d}.png"
    )
    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError(
            f"Missing image or mask for {track} frame {frame_id:04d}"
        )

    image_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    image = (image - mean) / std
    mask = np.array(Image.open(mask_path), dtype=np.int64)
    return image, mask


class InferenceEngine:
    """Reusable checkpoint loader and predictor for segmentation visualization.

    Args:
        device: Torch device for inference.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.current_ckpt: str | None = None
        self.model: SegmentationLightningModule | None = None
        self.metadata_cache: dict[str, dict[str, Any]] = {}
        self.prediction_cache: dict[tuple[str, str, int], np.ndarray] = {}

    def load(self, checkpoint_path: str) -> None:
        """Load model checkpoint if needed.

        Args:
            checkpoint_path: Path to checkpoint file.
        """

        if self.current_ckpt == checkpoint_path and self.model is not None:
            return
        self.model = SegmentationLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
        )
        self.model.eval()
        self.model.to(self.device)
        self.current_ckpt = checkpoint_path

    def _load_hparams_from_version_dir(self, checkpoint_path: str) -> dict[str, Any]:
        """Load saved run hyperparameters from Lightning version directory.

        Args:
            checkpoint_path: Checkpoint path.

        Returns:
            Hyperparameter dictionary loaded from hparams.yaml.
        """

        version_dir = Path(checkpoint_path).parent.parent
        hparams_path = version_dir / "hparams.yaml"
        if not hparams_path.exists():
            return {}
        with hparams_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        return loaded if isinstance(loaded, dict) else {}

    def _compute_validation_metrics(self, checkpoint_path: str) -> dict[str, float]:
        """Compute validation metrics for one checkpoint.

        Args:
            checkpoint_path: Checkpoint path.

        Returns:
            Dictionary with scalar test metrics.
        """

        if self.model is None:
            return {}

        run_hparams = self._load_hparams_from_version_dir(checkpoint_path)
        data_root = run_hparams.get("data_root", "")
        val_fraction = float(run_hparams.get("val_fraction", 0.2))
        val_minority_fraction = float(run_hparams.get("val_minority_fraction", 0.5))
        batch_size = int(run_hparams.get("batch_size", 8))
        crop_size = int(run_hparams.get("crop_size", 256))
        seed = int(run_hparams.get("seed", 42))

        data_root_path = Path(data_root) if data_root else None
        datamodule = DenseSegmentationDataModule(
            data_root=data_root_path,
            batch_size=batch_size,
            crop_size=crop_size,
            val_fraction=val_fraction,
            val_minority_fraction=val_minority_fraction,
            seed=seed,
        )
        datamodule.prepare_data()

        accelerator = "gpu" if self.device.type == "cuda" else "cpu"
        trainer = L.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        val_results = trainer.validate(
            model=self.model, datamodule=datamodule, verbose=False
        )
        # Lightning validate teardown can move model tensors back to CPU.
        # Restore the model to the configured inference device.
        self.model.to(self.device)
        self.model.eval()
        if not val_results:
            return {}
        output = {}
        for key, value in val_results[0].items():
            if isinstance(value, (int, float)):
                output[key] = float(value)
        return output

    def get_checkpoint_metadata(self, checkpoint_path: str) -> dict[str, Any]:
        """Get cached metadata with hyperparameters and validation metrics.

        Args:
            checkpoint_path: Checkpoint path.

        Returns:
            Metadata dictionary.
        """

        if checkpoint_path in self.metadata_cache:
            return self.metadata_cache[checkpoint_path]

        self.load(checkpoint_path)
        run_hparams = self._load_hparams_from_version_dir(checkpoint_path)
        model_hparams = dict(self.model.hparams) if self.model is not None else {}
        validation_metrics = self._compute_validation_metrics(checkpoint_path)

        metadata = {
            "checkpoint_name": Path(checkpoint_path).name,
            "checkpoint_path": checkpoint_path,
            "run_hparams": run_hparams,
            "model_hparams": model_hparams,
            "validation_metrics": validation_metrics,
        }
        self.metadata_cache[checkpoint_path] = metadata
        return metadata

    def predict(self, image: torch.Tensor) -> np.ndarray:
        """Run model prediction for one image.

        Args:
            image: Normalized CHW image tensor.

        Returns:
            HxW predicted class-index mask.

        Raises:
            RuntimeError: If model is not loaded.
        """

        if self.model is None:
            raise RuntimeError("Model checkpoint has not been loaded.")

        # Defensive guard in case any previous step moved the model.
        self.model.to(self.device)
        self.model.eval()

        # SMP backbones typically require H and W divisible by the encoder output stride.
        # Pad to the next multiple of 32, then crop prediction back to original size.
        def _pad_to_multiple_of_32(chw: torch.Tensor) -> tuple[torch.Tensor, int, int]:
            _, height, width = chw.shape
            target_height = ((height + 31) // 32) * 32
            target_width = ((width + 31) // 32) * 32
            pad_bottom = target_height - height
            pad_right = target_width - width
            if pad_bottom == 0 and pad_right == 0:
                return chw, 0, 0
            padded = F.pad(chw, (0, pad_right, 0, pad_bottom), mode="constant", value=0.0)
            return padded, pad_bottom, pad_right

        with torch.no_grad():
            padded_image, pad_bottom, pad_right = _pad_to_multiple_of_32(image)
            logits = self.model(padded_image.unsqueeze(0).to(self.device))
            pred_tensor = torch.argmax(logits, dim=1).squeeze(0)
            if pad_bottom > 0:
                pred_tensor = pred_tensor[:-pad_bottom, :]
            if pad_right > 0:
                pred_tensor = pred_tensor[:, :-pad_right]
            pred = pred_tensor.cpu().numpy()
        return pred

    def predict_with_cache(
        self,
        checkpoint_path: str,
        track: str,
        frame_id: int,
        image: torch.Tensor,
    ) -> np.ndarray:
        """Predict one frame with a cache keyed by checkpoint/track/frame.

        Args:
            checkpoint_path: Model checkpoint path.
            track: Track name.
            frame_id: Frame id.
            image: Normalized frame tensor.

        Returns:
            Predicted class-index mask.
        """

        cache_key = (checkpoint_path, track, int(frame_id))
        cached = self.prediction_cache.get(cache_key)
        if cached is not None:
            return cached

        self.load(checkpoint_path)
        pred = self.predict(image)
        self.prediction_cache[cache_key] = pred
        return pred

    def warmup_track_predictions(
        self,
        checkpoint_path: str,
        track: str,
        frame_ids: list[int],
    ) -> tuple[int, int]:
        """Precompute predictions for all selected frames in one track.

        Args:
            checkpoint_path: Model checkpoint path.
            track: Track name.
            frame_ids: Frame ids to precompute.

        Returns:
            Tuple with (newly_cached, already_cached).
        """

        self.load(checkpoint_path)
        newly_cached = 0
        already_cached = 0
        for frame_id in frame_ids:
            cache_key = (checkpoint_path, track, int(frame_id))
            if cache_key in self.prediction_cache:
                already_cached += 1
                continue
            image, _ = load_sample(track=track, frame_id=int(frame_id))
            self.prediction_cache[cache_key] = self.predict(image)
            newly_cached += 1
        return newly_cached, already_cached


def list_available_frames(track: str) -> list[int]:
    """List valid frame ids for one track.

    Args:
        track: Track name.

    Returns:
        Sorted frame ids with both image and combined mask available.
    """

    frame_dir = DATA_DIR / "dense_data" / track / "frame"
    combined_dir = DATA_DIR / "dense_data" / track / "combined"
    if not frame_dir.exists() or not combined_dir.exists():
        return []

    available: list[int] = []
    for frame_id in range(250):
        image_path = frame_dir / f"frame_{frame_id:04d}.png"
        mask_path = combined_dir / f"mask_combined_{frame_id:04d}.png"
        if image_path.exists() and mask_path.exists():
            available.append(frame_id)
    return available


def render_comparison(
    image: torch.Tensor,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Render frame, masks, and overlay side-by-side.

    Args:
        image: Normalized CHW image tensor.
        ground_truth: HxW class-index mask.
        prediction: HxW class-index mask.
    Returns:
        Rendered RGB image array.
    """

    image_rgb = denormalize_image(image)
    gt_color = colorize_mask(ground_truth)
    pred_color = colorize_mask(prediction)
    overlay = np.clip(0.65 * image_rgb + 0.35 * (pred_color / 255.0), 0.0, 1.0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Frame")
    axes[1].imshow(gt_color)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_color)
    axes[2].set_title("Prediction")
    axes[3].imshow(overlay)
    axes[3].set_title("Prediction Overlay")

    for axis in axes:
        axis.axis("off")
    fig.tight_layout()

    fig.canvas.draw()
    output = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[attr-defined]
    output = output.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return output


def render_video_frame(
    image: torch.Tensor,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Render compact side-by-side frame for exported track playback.

    Args:
        image: Normalized CHW image tensor.
        ground_truth: HxW class-index mask.
        prediction: HxW class-index mask.

    Returns:
        HxWx3 uint8 frame ready for video writing.
    """

    image_rgb = (denormalize_image(image) * 255.0).astype(np.uint8)
    gt_color = colorize_mask(ground_truth)
    pred_color = colorize_mask(prediction)
    overlay = np.clip(0.65 * image_rgb + 0.35 * pred_color, 0.0, 255.0).astype(np.uint8)
    return np.concatenate([image_rgb, gt_color, pred_color, overlay], axis=1)


def gif_preview_html(gif_path: Path) -> str:
    """Build inline animated preview HTML for a generated GIF.

    Args:
        gif_path: Path to GIF file.

    Returns:
        HTML snippet containing an embedded GIF preview.
    """

    encoded = base64.b64encode(gif_path.read_bytes()).decode("ascii")
    return (
        "<div style='padding:8px;border:1px solid #ddd;border-radius:8px;'>"
        "<div style='font-weight:600;margin-bottom:8px;'>Generated GIF Preview</div>"
        f"<img src='data:image/gif;base64,{encoded}' style='max-width:100%;height:auto;border-radius:6px;'/>"
        f"<div style='margin-top:8px;font-size:0.9rem;color:#444;'>Saved at: {gif_path}</div>"
        "</div>"
    )


def list_checkpoints(log_root: Path) -> list[str]:
    """List available Lightning checkpoints under logs directory.

    Args:
        log_root: Root logs directory.

    Returns:
        Sorted checkpoint path list.
    """

    checkpoints = sorted(log_root.glob("**/checkpoints/*.ckpt"))
    return [str(path) for path in checkpoints]


def checkpoint_choices(log_root: Path) -> list[tuple[str, str]]:
    """Create human-readable dropdown choices for checkpoints.

    Args:
        log_root: Logs root directory.

    Returns:
        List of tuples (label, value) for gradio dropdown.
    """

    choices: list[tuple[str, str]] = []
    for checkpoint in list_checkpoints(log_root):
        path = Path(checkpoint)
        version_dir = path.parent.parent
        exp_name = version_dir.parent.name
        label = f"{exp_name}/{version_dir.name}/{path.name}"
        choices.append((label, checkpoint))
    return choices


def legend_html() -> str:
    """Build external legend html for class colors.

    Returns:
        HTML string with colored labels.
    """

    chips = []
    for class_id, rgb in DEFAULT_COLORS.items():
        name = (
            CLASS_NAMES[class_id]
            if class_id < len(CLASS_NAMES)
            else f"class_{class_id}"
        )
        chip = (
            "<span style='display:inline-flex;align-items:center;margin-right:12px;margin-bottom:8px;'>"
            f"<span style='width:16px;height:16px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});"
            "border:1px solid #444;display:inline-block;margin-right:6px;'></span>"
            f"{class_id}: {name}</span>"
        )
        chips.append(chip)
    return "<div style='padding:8px 0;'>" + "".join(chips) + "</div>"


def experiment_summary_markdown(
    metadata: dict[str, Any], track: str, frame_id: int, device: str
) -> str:
    """Format compact experiment summary for selected checkpoint/frame.

    Args:
        metadata: Checkpoint metadata.
        track: Selected track.
        frame_id: Selected frame id.
        device: Inference device.

    Returns:
        Markdown string focused on experiment identity and key hyperparameters.
    """

    run_hparams = metadata.get("run_hparams", {})
    model_hparams = metadata.get("model_hparams", {})
    architecture = model_hparams.get(
        "architecture", run_hparams.get("architecture", "unknown")
    )
    encoder_name = model_hparams.get(
        "encoder_name", run_hparams.get("encoder_name", "unknown")
    )
    loss_name = model_hparams.get("loss_name", run_hparams.get("loss_name", "unknown"))

    lines = [
        "## Experiment Summary",
        f"- Checkpoint: {metadata.get('checkpoint_name', 'unknown')}",
        f"- Backbone: {encoder_name}",
        f"- Architecture: {architecture}",
        f"- Loss: {loss_name}",
        f"- LR: {model_hparams.get('learning_rate', run_hparams.get('learning_rate', 'unknown'))}",
        f"- Batch/Crop: {run_hparams.get('batch_size', 'unknown')} / {run_hparams.get('crop_size', 'unknown')}",
        f"- Precision: {run_hparams.get('precision', 'unknown')}",
        f"- Sample: {track} / frame_{int(frame_id):04d}",
        f"- Device: {device}",
    ]

    return "\n".join(lines)


def aggregate_metrics_markdown(metadata: dict[str, Any]) -> str:
    """Format aggregate validation metrics.

    Args:
        metadata: Checkpoint metadata dictionary.

    Returns:
        Markdown string with aggregate validation metrics.
    """

    metrics = metadata.get("validation_metrics", {})
    if not metrics:
        return "## Aggregate Validation Metrics\n- Validation metrics not available for this checkpoint."

    lines = ["## Aggregate Validation Metrics"]
    aggregate_keys = ["val_loss", "val_mean_iou", "val_mean_acc", "val_pixel_acc"]
    for key in aggregate_keys:
        if key in metrics:
            lines.append(f"- {key}: {float(metrics[key]):.4f}")

    return "\n".join(lines)


def per_class_metrics_markdown(metadata: dict[str, Any]) -> str:
    """Format per-class validation metrics table.

    Args:
        metadata: Checkpoint metadata dictionary.

    Returns:
        Markdown string with per-class IoU and accuracy.
    """

    metrics = metadata.get("validation_metrics", {})
    if not metrics:
        return "## Per-Class Validation Metrics\n- Validation metrics not available for this checkpoint."

    lines = [
        "## Per-Class Validation Metrics",
        "| Class | IoU | Accuracy |",
        "|---|---:|---:|",
    ]

    for class_name in CLASS_NAMES:
        iou_key = f"val_iou_{class_name}"
        acc_key = f"val_acc_{class_name}"
        iou_value = metrics.get(iou_key)
        acc_value = metrics.get(acc_key)
        if iou_value is None and acc_value is None:
            continue
        iou_str = f"{float(iou_value):.4f}" if iou_value is not None else "-"
        acc_str = f"{float(acc_value):.4f}" if acc_value is not None else "-"
        lines.append(f"| {class_name} | {iou_str} | {acc_str} |")

    return "\n".join(lines)


def leaderboard_choices(log_root: Path) -> list[tuple[str, str]]:
    """List available leaderboard reports from grid runs.

    Args:
        log_root: Root logs directory.

    Returns:
        Tuple list where label is user-friendly and value is CSV path.
    """

    choices: list[tuple[str, str]] = []
    grid_root = log_root / "grid_runs"
    if not grid_root.exists():
        return choices
    candidates = sorted(grid_root.glob("*/leaderboard.csv"), reverse=True)
    for path in candidates:
        run_name = path.parent.name
        choices.append((run_name, str(path)))
    return choices


def load_leaderboard(path: str, top_k: int = 20) -> tuple[list[list[Any]], str]:
    """Load leaderboard csv rows for gradio table.

    Args:
        path: Leaderboard CSV path.
        top_k: Maximum rows to return.

    Returns:
        Table rows and markdown summary.
    """

    if not path:
        return [], "No leaderboard file selected."
    leaderboard_path = Path(path)
    if not leaderboard_path.exists():
        return [], f"Leaderboard not found: {leaderboard_path}"

    rows: list[dict[str, Any]] = []
    with leaderboard_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        return [], "Leaderboard is empty."

    sliced = rows[: max(1, top_k)]
    table_rows: list[list[Any]] = []
    for row in sliced:
        table_rows.append(
            [
                row.get("rank", ""),
                row.get("trial_name", ""),
                row.get("architecture", ""),
                row.get("encoder_name", ""),
                row.get("loss_name", ""),
                row.get("best_val_mean_iou", ""),
                row.get("val_mean_acc", ""),
                row.get("val_pixel_acc", ""),
                row.get("best_model_path", ""),
            ]
        )

    best = rows[0]
    summary = "\n".join(
        [
            "## Leaderboard Overview",
            f"- Source: {leaderboard_path}",
            f"- Total successful trials: {len(rows)}",
            f"- Best backbone: {best.get('encoder_name', 'unknown')}",
            f"- Best loss: {best.get('loss_name', 'unknown')}",
            f"- Best validation mIoU: {best.get('best_val_mean_iou', 'unknown')}",
            f"- Best validation mean accuracy: {best.get('val_mean_acc', 'unknown')}",
        ]
    )
    return table_rows, summary


def build_gradio_app() -> gr.Blocks:
    """Create the interactive Gradio app for result inspection.

    Returns:
        Configured Gradio Blocks instance.
    """

    tracks = sorted(
        [path.name for path in (DATA_DIR / "dense_data").iterdir() if path.is_dir()]
    )
    choices = checkpoint_choices(LOGS_DIR)
    if not choices:
        choices = [("No checkpoints found", "")]
    report_choices = leaderboard_choices(LOGS_DIR)
    if not report_choices:
        report_choices = [("No leaderboard found", "")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = InferenceEngine(device=device)

    def _predict(
        ckpt_path: str, track: str, frame_id: int
    ) -> tuple[np.ndarray, str, str, str]:
        """Infer and render one selection from UI controls."""

        if not ckpt_path:
            blank = np.zeros((450, 1450, 3), dtype=np.uint8)
            msg = "## No checkpoint found in logs yet."
            return blank, msg, msg, msg

        image, mask = load_sample(track=track, frame_id=int(frame_id))
        metadata = engine.get_checkpoint_metadata(ckpt_path)
        pred = engine.predict_with_cache(
            checkpoint_path=ckpt_path,
            track=track,
            frame_id=int(frame_id),
            image=image,
        )
        render = render_comparison(
            image=image,
            ground_truth=mask,
            prediction=pred,
        )
        experiment_text = experiment_summary_markdown(
            metadata=metadata,
            track=track,
            frame_id=int(frame_id),
            device=device,
        )
        aggregate_text = aggregate_metrics_markdown(metadata=metadata)
        per_class_text = per_class_metrics_markdown(metadata=metadata)
        return render, experiment_text, aggregate_text, per_class_text

    def _refresh_checkpoints() -> gr.Dropdown:
        """Refresh checkpoint dropdown choices from logs."""

        refreshed = checkpoint_choices(LOGS_DIR)
        if not refreshed:
            refreshed = [("No checkpoints found", "")]
        return gr.Dropdown(choices=refreshed, value=refreshed[0][1])

    def _refresh_leaderboards() -> gr.Dropdown:
        """Refresh leaderboard choices from grid outputs."""

        refreshed = leaderboard_choices(LOGS_DIR)
        if not refreshed:
            refreshed = [("No leaderboard found", "")]
        return gr.Dropdown(choices=refreshed, value=refreshed[0][1])

    def _load_report(path: str, top_k: int) -> tuple[list[list[Any]], str]:
        """Load leaderboard report data for UI widgets."""

        return load_leaderboard(path=path, top_k=int(top_k))

    def _select_model_from_report(
        report_rows: Any,
        evt: gr.SelectData,
    ) -> tuple[gr.Dropdown, str, gr.Tabs]:
        """Use selected leaderboard row to populate inference checkpoint.

        Args:
            report_rows: Rendered leaderboard table rows.
            evt: Selection event payload from dataframe.

        Returns:
            Updated checkpoint dropdown, status markdown, and selected tab.
        """

        normalized_rows: list[list[Any]]
        if report_rows is None:
            normalized_rows = []
        elif hasattr(report_rows, "empty") and hasattr(report_rows, "values"):
            if bool(report_rows.empty):
                normalized_rows = []
            else:
                normalized_rows = report_rows.values.tolist()
        elif isinstance(report_rows, list):
            normalized_rows = report_rows
        else:
            try:
                normalized_rows = list(report_rows)
            except TypeError:
                normalized_rows = []

        if len(normalized_rows) == 0:
            return gr.Dropdown(), "No report rows loaded yet.", gr.Tabs()

        row_index_raw = evt.index
        if isinstance(row_index_raw, tuple):
            row_index = int(row_index_raw[0])
        elif isinstance(row_index_raw, list):
            row_index = int(row_index_raw[0])
        else:
            row_index = int(row_index_raw)
        if row_index < 0 or row_index >= len(normalized_rows):
            return gr.Dropdown(), "Selected row is out of bounds.", gr.Tabs()

        selected_row = normalized_rows[row_index]
        if len(selected_row) < 9:
            return (
                gr.Dropdown(),
                "Selected row is missing checkpoint information.",
                gr.Tabs(),
            )

        checkpoint_path = str(selected_row[8])
        if not checkpoint_path:
            return gr.Dropdown(), "Selected row has an empty checkpoint path.", gr.Tabs()

        if not Path(checkpoint_path).exists():
            return (
                gr.Dropdown(),
                f"Checkpoint not found on disk: {checkpoint_path}",
                gr.Tabs(),
            )

        return (
            gr.Dropdown(value=checkpoint_path),
            f"Selected model loaded in Inference tab: {selected_row[1]}",
            gr.Tabs(selected="inference_tab"),
        )

    def _tensorboard_iframe(url: str) -> str:
        """Build embeddable iframe markup for TensorBoard.

        Args:
            url: TensorBoard URL.

        Returns:
            HTML string containing iframe or usage message.
        """

        normalized = (url or "").strip()
        if not normalized:
            return (
                "<div style='padding:12px;border:1px solid #ccc;border-radius:8px;'>"
                "Provide a TensorBoard URL (for example http://localhost:6006)."
                "</div>"
            )
        if not normalized.startswith(("http://", "https://")):
            normalized = f"http://{normalized}"
        return (
            "<iframe "
            f"src='{normalized}' "
            "style='width:100%;height:760px;border:1px solid #ddd;border-radius:8px;' "
            "referrerpolicy='no-referrer' "
            "loading='lazy'>"
            "</iframe>"
        )

    def _on_track_change(track: str) -> tuple[gr.Slider, list[int], str]:
        """Update frame controls for selected track.

        Args:
            track: Selected track.

        Returns:
            Updated slider, frame list state, and status text.
        """

        frame_ids = list_available_frames(track)
        if not frame_ids:
            return gr.Slider(value=0), [], f"No valid frames found for track: {track}"
        return (
            gr.Slider(value=frame_ids[0]),
            frame_ids,
            f"Loaded {len(frame_ids)} valid frames for {track}.",
        )

    def _step_frame(current_frame: int, frame_ids: list[int], direction: int) -> int:
        """Move one step over available frame ids.

        Args:
            current_frame: Current frame value.
            frame_ids: Available frame ids.
            direction: Step direction (-1 or +1).

        Returns:
            Next frame id.
        """

        if not frame_ids:
            return int(current_frame)
        current = int(current_frame)
        if current not in frame_ids:
            return frame_ids[0]
        index = frame_ids.index(current)
        return frame_ids[(index + direction) % len(frame_ids)]

    def _warmup_track_cache(ckpt_path: str, track: str, frame_ids: list[int]) -> str:
        """Precompute and cache predictions for current track.

        Args:
            ckpt_path: Checkpoint path.
            track: Selected track.
            frame_ids: Frames to precompute.

        Returns:
            Status summary string.
        """

        if not ckpt_path:
            return "Select a checkpoint before caching predictions."
        if not frame_ids:
            return "No valid frames to cache for this track."
        newly_cached, already_cached = engine.warmup_track_predictions(
            checkpoint_path=ckpt_path,
            track=track,
            frame_ids=frame_ids,
        )
        return (
            f"Caching done for {track}: new={newly_cached}, already_cached={already_cached}, "
            f"total_frames={len(frame_ids)}"
        )

    def _generate_track_video(
        ckpt_path: str,
        track: str,
        frame_ids: list[int],
        fps: int,
        stride: int,
        max_frames: int,
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ) -> tuple[Any, str, str]:
        """Generate a GIF playback file from a full track.

        Args:
            ckpt_path: Selected checkpoint path.
            track: Selected track name.
            frame_ids: Available frame ids for this track.
            fps: Output frame rate.
            stride: Keep one every N frames.
            max_frames: Optional max number of frames (0 means all).

        Returns:
            GIF file update, animated preview HTML, and status message.
        """

        if not ckpt_path:
            return (
                gr.update(value=None, visible=False),
                "",
                "Select a checkpoint before generating playback.",
            )
        if not frame_ids:
            return (
                gr.update(value=None, visible=False),
                "",
                "No valid frames available for this track.",
            )

        effective_stride = max(1, int(stride))
        selected_ids = frame_ids[::effective_stride]
        if max_frames > 0:
            selected_ids = selected_ids[: int(max_frames)]
        if not selected_ids:
            return (
                gr.update(value=None, visible=False),
                "",
                "No frames selected with current stride/max settings.",
            )

        progress(0.0, desc="Preparing frames for GIF export...")
        frames: list[np.ndarray] = []
        total = len(selected_ids)
        for idx, frame_id in enumerate(selected_ids, start=1):
            image, mask = load_sample(track=track, frame_id=int(frame_id))
            pred = engine.predict_with_cache(
                checkpoint_path=ckpt_path,
                track=track,
                frame_id=int(frame_id),
                image=image,
            )
            frames.append(
                render_video_frame(image=image, ground_truth=mask, prediction=pred)
            )
            progress(idx / total, desc=f"Rendering GIF frames... ({idx}/{total})")

        output_dir = LOGS_DIR / "visualizer_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_name = f"{Path(ckpt_path).stem}-{track}-{timestamp}"
        fps_value = max(1, int(fps))

        gif_path = output_dir / f"{base_name}.gif"
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(20, int(1000 / fps_value)),
            loop=0,
        )
        return (
            gr.update(value=str(gif_path), visible=True),
            gif_preview_html(gif_path),
            f"Track GIF generated: {gif_path} ({len(frames)} frames at {fps_value} FPS).",
        )

    def _prepare_gif_export() -> tuple[Any, str, str]:
        """Show immediate loading feedback before GIF generation starts.

        Returns:
            Cleared GIF file/preview and loading message.
        """

        return (
            gr.update(value=None, visible=False),
            "",
            "Generating GIF, please wait...",
        )

    app_css = """
    .big-frame-btn button {
        min-height: 56px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 0.8rem 1rem !important;
    }
    """

    with gr.Blocks(title="Taller 2 Segmentation Visualizer") as app:
        gr.HTML(f"<style>{app_css}</style>")
        gr.Markdown("# Taller 2 Semantic Segmentation Visualizer")
        gr.Markdown(
            "Compare predictions and inspect experiment reports from grid runs. "
            "Use the tabs to switch between sample inference and leaderboard analysis."
        )

        with gr.Tabs(selected="inference_tab") as main_tabs:
            with gr.Tab("Inference", id="inference_tab"):
                gr.HTML(value=legend_html(), label="Class Legend")
                initial_frames = list_available_frames(tracks[0]) if tracks else []
                initial_frame = initial_frames[0] if initial_frames else 0
                frame_ids_state = gr.State(initial_frames)

                with gr.Row():
                    ckpt_input = gr.Dropdown(
                        choices=choices,
                        value=choices[0][1],
                        label="Checkpoint",
                    )
                    track_input = gr.Dropdown(
                        choices=tracks, value=tracks[0], label="Track"
                    )
                    frame_input = gr.Slider(
                        minimum=0,
                        maximum=249,
                        value=initial_frame,
                        step=1,
                        label="Frame ID",
                    )

                with gr.Row():
                    refresh_button = gr.Button("Refresh Checkpoints")
                    run_button = gr.Button("Run Inference", variant="primary")

                with gr.Row():
                    prev_button = gr.Button(
                        "◀ Prev Frame",
                        variant="secondary",
                        elem_classes=["big-frame-btn"],
                    )
                    next_button = gr.Button(
                        "Next Frame ▶",
                        variant="secondary",
                        elem_classes=["big-frame-btn"],
                    )
                    cache_track_button = gr.Button("Cache Track Predictions")
                    export_video_button = gr.Button(
                        "Generate Track GIF", variant="primary"
                    )

                with gr.Row():
                    export_fps_input = gr.Slider(
                        minimum=2, maximum=30, value=8, step=1, label="GIF FPS"
                    )
                    export_stride_input = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=1,
                        step=1,
                        label="Frame Stride (1 = all frames)",
                    )
                    export_max_frames_input = gr.Slider(
                        minimum=0,
                        maximum=250,
                        value=0,
                        step=1,
                        label="Max Frames (0 = all)",
                    )

                playback_status = gr.Markdown(
                    value="Manual mode. Use Prev/Next and slider for navigation."
                )
                export_status = gr.Markdown(
                    value="Generate a track GIF for smooth qualitative review."
                )

                output_image = gr.Image(label="Visualization", type="numpy")
                output_gif = gr.Image(
                    label="Generated Track GIF", type="filepath", visible=False
                )
                output_gif_preview = gr.HTML(label="GIF Playback")

                with gr.Row():
                    experiment_text = gr.Markdown(label="Experiment")
                    aggregate_metrics_text = gr.Markdown(
                        label="Aggregate Validation Metrics"
                    )
                    per_class_metrics_text = gr.Markdown(
                        label="Per-Class Validation Metrics"
                    )

                refresh_button.click(
                    fn=_refresh_checkpoints,
                    outputs=[ckpt_input],
                )

                run_button.click(
                    fn=_predict,
                    inputs=[ckpt_input, track_input, frame_input],
                    outputs=[
                        output_image,
                        experiment_text,
                        aggregate_metrics_text,
                        per_class_metrics_text,
                    ],
                )

                frame_input.change(
                    fn=_predict,
                    inputs=[ckpt_input, track_input, frame_input],
                    outputs=[
                        output_image,
                        experiment_text,
                        aggregate_metrics_text,
                        per_class_metrics_text,
                    ],
                )

                track_input.change(
                    fn=_on_track_change,
                    inputs=[track_input],
                    outputs=[frame_input, frame_ids_state, playback_status],
                ).then(
                    fn=_predict,
                    inputs=[ckpt_input, track_input, frame_input],
                    outputs=[
                        output_image,
                        experiment_text,
                        aggregate_metrics_text,
                        per_class_metrics_text,
                    ],
                )

                prev_button.click(
                    fn=lambda frame_id, frame_ids: _step_frame(frame_id, frame_ids, -1),
                    inputs=[frame_input, frame_ids_state],
                    outputs=[frame_input],
                ).then(
                    fn=_predict,
                    inputs=[ckpt_input, track_input, frame_input],
                    outputs=[
                        output_image,
                        experiment_text,
                        aggregate_metrics_text,
                        per_class_metrics_text,
                    ],
                )

                next_button.click(
                    fn=lambda frame_id, frame_ids: _step_frame(frame_id, frame_ids, 1),
                    inputs=[frame_input, frame_ids_state],
                    outputs=[frame_input],
                ).then(
                    fn=_predict,
                    inputs=[ckpt_input, track_input, frame_input],
                    outputs=[
                        output_image,
                        experiment_text,
                        aggregate_metrics_text,
                        per_class_metrics_text,
                    ],
                )

                cache_track_button.click(
                    fn=_warmup_track_cache,
                    inputs=[ckpt_input, track_input, frame_ids_state],
                    outputs=[playback_status],
                )

                export_video_button.click(
                    fn=_prepare_gif_export,
                    outputs=[output_gif, output_gif_preview, export_status],
                ).then(
                    fn=_generate_track_video,
                    inputs=[
                        ckpt_input,
                        track_input,
                        frame_ids_state,
                        export_fps_input,
                        export_stride_input,
                        export_max_frames_input,
                    ],
                    outputs=[output_gif, output_gif_preview, export_status],
                    show_progress="full",
                )

            with gr.Tab("Leaderboard Report", id="leaderboard_tab"):
                gr.Markdown("Browse ranked results generated by the grid runner.")
                with gr.Row():
                    report_input = gr.Dropdown(
                        choices=report_choices,
                        value=report_choices[0][1],
                        label="Grid Run Report",
                    )
                    topk_input = gr.Slider(
                        minimum=5, maximum=100, value=20, step=1, label="Top K rows"
                    )

                with gr.Row():
                    refresh_report_button = gr.Button("Refresh Reports")
                    load_report_button = gr.Button("Load Report", variant="primary")

                report_table = gr.Dataframe(
                    headers=[
                        "rank",
                        "trial_name",
                        "architecture",
                        "encoder",
                        "loss",
                        "val_miou",
                        "val_mean_acc",
                        "val_pixel_acc",
                        "checkpoint",
                    ],
                    datatype=[
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                        "str",
                    ],
                    row_count=20,
                    column_count=9,
                    wrap=True,
                    interactive=False,
                    label="Leaderboard",
                )
                report_summary = gr.Markdown(label="Report Summary")
                report_status = gr.Markdown(
                    value="Click any leaderboard row to load its checkpoint in the Inference tab.",
                    label="Selection Status",
                )

                refresh_report_button.click(
                    fn=_refresh_leaderboards,
                    outputs=[report_input],
                )

                load_report_button.click(
                    fn=_load_report,
                    inputs=[report_input, topk_input],
                    outputs=[report_table, report_summary],
                )

                report_table.select(
                    fn=_select_model_from_report,
                    inputs=[report_table],
                    outputs=[ckpt_input, report_status, main_tabs],
                )

            with gr.Tab("TensorBoard"):
                gr.Markdown(
                    "View TensorBoard directly inside the app. "
                    "Start TensorBoard first with `make tensorboard`, then load the URL below."
                )
                with gr.Row():
                    tb_url = gr.Textbox(
                        value=os.getenv("TENSORBOARD_URL"), label="TensorBoard URL"
                    )
                    tb_load = gr.Button("Load TensorBoard", variant="primary")

                tb_frame = gr.HTML(
                    value=_tensorboard_iframe(
                        os.getenv("TENSORBOARD_URL", "http://localhost:6006")
                    ),
                    label="TensorBoard Viewer",
                )
                tb_load.click(
                    fn=_tensorboard_iframe, inputs=[tb_url], outputs=[tb_frame]
                )

    return app


def main() -> None:
    """Launch Gradio visualizer."""

    parser = argparse.ArgumentParser(description="Launch segmentation visualizer.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable public Gradio sharing link.",
    )
    args = parser.parse_args()

    app = build_gradio_app()
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
