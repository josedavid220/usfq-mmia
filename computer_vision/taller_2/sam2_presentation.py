"""Presentation-optimized Gradio app for SAM2 transfer segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import lightning as L
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from computer_vision.taller_2.dataset import CLASS_NAMES, DenseSegmentationDataModule
    from computer_vision.taller_2.sam2_model import SAM2TransferLightningModule
    from computer_vision.taller_2.visualization import (
        list_available_frames,
        load_sample,
        render_comparison,
        render_video_frame,
    )
except ModuleNotFoundError:
    # Local fallback for direct execution from this directory.
    from dataset import CLASS_NAMES, DenseSegmentationDataModule
    from sam2_model import SAM2TransferLightningModule
    from visualization import (
        list_available_frames,
        load_sample,
        render_comparison,
        render_video_frame,
    )
from settings import DATA_DIR, LOGS_DIR


def list_checkpoints() -> list[str]:
    """Return sorted SAM2 checkpoint paths."""

    checkpoints = sorted(LOGS_DIR.glob("**/checkpoints/*.ckpt"))
    return [str(path) for path in checkpoints if "sam2" in str(path).lower()]


def checkpoint_choices() -> list[tuple[str, str]]:
    """Build dropdown choices from available checkpoints."""

    choices: list[tuple[str, str]] = []
    for ckpt in list_checkpoints():
        path = Path(ckpt)
        version_dir = path.parent.parent
        label = f"{version_dir.parent.name}/{version_dir.name}/{path.name}"
        choices.append((label, ckpt))
    return choices


class PresentationEngine:
    """Small inference and metric cache for SAM2 presentation app."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.current_ckpt: str | None = None
        self.model: SAM2TransferLightningModule | None = None
        self.pred_cache: dict[tuple[str, str, int], np.ndarray] = {}
        self.metric_cache: dict[str, dict[str, float]] = {}

    def load(self, checkpoint_path: str) -> None:
        """Load SAM2 transfer checkpoint if not already active."""

        if self.current_ckpt == checkpoint_path and self.model is not None:
            return
        self.model = SAM2TransferLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
            class_weights=None,
        )
        self.model.to(self.device)
        self.model.eval()
        self.current_ckpt = checkpoint_path

    def predict(self, checkpoint_path: str, track: str, frame_id: int) -> np.ndarray:
        """Predict one frame and cache result by checkpoint/track/frame."""

        key = (checkpoint_path, track, int(frame_id))
        cached = self.pred_cache.get(key)
        if cached is not None:
            return cached

        self.load(checkpoint_path)
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        image, _ = load_sample(track=track, frame_id=int(frame_id))
        with torch.no_grad():
            logits = self.model(image.unsqueeze(0).to(self.device))
            logits = torch.nn.functional.interpolate(
                logits,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        self.pred_cache[key] = pred
        return pred

    def validation_metrics(self, checkpoint_path: str) -> dict[str, float]:
        """Compute and cache validation metrics for one checkpoint."""

        cached = self.metric_cache.get(checkpoint_path)
        if cached is not None:
            return cached

        self.load(checkpoint_path)
        if self.model is None:
            return {}

        datamodule = DenseSegmentationDataModule()
        datamodule.prepare_data()

        trainer = L.Trainer(
            accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        val_results = trainer.validate(self.model, datamodule=datamodule, verbose=False)
        output: dict[str, float] = {}
        if val_results:
            for key, value in val_results[0].items():
                if isinstance(value, (int, float)):
                    output[key] = float(value)
        self.metric_cache[checkpoint_path] = output
        return output


def metric_markdown(metrics: dict[str, float]) -> str:
    """Build compact markdown summary for key metrics."""

    if not metrics:
        return "## Metricas\n- No hay metricas disponibles para este checkpoint."

    lines = ["## Metricas"]
    for key in ["val_loss", "val_mean_iou", "val_mean_acc", "val_pixel_acc"]:
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]:.4f}")

    lines.append("## IoU por clase")
    for class_name in CLASS_NAMES:
        metric_key = f"val_iou_{class_name}"
        if metric_key in metrics:
            lines.append(f"- {class_name}: {metrics[metric_key]:.4f}")
    return "\n".join(lines)


def build_app() -> tuple[gr.Blocks, str]:
    """Create presentation-focused Gradio interface."""

    track_options = sorted([p.name for p in (DATA_DIR / "dense_data").iterdir() if p.is_dir()])
    ckpt_options = checkpoint_choices()
    if not ckpt_options:
        ckpt_options = [("No checkpoints found", "")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = PresentationEngine(device=device)

    def refresh_ckpts() -> gr.Dropdown:
        refreshed = checkpoint_choices()
        if not refreshed:
            refreshed = [("No checkpoints found", "")]
        return gr.Dropdown(choices=refreshed, value=refreshed[0][1])

    def on_track_change(track: str) -> tuple[gr.Slider, list[int]]:
        frame_ids = list_available_frames(track)
        if not frame_ids:
            return gr.Slider(minimum=0, maximum=0, step=1, value=0), []
        return (
            gr.Slider(
                minimum=frame_ids[0],
                maximum=frame_ids[-1],
                step=1,
                value=frame_ids[0],
            ),
            frame_ids,
        )

    def step_frame(current: int, frame_ids: list[int], direction: int) -> int:
        if not frame_ids:
            return int(current)
        if current not in frame_ids:
            return frame_ids[0]
        idx = frame_ids.index(current)
        next_idx = max(0, min(len(frame_ids) - 1, idx + direction))
        return frame_ids[next_idx]

    def run_inference(ckpt: str, track: str, frame_id: int) -> tuple[np.ndarray, str]:
        if not ckpt:
            return np.zeros((256, 256, 3), dtype=np.uint8), "## Estado\n- Selecciona un checkpoint."
        image, mask = load_sample(track=track, frame_id=int(frame_id))
        pred = engine.predict(ckpt, track, int(frame_id))
        panel = render_comparison(image=image, ground_truth=mask, prediction=pred)
        summary = metric_markdown(engine.validation_metrics(ckpt))
        return panel, summary

    def export_track_gif(ckpt: str, track: str, frame_ids: list[int]) -> tuple[Any, str]:
        if not ckpt:
            return None, "Selecciona un checkpoint primero."
        if not frame_ids:
            return None, "No hay frames disponibles para este track."

        selected = frame_ids[:: max(1, len(frame_ids) // 25)][:25]
        rendered: list[Image.Image] = []
        for frame_id in selected:
            image, mask = load_sample(track=track, frame_id=int(frame_id))
            pred = engine.predict(ckpt, track, int(frame_id))
            frame = render_video_frame(image=image, ground_truth=mask, prediction=pred)
            rendered.append(Image.fromarray(frame))

        out_dir = LOGS_DIR / "sam2_presentation" / "gifs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{track}.gif"
        rendered[0].save(
            out_path,
            save_all=True,
            append_images=rendered[1:],
            duration=220,
            loop=0,
        )
        return str(out_path), f"GIF generado: {out_path}"

    app_css = """
    .big-btn button {
        min-height: 56px !important;
        font-size: 1.12rem !important;
        font-weight: 700 !important;
    }
    """

    with gr.Blocks(title="SAM2 Homework Presentation") as app:
        gr.Markdown("# SAM2 Transfer Segmentation - Presentacion")

        frame_ids_state = gr.State([])

        with gr.Tabs():
            with gr.Tab("Inference"):
                with gr.Row():
                    ckpt_dd = gr.Dropdown(
                        label="Checkpoint",
                        choices=ckpt_options,
                        value=ckpt_options[0][1],
                    )
                    refresh_btn = gr.Button("Refresh Checkpoints")
                with gr.Row():
                    track_dd = gr.Dropdown(label="Track", choices=track_options, value=track_options[0])
                    frame_slider = gr.Slider(label="Frame", minimum=0, maximum=249, step=1, value=0)
                with gr.Row():
                    prev_btn = gr.Button("Prev Frame", elem_classes=["big-btn"])
                    next_btn = gr.Button("Next Frame", elem_classes=["big-btn"])
                infer_btn = gr.Button("Run Inference", variant="primary")

                preview_image = gr.Image(label="Frame | GT | Prediction | Overlay")
                metric_md = gr.Markdown()

                with gr.Row():
                    gif_btn = gr.Button("Generate Track GIF")
                    gif_file = gr.File(label="GIF")
                gif_status = gr.Markdown()

            with gr.Tab("Highlights"):
                gr.Markdown(
                    "\n".join(
                        [
                            "# Highlights",
                            "- Reutilizamos el DataModule del taller para mantener reproducibilidad.",
                            "- Usamos weighted cross entropy para mitigar desbalance de clases.",
                            "- Backbone SAM2 congelado y decoder pequeno para entrenamiento simple.",
                            "- Seguimiento en TensorBoard con metricas agregadas, por clase y ejemplos fijos por epoca.",
                            "- Visualizacion final con grid aleatorio y GIFs por track para presentacion.",
                        ]
                    )
                )

        refresh_btn.click(refresh_ckpts, outputs=[ckpt_dd])
        track_dd.change(on_track_change, inputs=[track_dd], outputs=[frame_slider, frame_ids_state])
        prev_btn.click(step_frame, inputs=[frame_slider, frame_ids_state, gr.State(-1)], outputs=[frame_slider])
        next_btn.click(step_frame, inputs=[frame_slider, frame_ids_state, gr.State(1)], outputs=[frame_slider])

        infer_btn.click(
            run_inference,
            inputs=[ckpt_dd, track_dd, frame_slider],
            outputs=[preview_image, metric_md],
        )
        frame_slider.release(
            run_inference,
            inputs=[ckpt_dd, track_dd, frame_slider],
            outputs=[preview_image, metric_md],
        )
        gif_btn.click(
            export_track_gif,
            inputs=[ckpt_dd, track_dd, frame_ids_state],
            outputs=[gif_file, gif_status],
        )

    return app, app_css


def main() -> None:
    """Launch the SAM2 presentation app."""

    parser = argparse.ArgumentParser(description="Launch SAM2 presentation app.")
    parser.add_argument("--share", action="store_true", help="Enable public Gradio sharing link.")
    args = parser.parse_args()

    app, app_css = build_app()
    app.launch(share=args.share, css=app_css)


if __name__ == "__main__":
    main()
