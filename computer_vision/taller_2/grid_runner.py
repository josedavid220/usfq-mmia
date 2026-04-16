"""Grid runner for segmentation experiments on local or high-end GPU setups."""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrialConfig:
    """Container with per-trial hyperparameters.

    Args:
        architecture: SMP architecture.
        encoder_name: Backbone encoder.
        loss_name: Loss function name.
        learning_rate: Optimizer learning rate.
        batch_size: Batch size.
        crop_size: Input crop size.
        seed: Random seed.
    """

    architecture: str
    encoder_name: str
    loss_name: str
    learning_rate: float
    batch_size: int
    crop_size: int
    seed: int


def parse_csv_values(raw: str, cast_type: type) -> list:
    """Parse comma-separated argument values into typed list.

    Args:
        raw: Raw comma-separated string.
        cast_type: Type used to cast each value.

    Returns:
        Typed list with stripped entries.

    Raises:
        ValueError: If no values are provided.
    """

    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value.")
    return [cast_type(item) for item in items]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for grid execution.

    Returns:
        Parsed namespace with grid and infra parameters.
    """

    parser = argparse.ArgumentParser(
        description="Run a grid of segmentation experiments locally or on multi-GPU hardware.",
    )
    parser.add_argument("--architectures", type=str, default="Unet")
    parser.add_argument("--encoders", type=str, default="resnet18,resnet34")
    parser.add_argument("--losses", type=str, default="ce_dice,cross_entropy,focal,dice")
    parser.add_argument("--learning-rates", type=str, default="0.001")
    parser.add_argument("--batch-sizes", type=str, default="8")
    parser.add_argument("--crop-sizes", type=str, default="256")
    parser.add_argument("--seeds", type=str, default="42")

    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--fixed-val-samples", type=int, default=5)
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-minority-fraction", type=float, default=0.5)
    parser.add_argument("--data-root", type=str, default="")

    parser.add_argument("--infra", type=str, default="local", choices=["local", "highend"])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)

    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--parallel-workers", type=int, default=1)

    parser.add_argument("--experiment-prefix", type=str, default="t2")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--shuffle-grid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_grid(args: argparse.Namespace) -> list[TrialConfig]:
    """Build trial grid from CLI arguments.

    Args:
        args: Parsed arguments.

    Returns:
        Trial configuration list.
    """

    architectures = parse_csv_values(args.architectures, str)
    encoders = parse_csv_values(args.encoders, str)
    losses = parse_csv_values(args.losses, str)
    learning_rates = parse_csv_values(args.learning_rates, float)
    batch_sizes = parse_csv_values(args.batch_sizes, int)
    crop_sizes = parse_csv_values(args.crop_sizes, int)
    seeds = parse_csv_values(args.seeds, int)

    trials = [
        TrialConfig(*combo)
        for combo in itertools.product(
            architectures,
            encoders,
            losses,
            learning_rates,
            batch_sizes,
            crop_sizes,
            seeds,
        )
    ]
    if args.shuffle_grid:
        import random

        random.shuffle(trials)
    if args.max_trials > 0:
        trials = trials[: args.max_trials]
    return trials


def trial_name(prefix: str, index: int, config: TrialConfig) -> str:
    """Create deterministic run name for one trial.

    Args:
        prefix: Experiment prefix.
        index: Trial index.
        config: Trial configuration.

    Returns:
        Unique trial name.
    """

    encoder_short = config.encoder_name.replace("timm-", "").replace("_", "-")
    loss_short = config.loss_name.replace("_", "")
    return f"{prefix}-t{index:03d}-{encoder_short}-{loss_short}"


def build_train_command(
    config: TrialConfig,
    run_name: str,
    args: argparse.Namespace,
    gpu_id: str | None,
) -> tuple[list[str], dict[str, str]]:
    """Build subprocess command for one training trial.

    Args:
        config: Trial configuration.
        run_name: Experiment name.
        args: Global parsed args.
        gpu_id: Optional GPU ID for pinned execution.

    Returns:
        Tuple with command list and environment mapping.
    """

    accelerator = args.accelerator
    devices = args.devices
    if gpu_id is not None:
        accelerator = "gpu"
        devices = "1"

    command = [
        sys.executable,
        "-m",
        "computer_vision.taller_2.train",
        "--architecture",
        config.architecture,
        "--encoder-name",
        config.encoder_name,
        "--encoder-weights",
        args.encoder_weights,
        "--loss-name",
        config.loss_name,
        "--batch-size",
        str(config.batch_size),
        "--crop-size",
        str(config.crop_size),
        "--max-epochs",
        str(args.max_epochs),
        "--learning-rate",
        str(config.learning_rate),
        "--seed",
        str(config.seed),
        "--patience",
        str(args.patience),
        "--fixed-val-samples",
        str(args.fixed_val_samples),
        "--val-fraction",
        str(args.val_fraction),
        "--val-minority-fraction",
        str(args.val_minority_fraction),
        "--devices",
        devices,
        "--accelerator",
        accelerator,
        "--precision",
        args.precision,
        "--strategy",
        args.strategy,
        "--accumulate-grad-batches",
        str(args.accumulate_grad_batches),
        "--num-sanity-val-steps",
        "0",
        "--experiment-name",
        run_name,
    ]

    if args.data_root:
        command.extend(["--data-root", args.data_root])

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    return command, env


def extract_last_json(stdout: str) -> dict[str, Any]:
    """Extract final JSON object from training stdout.

    Args:
        stdout: Complete process stdout.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: If no JSON object is found.
    """

    brace_positions = [idx for idx, char in enumerate(stdout) if char == "{"]
    for start in reversed(brace_positions):
        candidate = stdout[start:].strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON results from train.py output.")


def read_completed_trials(results_csv: Path) -> set[str]:
    """Read already completed trial names from results CSV.

    Args:
        results_csv: Path to accumulated results CSV.

    Returns:
        Set of completed trial names.
    """

    if not results_csv.exists():
        return set()
    completed: set[str] = set()
    with results_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("trial_name", "")
            status = row.get("status", "")
            if name and status == "success":
                completed.add(name)
    return completed


def append_result(results_csv: Path, row: dict[str, Any]) -> None:
    """Append one trial result row to CSV.

    Args:
        results_csv: Output CSV path.
        row: Result dictionary to append.
    """

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with results_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_trial(
    index: int,
    config: TrialConfig,
    args: argparse.Namespace,
    output_dir: Path,
    gpu_id: str | None,
) -> dict[str, Any]:
    """Run one trial and return normalized result dictionary.

    Args:
        index: Trial index.
        config: Trial configuration.
        args: Global run arguments.
        output_dir: Grid output folder.
        gpu_id: Optional GPU pinning id.

    Returns:
        Result dictionary with metrics and metadata.
    """

    run_name = trial_name(args.experiment_prefix, index, config)
    started_at = time.time()
    command, env = build_train_command(config, run_name, args, gpu_id)

    if args.dry_run:
        return {
            "trial_name": run_name,
            "status": "dry_run",
            "gpu_id": gpu_id or "",
            "command": " ".join(command),
            "elapsed_seconds": 0.0,
            "best_val_mean_iou": None,
            "val_mean_acc": None,
            "val_pixel_acc": None,
            "best_model_path": "",
            "log_dir": "",
            "error": "",
        }

    process = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    elapsed = time.time() - started_at
    stdout_path = output_dir / "trial_logs" / f"{run_name}.stdout.log"
    stderr_path = output_dir / "trial_logs" / f"{run_name}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")

    if process.returncode != 0:
        return {
            "trial_name": run_name,
            "status": "failed",
            "gpu_id": gpu_id or "",
            "command": " ".join(command),
            "elapsed_seconds": round(elapsed, 2),
            "best_val_mean_iou": None,
            "val_mean_acc": None,
            "val_pixel_acc": None,
            "best_model_path": "",
            "log_dir": "",
            "error": f"Return code {process.returncode}",
        }

    try:
        parsed = extract_last_json(process.stdout)
        return {
            "trial_name": run_name,
            "status": "success",
            "gpu_id": gpu_id or "",
            "command": " ".join(command),
            "elapsed_seconds": round(elapsed, 2),
            "best_val_mean_iou": parsed.get("best_val_mean_iou"),
            "val_mean_acc": parsed.get("val_mean_acc"),
            "val_pixel_acc": parsed.get("val_pixel_acc"),
            "best_model_path": parsed.get("best_model_path", ""),
            "log_dir": parsed.get("log_dir", ""),
            "error": "",
        }
    except ValueError as error:
        return {
            "trial_name": run_name,
            "status": "failed",
            "gpu_id": gpu_id or "",
            "command": " ".join(command),
            "elapsed_seconds": round(elapsed, 2),
            "best_val_mean_iou": None,
            "val_mean_acc": None,
            "val_pixel_acc": None,
            "best_model_path": "",
            "log_dir": "",
            "error": str(error),
        }


def write_leaderboard(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write sorted leaderboard CSV for successful trials.

    Args:
        output_dir: Grid run output directory.
        rows: Collected trial rows.

    Returns:
        Path to generated leaderboard CSV.
    """

    success = [row for row in rows if row.get("status") == "success" and row.get("best_val_mean_iou") is not None]
    success.sort(key=lambda item: float(item["best_val_mean_iou"]), reverse=True)

    leaderboard_path = output_dir / "leaderboard.csv"
    fieldnames = [
        "rank",
        "trial_name",
        "architecture",
        "encoder_name",
        "loss_name",
        "learning_rate",
        "batch_size",
        "crop_size",
        "seed",
        "best_val_mean_iou",
        "val_mean_acc",
        "val_pixel_acc",
        "elapsed_seconds",
        "best_model_path",
        "log_dir",
    ]

    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(success, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "trial_name": row.get("trial_name", ""),
                    "architecture": row.get("architecture", ""),
                    "encoder_name": row.get("encoder_name", ""),
                    "loss_name": row.get("loss_name", ""),
                    "learning_rate": row.get("learning_rate", ""),
                    "batch_size": row.get("batch_size", ""),
                    "crop_size": row.get("crop_size", ""),
                    "seed": row.get("seed", ""),
                    "best_val_mean_iou": row.get("best_val_mean_iou", ""),
                    "val_mean_acc": row.get("val_mean_acc", ""),
                    "val_pixel_acc": row.get("val_pixel_acc", ""),
                    "elapsed_seconds": row.get("elapsed_seconds", ""),
                    "best_model_path": row.get("best_model_path", ""),
                    "log_dir": row.get("log_dir", ""),
                }
            )
    return leaderboard_path


def choose_output_dir(args: argparse.Namespace) -> Path:
    """Resolve output directory for this grid run.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path where summary files will be stored.
    """

    if args.output_dir:
        return Path(args.output_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("logs") / "grid_runs" / f"{args.experiment_prefix}-{timestamp}"


def main() -> None:
    """Grid runner CLI entrypoint."""

    args = parse_args()
    trials = build_grid(args)
    output_dir = choose_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_path = output_dir / "grid.json"
    grid_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "trials": [asdict(trial) for trial in trials],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    gpu_ids = parse_csv_values(args.gpu_ids, str) if args.gpu_ids else []
    if args.infra == "highend" and not gpu_ids:
        print("[info] infra=highend without --gpu-ids. Running sequentially with provided accelerator/devices.")

    results_csv = output_dir / "results.csv"
    completed = read_completed_trials(results_csv) if args.resume else set()
    total = len(trials)
    print(f"[info] total trials: {total}")
    print(f"[info] output dir: {output_dir}")

    pending_items: list[tuple[int, TrialConfig]] = []
    for idx, config in enumerate(trials, start=1):
        name = trial_name(args.experiment_prefix, idx, config)
        if name in completed:
            print(f"[skip] {idx}/{total} {name} already completed")
            continue
        pending_items.append((idx, config))

    if args.parallel_workers > 1 and not gpu_ids:
        print("[warn] parallel-workers > 1 without --gpu-ids can oversubscribe one device. Falling back to sequential.")
    use_parallel = args.parallel_workers > 1 and len(gpu_ids) > 1 and not args.dry_run

    success_scores: list[tuple[str, float]] = []
    all_results: list[dict[str, Any]] = []

    if use_parallel:
        worker_count = min(args.parallel_workers, len(gpu_ids))
        buckets: list[list[tuple[int, TrialConfig]]] = [[] for _ in range(worker_count)]
        for position, item in enumerate(pending_items):
            buckets[position % worker_count].append(item)

        def _worker_run(bucket: list[tuple[int, TrialConfig]], worker_gpu: str) -> list[dict[str, Any]]:
            worker_results: list[dict[str, Any]] = []
            for idx, config in bucket:
                name = trial_name(args.experiment_prefix, idx, config)
                print(f"[run] {idx}/{total} {name} gpu={worker_gpu}")
                result = run_trial(
                    index=idx,
                    config=config,
                    args=args,
                    output_dir=output_dir,
                    gpu_id=worker_gpu,
                )
                merged = {
                    "trial_index": idx,
                    **asdict(config),
                    **result,
                }
                worker_results.append(merged)
            return worker_results

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for worker_idx in range(worker_count):
                bucket = buckets[worker_idx]
                if not bucket:
                    continue
                futures.append(executor.submit(_worker_run, bucket, gpu_ids[worker_idx]))

            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
    else:
        for idx, config in pending_items:
            gpu_id = None
            if gpu_ids:
                gpu_id = gpu_ids[(idx - 1) % len(gpu_ids)]

            name = trial_name(args.experiment_prefix, idx, config)
            print(f"[run] {idx}/{total} {name} gpu={gpu_id or 'auto'}")
            result = run_trial(
                index=idx,
                config=config,
                args=args,
                output_dir=output_dir,
                gpu_id=gpu_id,
            )
            merged = {
                "trial_index": idx,
                **asdict(config),
                **result,
            }
            all_results.append(merged)

    all_results.sort(key=lambda row: int(row["trial_index"]))
    for merged in all_results:
        append_result(results_csv, merged)
        if merged["status"] == "success" and merged["best_val_mean_iou"] is not None:
            success_scores.append((merged["trial_name"], float(merged["best_val_mean_iou"])))
            print(
                f"[ok] {merged['trial_name']} "
                f"best_val_mean_iou={float(merged['best_val_mean_iou']):.4f}"
            )
        elif merged["status"] == "dry_run":
            print(f"[dry] {merged['trial_name']}")
        else:
            print(f"[fail] {merged['trial_name']} error={merged['error']}")

    summary = {
        "output_dir": str(output_dir),
        "results_csv": str(results_csv),
        "leaderboard_csv": "",
        "total_trials": total,
        "completed_successfully": len(success_scores),
        "best_trial": None,
    }
    if success_scores:
        best_name, best_score = max(success_scores, key=lambda item: item[1])
        summary["best_trial"] = {
            "trial_name": best_name,
            "best_val_mean_iou": best_score,
        }

    leaderboard_path = write_leaderboard(output_dir=output_dir, rows=all_results)
    summary["leaderboard_csv"] = str(leaderboard_path)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
