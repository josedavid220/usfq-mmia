UV ?= uv
PYTHON ?= $(UV) run python
PROJECT_ROOT ?= $(CURDIR)

# Default high-end experiment settings.
EXPERIMENT_PREFIX ?= t2
ARCHITECTURES ?= UnetPlusPlus,FPN,PSPNet,DeepLabV3Plus
ENCODERS ?= resnet101,timm-efficientnet-b5
LOSSES ?= ce_dice,cross_entropy,focal,dice
LEARNING_RATES ?= 0.001, 0.003
BATCH_SIZES ?= 32
CROP_SIZES ?= 256
SEEDS ?= 42
MAX_EPOCHS ?= 100
PATIENCE ?= 10
GPU_IDS ?= 0,1,2,3,4,5,6
PARALLEL_WORKERS ?= 7
PRECISION ?= 16-mixed
STRATEGY ?= auto

.PHONY: help install clean-logs train-smoke train-local grid-local grid-highend grid-dry grid-resume tensorboard gradio deploy leaderboard lint-fix format type-check run-hooks

.DEFAULT_GOAL := grid-highend

help:
	@echo "Available commands:"
	@echo "  make grid-highend   # default: high-end multi-GPU grid"
	@echo "  make grid-local     # local single-GPU grid"
	@echo "  make grid-dry       # dry-run without training"
	@echo "  make train-local    # single training run"
	@echo "  make train-smoke    # minimal quick sanity run"
	@echo "  make clean-logs     # remove logs content except .gitkeep"
	@echo "  make tensorboard    # open logs in tensorboard"
	@echo "  make gradio         # run visualizer"
	@echo "  make deploy         # run visualizer with public sharing"
	@echo "  make leaderboard    # print latest grid leaderboard top rows"
	@echo "  make lint-fix       # ruff fix + format"
	@echo "  make format         # formatting only"
	@echo "  make type-check     # pyright static check"
	@echo "  make run-hooks      # pre-commit hooks"

install:
	cd $(PROJECT_ROOT) && uv sync

clean-logs:
	cd $(PROJECT_ROOT) && mkdir -p logs && find logs -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +

train-smoke:
	cd $(PROJECT_ROOT) && $(PYTHON) -m computer_vision.taller_2.train \
		--experiment-name taller2-smoke \
		--architecture Unet \
		--encoder-name resnet18 \
		--loss-name ce_dice \
		--batch-size 4 \
		--crop-size 128 \
		--max-epochs 1 \
		--patience 2 \
		--accelerator gpu \
		--devices 1 \
		--precision 16-mixed \
		--num-sanity-val-steps 0

train-local:
	cd $(PROJECT_ROOT) && $(PYTHON) -m computer_vision.taller_2.train \
		--experiment-name taller2-local-run \
		--architecture Unet \
		--encoder-name resnet34 \
		--loss-name ce_dice \
		--batch-size 8 \
		--crop-size 256 \
		--max-epochs 25 \
		--patience 6 \
		--accelerator gpu \
		--devices 1 \
		--precision 16-mixed \
		--num-sanity-val-steps 0

grid-local:
	cd $(PROJECT_ROOT) && $(PYTHON) -m computer_vision.taller_2.grid_runner \
		--infra local \
		--experiment-prefix taller2-local-grid \
		--architectures Unet \
		--encoders resnet18,resnet34,efficientnet-b0 \
		--losses ce_dice,cross_entropy,focal \
		--learning-rates 0.001,0.0003 \
		--batch-sizes 4,8 \
		--crop-sizes 256 \
		--seeds 42 \
		--max-epochs 25 \
		--patience 6 \
		--accelerator gpu \
		--devices 1 \
		--precision 16-mixed \
		--strategy auto

grid-highend:
	cd $(PROJECT_ROOT) && $(PYTHON) -m computer_vision.taller_2.grid_runner \
		--infra highend \
		--experiment-prefix $(EXPERIMENT_PREFIX) \
		--architectures $(ARCHITECTURES) \
		--encoders $(ENCODERS) \
		--losses $(LOSSES) \
		--learning-rates $(LEARNING_RATES) \
		--batch-sizes $(BATCH_SIZES) \
		--crop-sizes $(CROP_SIZES) \
		--seeds $(SEEDS) \
		--max-epochs $(MAX_EPOCHS) \
		--patience $(PATIENCE) \
		--gpu-ids $(GPU_IDS) \
		--parallel-workers $(PARALLEL_WORKERS) \
		--precision $(PRECISION) \
		--strategy $(STRATEGY)

grid-dry:
	cd $(PROJECT_ROOT) && $(PYTHON) -m computer_vision.taller_2.grid_runner \
		--infra highend \
		--experiment-prefix $(EXPERIMENT_PREFIX)-dry \
		--architectures $(ARCHITECTURES) \
		--encoders $(ENCODERS) \
		--losses $(LOSSES) \
		--learning-rates $(LEARNING_RATES) \
		--batch-sizes $(BATCH_SIZES) \
		--crop-sizes $(CROP_SIZES) \
		--seeds $(SEEDS) \
		--gpu-ids $(GPU_IDS) \
		--parallel-workers $(PARALLEL_WORKERS) \
		--precision $(PRECISION) \
		--strategy $(STRATEGY) \
		--dry-run

grid-resume:
	@if [ -z "$(OUTPUT_DIR)" ]; then echo "Usage: make grid-resume OUTPUT_DIR=logs/grid_runs/<run-dir>"; exit 1; fi
	cd $(PROJECT_ROOT) && $(PYTHON) -m computer_vision.taller_2.grid_runner \
		--infra highend \
		--output-dir $(OUTPUT_DIR) \
		--experiment-prefix $(EXPERIMENT_PREFIX) \
		--architectures $(ARCHITECTURES) \
		--encoders $(ENCODERS) \
		--losses $(LOSSES) \
		--learning-rates $(LEARNING_RATES) \
		--batch-sizes $(BATCH_SIZES) \
		--crop-sizes $(CROP_SIZES) \
		--seeds $(SEEDS) \
		--max-epochs $(MAX_EPOCHS) \
		--patience $(PATIENCE) \
		--gpu-ids $(GPU_IDS) \
		--parallel-workers $(PARALLEL_WORKERS) \
		--precision $(PRECISION) \
		--strategy $(STRATEGY) \
		--resume

tensorboard:
	cd $(PROJECT_ROOT) && $(PYTHON) -m tensorboard.main --logdir $(PROJECT_ROOT)/logs --port 6006

gradio:
	cd $(PROJECT_ROOT) && $(PYTHON) -m computer_vision.taller_2.visualization

deploy:
	cd $(PROJECT_ROOT) && $(PYTHON) -m computer_vision.taller_2.visualization --share

leaderboard:
	cd $(PROJECT_ROOT) && $(PYTHON) -c "from pathlib import Path; import csv; files=sorted(Path('logs/grid_runs').glob('*/leaderboard.csv'), reverse=True); print('No leaderboard.csv found under logs/grid_runs') if not files else None; path=files[0] if files else None; rows=[] if path is None else list(csv.DictReader(path.open('r', encoding='utf-8'))); print(f'Latest leaderboard: {path}') if path else None; print('Leaderboard is empty.') if path and not rows else None; [print(f\"#{r.get('rank')} | {r.get('encoder_name')} | {r.get('loss_name')} | val_miou={r.get('best_val_mean_iou')} | val_mean_acc={r.get('val_mean_acc')}\") for r in rows[:10]]"

lint-fix:
	cd $(PROJECT_ROOT) && $(PYTHON) -m ruff check . --fix && $(PYTHON) -m ruff format .

format:
	cd $(PROJECT_ROOT) && $(PYTHON) -m ruff format .

type-check:
	cd $(PROJECT_ROOT) && $(PYTHON) -m pyright

run-hooks:
	cd $(PROJECT_ROOT) && $(PYTHON) -m pre_commit run --all-files
