# train-stageN, continue-stageN and test-N are pattern rules; listing them in .PHONY would disable them.
.PHONY: help install run view view-best view-baseline test tests test-list eval train training run-train \
	train-full train-full-multi train-smoke eval-smoke eval-stage1 eval-stage2 eval-stage3 \
	showcase view-learned eval-heldout eval-noisy eval-dense

PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	elif [ -x .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe; \
	else echo python3; fi)
RUNS := runs
MODELS := models
SEED ?= 0
SEEDS ?= 0 1 2
SEEDS_N ?= 20

help:
	@echo "SwarmFlow commands (see guide.md):"
	@echo "  make training             Full training: stage 1 -> 2 -> 3 for SEED=$(SEED), then store the best"
	@echo "                            model in models/showcase (re-run to continue after an interruption)"
	@echo "  make run train            Same training, plus the viewer: live progress, then the best model"
	@echo "                            (same as make run-train)"
	@echo "  make run                  Start the viewer (uses the best model under models/ automatically)"
	@echo ""
	@echo "  make install              Install/update the Python environment"
	@echo "  make view-baseline        Start the viewer without any trained model"
	@echo "  make test                 Run the unit tests"
	@echo "  make tests                Pick which numbered validation tests to run (menu)"
	@echo "  make test-N               Run one numbered test, e.g. make test-2"
	@echo ""
	@echo "  make train-full           Same as make training"
	@echo "  make train-full-multi     Training for SEEDS=\"$(SEEDS)\"; the showcase keeps the best seed"
	@echo "  make train-stage1/2/3     Train one curriculum stage for SEED into models/stageN-seedSEED"
	@echo "  make continue-stage1/2/3  Finish an interrupted stage from its latest.pt"
	@echo "  make eval-stage1/2/3      Evaluate a stage's best.pt on 20 development seeds"
	@echo "  make showcase             Test the best model on held-out seeds and store it in models/showcase"
	@echo "  make view-best            Open the viewer with models/showcase/best.pt"
	@echo "  make train-smoke          Tiny run that only checks the training machinery"
	@echo ""
	@echo "  make eval                 Evaluate the predictive baseline"
	@echo "  make eval-heldout         Baseline eval on seeds 100000+ (SEEDS_N=$(SEEDS_N))"
	@echo "  make eval-noisy           Baseline eval with noise 0.05 and dropout 0.1"
	@echo "  make eval-dense           Baseline eval on the dense sphere scenario"

install:
	uv sync --python 3.12 --extra dev --cache-dir .uv-cache

# `make run train` must train with the viewer open instead of starting the viewer and then training.
RUN_AND_TRAIN := $(and $(filter run,$(MAKECMDGOALS)),$(filter train training,$(MAKECMDGOALS)))

ifdef RUN_AND_TRAIN
run train training: run-train
	@:
else
run: view

train: training

training:
	$(PYTHON) -m swarmflow pipeline --seeds $(SEED)
endif

run-train:
	$(PYTHON) -m swarmflow pipeline --seeds $(SEED) --view

view:
	$(PYTHON) -m swarmflow view

view-baseline:
	$(PYTHON) -m swarmflow view --baseline-only

view-best view-learned:
	$(PYTHON) -m swarmflow view --checkpoint $(MODELS)/showcase/best.pt

test:
	$(PYTHON) -m pytest -q

tests:
	$(PYTHON) -m swarmflow test $(T)

test-list:
	$(PYTHON) -m swarmflow test --list

test-%:
	$(PYTHON) -m swarmflow test $*

eval:
	$(PYTHON) -m swarmflow eval \
		--output $(RUNS)/baseline.json

train-full: training

train-full-multi:
	$(PYTHON) -m swarmflow pipeline --seeds $(SEEDS)

train-stage%:
	$(PYTHON) -m swarmflow train --stage $* --seed $(SEED)

continue-stage%:
	$(PYTHON) -m swarmflow train --stage $* --seed $(SEED) --continue

eval-stage1:
	$(PYTHON) -m swarmflow eval --policy learned \
		--checkpoint $(MODELS)/stage1-seed$(SEED)/best.pt \
		--scenarios hold follow turn --seeds $(SEEDS_N) \
		--output $(MODELS)/stage1-seed$(SEED)/dev-evaluation.json

eval-stage2:
	$(PYTHON) -m swarmflow eval --policy learned \
		--checkpoint $(MODELS)/stage2-seed$(SEED)/best.pt \
		--scenarios hold follow turn crossing head_on --seeds $(SEEDS_N) \
		--output $(MODELS)/stage2-seed$(SEED)/dev-evaluation.json

eval-stage3:
	$(PYTHON) -m swarmflow eval --policy learned \
		--checkpoint $(MODELS)/stage3-seed$(SEED)/best.pt \
		--scenarios hold follow crossing head_on double_crossing slalom turn gust --seeds $(SEEDS_N) \
		--output $(MODELS)/stage3-seed$(SEED)/dev-evaluation.json

showcase:
	$(PYTHON) -m swarmflow showcase

train-smoke:
	$(PYTHON) -m swarmflow train \
		--steps 512 \
		--horizon 64 \
		--epochs 2 \
		--clone-steps 128 \
		--clone-epochs 3 \
		--eval-seeds 1 \
		--scenarios hold follow crossing \
		--output $(RUNS)/my-smoke

eval-smoke:
	$(PYTHON) -m swarmflow eval \
		--policy learned \
		--checkpoint $(RUNS)/my-smoke/best.pt \
		--scenarios follow crossing \
		--seeds 3 \
		--output $(RUNS)/my-smoke/evaluation.json

eval-heldout:
	$(PYTHON) -m swarmflow eval \
		--seeds $(SEEDS_N) \
		--seed-start 100000 \
		--output $(RUNS)/heldout.json

eval-noisy:
	$(PYTHON) -m swarmflow eval \
		--noise 0.05 \
		--dropout 0.1 \
		--output $(RUNS)/noisy.json

eval-dense:
	$(PYTHON) -m swarmflow eval \
		--scenarios dense slalom \
		--obstacles 12 \
		--output $(RUNS)/dense.json
