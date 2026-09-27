.PHONY: help install run view test eval train \
	train-smoke eval-smoke train-stage1 eval-stage1 \
	train-stage2 eval-stage2 view-learned \
	train-stage1-multi eval-stage1-multi \
	train-stage2-multi eval-stage2-multi \
	eval-heldout eval-noisy eval-dense

PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	elif [ -x .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe; \
	else echo python3; fi)
RUNS := runs
SEEDS ?= 0 1 2
SEEDS_N ?= 20

help:
	@echo "SwarmFlow commands:"
	@echo "  make install              Install/update the Python environment"
	@echo "  make run                  Start the interactive viewer"
	@echo "  make test                 Run the test suite"
	@echo "  make eval                 Evaluate the predictive baseline"
	@echo "  make train-smoke          Run a short training smoke test"
	@echo "  make eval-smoke           Evaluate the smoke-test checkpoint"
	@echo "  make train-stage1         Train hold/follow policy (seed 0)"
	@echo "  make eval-stage1          Evaluate the stage 1 checkpoint"
	@echo "  make train-stage2         Resume with crossing and head-on scenarios"
	@echo "  make eval-stage2          Evaluate the stage 2 checkpoint"
	@echo "  make view-learned         View the stage 2 learned policy"
	@echo "  make train-stage1-multi   Train stage 1 for SEEDS (default 0 1 2)"
	@echo "  make eval-stage1-multi    Evaluate each stage 1 seed"
	@echo "  make train-stage2-multi   Train stage 2 for SEEDS"
	@echo "  make eval-stage2-multi    Evaluate each stage 2 seed"
	@echo "  make eval-heldout         Baseline eval on seeds 100000+ (SEEDS_N=$(SEEDS_N))"
	@echo "  make eval-noisy           Baseline eval with noise 0.05 and dropout 0.1"
	@echo "  make eval-dense           Baseline eval on the dense sphere scenario"

install:
	uv sync --python 3.12 --extra dev --cache-dir .uv-cache

run: view

view:
	$(PYTHON) -m swarmflow view

test:
	$(PYTHON) -m pytest -q

eval:
	$(PYTHON) -m swarmflow eval \
		--output $(RUNS)/baseline.json

train: train-stage1

train-smoke:
	$(PYTHON) -m swarmflow train \
		--steps 512 \
		--horizon 64 \
		--epochs 2 \
		--clone-steps 128 \
		--clone-epochs 3 \
		--scenarios hold follow crossing \
		--output $(RUNS)/my-smoke

eval-smoke:
	$(PYTHON) -m swarmflow eval \
		--policy learned \
		--checkpoint $(RUNS)/my-smoke/latest.pt \
		--scenarios follow crossing \
		--seeds 3 \
		--output $(RUNS)/my-smoke/evaluation.json

train-stage1:
	$(PYTHON) -m swarmflow train \
		--steps 50000 \
		--clone-steps 2000 \
		--scenarios hold follow \
		--seed 0 \
		--output $(RUNS)/stage1-seed0

eval-stage1:
	$(PYTHON) -m swarmflow eval \
		--policy learned \
		--checkpoint $(RUNS)/stage1-seed0/latest.pt \
		--scenarios hold follow \
		--seeds 20 \
		--output $(RUNS)/stage1-seed0/validation.json

train-stage2:
	$(PYTHON) -m swarmflow train \
		--resume $(RUNS)/stage1-seed0/latest.pt \
		--steps 100000 \
		--scenarios hold follow crossing head_on \
		--seed 0 \
		--output $(RUNS)/stage2-seed0

eval-stage2:
	$(PYTHON) -m swarmflow eval \
		--policy learned \
		--checkpoint $(RUNS)/stage2-seed0/latest.pt \
		--seeds 20 \
		--output $(RUNS)/stage2-seed0/validation.json

view-learned:
	$(PYTHON) -m swarmflow view \
		--checkpoint $(RUNS)/stage2-seed0/latest.pt

train-stage1-multi:
	set -e; for S in $(SEEDS); do \
		$(PYTHON) -m swarmflow train \
			--steps 50000 \
			--clone-steps 2000 \
			--scenarios hold follow \
			--seed $$S \
			--output $(RUNS)/stage1-seed$$S; \
	done

eval-stage1-multi:
	set -e; for S in $(SEEDS); do \
		$(PYTHON) -m swarmflow eval \
			--policy learned \
			--checkpoint $(RUNS)/stage1-seed$$S/latest.pt \
			--scenarios hold follow \
			--seeds $(SEEDS_N) \
			--output $(RUNS)/stage1-seed$$S/validation.json; \
	done

train-stage2-multi:
	set -e; for S in $(SEEDS); do \
		$(PYTHON) -m swarmflow train \
			--resume $(RUNS)/stage1-seed$$S/latest.pt \
			--steps 100000 \
			--scenarios hold follow crossing head_on \
			--seed $$S \
			--output $(RUNS)/stage2-seed$$S; \
	done

eval-stage2-multi:
	set -e; for S in $(SEEDS); do \
		$(PYTHON) -m swarmflow eval \
			--policy learned \
			--checkpoint $(RUNS)/stage2-seed$$S/latest.pt \
			--seeds $(SEEDS_N) \
			--output $(RUNS)/stage2-seed$$S/validation.json; \
	done

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
