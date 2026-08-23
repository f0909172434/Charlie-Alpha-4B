UV ?= uv
CLI := $(UV) run charlie-alpha

.PHONY: setup export-setup data distill mix baseline pilot train eval export gguf clean-load overnight chat serve test lint release-check publish-hf

setup:
	$(UV) sync --extra eval --group dev

export-setup:
	$(UV) sync --extra eval --extra export --group dev

data:
	$(CLI) data prepare --config configs/pipeline.yaml

distill:
	$(CLI) data distill --config configs/pipeline.yaml

mix:
	$(CLI) data mix --config configs/pipeline.yaml

baseline:
	$(CLI) eval run --variant base --config configs/pipeline.yaml

pilot:
	$(CLI) train pilot --config configs/pipeline.yaml

train:
	$(CLI) train run --config configs/pipeline.yaml

eval:
	$(CLI) eval run --variant adapter --config configs/pipeline.yaml

export:
	$(CLI) export all --config configs/pipeline.yaml

gguf:
	$(CLI) export gguf --config configs/pipeline.yaml

clean-load:
	$(CLI) export validate-clean --config configs/pipeline.yaml

overnight:
	$(CLI) overnight --config configs/pipeline.yaml

chat:
	$(CLI) chat --config configs/pipeline.yaml

serve:
	$(CLI) serve --config configs/pipeline.yaml

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests

release-check:
	$(CLI) release check --config configs/pipeline.yaml

publish-hf:
	$(CLI) release publish-hf --config configs/pipeline.yaml
