UV ?= uv
CLI := $(UV) run charlie-alpha

.PHONY: setup export-setup data distill mix baseline pilot train eval export gguf clean-load overnight chat serve test lint release-check publish-hf publish-github forge forge-lock forge-prepare forge-score forge-select forge-distill forge-build forge-pilot forge-train forge-dev forge-freeze forge-final forge-chat forge-serve forge-export forge-clean-load forge-release-check forge-publish-github

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

publish-github:
	$(CLI) release publish-github --config configs/pipeline.yaml

forge:
	$(CLI) forge overnight --config configs/pipeline.v2.yaml

forge-lock:
	$(CLI) forge lock-eval --config configs/pipeline.v2.yaml

forge-prepare:
	$(CLI) forge prepare --config configs/pipeline.v2.yaml

forge-score:
	$(CLI) forge score --config configs/pipeline.v2.yaml

forge-select:
	$(CLI) forge select --config configs/pipeline.v2.yaml

forge-distill:
	$(CLI) forge distill --config configs/pipeline.v2.yaml

forge-build:
	$(CLI) forge build --config configs/pipeline.v2.yaml

forge-pilot:
	$(CLI) forge pilot --config configs/pipeline.v2.yaml

forge-train:
	$(CLI) forge train --config configs/pipeline.v2.yaml

forge-dev:
	$(CLI) forge eval --suite dev --variant forge --config configs/pipeline.v2.yaml

forge-freeze:
	$(CLI) forge freeze --config configs/pipeline.v2.yaml

forge-final:
	$(CLI) forge eval --suite final --variant qwen35-base --config configs/pipeline.v2.yaml
	$(CLI) forge eval --suite final --variant forge --config configs/pipeline.v2.yaml
	$(CLI) forge compare --suite final --config configs/pipeline.v2.yaml

forge-chat:
	$(CLI) chat --config configs/pipeline.v2.yaml

forge-serve:
	$(CLI) serve --config configs/pipeline.v2.yaml

forge-export:
	$(CLI) export all --config configs/pipeline.v2.yaml

forge-clean-load:
	$(CLI) export validate-clean --config configs/pipeline.v2.yaml

forge-release-check:
	$(CLI) release check --config configs/pipeline.v2.yaml

forge-publish-github:
	$(CLI) release publish-github --config configs/pipeline.v2.yaml
