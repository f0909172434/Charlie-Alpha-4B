UV ?= uv
CLI := $(UV) run charlie-alpha

.PHONY: setup export-setup data distill mix baseline pilot train eval export gguf clean-load overnight chat serve test lint release-check publish-hf publish-github stats-setup stats-simulate stats-data stats-distill stats-lock stats-baseline stats-pilot stats-train stats-eval stats-export stats-chat stats-serve stats-release-check stats-publish-hf stats-publish-github evolve evolve-prepare evolve-status evolve-bakeoff evolve-project-prepare evolve-project evolve-project-balanced evolve-diagnose evolve-cone evolve-cone-confirm evolve-calibrate evolve-block evolve-family-route-prepare evolve-family-route evolve-family-experts-train evolve-family-experts evolve-router-prepare evolve-router evolve-llm-router-prepare evolve-llm-router evolve-llm-router-promote evolve-llm-router-final evolve-robust-experts-prepare evolve-robust-experts-data evolve-robust-experts-train evolve-robust-experts-select evolve-robust-experts evolve-targeted-repair-prepare evolve-targeted-repair-data evolve-targeted-repair-train evolve-targeted-repair-select evolve-targeted-repair evolve-cone-promote forge forge-lock forge-prepare forge-score forge-select forge-distill forge-build forge-pilot forge-train forge-calibrate forge-dev forge-freeze forge-final forge-router-lock forge-router-freeze forge-router-eval forge-router-verify forge-chat forge-serve forge-export forge-clean-load forge-release-check forge-publish-github

setup:
	$(UV) sync --extra eval --group dev
	$(CLI) stats setup --config configs/pipeline.stats.yaml

export-setup:
	$(UV) sync --extra eval --extra export --group dev

data:
	$(CLI) stats data --config configs/pipeline.stats.yaml

distill:
	$(CLI) stats distill --config configs/pipeline.stats.yaml
	$(CLI) stats data --config configs/pipeline.stats.yaml

mix:
	$(CLI) stats data --config configs/pipeline.stats.yaml

baseline:
	$(CLI) stats baseline --config configs/pipeline.stats.yaml

pilot:
	$(CLI) stats pilot --config configs/pipeline.stats.yaml

train:
	$(CLI) stats train --config configs/pipeline.stats.yaml

eval:
	$(CLI) stats eval --variant all --config configs/pipeline.stats.yaml

export:
	$(CLI) stats export --config configs/pipeline.stats.yaml

gguf:
	$(CLI) stats export --gguf --config configs/pipeline.stats.yaml

clean-load:
	$(CLI) stats export --config configs/pipeline.stats.yaml

overnight:
	$(CLI) stats overnight --config configs/pipeline.stats.yaml

chat:
	$(CLI) stats chat --config configs/pipeline.stats.yaml

serve:
	$(CLI) stats serve --config configs/pipeline.stats.yaml

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests

release-check:
	$(CLI) stats release-check --config configs/pipeline.stats.yaml

publish-hf:
	$(CLI) stats publish-hf --config configs/pipeline.stats.yaml

publish-github:
	$(CLI) stats publish-github --config configs/pipeline.stats.yaml

stats-setup:
	$(CLI) stats setup --config configs/pipeline.stats.yaml

stats-simulate:
	$(CLI) stats simulate --config configs/pipeline.stats.yaml

stats-data:
	$(CLI) stats data --config configs/pipeline.stats.yaml

stats-distill:
	$(CLI) stats distill --config configs/pipeline.stats.yaml
	$(CLI) stats data --config configs/pipeline.stats.yaml

stats-lock:
	$(CLI) stats lock-eval --config configs/pipeline.stats.yaml

stats-baseline:
	$(CLI) stats baseline --config configs/pipeline.stats.yaml

stats-pilot:
	$(CLI) stats pilot --config configs/pipeline.stats.yaml

stats-train:
	$(CLI) stats train --config configs/pipeline.stats.yaml

stats-eval:
	$(CLI) stats eval --variant all --config configs/pipeline.stats.yaml

stats-export:
	$(CLI) stats export --config configs/pipeline.stats.yaml

stats-chat:
	$(CLI) stats chat --config configs/pipeline.stats.yaml

stats-serve:
	$(CLI) stats serve --config configs/pipeline.stats.yaml

stats-release-check:
	$(CLI) stats release-check --config configs/pipeline.stats.yaml

stats-publish-hf:
	$(CLI) stats publish-hf --config configs/pipeline.stats.yaml

stats-publish-github:
	$(CLI) stats publish-github --config configs/pipeline.stats.yaml

evolve:
	$(CLI) stats iterate --config configs/pipeline.evolve.yaml

evolve-prepare:
	$(CLI) stats iterate --prepare-only --config configs/pipeline.evolve.yaml

evolve-status:
	$(CLI) stats evolve-status --config configs/pipeline.evolve.yaml

evolve-bakeoff:
	$(CLI) stats base-bakeoff --config configs/pipeline.evolve.yaml

evolve-project-prepare:
	$(CLI) stats policy-project --prepare-only --config configs/pipeline.evolve.yaml

evolve-project:
	$(CLI) stats policy-project --config configs/pipeline.evolve.yaml

evolve-project-balanced:
	$(CLI) stats policy-project --balanced --config configs/pipeline.evolve.yaml

evolve-diagnose:
	$(CLI) stats policy-diagnose --config configs/pipeline.evolve.yaml

evolve-cone:
	$(CLI) stats policy-cone --config configs/pipeline.evolve.yaml

evolve-cone-confirm:
	$(CLI) stats policy-cone-confirm --config configs/pipeline.evolve.yaml

evolve-calibrate:
	$(CLI) stats policy-calibrate --config configs/pipeline.evolve.yaml

evolve-block:
	$(CLI) stats policy-block --config configs/pipeline.evolve.yaml

evolve-family-route:
	$(CLI) stats policy-family-route --config configs/pipeline.evolve.yaml

evolve-family-route-prepare:
	$(CLI) stats policy-family-route --selection-only --config configs/pipeline.evolve.yaml

evolve-family-experts-train:
	$(CLI) stats policy-family-experts --train-only --config configs/pipeline.evolve.yaml

evolve-family-experts:
	$(CLI) stats policy-family-experts --config configs/pipeline.evolve.yaml

evolve-router-prepare:
	$(CLI) stats policy-router-prepare --config configs/pipeline.evolve.yaml

evolve-router:
	$(CLI) stats policy-router --config configs/pipeline.evolve.yaml

evolve-llm-router-prepare:
	$(CLI) stats policy-llm-router --selection-only --config configs/pipeline.evolve.yaml

evolve-llm-router:
	$(CLI) stats policy-llm-router --config configs/pipeline.evolve.yaml

evolve-llm-router-promote:
	$(CLI) stats policy-llm-router-promote --config configs/pipeline.evolve.yaml

evolve-llm-router-final:
	$(CLI) stats policy-llm-router-final --config configs/pipeline.evolve.yaml

evolve-robust-experts-prepare:
	$(CLI) stats robust-experts-prepare --config configs/pipeline.evolve.yaml

evolve-robust-experts-data:
	$(CLI) stats robust-experts-data --config configs/pipeline.evolve.yaml

evolve-robust-experts-train:
	$(CLI) stats robust-experts-train --config configs/pipeline.evolve.yaml

evolve-robust-experts-select:
	$(CLI) stats robust-experts-select --config configs/pipeline.evolve.yaml

evolve-robust-experts: evolve-robust-experts-train evolve-robust-experts-select

evolve-targeted-repair-prepare:
	$(CLI) stats targeted-repair-prepare --config configs/pipeline.evolve.yaml

evolve-targeted-repair-data:
	$(CLI) stats targeted-repair-data --config configs/pipeline.evolve.yaml

evolve-targeted-repair-train:
	$(CLI) stats targeted-repair-train --config configs/pipeline.evolve.yaml

evolve-targeted-repair-select:
	$(CLI) stats targeted-repair-select --config configs/pipeline.evolve.yaml

evolve-targeted-repair: evolve-targeted-repair-train evolve-targeted-repair-select

evolve-cone-promote:
	$(CLI) stats policy-cone-promote --config configs/pipeline.evolve.yaml

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

forge-calibrate:
	$(CLI) forge calibrate --config configs/pipeline.v2.yaml

forge-dev:
	$(CLI) forge eval --suite dev --variant forge --config configs/pipeline.v2.yaml

forge-freeze:
	$(CLI) forge freeze --config configs/pipeline.v2.yaml

forge-final:
	$(CLI) forge eval --suite final --variant qwen35-base --config configs/pipeline.v2.yaml
	$(CLI) forge eval --suite final --variant forge --config configs/pipeline.v2.yaml
	$(CLI) forge compare --suite final --config configs/pipeline.v2.yaml

forge-router-lock:
	$(CLI) forge router-lock --config configs/pipeline.v2.yaml

forge-router-freeze:
	$(CLI) forge router-freeze --config configs/pipeline.v2.yaml

forge-router-eval:
	$(CLI) forge router-eval --variant qwen35-base --config configs/pipeline.v2.yaml
	$(CLI) forge router-eval --variant routed --config configs/pipeline.v2.yaml
	$(CLI) forge router-compare --config configs/pipeline.v2.yaml

forge-router-verify:
	$(CLI) forge router-verify --config configs/pipeline.v2.yaml

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
