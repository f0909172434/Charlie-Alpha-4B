UV ?= uv
CLI := $(UV) run charlie-alpha

.PHONY: setup export-setup data distill mix baseline pilot train eval export gguf clean-load overnight chat serve test lint release-check publish-hf publish-github stats-setup stats-simulate stats-data stats-distill stats-lock stats-baseline stats-pilot stats-train stats-eval stats-export stats-chat stats-serve stats-release-check stats-publish-hf stats-publish-github evolve evolve-prepare evolve-status evolve-bakeoff evolve-project-prepare evolve-project evolve-project-balanced evolve-diagnose evolve-cone evolve-cone-confirm evolve-calibrate evolve-block evolve-family-route-prepare evolve-family-route evolve-family-experts-train evolve-family-experts evolve-router-prepare evolve-router evolve-llm-router-prepare evolve-llm-router evolve-llm-router-promote evolve-llm-router-final evolve-representation-probe-prepare evolve-representation-probe-data evolve-representation-probe-select evolve-representation-probe-confirm evolve-selector-head-prepare evolve-selector-head-data evolve-selector-head-pilot evolve-selector-head-confirm evolve-selector-runtime-freeze evolve-selector-runtime-verify evolve-selector-runtime-predict evolve-selector-external-prepare evolve-selector-external-data evolve-selector-external-run evolve-selector-external-amend evolve-selector-external-run-amended evolve-style-invariance-prepare evolve-style-invariance-data evolve-style-invariance-select evolve-style-invariance-confirm evolve-external-representation-diagnostic evolve-selector-sufficiency-prepare evolve-selector-sufficiency-data evolve-selector-sufficiency-select evolve-selector-sufficiency-confirm evolve-selector-sufficiency-historical-e3 evolve-selective-external-qualify evolve-selective-external-prepare evolve-selective-external-data evolve-selective-external-run evolve-external-domain-bridge-prepare evolve-external-domain-bridge-data evolve-external-domain-bridge-train evolve-external-domain-bridge-amend evolve-external-domain-bridge-train-amended evolve-external-exemplar-router-prepare evolve-external-exemplar-router-train evolve-guarded-weight-bridge-prepare evolve-guarded-weight-bridge-data evolve-guarded-weight-bridge-train evolve-guarded-external-prepare evolve-guarded-external-screen evolve-guarded-external-data evolve-guarded-external-run evolve-guarded-external-metadata-screen evolve-opened-source-residual-prepare evolve-opened-source-residual-data evolve-opened-source-residual-opportunity evolve-invalid-control-catalog-fallback-prepare evolve-invalid-control-catalog-fallback-data evolve-invalid-control-catalog-fallback-run evolve-catalog-fallback-external-prepare evolve-catalog-fallback-external-source evolve-catalog-fallback-external-freeze-child evolve-catalog-fallback-external-selected-data evolve-catalog-fallback-external-data evolve-catalog-fallback-external-run evolve-external-catalog-prepare evolve-external-catalog-data evolve-external-catalog-run evolve-external-catalog-v2-prepare evolve-external-catalog-v2-data evolve-external-catalog-v2-run evolve-robust-experts-prepare evolve-robust-experts-data evolve-robust-experts-train evolve-robust-experts-select evolve-robust-experts evolve-targeted-repair-prepare evolve-targeted-repair-data evolve-targeted-repair-train evolve-targeted-repair-select evolve-targeted-repair evolve-cone-promote forge forge-lock forge-prepare forge-score forge-select forge-distill forge-build forge-pilot forge-train forge-calibrate forge-dev forge-freeze forge-final forge-router-lock forge-router-freeze forge-router-eval forge-router-verify forge-chat forge-serve forge-export forge-clean-load forge-release-check forge-publish-github

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

evolve-llm-router-replication-prepare:
	$(CLI) stats policy-llm-router-replication-prepare --config configs/pipeline.evolve.yaml

evolve-llm-router-replicate:
	$(CLI) stats policy-llm-router-replicate --config configs/pipeline.evolve.yaml

evolve-llm-router-replication-diagnose:
	$(CLI) stats policy-llm-router-replication-diagnose --config configs/pipeline.evolve.yaml

evolve-llm-router-reduced-prepare:
	$(CLI) stats policy-llm-router-reduced-prepare --config configs/pipeline.evolve.yaml

evolve-llm-router-reduced-confirm:
	$(CLI) stats policy-llm-router-reduced-confirm --config configs/pipeline.evolve.yaml

evolve-sufficiency-guard-prepare:
	$(CLI) stats policy-sufficiency-guard-prepare --config configs/pipeline.evolve.yaml

evolve-sufficiency-guard-confirm:
	$(CLI) stats policy-sufficiency-guard-confirm --config configs/pipeline.evolve.yaml

evolve-sufficiency-guard-diagnose:
	$(CLI) stats policy-sufficiency-guard-diagnose --config configs/pipeline.evolve.yaml

evolve-sufficiency-guard-thresholded-prepare:
	$(CLI) stats policy-sufficiency-guard-thresholded-prepare --config configs/pipeline.evolve.yaml

evolve-sufficiency-guard-thresholded-confirm:
	$(CLI) stats policy-sufficiency-guard-thresholded-confirm --config configs/pipeline.evolve.yaml

evolve-router-historical-external-prepare:
	$(CLI) stats policy-router-historical-external-prepare --config configs/pipeline.evolve.yaml

evolve-router-historical-external:
	$(CLI) stats policy-router-historical-external --config configs/pipeline.evolve.yaml

evolve-representation-probe-prepare:
	$(CLI) stats representation-probe-prepare --config configs/pipeline.evolve.yaml

evolve-representation-probe-data:
	$(CLI) stats representation-probe-data --config configs/pipeline.evolve.yaml

evolve-representation-probe-select:
	$(CLI) stats representation-probe-select --config configs/pipeline.evolve.yaml

evolve-representation-probe-confirm:
	$(CLI) stats representation-probe-confirm --config configs/pipeline.evolve.yaml

evolve-selector-head-prepare:
	$(CLI) stats selector-head-prepare --config configs/pipeline.evolve.yaml

evolve-selector-head-data:
	$(CLI) stats selector-head-data --config configs/pipeline.evolve.yaml

evolve-selector-head-pilot:
	$(CLI) stats selector-head-pilot --config configs/pipeline.evolve.yaml

evolve-selector-head-confirm:
	$(CLI) stats selector-head-confirm --config configs/pipeline.evolve.yaml

evolve-selector-runtime-freeze:
	$(CLI) stats selector-runtime-freeze --config configs/pipeline.evolve.yaml

evolve-selector-runtime-verify:
	$(CLI) stats selector-runtime-verify --config configs/pipeline.evolve.yaml

evolve-selector-runtime-predict:
	$(CLI) stats selector-runtime-predict --config configs/pipeline.evolve.yaml

evolve-selector-external-prepare:
	$(CLI) stats selector-external-prepare --config configs/pipeline.evolve.yaml

evolve-selector-external-data:
	$(CLI) stats selector-external-data --config configs/pipeline.evolve.yaml

evolve-selector-external-run:
	$(CLI) stats selector-external-run --config configs/pipeline.evolve.yaml

evolve-selector-external-amend:
	$(CLI) stats selector-external-amend --config configs/pipeline.evolve.yaml

evolve-selector-external-run-amended:
	$(CLI) stats selector-external-run-amended --config configs/pipeline.evolve.yaml

evolve-style-invariance-prepare:
	$(CLI) stats style-invariance-prepare --config configs/pipeline.evolve.yaml

evolve-style-invariance-data:
	$(CLI) stats style-invariance-data --config configs/pipeline.evolve.yaml

evolve-style-invariance-select:
	$(CLI) stats style-invariance-select --config configs/pipeline.evolve.yaml

evolve-style-invariance-confirm:
	$(CLI) stats style-invariance-confirm --config configs/pipeline.evolve.yaml

evolve-external-representation-diagnostic:
	$(CLI) stats external-representation-diagnostic --config configs/pipeline.evolve.yaml

evolve-selector-sufficiency-prepare:
	$(CLI) stats selector-sufficiency-prepare --config configs/pipeline.evolve.yaml

evolve-selector-sufficiency-data:
	$(CLI) stats selector-sufficiency-data --config configs/pipeline.evolve.yaml

evolve-selector-sufficiency-select:
	$(CLI) stats selector-sufficiency-select --config configs/pipeline.evolve.yaml

evolve-selector-sufficiency-confirm:
	$(CLI) stats selector-sufficiency-confirm --config configs/pipeline.evolve.yaml

evolve-selector-sufficiency-historical-e3:
	$(CLI) stats selector-sufficiency-historical-e3 --config configs/pipeline.evolve.yaml

evolve-selective-external-qualify:
	$(CLI) stats selective-external-qualify --config configs/pipeline.evolve.yaml

evolve-selective-external-prepare:
	$(CLI) stats selective-external-prepare --config configs/pipeline.evolve.yaml

evolve-selective-external-data:
	$(CLI) stats selective-external-data --config configs/pipeline.evolve.yaml

evolve-selective-external-run:
	$(CLI) stats selective-external-run --config configs/pipeline.evolve.yaml

evolve-external-domain-bridge-prepare:
	$(CLI) stats external-domain-bridge-prepare --config configs/pipeline.evolve.yaml

evolve-external-domain-bridge-data:
	$(CLI) stats external-domain-bridge-data --config configs/pipeline.evolve.yaml

evolve-external-domain-bridge-train:
	$(CLI) stats external-domain-bridge-train --config configs/pipeline.evolve.yaml

evolve-external-domain-bridge-amend:
	$(CLI) stats external-domain-bridge-amend --config configs/pipeline.evolve.yaml

evolve-external-domain-bridge-train-amended:
	$(CLI) stats external-domain-bridge-train-amended --config configs/pipeline.evolve.yaml

evolve-external-exemplar-router-prepare:
	$(CLI) stats external-exemplar-router-prepare --config configs/pipeline.evolve.yaml

evolve-external-exemplar-router-train:
	$(CLI) stats external-exemplar-router-train --config configs/pipeline.evolve.yaml

evolve-guarded-weight-bridge-prepare:
	$(CLI) stats guarded-weight-bridge-prepare --config configs/pipeline.evolve.yaml

evolve-guarded-weight-bridge-data:
	$(CLI) stats guarded-weight-bridge-data --config configs/pipeline.evolve.yaml

evolve-guarded-weight-bridge-train:
	$(CLI) stats guarded-weight-bridge-train --config configs/pipeline.evolve.yaml

evolve-guarded-external-prepare:
	$(CLI) stats guarded-external-prepare --config configs/pipeline.evolve.yaml

evolve-guarded-external-screen:
	$(CLI) stats guarded-external-screen --config configs/pipeline.evolve.yaml

evolve-guarded-external-data:
	$(CLI) stats guarded-external-data --config configs/pipeline.evolve.yaml

evolve-guarded-external-run:
	$(CLI) stats guarded-external-run --config configs/pipeline.evolve.yaml

evolve-guarded-external-metadata-screen:
	$(CLI) stats guarded-external-metadata-screen --config configs/pipeline.evolve.yaml

evolve-opened-source-residual-prepare:
	$(CLI) stats opened-source-residual-prepare --config configs/pipeline.evolve.yaml

evolve-opened-source-residual-data:
	$(CLI) stats opened-source-residual-data --config configs/pipeline.evolve.yaml

evolve-opened-source-residual-opportunity:
	$(CLI) stats opened-source-residual-opportunity --config configs/pipeline.evolve.yaml

evolve-invalid-control-catalog-fallback-prepare:
	$(CLI) stats invalid-control-catalog-fallback-prepare --config configs/pipeline.evolve.yaml

evolve-invalid-control-catalog-fallback-data:
	$(CLI) stats invalid-control-catalog-fallback-data --config configs/pipeline.evolve.yaml

evolve-invalid-control-catalog-fallback-run:
	$(CLI) stats invalid-control-catalog-fallback-run --config configs/pipeline.evolve.yaml

evolve-catalog-fallback-external-prepare:
	$(CLI) stats catalog-fallback-external-prepare --config configs/pipeline.evolve.yaml

evolve-catalog-fallback-external-source:
	$(CLI) stats catalog-fallback-external-source --config configs/pipeline.evolve.yaml

evolve-catalog-fallback-external-freeze-child:
	$(CLI) stats catalog-fallback-external-freeze-child --metadata $(METADATA) --config configs/pipeline.evolve.yaml

evolve-catalog-fallback-external-selected-data:
	$(CLI) stats catalog-fallback-external-selected-data --config configs/pipeline.evolve.yaml

evolve-catalog-fallback-external-data:
	$(CLI) stats catalog-fallback-external-data --config configs/pipeline.evolve.yaml

evolve-catalog-fallback-external-run:
	$(CLI) stats catalog-fallback-external-run --config configs/pipeline.evolve.yaml

evolve-external-catalog-prepare:
	$(CLI) stats external-catalog-prepare --config configs/pipeline.evolve.yaml

evolve-external-catalog-data:
	$(CLI) stats external-catalog-data --config configs/pipeline.evolve.yaml

evolve-external-catalog-run:
	$(CLI) stats external-catalog-run --config configs/pipeline.evolve.yaml

evolve-external-catalog-v2-prepare:
	$(CLI) stats external-catalog-v2-prepare --config configs/pipeline.evolve.yaml

evolve-external-catalog-v2-data:
	$(CLI) stats external-catalog-v2-data --config configs/pipeline.evolve.yaml

evolve-external-catalog-v2-run:
	$(CLI) stats external-catalog-v2-run --config configs/pipeline.evolve.yaml

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
