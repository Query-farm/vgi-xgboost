# XGBoost VGI worker — dev, test, and deploy targets.
#
# Usage:
#   make venv         # create .venv with vgi + xgboost + scikit-learn (from PyPI)
#   make lint         # ruff + mypy
#   make test         # lint + pytest + SQL (stdio, launch, http)
#   make test-sql     # SQL tests over stdio, launch and http (TRANSPORTS=... for one)
#   make test-stdio   # SQL tests with the worker as a subprocess
#   make test-http    # SQL tests against the worker run as an HTTP server
#   make test-cloud   # SQL tests against the deployed Fly.io service
#   make deploy       # build locally, smoke-test, push, deploy to Fly.io

# SQL tests: ci/run-integration.sh (adapted from the vgi-starter template) runs
# test/sql/*.test through the published haybarn-unittest with the signed community
# `vgi` extension, over stdio, launch (AF_UNIX, pooled) and http, with an isolated
# model registry per run. CI runs the same script per transport. ci/worker execs
# the venv interpreter on xgboost_worker.py.
HAYBARN_VERSION  ?= 2.0.1rc3
WORKER           ?= $(CURDIR)/ci/worker
TOOLS            := $(CURDIR)/.tools
HAYBARN_UNITTEST := $(TOOLS)/bin/haybarn-unittest
TRANSPORTS       ?= stdio launch http
WORKER_CLOUD     ?= https://$(FLY_APP).fly.dev

# Fly.io config
FLY_APP        ?= vgi-xgboost

.PHONY: lint check test pytest sync tools test-sql test-stdio test-http test-cloud build push smoke-test deploy venv

venv:
	uv venv --python 3.14
	uv pip install --python .venv \
		"vgi-python[http,oauth]>=0.52.2,<0.53" \
		"vgi-rpc[sentry]>=0.53.1" \
		"xgboost>=2.0" "scikit-learn>=1.5" numpy pandas \
		pytest ruff mypy haybarn

lint:
	.venv/bin/ruff check .
	.venv/bin/mypy

pytest:
	.venv/bin/pytest tests/ --rootdir=. -o "addopts=" -q

sync:
	uv sync --locked --all-extras

tools:
	@test -x $(HAYBARN_UNITTEST) || { uv venv -q $(TOOLS) && uv pip install -q -p $(TOOLS) haybarn-unittest==$(HAYBARN_VERSION); }

test-sql: sync tools
	@for t in $(TRANSPORTS); do \
		WORKER="$(WORKER)" HAYBARN_UNITTEST="$(HAYBARN_UNITTEST)" TRANSPORT=$$t ci/run-integration.sh || exit 1; \
	done

test-stdio:
	$(MAKE) test-sql TRANSPORTS=stdio

test-http:
	$(MAKE) test-sql TRANSPORTS=http

test-cloud: tools
	WORKER_URL="$(WORKER_CLOUD)" HAYBARN_UNITTEST="$(HAYBARN_UNITTEST)" TRANSPORT=http ci/run-integration.sh

test: sync lint pytest test-sql

check: test

GIT_COMMIT     := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
IMAGE_TAG      := $(GIT_COMMIT)-$(shell date +%Y%m%d%H%M%S)
IMAGE          := registry.fly.io/$(FLY_APP):$(IMAGE_TAG)

build:
	docker build --platform linux/amd64 --build-arg GIT_COMMIT=$(GIT_COMMIT) -t $(IMAGE) .

smoke-test: build
	@echo "Smoke-testing $(IMAGE)..."
	@docker run --rm --platform linux/amd64 -e VGI_SIGNING_KEY=dev $(IMAGE) \
		python -c "from xgboost_worker import XGBoostWorker; import serve; print('imports OK')"
	@CID=$$(docker run -d --platform linux/amd64 -e VGI_SIGNING_KEY=dev -p 18000:8000 $(IMAGE)); \
		trap "docker rm -f $$CID >/dev/null" EXIT; \
		for i in 1 2 3 4 5 6 7 8 9 10; do \
			if curl -fsS -o /dev/null -w "%{http_code}\n" http://localhost:18000/health 2>/dev/null | grep -qE '^(200|401|403|404)$$'; then \
				echo "HTTP server responding"; exit 0; \
			fi; \
			sleep 1; \
		done; \
		echo "ERROR: container did not respond on /health within 10s" >&2; \
		docker logs $$CID >&2; \
		exit 1

push: smoke-test
	fly auth docker
	docker push $(IMAGE)

deploy: push
	fly deploy --image $(IMAGE) --app $(FLY_APP)
