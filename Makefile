# XGBoost VGI worker — dev, test, and deploy targets.
#
# Usage:
#   make venv         # create .venv with vgi + xgboost + scikit-learn (from PyPI)
#   make lint         # ruff + mypy
#   make test         # pytest unit/integration + SQL (stdio/http)
#   make test-stdio   # SQL tests with the worker as a subprocess
#   make test-http    # start a local HTTP server, run SQL tests, stop it
#   make test-cloud   # SQL tests against the deployed Fly.io service
#   make deploy       # build locally, smoke-test, push, deploy to Fly.io

VGI_BUILD_DIR  ?= $(HOME)/Development/vgi/build/release
TEST_RUNNER     = $(VGI_BUILD_DIR)/test/unittest
TEST_DIR        = .
TEST_PATTERN    = test/sql/*

# Worker paths (overridable)
WORKER_STDIO   ?= uv run --python 3.13 xgboost_worker.py
WORKER_HTTP    ?= http://localhost:8000
WORKER_CLOUD   ?= https://$(FLY_APP).fly.dev
HTTP_PORT      ?= 8000

# Fly.io config
FLY_APP        ?= vgi-xgboost

# Isolated model registry for local SQL tests (stdio/http workers inherit this).
TEST_MODELS_DIR ?= $(CURDIR)/.test-models

.PHONY: lint test pytest test-sql test-stdio test-http test-cloud build push smoke-test deploy venv

venv:
	uv venv --python 3.13
	uv pip install --python .venv \
		"vgi-python[http,oauth]>=0.8.2" \
		"vgi-rpc[sentry]>=0.20.4" \
		"xgboost>=2.0" "scikit-learn>=1.5" numpy \
		pytest ruff mypy haybarn

lint:
	.venv/bin/ruff check .
	.venv/bin/mypy

pytest:
	.venv/bin/pytest tests/ --rootdir=. -o "addopts=" -q --ignore=tests/test_sql_haybarn.py

# CI-portable SQL suite: replays test/sql/*.test in-process via haybarn (no
# custom DuckDB build needed). The unittest-based test-stdio/test-http remain
# the local authority.
test-sql:
	VGI_SQL_HAYBARN=1 .venv/bin/pytest tests/test_sql_haybarn.py --rootdir=. -o "addopts=" -q

test: lint pytest test-sql test-stdio test-http

test-stdio:
	rm -rf "$(TEST_MODELS_DIR)"
	XGBOOST_MODELS_DIR="$(TEST_MODELS_DIR)" VGI_XGBOOST_WORKER="$(WORKER_STDIO)" \
		$(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

test-http:
	@if lsof -iTCP:$(HTTP_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "ERROR: port $(HTTP_PORT) is already in use" >&2; \
		echo "  Kill the existing process: kill $$(lsof -iTCP:$(HTTP_PORT) -sTCP:LISTEN -t)" >&2; \
		exit 1; \
	fi
	@rm -rf "$(TEST_MODELS_DIR)"
	@XGBOOST_MODELS_DIR="$(TEST_MODELS_DIR)" VGI_SIGNING_KEY=dev .venv/bin/python serve.py --port $(HTTP_PORT) & \
		SERVER_PID=$$!; \
		for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
			curl -fsS -o /dev/null "http://localhost:$(HTTP_PORT)/health" 2>/dev/null && break; \
			sleep 1; \
		done; \
		VGI_XGBOOST_WORKER="$(WORKER_HTTP)" $(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"; \
		TEST_EXIT=$$?; \
		kill $$SERVER_PID 2>/dev/null; \
		wait $$SERVER_PID 2>/dev/null; \
		exit $$TEST_EXIT

test-cloud:
	VGI_XGBOOST_WORKER="$(WORKER_CLOUD)" $(TEST_RUNNER) --test-dir "$(TEST_DIR)" "$(TEST_PATTERN)"

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
