# Makefile for ADAS Core.
#
# Everything here runs with PYTHONPATH=src against the system interpreter. That is
# not a shortcut — it is the only supported invocation on the target board, where
# python3-venv is unavailable and pip-installing into the system interpreter is
# forbidden by the project's dependency policy (see docs/JETSON.md). `make install`
# is for a development machine only and says so.

PYTHON ?= python3
PYTHONPATH_SRC = PYTHONPATH=src
HEALTH_URL ?= http://127.0.0.1:8090
# Other agents share this board's GPU; every command that touches TensorRT must hold
# the mutex. Harmless off-board (flock creates the file).
FLOCK ?= flock /tmp/jetson-gpu.lock -c

.PHONY: help install test test-ops lint format clean \
        lint-all docker docker-test docker-jetson compose-ci \
        run run-debug run-json run-trt \
        health metrics events deploy-check

help:
	@echo "ADAS Core"
	@echo ""
	@echo "Development (no GPU needed):"
	@echo "  make test          Full test suite (PYTHONPATH=src, system python3)"
	@echo "  make test-ops      Only the ops-layer tests (health, events, metrics)"
	@echo "  make lint          ruff check, errors only (hard gate)"
	@echo "  make lint-all      ruff check, full rule set (advisory)"
	@echo "  make format        ruff check --fix + ruff format"
	@echo "  make clean         Remove caches and build artefacts"
	@echo "  make install       pip install -e '.[dev]'  -- DEV MACHINES ONLY, never the board"
	@echo ""
	@echo "Running:"
	@echo "  make run           Mock pipeline, 10 frames"
	@echo "  make run-debug     ...with --log-level DEBUG (this now actually works)"
	@echo "  make run-json      ...with JSON structured logging"
	@echo "  make run-trt       TensorRT detector on example.mp4 (needs an engine; holds the GPU lock)"
	@echo ""
	@echo "Operations:"
	@echo "  make health        curl \$$HEALTH_URL/healthz  (default $(HEALTH_URL))"
	@echo "  make metrics       curl \$$HEALTH_URL/metrics"
	@echo "  make events        tail the safety-event JSONL"
	@echo "  make deploy-check  bash -n + shellcheck + --dry-run of deploy/setup_jetson.sh"
	@echo ""
	@echo "Containers:"
	@echo "  make docker        Build the CI image (CPU, tests only)"
	@echo "  make docker-test   Build it and run the suite inside it"
	@echo "  make docker-jetson Build the arm64/L4T runtime image (~5 GB base pull)"

# DEV MACHINES ONLY. On the Jetson this would install into the system interpreter,
# which the project forbids; use PYTHONPATH=src there instead.
install:
	@echo "NOTE: dev machines only. On the Jetson use 'PYTHONPATH=src python3 -m ...'."
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHONPATH_SRC) $(PYTHON) -m pytest tests/ -q

test-ops:
	$(PYTHONPATH_SRC) $(PYTHON) -m pytest tests/test_health.py tests/test_events.py \
		tests/test_metrics.py -q

# Hard gate: real defects (syntax errors, undefined/unused names, bad comparisons).
lint:
	ruff check --select E9,F src tests

# Everything ruff has an opinion about. Advisory: the tree is not clean yet
# (~900 findings, mostly UP006/UP031 modernisation), so this is a report, not a gate.
lint-all:
	-ruff check --statistics src tests

format:
	ruff check --fix src tests
	ruff format src tests

clean:
	rm -rf build/ dist/ src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null || true

# ------------------------------------------------------------------- running

run:
	$(PYTHONPATH_SRC) $(PYTHON) -m adas.cli --frames 10

run-debug:
	$(PYTHONPATH_SRC) $(PYTHON) -m adas.cli --frames 10 --log-level DEBUG

run-json:
	$(PYTHONPATH_SRC) ADAS_LOG_FORMAT=json $(PYTHON) -m adas.cli --frames 10

# Holds the board-wide GPU mutex: other agents build engines on this machine.
run-trt:
	$(FLOCK) "$(PYTHONPATH_SRC) $(PYTHON) -m adas.cli --detector tensorrt \
		--source Ultra-Fast-Lane-Detection-v2/example.mp4 --frames 60"

# ---------------------------------------------------------------- operations

health:
	@curl -fsS $(HEALTH_URL)/healthz | $(PYTHON) -m json.tool \
		|| echo "no health endpoint on $(HEALTH_URL) (is the service running?)"

metrics:
	@curl -fsS $(HEALTH_URL)/metrics || echo "no metrics endpoint on $(HEALTH_URL)"

events:
	@tail -n 20 -f /var/lib/adas/events.jsonl 2>/dev/null \
		|| tail -n 20 -f data/events.jsonl 2>/dev/null \
		|| echo "no event log at /var/lib/adas/events.jsonl or data/events.jsonl"

# Everything about the installer that can be checked without root.
deploy-check:
	bash -n deploy/setup_jetson.sh
	@command -v shellcheck >/dev/null 2>&1 \
		&& shellcheck -S warning deploy/setup_jetson.sh \
		|| echo "shellcheck not installed; skipped"
	bash deploy/setup_jetson.sh --dry-run >/dev/null
	@echo "deploy/setup_jetson.sh: syntax ok, dry run ok"
	@$(PYTHON) -c "import re,sys; u=open('deploy/adas.service').read(); \
	  req=['Type=notify','WatchdogSec=','NoNewPrivileges=true','ProtectSystem=strict', \
	       'ProtectHome=true','PrivateTmp=true','ReadWritePaths=','User=adas','Restart=']; \
	  miss=[k for k in req if k not in u]; \
	  sys.exit('deploy/adas.service is missing: %s' % miss) if miss else None; \
	  sys.exit('deploy/adas.service has no [Install] section') if not re.search(r'^\[Install\]', u, re.M) else None; \
	  print('deploy/adas.service: hardening directives present')"

# ---------------------------------------------------------------- containers

docker:
	docker build -t adas-core:ci .

docker-test: docker
	docker run --rm adas-core:ci

# ~5 GB base image pull on first build. arm64 only.
docker-jetson:
	docker build -f deploy/Dockerfile.jetson -t adas-core:jetson .

compose-ci:
	docker compose --profile ci run --rm adas-ci

# CI convenience targets
ci-test: test lint
ci-build: clean docker
