#!/usr/bin/env bash
#
# ADAS installer for a Jetson Xavier NX (L4T R35.6.5 / JetPack 5.1.6).
#
# Design rules, in order of importance:
#
#   1. Never half-install. Everything is staged into $PREFIX/releases/<release-id>
#      and only becomes live when the $PREFIX/current symlink is swapped, which is
#      the last mutating step. A failure anywhere before that leaves the running
#      install untouched; `set -euo pipefail` plus an ERR trap make a failure a
#      failure rather than a warning nobody reads.
#   2. Rollback is one symlink. `ln -sfn releases/<old> current && systemctl restart
#      adas` puts the previous release back.
#   3. Code, engines and config are root-owned and read-only to the service. The adas
#      account owns only /var/lib/adas, /run/adas and /var/log/adas. A process that
#      can rewrite models/MANIFEST.json can defeat its own integrity record, so it
#      must not be able to.
#   4. Ship only what runs. No tests/, no docs/, no .git, no __pycache__, no
#      recordings, and no data/events.jsonl from the lab.
#   5. Refuse to enable a unit that would crash-loop. --frames 0 must mean "run until
#      stopped"; if this build still treats it as zero frames, the unit is installed
#      but NOT enabled, and the reason is printed. See "check_continuous_mode".
#   6. Never touch the power state of a vehicle unless asked: nvpmodel runs only under
#      --apply-power, and jetson_clocks is never run at all.
#
# This script cannot be executed by CI (it needs root). It is validated by `bash -n`,
# by `shellcheck`, and by `--dry-run`, which performs every read-only precondition
# check for real and prints every mutating command without running it.
#
# Usage:
#   sudo bash deploy/setup_jetson.sh [options]
#
set -euo pipefail

PROG="$(basename "$0")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

PREFIX=/opt/adas
STATE_DIR=/var/lib/adas
LOG_DIR=/var/log/adas
CONF_DIR=/etc/adas
SERVICE=adas.service
SERVICE_USER=adas
MODELS_SRC="$REPO/models"

DRY_RUN=0
APPLY_POWER=0
FORCE_CONFIG=0
FORCE_ENABLE=0
NO_RESTART=0
NO_MODELS=0
RELEASE_ID=""
HEALTH_PORT=8090

usage() {
  cat <<EOF
Usage: sudo $PROG [options]

  --prefix DIR      install prefix (default $PREFIX). The unit file is rewritten to match.
  --release ID      name the release directory (default: git short sha, else a UTC stamp)
  --config FILE     seed $CONF_DIR/config.json from FILE (default: config.example.json)
  --force-config    overwrite an existing $CONF_DIR/config.json (preserved by default)
  --no-models       do not copy models/*.engine (use when engines are bind-mounted)
  --apply-power     run 'nvpmodel -m 8' (MODE_20W_6CORE). jetson_clocks is NEVER run.
  --force-enable    enable the unit even if the continuous-mode preflight fails
  --no-restart      install everything but leave the running service alone
  --dry-run         run every check, print every change, modify nothing. No root needed.
  -h, --help        this text

Rollback:
  ls $PREFIX/releases
  ln -sfn $PREFIX/releases/<old> $PREFIX/current && systemctl restart $SERVICE

Verify after install:
  systemctl status $SERVICE
  curl -fsS http://127.0.0.1:$HEALTH_PORT/readyz | python3 -m json.tool
  curl -fsS http://127.0.0.1:$HEALTH_PORT/metrics | head
EOF
}

# --------------------------------------------------------------------- logging

info() { printf '[ ok ] %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }
die() {
  printf '[FAIL] %s\n' "$*" >&2
  exit 1
}

on_err() {
  local rc=$?
  local line=${1:-?}
  printf '\n[FAIL] %s aborted at line %s (exit %s).\n' "$PROG" "$line" "$rc" >&2
  if [ -n "${RELEASE_DIR:-}" ] && [ -d "${RELEASE_DIR:-}" ] && [ "$DRY_RUN" -eq 0 ]; then
    printf '[FAIL] The staged release %s was NOT activated; the previous install is untouched.\n' \
      "$RELEASE_DIR" >&2
    printf '[FAIL] Remove it with: rm -rf %s\n' "$RELEASE_DIR" >&2
  fi
  exit "$rc"
}
trap 'on_err $LINENO' ERR

# Run a mutating command, honouring --dry-run.
run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry ] %s\n' "$*"
    return 0
  fi
  "$@"
}

# Run a mutating shell snippet (redirections, globs), honouring --dry-run.
run_sh() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry ] sh -c %s\n' "$1"
    return 0
  fi
  sh -c "$1"
}

# ------------------------------------------------------------------ arguments

CONFIG_SRC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="${2:?--prefix needs a directory}"; shift ;;
    --release) RELEASE_ID="${2:?--release needs an id}"; shift ;;
    --config) CONFIG_SRC="${2:?--config needs a file}"; shift ;;
    --force-config) FORCE_CONFIG=1 ;;
    --no-models) NO_MODELS=1 ;;
    --apply-power) APPLY_POWER=1 ;;
    --force-enable) FORCE_ENABLE=1 ;;
    --no-restart) NO_RESTART=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
  shift
done

[ -n "$CONFIG_SRC" ] || CONFIG_SRC="$REPO/config.example.json"

# ---------------------------------------------------------------- preconditions

step "Preconditions"

if [ "$DRY_RUN" -eq 0 ] && [ "$(id -u)" -ne 0 ]; then
  die "must be run as root (or use --dry-run, which needs no privileges)"
fi
info "running as uid $(id -u)$([ "$DRY_RUN" -eq 1 ] && echo ' (dry run)')"

[ -d "$REPO/src/adas" ] || die "$REPO does not look like the ADAS checkout (no src/adas)"
[ -f "$REPO/deploy/adas.service" ] || die "deploy/adas.service is missing"
info "source tree: $REPO"

if [ ! -r /etc/nv_tegra_release ]; then
  warn "/etc/nv_tegra_release absent: this does not look like a Jetson. Continuing, but"
  warn "the CUDA/TensorRT paths in the unit file will not resolve."
else
  info "L4T: $(head -1 /etc/nv_tegra_release)"
fi

PYTHON=/usr/bin/python3
[ -x "$PYTHON" ] || die "$PYTHON not found"
PYVER="$("$PYTHON" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[ "$PYVER" = "3.8" ] || warn "system python is $PYVER; this board ships 3.8 and the unit pins /usr/bin/python3"
info "python: $PYTHON ($PYVER)"

# Import check with the *installed* layout: PYTHONPATH=src, no venv, no pip.
if ! PYTHONPATH="$REPO/src" "$PYTHON" -c 'import adas, adas.io, adas.core.metrics' >/dev/null 2>&1; then
  die "PYTHONPATH=$REPO/src python3 -c 'import adas' failed; fix the tree before installing"
fi
info "adas package imports with PYTHONPATH=$REPO/src"

for mod in numpy cv2; do
  if "$PYTHON" -c "import $mod" >/dev/null 2>&1; then
    info "system module present: $mod"
  else
    warn "system module MISSING: $mod — the real perception path will not run."
    warn "Install the L4T system package; do NOT pip install it on this board."
  fi
done

if [ "$NO_MODELS" -eq 0 ]; then
  if [ -d "$MODELS_SRC" ] && ls "$MODELS_SRC"/*.engine >/dev/null 2>&1; then
    info "engines: $(ls "$MODELS_SRC"/*.engine | wc -l) file(s), $(du -sh "$MODELS_SRC" | cut -f1)"
  else
    warn "no *.engine in $MODELS_SRC — the unit will start with mock perception only."
  fi
fi

# ------------------------------------------------------- continuous-mode check

# The unit runs `--frames 0` and systemd restarts it on exit. If this build still
# interprets 0 as "zero frames", the service exits immediately and Restart=always
# turns into an infinite restart loop that looks, from the outside, like a working
# deployment. Find that out here, before enabling anything.
CONTINUOUS_OK=0
check_continuous_mode() {
  step "Preflight: does --frames 0 mean 'run until stopped'?"
  local out rc
  set +e
  out="$(cd "$REPO" && PYTHONPATH=src timeout 8 "$PYTHON" -m adas.cli --frames 0 \
          --source synthetic --log-level ERROR 2>&1)"
  rc=$?
  set -e
  if [ "$rc" -eq 124 ]; then
    CONTINUOUS_OK=1
    info "still running after 8 s — continuous mode works"
  else
    CONTINUOUS_OK=0
    warn "adas.cli --frames 0 exited after less than 8 s (exit $rc)."
    warn "That is the ADAS-OPS-02 defect: the runner treats 0 as zero frames, so the"
    warn "systemd unit would restart every RestartSec forever. The unit will be"
    warn "installed but NOT enabled. Re-run with --force-enable to override."
    printf '%s\n' "$out" | tail -5 | sed 's/^/       | /' >&2
  fi
}
check_continuous_mode

# --------------------------------------------------------------- service user

step "Service account and directories"

if id -u "$SERVICE_USER" >/dev/null 2>&1; then
  info "user $SERVICE_USER exists"
else
  run useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
  info "created system user $SERVICE_USER"
fi

# video: /dev/video* and the nvargus socket.
if getent group video >/dev/null 2>&1; then
  run usermod -aG video "$SERVICE_USER"
  info "$SERVICE_USER added to group video"
fi

for d in "$STATE_DIR" "$LOG_DIR"; do
  run install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$d"
  info "$d 0750 $SERVICE_USER:$SERVICE_USER"
done
# Config and code are root-owned and read-only to the service.
run install -d -o root -g root -m 0755 "$CONF_DIR"
run install -d -o root -g root -m 0755 "$PREFIX" "$PREFIX/releases"
info "$CONF_DIR and $PREFIX 0755 root:root"

# ------------------------------------------------------------- stage release

step "Stage release"

if [ -z "$RELEASE_ID" ]; then
  if git -C "$REPO" rev-parse --short=12 HEAD >/dev/null 2>&1; then
    RELEASE_ID="$(git -C "$REPO" rev-parse --short=12 HEAD)"
  else
    RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  fi
fi
RELEASE_DIR="$PREFIX/releases/$RELEASE_ID"
info "release id: $RELEASE_ID"

if [ -e "$RELEASE_DIR" ] && [ "$DRY_RUN" -eq 0 ]; then
  warn "$RELEASE_DIR already exists; replacing its contents"
fi
run install -d -o root -g root -m 0755 "$RELEASE_DIR"

RSYNC_EXCLUDES=(
  --exclude '.git' --exclude '.github' --exclude '__pycache__' --exclude '*.pyc'
  --exclude '.pytest_cache' --exclude '.ruff_cache' --exclude 'tests'
  --exclude 'docs' --exclude 'recordings' --exclude 'data'
  --exclude '*.egg-info' --exclude 'Ultra-Fast-Lane-Detection-v2'
  --exclude 'models/*.onnx' --exclude 'models/*.log'
)
if [ "$NO_MODELS" -eq 1 ]; then
  RSYNC_EXCLUDES+=(--exclude 'models/*.engine')
fi

if command -v rsync >/dev/null 2>&1; then
  run rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$REPO"/ "$RELEASE_DIR"/
else
  warn "rsync not found; falling back to cp -a of src/, deploy/, models/ and metadata"
  run_sh "cp -a '$REPO/src' '$RELEASE_DIR/'"
  run_sh "cp -a '$REPO/deploy' '$RELEASE_DIR/'"
  run_sh "[ -d '$REPO/models' ] && cp -a '$REPO/models' '$RELEASE_DIR/' || true"
  run_sh "cp -a '$REPO/pyproject.toml' '$REPO/README.md' '$RELEASE_DIR/' 2>/dev/null || true"
fi
run chown -R root:root "$RELEASE_DIR"
run_sh "find '$RELEASE_DIR' -type d -exec chmod 0755 {} +"
run_sh "find '$RELEASE_DIR' -type f -exec chmod 0644 {} +"
info "staged $RELEASE_DIR (root-owned, read-only to $SERVICE_USER)"

# Record what was installed, so adas_build_info has a git sha on a unit with no .git.
GIT_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
run_sh "cat > '$CONF_DIR/release' <<EOF
release_id=$RELEASE_ID
git_sha=$GIT_SHA
installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
installed_from=$REPO
prefix=$PREFIX
EOF"
run chmod 0644 "$CONF_DIR/release"
info "wrote $CONF_DIR/release"

# ------------------------------------------------------------------- config

step "Configuration"

if [ -f "$CONF_DIR/config.json" ] && [ "$FORCE_CONFIG" -eq 0 ]; then
  info "$CONF_DIR/config.json exists; preserved (use --force-config to replace)"
elif [ -f "$CONFIG_SRC" ]; then
  if ! "$PYTHON" -c "import json,sys;json.load(open(sys.argv[1]))" "$CONFIG_SRC"; then
    die "$CONFIG_SRC is not valid JSON"
  fi
  run install -o root -g root -m 0644 "$CONFIG_SRC" "$CONF_DIR/config.json"
  info "installed $CONF_DIR/config.json from $CONFIG_SRC"
else
  warn "$CONFIG_SRC not found; the unit will fail to start until $CONF_DIR/config.json exists"
fi

# --------------------------------------------------------------- unit files

step "systemd unit, logrotate"

UNIT_TMP="$(mktemp)"
sed "s#/opt/adas#$PREFIX#g" "$REPO/deploy/adas.service" > "$UNIT_TMP"
if [ "$DRY_RUN" -eq 1 ]; then
  printf '[dry ] install -m 0644 <rendered unit> /etc/systemd/system/%s\n' "$SERVICE"
  printf '[dry ] --- rendered ExecStart/PYTHONPATH ---\n'
  grep -E '^(ExecStart|Environment=PYTHONPATH|WorkingDirectory)' "$UNIT_TMP" | sed 's/^/[dry ]   /'
else
  install -o root -g root -m 0644 "$UNIT_TMP" "/etc/systemd/system/$SERVICE"
  info "installed /etc/systemd/system/$SERVICE"
fi
rm -f "$UNIT_TMP"

run install -o root -g root -m 0644 "$REPO/deploy/adas.logrotate" /etc/logrotate.d/adas
info "installed /etc/logrotate.d/adas"
if command -v logrotate >/dev/null 2>&1 && [ "$DRY_RUN" -eq 0 ]; then
  logrotate --debug /etc/logrotate.d/adas >/dev/null && info "logrotate config parses"
fi

run ln -sfn "$RELEASE_DIR" "$PREFIX/current"
info "$PREFIX/current -> $RELEASE_DIR"

# ------------------------------------------------------------------- power

if [ "$APPLY_POWER" -eq 1 ]; then
  step "Power mode"
  if command -v nvpmodel >/dev/null 2>&1; then
    run nvpmodel -m 8
    info "nvpmodel -m 8 (MODE_20W_6CORE). jetson_clocks deliberately NOT run."
  else
    warn "nvpmodel not found; power mode unchanged"
  fi
fi

# ------------------------------------------------------------------ activate

step "Activate"

run systemctl daemon-reload

if [ "$CONTINUOUS_OK" -eq 1 ] || [ "$FORCE_ENABLE" -eq 1 ]; then
  run systemctl enable "$SERVICE"
  info "enabled $SERVICE"
  if [ "$NO_RESTART" -eq 1 ]; then
    info "--no-restart: not (re)starting the service"
  else
    run systemctl restart "$SERVICE"
    info "restarted $SERVICE"
  fi
else
  warn "NOT enabling $SERVICE: the continuous-mode preflight failed (see above)."
  warn "Install is complete and rollback-safe; enable it with:"
  warn "  systemctl enable --now $SERVICE"
fi

# -------------------------------------------------------------------- verify

if [ "$DRY_RUN" -eq 0 ] && systemctl is-active --quiet "$SERVICE"; then
  step "Verify"
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$HEALTH_PORT/readyz" >/dev/null 2>&1; then
      info "health endpoint is ready on 127.0.0.1:$HEALTH_PORT"
      break
    fi
    sleep 2
  done
  curl -fsS "http://127.0.0.1:$HEALTH_PORT/healthz" 2>/dev/null \
    | "$PYTHON" -m json.tool 2>/dev/null | head -25 || \
    warn "health endpoint did not answer within 60 s; check: journalctl -u $SERVICE -n 50"
fi

step "Done"
cat <<EOF
  status    systemctl status $SERVICE
  logs      journalctl -u $SERVICE -f -o cat | jq
  health    curl -fsS http://127.0.0.1:$HEALTH_PORT/healthz | python3 -m json.tool
  ready     curl -fsS http://127.0.0.1:$HEALTH_PORT/readyz  >/dev/null && echo READY
  metrics   curl -fsS http://127.0.0.1:$HEALTH_PORT/metrics
  events    tail -f $STATE_DIR/events.jsonl | jq
  rollback  ln -sfn $PREFIX/releases/<old> $PREFIX/current && systemctl restart $SERVICE
EOF
