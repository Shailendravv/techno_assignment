#!/usr/bin/env bash
# =============================================================================
#  start.sh — One-command startup for the Runbook Agent
#
#  Starts in order:
#    1. Backend FastAPI   (port 8000)
#    2. Frontend Vite     (port 5173)
#
#  On CTRL+C everything is shut down cleanly in reverse order.
#
#  Target OS:  Linux / macOS / Windows (Git Bash)
#  Shell:      bash
#
#  Prerequisites:
#    - Python 3.10+
#    - Node.js 18+
#    - npm
#
#  Secrets are never hardcoded. They are read, in priority order, from:
#    1. the environment            (export GROQ_API_KEY=... ./start.sh)
#    2. ~/.runbook-agent/secrets.env   (outside the workspace)
#    3. backend/.env
#    4. an interactive masked prompt, which offers to save to backend/.env
#
#  Flags:
#    --backend-only / --frontend-only   start just one side
#    --skip-install                     assume dependencies are already there
#    --install-only                     set everything up, start nothing
#    --no-free-ports                    abort on a busy port instead of freeing it
#    -h | --help
#
#  Environment:
#    APP_ENV=local|dev   which backend/config/<profile>.json to run against
#    BACKEND_PORT        default 8000
#    FRONTEND_PORT       default 5173
# =============================================================================

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${SCRIPT_DIR}/backend"
FRONTEND_DIR="${SCRIPT_DIR}/frontend"
LOGS_DIR="${SCRIPT_DIR}/logs"
ENV_FILE="${BACKEND_DIR}/.env"
SECRETS_FILE="${HOME}/.runbook-agent/secrets.env"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

RUN_BACKEND=true
RUN_FRONTEND=true
DO_INSTALL=true
START_SERVERS=true
FREE_PORTS=true

# ── Colors (disabled when stdout is not a terminal) ──────────────────────────
#
# ANSI-C quoting, so the escape is a real ESC byte. '\033[0;31m' in a plain
# string only works through printf, and breaks the moment it reaches echo.
if [ -t 1 ]; then
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'
    CYAN=$'\033[0;36m'
    DIM=$'\033[2m'
    BOLD=$'\033[1m'
    NC=$'\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; DIM=''; BOLD=''; NC=''
fi

STARTUP_LOG="${LOGS_DIR}/startup.log"
BACKEND_LOG="${LOGS_DIR}/backend.log"
FRONTEND_LOG="${LOGS_DIR}/frontend.log"

case "$(uname -s 2>/dev/null || echo Unknown)" in
    MINGW* | MSYS* | CYGWIN* | Windows_NT) IS_WINDOWS=true ;;
    *) IS_WINDOWS=false ;;
esac

# ==============================================================================
# LOGGING HELPERS  (print to terminal AND append to the startup log)
# ==============================================================================

# Colors are passed as arguments, never embedded in the format string: a
# message containing a % would otherwise be eaten by printf.
_log_tee() {
    local color="$1" label="$2"
    shift 2
    if [ -d "$LOGS_DIR" ]; then
        printf '%s[%s]%s  %s\n' "$color" "$label" "$NC" "$*" | tee -a "$STARTUP_LOG"
    else
        printf '%s[%s]%s  %s\n' "$color" "$label" "$NC" "$*"
    fi
}

info()  { _log_tee "$GREEN"  "INFO"  "$*"; }
warn()  { _log_tee "$YELLOW" "WARN"  "$*"; }
error() { _log_tee "$RED"    "ERROR" "$*"; }
ok()    { _log_tee "$GREEN"  "OK"    "$*"; }
fail()  { _log_tee "$RED"    "FAIL"  "$*"; }

step() {
    if [ -d "$LOGS_DIR" ]; then
        printf '\n%s%s-- %s --%s\n' "$BLUE" "$BOLD" "$*" "$NC" | tee -a "$STARTUP_LOG"
    else
        printf '\n%s%s-- %s --%s\n' "$BLUE" "$BOLD" "$*" "$NC"
    fi
}

die() { error "$*"; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

usage() {
    # The header comment is the help text; there is no second copy to drift.
    # Lines 3 up to the divider that closes the header block, minus the
    # divider itself, with the leading comment marker stripped.
    sed -n '3,/^# =\{10,\}/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^#\{1,\} \{0,1\}//'
    exit 0
}

# ==============================================================================
# OS DETECTION
# ==============================================================================

_detect_os() {
    local os
    os="$(uname -s 2>/dev/null || echo Unknown)"
    case "$os" in
        Linux*)                 info "Detected OS: Linux" ;;
        Darwin*)                info "Detected OS: macOS" ;;
        MINGW* | MSYS* | CYGWIN*) info "Detected OS: Windows (Git Bash / MSYS2)" ;;
        *)                      warn "Detected OS: ${os} (not fully tested)" ;;
    esac
}

# ==============================================================================
# PID TRACKING
# ==============================================================================

_PIDS=()
_PID_NAMES=()

_track_pid() {
    local name="$1" pid="$2"
    _PIDS+=("$pid")
    _PID_NAMES+=("$name")
}

# Kill a process and everything below it.
#
# Windows needs taskkill /T against the *Windows* pid: `npm run dev` forks
# node and `uvicorn --reload` forks a worker, and a plain kill of the launcher
# leaves both of those orphaned, still holding the port.
_kill_tree() {
    local pid="${1:-}" force="${2:-false}"
    [ -n "$pid" ] || return 0
    kill -0 "$pid" 2>/dev/null || return 0

    if [ "$IS_WINDOWS" = true ]; then
        local wpid=""
        if have ps; then
            # Git Bash `ps -W` columns: PID PPID PGID WINPID TTY ...
            wpid="$(ps -W 2>/dev/null | awk -v p="$pid" '$1 == p { print $4; exit }')" || wpid=""
        fi
        [ -n "$wpid" ] || wpid="$pid"
        taskkill //PID "$wpid" //T //F >/dev/null 2>&1 || kill -TERM "$pid" 2>/dev/null || true
    elif [ "$force" = true ]; then
        pkill -KILL -P "$pid" 2>/dev/null || true
        kill -KILL "$pid" 2>/dev/null || true
    else
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
    fi
}

_kill_all() {
    [ "${#_PIDS[@]}" -gt 0 ] || return 0
    info "Shutting down services ..."

    local i pid name
    # Reverse order - frontend first, then backend.
    for (( i=${#_PIDS[@]}-1; i>=0; i-- )); do
        pid="${_PIDS[$i]}"
        name="${_PID_NAMES[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
            info "Stopping ${name} (PID ${pid}) ..."
            _kill_tree "$pid" false
        fi
    done

    sleep 1

    # Force-kill survivors.
    for (( i=${#_PIDS[@]}-1; i>=0; i-- )); do
        pid="${_PIDS[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
            _kill_tree "$pid" true
        fi
    done

    info "All services stopped."
}

# EXIT is trapped as well as INT/TERM. Without it, any `die` after a service
# has started would leave that service running with nothing supervising it.
_cleanup() {
    local code=$?
    trap - INT TERM EXIT
    set +e
    if [ "${#_PIDS[@]}" -gt 0 ]; then
        echo ""
        info "Received shutdown signal ..."
        _kill_all
        info "Goodbye!"
    fi
    exit "$code"
}
trap _cleanup INT TERM EXIT

# ==============================================================================
# ARGUMENTS
# ==============================================================================

while [ $# -gt 0 ]; do
    case "$1" in
        --backend-only)  RUN_FRONTEND=false ;;
        --frontend-only) RUN_BACKEND=false ;;
        --skip-install)  DO_INSTALL=false ;;
        --install-only)  START_SERVERS=false ;;
        --no-free-ports) FREE_PORTS=false ;;
        -h | --help)     usage ;;
        *) printf '%serror:%s unknown option: %s  (try --help)\n' "$RED" "$NC" "$1" >&2; exit 1 ;;
    esac
    shift
done

# ==============================================================================
# PREREQUISITE CHECKS
# ==============================================================================

# Find a working Python 3, bypassing the Microsoft Store stub on Windows.
#
# That stub is named python3.exe, sits on PATH, and exits without printing a
# version - so "does `python3` exist" is the wrong question. Asking each
# candidate for its version and requiring 3.10+ is what actually separates a
# usable interpreter from the stub.
_find_python() {
    local candidates=("python3" "py -3" "py" "python")
    local c base out
    for c in "${candidates[@]}"; do
        base="${c%% *}"          # "py -3" -> "py"
        have "$base" || continue
        out="$($c --version 2>&1 || true)"
        echo "$out" | grep -qiE "Python [0-9]+\.[0-9]+" 2>/dev/null || continue
        # 3.10+ is the floor: agent/config.py and the pinned wheels assume it.
        $c -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null || continue
        PYTHON="$base"
        PYTHON_ARGS="${c#"$base"}"
        PYTHON_VERSION="$out"
        return 0
    done
    return 1
}

_check_prerequisites() {
    step "Checking prerequisites"

    if [ "$RUN_BACKEND" = true ]; then
        if _find_python; then
            ok "Python: ${PYTHON_VERSION}"
        else
            die "Python 3.10+ is not installed or does not work. Get it from https://python.org and try again."
        fi
    fi

    if [ "$RUN_FRONTEND" = true ]; then
        if have node; then
            ok "Node.js: $(node --version 2>&1)"
        else
            die "Node.js is not installed. Install Node.js 18+ and try again."
        fi

        if have npm; then
            ok "npm: $(npm --version 2>&1)"
        else
            die "npm is not installed."
        fi
    fi

    # Not fatal: every wait falls back to /dev/tcp when neither is present.
    if have curl || have wget; then
        :
    else
        warn "Neither curl nor wget found - health checks fall back to a TCP probe."
    fi
}

# ==============================================================================
# HTTP-REQUEST HELPERS  (curl -> wget fallback -> /dev/tcp)
# ==============================================================================

# Is a URL reachable and returning success? 0 = yes.
_http_ok() {
    local url="$1"
    if have curl; then
        curl -sf --max-time 5 "$url" >/dev/null 2>&1 && return 0
    elif have wget; then
        wget -q --timeout=5 --spider "$url" >/dev/null 2>&1 && return 0
    fi
    return 1
}

# Is a TCP port accepting connections? Ignores HTTP status codes entirely.
_tcp_ok() {
    local host="$1" port="$2"
    (echo > "/dev/tcp/${host}/${port}") 2>/dev/null && return 0
    if have curl; then
        curl -s -o /dev/null --connect-timeout 5 "http://${host}:${port}/" >/dev/null 2>&1 && return 0
    fi
    return 1
}

# Kill whatever is listening on a TCP port.
#
# The common case is a uvicorn or vite left over from a previous run, and
# aborting the whole startup over it is unhelpful. What is killed is always
# named first, and --no-free-ports turns this into a hard stop instead.
_free_port() {
    local port="$1" label="$2"
    local pid=""

    if [ "$IS_WINDOWS" = true ]; then
        # netstat -ano: Proto  Local  Foreign  State  PID
        # Match on the *local* address and LISTENING only, so an outbound
        # connection to the same port number is never mistaken for a server.
        pid="$(netstat -ano 2>/dev/null \
            | awk -v p=":${port}$" '$1 ~ /^TCP/ && $2 ~ p && $4 == "LISTENING" { print $5; exit }')" || pid=""
    else
        if have lsof; then
            pid="$(lsof -ti :"$port" 2>/dev/null | head -1 || true)"
        fi
        if [ -z "$pid" ] && have ss; then
            pid="$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
        fi
    fi

    [ -n "$pid" ] && [ "$pid" != "0" ] || return 0

    if [ "$FREE_PORTS" != true ]; then
        die "Port ${port} (${label}) is in use by PID ${pid}, and --no-free-ports is set."
    fi

    warn "Port ${port} (${label}) is in use by PID ${pid}. Killing it ..."
    if [ "$IS_WINDOWS" = true ]; then
        taskkill //PID "$pid" //T //F >/dev/null 2>&1 || true
    else
        kill -9 "$pid" 2>/dev/null || true
    fi
    sleep 1
    ok "Freed port ${port}"
}

# ==============================================================================
# HEALTH / WAIT HELPERS
# ==============================================================================

# Print the tail of a service log, so a failed wait says *why* rather than
# only that it failed.
_tail_log() {
    local log_file="$1" lines="${2:-20}"
    [ -f "$log_file" ] || return 0
    printf '%s--- last %s lines of %s ---%s\n' "$DIM" "$lines" "$log_file" "$NC"
    tail -n "$lines" "$log_file" || true
    printf '%s--- end ---%s\n' "$DIM" "$NC"
}

# Wait for an HTTP URL to return success.
_wait_for_url() {
    local url="$1" name="$2"
    local max_attempts="${3:-24}" interval="${4:-5}" watch_pid="${5:-}"
    local attempt=1

    info "Waiting for ${name} at ${url} ..."
    while [ "$attempt" -le "$max_attempts" ]; do
        # No point waiting out the full timeout on a process that has died.
        if [ -n "$watch_pid" ] && ! kill -0 "$watch_pid" 2>/dev/null; then
            fail "${name} exited during startup."
            return 1
        fi
        if _http_ok "$url"; then
            ok "${name} is reachable at ${url}"
            return 0
        fi
        warn "  [${attempt}/${max_attempts}] ${name} not ready yet, retrying in ${interval}s ..."
        sleep "$interval"
        attempt=$((attempt + 1))
    done

    fail "${name} did not become reachable at ${url} within $((max_attempts * interval))s."
    return 1
}

# Wait for a TCP port to accept connections, for anything that does not serve
# a 200 on / - the Vite dev server among them, depending on version.
_wait_for_port() {
    local host="$1" port="$2" name="$3"
    local max_attempts="${4:-24}" interval="${5:-5}" watch_pid="${6:-}"
    local attempt=1

    info "Waiting for ${name} on port ${port} ..."
    while [ "$attempt" -le "$max_attempts" ]; do
        if [ -n "$watch_pid" ] && ! kill -0 "$watch_pid" 2>/dev/null; then
            fail "${name} exited during startup."
            return 1
        fi
        if _tcp_ok "$host" "$port"; then
            ok "${name} is running on port ${port}"
            return 0
        fi
        warn "  [${attempt}/${max_attempts}] ${name} not ready on port ${port}, retrying in ${interval}s ..."
        sleep "$interval"
        attempt=$((attempt + 1))
    done

    fail "${name} did not start on port ${port} within $((max_attempts * interval))s."
    return 1
}

# ==============================================================================
# LOGS DIRECTORY
# ==============================================================================

_create_logs_dir() {
    mkdir -p "$LOGS_DIR"
    local f
    for f in startup.log backend.log frontend.log; do
        : > "${LOGS_DIR}/${f}"
    done
}

# ==============================================================================
# BACKEND SETUP (venv + deps)
# ==============================================================================

_setup_backend_venv() {
    step "Setting up backend Python virtual environment"

    cd "$BACKEND_DIR"

    local venv_dir="" candidate
    for candidate in ".venv" "venv"; do
        if [ -d "$candidate" ] && { [ -f "$candidate/bin/activate" ] || [ -f "$candidate/Scripts/activate" ]; }; then
            venv_dir="$candidate"
            info "Found existing virtual environment: ${candidate}"
            break
        fi
    done

    if [ -z "$venv_dir" ]; then
        info "No virtual environment found. Creating .venv ..."
        $PYTHON ${PYTHON_ARGS:-} -m venv .venv \
            || die "python -m venv failed (on Debian/Ubuntu: apt install python3-venv)"
        venv_dir=".venv"
        ok "Created virtual environment: .venv"
    fi

    if [ -f "${venv_dir}/bin/activate" ]; then
        VENV_ACTIVATE="${venv_dir}/bin/activate"
    elif [ -f "${venv_dir}/Scripts/activate" ]; then
        VENV_ACTIVATE="${venv_dir}/Scripts/activate"
    else
        die "Cannot find an activate script in ${venv_dir}."
    fi

    # shellcheck source=/dev/null
    . "$VENV_ACTIVATE"

    # Activation that silently did not take is worth catching here rather than
    # discovering it as a missing module three steps later.
    local active_prefix
    active_prefix="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
    case "$active_prefix" in
        *"$venv_dir"* | *".venv"* | *"venv"*) ;;
        *) die "Activated ${VENV_ACTIVATE} but python still resolves to ${active_prefix:-nothing}." ;;
    esac

    VENV_DIR="$venv_dir"
    ok "Virtual environment activated: ${VENV_ACTIVATE}  (Python $(python -c 'import platform; print(platform.python_version())'))"
}

_install_backend_deps() {
    step "Installing backend Python dependencies"

    cd "$BACKEND_DIR"

    if [ "$DO_INSTALL" != true ]; then
        warn "--skip-install: leaving pip alone."
        return 0
    fi

    # Which requirements file follows from the profile, and this is not
    # cosmetic. config/local.json sets EMBEDDER=local, i.e. fastembed, and
    # uvicorn itself is a dev dependency - installing plain requirements.txt
    # locally would leave this script unable to start the server it just set up.
    local req
    case "$APP_ENV" in
        dev) req="requirements.txt" ;;
        *)   req="requirements-dev.txt" ;;
    esac
    [ -f "$req" ] || die "Missing ${BACKEND_DIR}/${req}"

    # pip is slow enough to notice on every start. Stamp the requirements
    # checksum after a successful install and skip while it matches; edit the
    # file and the next run reinstalls.
    local stamp="${VENV_DIR}/.deps-stamp"
    local want
    want="${req}:$(_checksum "$req")"
    if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
        ok "Dependencies already up to date (${req})."
        return 0
    fi

    info "Upgrading pip ..."
    python -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true

    info "Installing from ${req} - the first run pulls ~200MB of onnxruntime for the local embedder ..."
    python -m pip install -r "$req" || die "pip install failed."
    printf '%s' "$want" > "$stamp"
    ok "${req} installed."
}

# One checksum helper that works with whatever the machine has.
_checksum() {
    if   have sha256sum; then sha256sum "$1" | cut -d' ' -f1
    elif have shasum;    then shasum -a 256 "$1" | cut -d' ' -f1
    elif have md5sum;    then md5sum "$1" | cut -d' ' -f1
    else cksum "$1" | cut -d' ' -f1
    fi
}

# ==============================================================================
# ENVIRONMENT — secrets loaded from external sources only (never hardcoded)
# ==============================================================================

# There is no _export_defaults here on purpose. Every non-secret default -
# thresholds, model roles, which embedder, which store - lives in
# backend/config/<profile>.json, which is committed and is what
# agent/config.py reads. Duplicating those as exports would create a second
# source of truth that silently outranks the first.

# Read one key out of backend/.env. grep -E rather than sed's \? and \+, which
# GNU sed accepts and BSD sed (macOS) does not.
_env_file_value() {
    [ -f "$ENV_FILE" ] || return 1
    local line
    line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?$1[[:space:]]*=" "$ENV_FILE" | tail -n 1)" || return 1
    [ -n "$line" ] || return 1
    line="${line#*=}"
    line="${line#"${line%%[![:space:]]*}"}"   # ltrim
    line="${line%"${line##*[![:space:]]}"}"   # rtrim
    line="${line%\"}"; line="${line#\"}"
    line="${line%\'}"; line="${line#\'}"
    printf '%s' "$line"
}

# Secrets kept outside the workspace, where a coding assistant working in this
# repo cannot read them.
_load_secrets_file() {
    if [ -f "$SECRETS_FILE" ]; then
        set -a
        # shellcheck source=/dev/null
        . "$SECRETS_FILE"
        set +a
        return 0
    fi
    return 1
}

# Prompt for one secret, masked, interactive terminals only.
_prompt_secret() {
    local var_name="$1" prompt_msg="$2"
    local val
    read -r -s -p "  Enter ${prompt_msg} (Enter to skip): " val
    echo ""
    val="$(printf '%s' "$val" | tr -d '[:space:]')"
    [ -n "$val" ] || return 1
    export "${var_name}=${val}"
    return 0
}

# For each VAR:Description pair: environment first, then backend/.env, then
# ask. Anything entered is offered to backend/.env so the next run is silent.
_ensure_secrets() {
    local pairs=("$@")
    local pair var desc
    for pair in "${pairs[@]}"; do
        var="${pair%%:*}"
        desc="${pair#*:}"

        if [ -n "${!var:-}" ]; then
            ok "${var} is set (from the environment)."
            continue
        fi

        local from_file
        from_file="$(_env_file_value "$var" 2>/dev/null || true)"
        if [ -n "$from_file" ]; then
            export "${var}=${from_file}"
            ok "${var} is set (from backend/.env)."
            continue
        fi

        if [ -t 0 ]; then
            if _prompt_secret "$var" "$desc"; then
                _persist_secret "$var" "${!var}"
            else
                warn "${var} was left empty."
            fi
        else
            warn "${var} is not set (non-interactive terminal - skipped)."
        fi
    done
}

_persist_secret() {
    local var="$1" value="$2"

    # Start from the documented template when there is no .env yet, so the
    # file the user ends up with still explains every other setting.
    if [ ! -f "$ENV_FILE" ] && [ -f "${BACKEND_DIR}/.env.example" ]; then
        cp "${BACKEND_DIR}/.env.example" "$ENV_FILE"
    fi

    if [ -f "$ENV_FILE" ] && grep -qE "^[[:space:]]*${var}[[:space:]]*=" "$ENV_FILE"; then
        # sed -i differs between GNU and BSD, so write a temp file and move it.
        sed "s|^[[:space:]]*${var}[[:space:]]*=.*|${var}=${value}|" "$ENV_FILE" > "${ENV_FILE}.tmp" \
            && mv "${ENV_FILE}.tmp" "$ENV_FILE"
    else
        printf '%s=%s\n' "$var" "$value" >> "$ENV_FILE"
    fi
    ok "${var} saved to backend/.env (gitignored)."
}

_load_env() {
    step "Loading environment variables"

    # APP_ENV selects config/<profile>.json, and the profile decides which
    # requirements file gets installed, so it is resolved before pip runs.
    if [ -z "${APP_ENV:-}" ]; then
        APP_ENV="$(_env_file_value APP_ENV 2>/dev/null || true)"
    fi
    APP_ENV="${APP_ENV:-local}"
    export APP_ENV

    if [ -f "${BACKEND_DIR}/config/${APP_ENV}.json" ]; then
        info "Profile: APP_ENV=${APP_ENV}  (backend/config/${APP_ENV}.json)"
    else
        warn "Profile: APP_ENV=${APP_ENV} - backend/config/${APP_ENV}.json does not exist, defaults will be used."
    fi

    if _load_secrets_file; then
        ok "Secrets loaded from ${SECRETS_FILE}"
    else
        info "No ${SECRETS_FILE} found - falling back to backend/.env and prompts."
    fi

    if [ -f "$ENV_FILE" ]; then
        info "Reading backend/.env"
    else
        info "No backend/.env yet."
    fi

    # Only GROQ_API_KEY is asked for in the local profile. The rest are
    # genuinely optional here: retrieval, the metadata filter, the no_match
    # gate and the entire test suite run offline, and without a key /ask
    # answers 503 rather than guessing. The dev profile is the one that needs
    # Gemini and Supabase, so it asks for those too.
    local wanted=("GROQ_API_KEY:Groq API key - free at https://console.groq.com/keys")
    if [ "$APP_ENV" = "dev" ]; then
        wanted+=(
            "GEMINI_API_KEY:Google AI Studio key (embeddings for the dev profile)"
            "SUPABASE_URL:Supabase project URL"
            "SUPABASE_SERVICE_KEY:Supabase service key"
        )
    fi
    _ensure_secrets "${wanted[@]}"

    if [ -z "${GROQ_API_KEY:-}" ]; then
        warn "No GROQ_API_KEY: retrieval and no_match still work, but /ask will return 503."
    fi

    ok "Environment loaded."
}

_create_frontend_env() {
    step "Creating / updating frontend environment file (frontend/.env)"

    local api_url="http://localhost:${BACKEND_PORT}"

    if [ -f "${FRONTEND_DIR}/.env" ]; then
        if ! grep -qF "VITE_API_URL=" "${FRONTEND_DIR}/.env" 2>/dev/null; then
            printf 'VITE_API_URL=%s\n' "$api_url" >> "${FRONTEND_DIR}/.env"
            ok "frontend/.env updated with VITE_API_URL=${api_url}"
        else
            local current
            current="$(sed -n 's/^VITE_API_URL=//p' "${FRONTEND_DIR}/.env" | tail -n 1)"
            ok "frontend/.env already exists - keeping VITE_API_URL=${current}"
            # A mismatch here is the classic "why is the UI getting connection
            # refused" and is invisible otherwise.
            if [ "$current" != "$api_url" ]; then
                warn "The API will run on ${api_url}, but the frontend is pointed at ${current}."
            fi
        fi
        return
    fi

    printf 'VITE_API_URL=%s\n' "$api_url" > "${FRONTEND_DIR}/.env"
    ok "frontend/.env created (VITE_API_URL=${api_url})"
}

# ==============================================================================
# FRONTEND SETUP
# ==============================================================================

_setup_frontend() {
    step "Setting up frontend dependencies"

    cd "$FRONTEND_DIR"

    if [ "$DO_INSTALL" != true ]; then
        warn "--skip-install: leaving npm alone."
        return 0
    fi

    # npm rewrites node_modules/.package-lock.json on every install, so
    # comparing timestamps against it catches the case a bare -d check misses:
    # node_modules exists, but package.json changed since it was built.
    local marker="node_modules/.package-lock.json"
    if [ ! -d "node_modules" ]; then
        info "node_modules is missing - installing ..."
    elif [ ! -f "$marker" ]; then
        info "node_modules looks incomplete - reinstalling ..."
    elif [ "package-lock.json" -nt "$marker" ] || [ "package.json" -nt "$marker" ]; then
        info "Dependencies changed since the last install - reinstalling ..."
    else
        ok "node_modules exists and is current - skipping npm install."
        return 0
    fi

    npm install || die "npm install failed."
    ok "npm install completed."
}

# ==============================================================================
# BACKEND (FastAPI)
# ==============================================================================

_start_backend() {
    step "Starting Backend (FastAPI, port ${BACKEND_PORT})"

    _free_port "$BACKEND_PORT" "backend"

    cd "$BACKEND_DIR"

    # 127.0.0.1, not 0.0.0.0: this is a local development server and there is
    # no reason to expose it to the rest of the network.
    nohup python -m uvicorn app.main:app --reload --host 127.0.0.1 --port "$BACKEND_PORT" \
        > "$BACKEND_LOG" 2>&1 &
    local pid=$!
    _track_pid "backend" "$pid"
    ok "Backend started (PID ${pid}). Logs: ${BACKEND_LOG}"

    # 60 x 3s. The first local run downloads the bge-small model and builds the
    # index, which takes far longer than a warm start.
    if ! _wait_for_url "http://127.0.0.1:${BACKEND_PORT}/health" "Backend API" 60 3 "$pid"; then
        _tail_log "$BACKEND_LOG" 25
        die "The backend did not come up. See ${BACKEND_LOG}"
    fi
}

# ==============================================================================
# FRONTEND (Vite)
# ==============================================================================

_start_frontend() {
    step "Starting Frontend (Vite, port ${FRONTEND_PORT})"

    _free_port "$FRONTEND_PORT" "frontend"

    cd "$FRONTEND_DIR"

    # --strictPort so a busy port is an error rather than Vite quietly moving
    # to another one, which would leave the summary and the browser disagreeing.
    nohup npm run dev -- --port "$FRONTEND_PORT" --strictPort \
        > "$FRONTEND_LOG" 2>&1 &
    local pid=$!
    _track_pid "frontend" "$pid"
    ok "Frontend started (PID ${pid}). Logs: ${FRONTEND_LOG}"

    # A TCP probe, not an HTTP one: Vite binds ::1 as well as 127.0.0.1 and
    # which one answers first varies, so "is the port open" is the reliable
    # question here.
    if ! _wait_for_port "localhost" "$FRONTEND_PORT" "Frontend (Vite)" 30 2 "$pid"; then
        _tail_log "$FRONTEND_LOG" 25
        die "The frontend did not come up. See ${FRONTEND_LOG}"
    fi
}

# ==============================================================================
# STATUS SUMMARY
# ==============================================================================

_print_status() {
    local name="$1" url="$2" mode="${3:-http}" port="${4:-}"
    local up=false

    if [ "$mode" = "tcp" ]; then
        _tcp_ok "localhost" "$port" && up=true
    else
        _http_ok "$url" && up=true
    fi

    if [ "$up" = true ]; then
        printf '  %s%-18s %-38s OK  Running%s\n' "$GREEN" "$name" "$url" "$NC"
    else
        printf '  %s%-18s %-38s --  Not responding%s\n' "$YELLOW" "$name" "$url" "$NC"
    fi
}

_show_summary() {
    {
        printf '\n%s\n' "────────────────────────────────────────────────────────────────"
        printf '  %-18s %-38s %s\n' "Service" "URL" "Status"
        printf '  %-18s %-38s %s\n' "-------" "---" "------"
        if [ "$RUN_BACKEND" = true ]; then
            _print_status "Backend API" "http://localhost:${BACKEND_PORT}/health"
            _print_status "API docs"    "http://localhost:${BACKEND_PORT}/docs"
        fi
        if [ "$RUN_FRONTEND" = true ]; then
            _print_status "Frontend" "http://localhost:${FRONTEND_PORT}" tcp "$FRONTEND_PORT"
        fi
        printf '\n'
        printf '  %-18s %s\n' "Profile" "APP_ENV=${APP_ENV}"
        printf '  %-18s %s\n' "Logs" "${LOGS_DIR}"
        printf '\n'
    } | tee -a "$STARTUP_LOG"

    if [ "$RUN_FRONTEND" = true ]; then
        info "Open http://localhost:${FRONTEND_PORT} in your browser."
    fi
    info "Press CTRL+C to stop all services."
    echo ""
}

# ==============================================================================
# ── MAIN ──────────────────────────────────────────────────────────────────────
# ==============================================================================

printf '\n'
printf '%s%s╔══════════════════════════════════════════════════════════════╗%s\n' "$CYAN" "$BOLD" "$NC"
printf '%s%s║           Runbook Agent — Local Startup                      ║%s\n' "$CYAN" "$BOLD" "$NC"
printf '%s%s╚══════════════════════════════════════════════════════════════╝%s\n' "$CYAN" "$BOLD" "$NC"
printf '\n'

# 0.  Log directory (must come before any logging helper writes to it)
_create_logs_dir

# 1.  OS detection
_detect_os

# 2.  Prerequisites
_check_prerequisites

# 3.  Environment — profile, secrets file, backend/.env, prompts.
#     Before the venv, because APP_ENV decides which requirements file to use.
_load_env

# 4.  Backend Python virtual environment
if [ "$RUN_BACKEND" = true ]; then
    _setup_backend_venv
    _install_backend_deps
fi

# 5.  Frontend environment file and dependencies
if [ "$RUN_FRONTEND" = true ]; then
    _create_frontend_env
    _setup_frontend
fi

if [ "$START_SERVERS" != true ]; then
    step "Setup complete (--install-only, nothing started)"
    exit 0
fi

# 6.  Backend (port 8000)
if [ "$RUN_BACKEND" = true ]; then
    _start_backend
fi

# 7.  Frontend (port 5173)
if [ "$RUN_FRONTEND" = true ]; then
    _start_frontend
fi

# 8.  Summary
_show_summary

# ── Idle (wait for CTRL+C) ───────────────────────────────────────────────────
#
# Not a bare `sleep` loop: a service that dies here would otherwise leave the
# script sitting there reporting nothing while the summary above goes stale.
info "All services running. Waiting for shutdown signal ..."
while true; do
    for (( i=0; i<${#_PIDS[@]}; i++ )); do
        if ! kill -0 "${_PIDS[$i]}" 2>/dev/null; then
            fail "${_PID_NAMES[$i]} (PID ${_PIDS[$i]}) has exited."
            case "${_PID_NAMES[$i]}" in
                backend)  _tail_log "$BACKEND_LOG" 25 ;;
                frontend) _tail_log "$FRONTEND_LOG" 25 ;;
            esac
            exit 1
        fi
    done
    sleep 5
done
