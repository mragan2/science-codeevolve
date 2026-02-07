#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_ROOT="${APP_ROOT:-${APP_ROOT_DEFAULT}}"
LOG_DIR="${LOG_DIR:-${APP_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/dashboard.log}"
LOCK_FILE="${LOCK_FILE:-${LOG_DIR}/dashboard.lock}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-codeevolve}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DASHBOARD_PY="${APP_ROOT}/scripts/problem_dashboard.py"

mkdir -p "${LOG_DIR}"

{
  echo "---- $(date '+%Y-%m-%d %H:%M:%S') ----"
  echo "Launcher: ${0}"
  echo "APP_ROOT: ${APP_ROOT}"
  echo "USER: ${USER}"
  echo "ENV_NAME: ${ENV_NAME}"
  echo "PYTHON_BIN: ${PYTHON_BIN}"
} >> "${LOG_FILE}"

if [ ! -f "${DASHBOARD_PY}" ]; then
  echo "ERROR: Dashboard script not found: ${DASHBOARD_PY}" >> "${LOG_FILE}"
  exit 1
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "INFO: Dashboard lock active at ${LOCK_FILE}; another instance is likely running." >> "${LOG_FILE}"
    if command -v notify-send >/dev/null 2>&1; then
      notify-send "CodeEvolve Dashboard" "Dashboard already running (lock: ${LOCK_FILE})"
    fi
    exit 0
  fi
else
  echo "WARN: 'flock' not found; single-instance lock disabled." >> "${LOG_FILE}"
fi

cd "${APP_ROOT}" || {
  echo "ERROR: Failed to cd to ${APP_ROOT}" >> "${LOG_FILE}"
  exit 1
}

if [ -f "${CONDA_SH}" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  if ! conda activate "${ENV_NAME}"; then
    echo "WARN: conda activate ${ENV_NAME} failed; continuing with current shell python." >> "${LOG_FILE}"
    if command -v notify-send >/dev/null 2>&1; then
      notify-send "CodeEvolve Dashboard" "Conda env '${ENV_NAME}' activation failed. Falling back to current python."
    fi
  fi
else
  echo "WARN: conda.sh not found at ${CONDA_SH}; using current shell python." >> "${LOG_FILE}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python binary not found in PATH: ${PYTHON_BIN}" >> "${LOG_FILE}"
  exit 1
fi

export PYTHONUNBUFFERED=1
if [ -z "${TERM:-}" ]; then
  export TERM="xterm-256color"
fi
"${PYTHON_BIN}" "${DASHBOARD_PY}" >> "${LOG_FILE}" 2>&1
rc=$?
echo "Exit code: ${rc}" >> "${LOG_FILE}"
exit "${rc}"
