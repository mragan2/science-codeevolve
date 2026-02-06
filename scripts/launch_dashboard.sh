#!/usr/bin/env bash
set -u
set -o pipefail

APP_ROOT="/home/rag/Projects/science-codeevolve"
LOG_DIR="${APP_ROOT}/logs"
LOG_FILE="${LOG_DIR}/dashboard.log"
CONDA_SH="/home/rag/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="codeevolve"

mkdir -p "${LOG_DIR}"

{
  echo "---- $(date '+%Y-%m-%d %H:%M:%S') ----"
  echo "Launcher: ${0}"
  echo "APP_ROOT: ${APP_ROOT}"
  echo "USER: ${USER}"
} >> "${LOG_FILE}"

cd "${APP_ROOT}" || {
  echo "ERROR: Failed to cd to ${APP_ROOT}" >> "${LOG_FILE}"
  exit 1
}

if [ -f "${CONDA_SH}" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  if ! conda activate "${ENV_NAME}"; then
    echo "ERROR: conda activate ${ENV_NAME} failed" >> "${LOG_FILE}"
    if command -v notify-send >/dev/null 2>&1; then
      notify-send "CodeEvolve Dashboard" "Failed to activate conda env '${ENV_NAME}'. See log: ${LOG_FILE}"
    fi
    exit 1
  fi
else
  echo "ERROR: conda.sh not found at ${CONDA_SH}" >> "${LOG_FILE}"
  if command -v notify-send >/dev/null 2>&1; then
    notify-send "CodeEvolve Dashboard" "Conda not found. See log: ${LOG_FILE}"
  fi
  exit 1
fi

export PYTHONUNBUFFERED=1
if [ -z "${TERM:-}" ]; then
  export TERM="xterm-256color"
fi
python "${APP_ROOT}/scripts/problem_dashboard.py" >> "${LOG_FILE}" 2>&1
rc=$?
echo "Exit code: ${rc}" >> "${LOG_FILE}"
exit "${rc}"
