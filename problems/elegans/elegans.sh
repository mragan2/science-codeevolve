#!/usr/bin/env bash
set -euo pipefail

PROB_NAME="elegans"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROB_DIR="${REPO_ROOT}/problems/${PROB_NAME}"
INPT_DIR="${PROB_DIR}/input"
CFG_PATH_DEFAULT="${PROB_DIR}/configs/config.yaml"
CFG_PATH="${CFG_PATH:-${CFG_PATH_DEFAULT}}"
EXP_DIR="${REPO_ROOT}/experiments/${PROB_NAME}"
VIS_PATH="${PROB_DIR}/visualize_elegans.py"
PYTHON_BIN="${PYTHON:-python}"

usage() {
  cat <<'USAGE'
Usage: elegans.sh <command> [options]

Commands:
  run [--run runN|N] [--next] [--cpu LIST] [--load-ckpt N] [--cfg-path PATH]
      [--skip-dryad|--require-dryad] [--no-taskset]
  warmstart <runN|N> [island]
  winner [args...]
  analyze
  viz <runN|N> [island]
  tail <runN|N> [island]
  ls
  help
USAGE
}

err() {
  echo "Error: $*" >&2
  exit 1
}

require_env() {
  if [[ -z "${API_KEY:-}" || -z "${API_BASE:-}" ]]; then
    err "Export API_KEY and API_BASE before running CodeEvolve."
  fi
}

ensure_paths() {
  [[ -d "${INPT_DIR}" ]] || err "Input directory not found: ${INPT_DIR}"
  [[ -f "${CFG_PATH}" ]] || err "Config file not found: ${CFG_PATH}"
}

normalize_run() {
  local run_id="$1"
  if [[ -z "${run_id}" ]]; then
    err "Run id is required."
  fi
  if [[ "${run_id}" =~ ^[0-9]+$ ]]; then
    echo "run${run_id}"
  else
    echo "${run_id}"
  fi
}

next_run_name() {
  local max=0
  if [[ -d "${EXP_DIR}" ]]; then
    for d in "${EXP_DIR}"/run*; do
      [[ -d "${d}" ]] || continue
      local base
      base="$(basename "${d}")"
      if [[ "${base}" =~ ^run([0-9]+)$ ]]; then
        local n="${BASH_REMATCH[1]}"
        if (( n > max )); then
          max=${n}
        fi
      fi
    done
  fi
  echo "run$((max + 1))"
}

cmd_run() {
  local run_id=""
  local cpu_list="${CPU_LIST:-0-7}"
  local load_ckpt=0
  local skip_dryad=1
  local use_taskset=1
  local cfg_path="${CFG_PATH}"

  if [[ -n "${CE_SKIP_DRYAD:-}" ]]; then
    case "${CE_SKIP_DRYAD,,}" in
      0|false|no)
        skip_dryad=0
        ;;
      *)
        skip_dryad=1
        ;;
    esac
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run)
        run_id="$2"
        shift 2
        ;;
      --next)
        run_id=""
        shift
        ;;
      --cpu)
        cpu_list="$2"
        shift 2
        ;;
      --load-ckpt)
        load_ckpt="$2"
        shift 2
        ;;
      --cfg-path)
        cfg_path="$2"
        shift 2
        ;;
      --skip-dryad)
        skip_dryad=1
        shift
        ;;
      --require-dryad)
        skip_dryad=0
        shift
        ;;
      --no-taskset)
        use_taskset=0
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        err "Unknown option for run: $1"
        ;;
    esac
  done

  CFG_PATH="${cfg_path}"
  ensure_paths
  require_env

  if [[ -z "${run_id}" ]]; then
    run_id="$(next_run_name)"
  fi
  run_id="$(normalize_run "${run_id}")"

  local out_dir="${EXP_DIR}/${run_id}"

  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
  export CE_ATLAS_DIR="${PROB_DIR}/data"

  if [[ "${skip_dryad}" -eq 1 ]]; then
    export CE_SKIP_DRYAD=1
  else
    unset CE_SKIP_DRYAD
  fi

  local cmd=()
  if command -v codeevolve >/dev/null 2>&1; then
    cmd+=(codeevolve)
  else
    # Allow running from a source checkout without installing the console script.
    cmd+=("${PYTHON_BIN}" -m codeevolve.cli)
  fi
  cmd+=(
    --inpt_dir="${INPT_DIR}"
    --cfg_path="${CFG_PATH}"
    --out_dir="${out_dir}"
    --load_ckpt="${load_ckpt}"
    --terminal_logging
  )

  if [[ "${use_taskset}" -eq 1 ]] && command -v taskset >/dev/null 2>&1; then
    cmd=(taskset --cpu-list "${cpu_list}" "${cmd[@]}")
  fi

  echo "Starting ${run_id}"
  echo "Out dir: ${out_dir}"
  "${cmd[@]}"
}

cmd_warmstart() {
  local run_id
  run_id="$(normalize_run "${1:-}")"
  local island="${2:-0}"
  local src="${EXP_DIR}/${run_id}/${island}/best_sol.py"
  local dest="${INPT_DIR}/src/init_program.py"

  [[ -f "${src}" ]] || err "best_sol.py not found: ${src}"
  [[ -f "${dest}" ]] || err "init_program.py not found: ${dest}"

  cp "${src}" "${dest}"
  echo "Warm-started from ${src} -> ${dest}"
}

cmd_analyze() {
  ensure_paths
  "${PYTHON_BIN}" "${REPO_ROOT}/analyze_evolution.py" --all "${EXP_DIR}"
}

cmd_winner() {
  (cd "${PROB_DIR}" && "${PYTHON_BIN}" "${PROB_DIR}/find_winner.py" "$@")
}

cmd_viz() {
  local run_id
  run_id="$(normalize_run "${1:-}")"
  local island="${2:-0}"
  local run_dir="${EXP_DIR}/${run_id}/${island}"
  local best="${run_dir}/best_sol.py"

  [[ -f "${best}" ]] || err "best_sol.py not found: ${best}"
  [[ -f "${VIS_PATH}" ]] || err "Visualizer not found: ${VIS_PATH}"

  (cd "${run_dir}" && "${PYTHON_BIN}" best_sol.py | "${PYTHON_BIN}" "${VIS_PATH}")
}

cmd_tail() {
  local run_id
  run_id="$(normalize_run "${1:-}")"
  local island="${2:-0}"
  local log_path="${EXP_DIR}/${run_id}/${island}/results.log"

  [[ -f "${log_path}" ]] || err "results.log not found: ${log_path}"
  tail -f "${log_path}"
}

cmd_ls() {
  if [[ ! -d "${EXP_DIR}" ]]; then
    echo "No experiments directory: ${EXP_DIR}"
    return 0
  fi
  ls -1 "${EXP_DIR}"
}

main() {
  local cmd="${1:-help}"
  shift || true

  case "${cmd}" in
    run)
      cmd_run "$@"
      ;;
    warmstart)
      cmd_warmstart "$@"
      ;;
    analyze)
      cmd_analyze "$@"
      ;;
    winner)
      cmd_winner "$@"
      ;;
    viz|visualize)
      cmd_viz "$@"
      ;;
    tail)
      cmd_tail "$@"
      ;;
    ls|list)
      cmd_ls
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      err "Unknown command: ${cmd}"
      ;;
  esac
}

main "$@"
