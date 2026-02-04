#!/bin/bash
PROB_NAME="elegans"
BASE_DIR="problems/${PROB_NAME}"
INPT_DIR="${BASE_DIR}/input"
CFG_PATH="${BASE_DIR}/configs/config.yaml"
OUT_DIR="experiments/${PROB_NAME}/run1"
LOAD_CKPT="0"
CPU_LIST="0-7"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export CE_ATLAS_DIR="${BASE_DIR}/data"

taskset --cpu-list $CPU_LIST codeevolve --inpt_dir=$INPT_DIR --cfg_path=$CFG_PATH --out_dir=$OUT_DIR --load_ckpt=$LOAD_CKPT --terminal_logging
