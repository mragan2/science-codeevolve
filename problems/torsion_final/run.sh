#!/bin/bash
# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file provides a template for executing CodeEvolve in the terminal using bash.
#
# ===--------------------------------------------------------------------------------------===#

PROB_NAME="torsion_final"
BASE_DIR="problems/${PROB_NAME}"
INPT_DIR="${BASE_DIR}/input"
CFG_PATH="${BASE_DIR}/configs/config.yaml"
<<<<<<< HEAD
OUT_DIR="experiments/${PROB_NAME}/run_3"
=======
OUT_DIR="experiments/${PROB_NAME}/MR_$(date +%Y%m%d)"
>>>>>>> d2d13a86f765e0ca72d1cf4e845120ad4bae373b
LOAD_CKPT="0"
CPU_LIST="0-7"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

taskset --cpu-list $CPU_LIST codeevolve --inpt_dir=$INPT_DIR --cfg_path=$CFG_PATH --out_dir=$OUT_DIR --load_ckpt=$LOAD_CKPT --terminal_logging
