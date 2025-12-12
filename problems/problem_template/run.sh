#!/bin/bash
# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# Generic template for running CodeEvolve on any project in the problems directory.
#
# BEST PRACTICE: Copy this to your project folder (problems/YOUR_PROJECT/run.sh)
# This keeps everything self-contained and portable.
#
# Usage:
#   1. Copy this template to your project directory:
#      cp problems/run_template.sh problems/YOUR_PROJECT/run.sh
#   2. Edit and set PROJECT_NAME to your project path (relative to problems/)
#   3. Adjust CONFIG_NAME if using a different config file
#   4. Run from your project folder:
#      cd problems/YOUR_PROJECT && bash run.sh
#
# ===--------------------------------------------------------------------------------------===#

# ==================================
# CONFIGURATION - EDIT THESE VALUES
# ==================================

# Project name relative to the problems/ directory
# Examples:
#   - "F_time"
#   - "alphaevolve_math_problems/circle_packing_square/26"
#   - "problem_template"
PROJECT_NAME="problem_template"

# Config file name (without .yaml extension)
# Common options: config, config_mp_insp, config_insp, config_mp, config_no_evolve
CONFIG_NAME="config_mp_insp"

# Output directory name (will be created under experiments/)
OUTPUT_NAME="run_$(date +%Y%m%d_%H%M%S)"

# Checkpoint to load (-1 for no checkpoint, or epoch number to resume from)
LOAD_CKPT=-1

# CPU affinity (leave empty for no restriction, or specify like "0-7" or "0,2,4,6")
CPU_LIST=""

# ==================================
# API CONFIGURATION (OPTIONAL)
# ==================================
# You can set API credentials here or use environment variables
# If set here, they will override environment variables

# Option 1: Set API key directly (NOT RECOMMENDED for shared/public projects)
# API_KEY="your-api-key-here"
# API_BASE="https://api.openai.com/v1"

# Option 2: Use environment variables (RECOMMENDED)
# Leave commented out to use existing environment variables
# Or set them here to override:
# export API_KEY="${API_KEY:-your-default-key}"
# export API_BASE="${API_BASE:-https://api.openai.com/v1}"

# Option 3: Load from external file (MOST SECURE)
# Create a file with: export API_KEY="..." and export API_BASE="..."
# Then uncomment the line below:
# source ~/.codeevolve_api_keys

# ==================================
# AUTOMATIC PATH SETUP - DO NOT EDIT
# ==================================

# Get the absolute path to the science-codeevolve directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Construct paths based on the standard project structure:
# - init_program.py is always in: problems/PROJECT_NAME/input/src/
# - evaluate.py is always in: problems/PROJECT_NAME/input/
# - config.yaml is in: problems/PROJECT_NAME/configs/
BASE_DIR="${REPO_ROOT}/problems/${PROJECT_NAME}"
INPT_DIR="${BASE_DIR}/input/"
CFG_PATH="${BASE_DIR}/configs/${CONFIG_NAME}.yaml"
OUT_DIR="${REPO_ROOT}/experiments/${PROJECT_NAME}/${OUTPUT_NAME}"

# ==================================
# VALIDATION
# ==================================

echo "======================================"
echo "CodeEvolve Run Configuration"
echo "======================================"
echo "Project Name:    ${PROJECT_NAME}"
echo "Input Directory: ${INPT_DIR}"
echo "Config File:     ${CFG_PATH}"
echo "Output Directory: ${OUT_DIR}"
echo "Load Checkpoint: ${LOAD_CKPT}"
echo "CPU List:        ${CPU_LIST:-'(all CPUs)'}"
echo "======================================"
echo ""

# Check if required directories and files exist
if [ ! -d "${INPT_DIR}" ]; then
    echo "ERROR: Input directory does not exist: ${INPT_DIR}"
    echo "Expected structure: problems/${PROJECT_NAME}/input/"
    exit 1
fi

if [ ! -f "${CFG_PATH}" ]; then
    echo "ERROR: Config file does not exist: ${CFG_PATH}"
    echo "Available configs in ${BASE_DIR}/configs/:"
    ls -1 "${BASE_DIR}/configs/" 2>/dev/null || echo "  (directory not found)"
    exit 1
fi

if [ ! -f "${INPT_DIR}/evaluate.py" ]; then
    echo "ERROR: evaluate.py not found in ${INPT_DIR}"
    echo "Expected: ${INPT_DIR}/evaluate.py"
    exit 1
fi

if [ ! -f "${INPT_DIR}/src/init_program.py" ]; then
    echo "WARNING: init_program.py not found in ${INPT_DIR}/src/"
    echo "Expected: ${INPT_DIR}/src/init_program.py"
fi

# Check if codeevolve command is available
if ! command -v codeevolve &> /dev/null; then
    echo "ERROR: codeevolve command not found. Please install the package:"
    echo "  pip install -e ."
    exit 1
fi

# Create output directory
mkdir -p "${OUT_DIR}"

# ==================================
# API KEY SETUP
# ==================================

# Export API keys if they were set in the configuration section above
if [ ! -z "${API_KEY}" ]; then
    export API_KEY
    echo "Using API_KEY from run script configuration"
fi

if [ ! -z "${API_BASE}" ]; then
    export API_BASE
    echo "Using API_BASE from run script: ${API_BASE}"
fi

# Check if API keys are available (from any source)
if [ -z "${API_KEY}" ]; then
    echo "WARNING: API_KEY is not set. The run may fail if your LLM requires authentication."
    echo "Set it via:"
    echo "  1. Environment variable: export API_KEY='your-key'"
    echo "  2. In this run.sh file (see API CONFIGURATION section)"
    echo "  3. External file: source ~/.codeevolve_api_keys"
    echo ""
fi

# ==================================
# RUN CODEEVOLVE
# ==================================

echo "Starting CodeEvolve..."
echo ""

if [ -n "${CPU_LIST}" ]; then
    # Run with CPU affinity
    taskset --cpu-list "${CPU_LIST}" codeevolve \
        --inpt_dir="${INPT_DIR}" \
        --cfg_path="${CFG_PATH}" \
        --out_dir="${OUT_DIR}" \
        --load_ckpt="${LOAD_CKPT}" \
        --terminal_logging
else
    # Run without CPU affinity
    codeevolve \
        --inpt_dir="${INPT_DIR}" \
        --cfg_path="${CFG_PATH}" \
        --out_dir="${OUT_DIR}" \
        --load_ckpt="${LOAD_CKPT}" \
        --terminal_logging
fi

# ==================================
# COMPLETION
# ==================================

EXIT_CODE=$?
echo ""
echo "======================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "CodeEvolve completed successfully!"
    echo "Results saved to: ${OUT_DIR}"
else
    echo "CodeEvolve exited with error code: ${EXIT_CODE}"
fi
echo "======================================"

exit ${EXIT_CODE}
