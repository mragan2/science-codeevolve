#!/usr/bin/env python
"""
CodeEvolve Dashboard — dark-themed local GUI for running problem helpers.

Uses only the Python stdlib (Tkinter) so it works out-of-the-box.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import queue
import difflib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# =============================================================================
# Theme — Catppuccin Mocha
# =============================================================================
class C:
    """Colour palette."""
    BASE      = "#1e1e2e"
    MANTLE    = "#181825"
    CRUST     = "#11111b"
    SURFACE0  = "#313244"
    SURFACE1  = "#45475a"
    SURFACE2  = "#585b70"
    OVERLAY0  = "#6c7086"
    TEXT      = "#cdd6f4"
    SUBTEXT0  = "#a6adc8"
    SUBTEXT1  = "#bac2de"
    BLUE      = "#89b4fa"
    GREEN     = "#a6e3a1"
    RED       = "#f38ba8"
    YELLOW    = "#f9e2af"
    PEACH     = "#fab387"
    TEAL      = "#94e2d5"
    MAUVE     = "#cba6f7"
    PINK      = "#f5c2e7"
    SKY       = "#89dceb"
    LAVENDER  = "#b4befe"
    ROSEWATER = "#f5e0dc"
    FLAMINGO  = "#f2cdcd"

FONT_UI    = ("Ubuntu Sans", 11)
FONT_UI_B  = ("Ubuntu Sans", 11, "bold")
FONT_MONO  = ("Ubuntu Sans Mono", 10)
FONT_MONO_SM = ("Ubuntu Sans Mono", 9)
FONT_TITLE = ("Ubuntu Sans", 13, "bold")

# Known dropdown options for categorical config parameters
DROPDOWN_OPTIONS: dict[str, list[str]] = {
    "selection_policy": ["roulette", "tournament", "truncation", "epsilon_greedy"],
    "migration_topology": ["ring", "full", "random"],
    "type": ["PlateauScheduler", "CosineScheduler", "StepScheduler", "LinearScheduler"],
    "elite_map_type": ["grid", "cvt"],
    "fitness_key": ["combined_score", "fitness", "delta_bic", "chi2_total", "chi2_reduced"],
}

# Config keys handled elsewhere or not suitable for inline editing
SKIP_KEYS = frozenset({
    "SYS_MSG", "CODEBASE_PATH", "INIT_FILE_DATA", "EVAL_FILE_NAME",
    "EXPLORATION_ENSEMBLE", "EXPLOITATION_ENSEMBLE", "SAMPLER_AUX_LM",
    "EMBEDDING", "mp_start_marker", "mp_end_marker",
    "evolve_start_marker", "evolve_end_marker",
})

# Models tab constants
_NONE_MODEL = "(none)"
_LLM_FIELDS = (
    "model_name",
    "temp",
    "top_p",
    "max_tok",
    "retries",
    "request_timeout_s",
    "weight",
    "verify_ssl",
)
_LLM_DEFAULTS: dict[str, object] = {
    "model_name": _NONE_MODEL,
    "temp": 0.5,
    "top_p": 0.9,
    "max_tok": 4096,
    "retries": 3,
    "request_timeout_s": 240.0,
    "weight": 0.33,
    "verify_ssl": False,
}
_EMBED_FIELDS = ("model_name", "retries", "request_timeout_s", "verify_ssl")

RUN_SORT_OPTIONS = ("Activity", "Best Score", "Name")
RUN_STATUS_OPTIONS = ("All", "LIVE", "WARM", "IDLE", "NEW")
VIZ_VIEW_OPTIONS = ("Branching", "Performance", "List", "MAP-Elites")
VIZ_HIGHLIGHT_OPTIONS = ("Top score", "Improvement", "Migration", "Recent")

# ---- New-problem wizard templates ----
_EVAL_TEMPLATE = '''\
"""
Evaluator for {name}.
Interface: python evaluate.py <candidate.py> <results.json>
"""
import importlib.util
import sys
import json
import math


def evaluate(candidate_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("candidate", candidate_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        return {{"fitness": 0.0, "error": f"import: {{exc}}"}}

    # --- YOUR EVALUATION LOGIC HERE ---
    fitness = 0.0
    return {{"fitness": fitness}}


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <candidate.py> <results.json>")
        return 1
    metrics = evaluate(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_INIT_TEMPLATE = '''\
import math

# EVOLVE-BLOCK-START


def get_params():
    return {}


def compute(params):
    """Replace with your solution logic."""
    return 0.0


# EVOLVE-BLOCK-END
'''

_SYSMSG_TEMPLATE = '''\
# PROMPT-BLOCK-START
You are optimizing a Python program.

HARD CONSTRAINTS (violation = fitness 0):
1. ONLY allowed import: "import math" - do NOT add numpy, scipy, etc.
2. Do not add new "import" lines
3. Only ASCII characters - no Unicode, no Greek letters

STRATEGIES:
- Modify the compute() function to improve fitness
- Tune parameters in get_params()
- ONE structural change OR 2-3 related tweaks per mutation
# PROMPT-BLOCK-END\
'''

_SH_TEMPLATE = '''\
#!/usr/bin/env bash
set -euo pipefail

PROB_NAME="{name}"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../.." && pwd)"
PROB_DIR="${{REPO_ROOT}}/problems/${{PROB_NAME}}"
INPT_DIR="${{PROB_DIR}}/input"
CFG_PATH_DEFAULT="${{PROB_DIR}}/configs/config.yaml"
CFG_PATH="${{CFG_PATH:-${{CFG_PATH_DEFAULT}}}}"
EXP_DIR="${{REPO_ROOT}}/experiments/${{PROB_NAME}}"
PYTHON_BIN="${{PYTHON:-python}}"

usage() {{
  cat <<'USAGE'
Usage: {name}.sh <command> [options]

Commands:
  run [--run runN|N] [--next] [--cpu LIST] [--load-ckpt N] [--cfg-path PATH] [--no-taskset]
  warmstart <runN|N> [island]
  winner [args...]
  analyze
  tail <runN|N> [island]
  ls
  help
USAGE
}}

err() {{
  echo "Error: $*" >&2
  exit 1
}}

require_env() {{
  if [[ -z "${{API_KEY:-}}" || -z "${{API_BASE:-}}" ]]; then
    err "Export API_KEY and API_BASE before running CodeEvolve."
  fi
}}

ensure_paths() {{
  [[ -d "${{INPT_DIR}}" ]] || err "Input directory not found: ${{INPT_DIR}}"
  [[ -f "${{CFG_PATH}}" ]] || err "Config file not found: ${{CFG_PATH}}"
}}

normalize_run() {{
  local run_id="$1"
  if [[ -z "${{run_id}}" ]]; then
    err "Run id is required."
  fi
  if [[ "${{run_id}}" =~ ^[0-9]+$ ]]; then
    if [[ -d "${{EXP_DIR}}/run_${{run_id}}" ]]; then
      echo "run_${{run_id}}"
    else
      echo "run${{run_id}}"
    fi
  else
    echo "${{run_id}}"
  fi
}}

next_run_name() {{
  local max=0
  if [[ -d "${{EXP_DIR}}" ]]; then
    for d in "${{EXP_DIR}}"/run*; do
      [[ -d "${{d}}" ]] || continue
      local base
      base="$(basename "${{d}}")"
      if [[ "${{base}}" =~ ^run([0-9]+)$ ]]; then
        local n="${{BASH_REMATCH[1]}}"
        if (( n > max )); then
          max=${{n}}
        fi
      fi
    done
  fi
  echo "run$((max + 1))"
}}

cmd_run() {{
  local run_id=""
  local cpu_list="${{CPU_LIST:-0-7}}"
  local load_ckpt=0
  local use_taskset=1
  local cfg_path="${{CFG_PATH}}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run) run_id="$2"; shift 2 ;;
      --next) run_id=""; shift ;;
      --cpu) cpu_list="$2"; shift 2 ;;
      --load-ckpt) load_ckpt="$2"; shift 2 ;;
      --cfg-path) cfg_path="$2"; shift 2 ;;
      --no-taskset) use_taskset=0; shift ;;
      -h|--help) usage; exit 0 ;;
      *) err "Unknown option for run: $1" ;;
    esac
  done

  CFG_PATH="${{cfg_path}}"
  ensure_paths
  require_env

  if [[ -z "${{run_id}}" ]]; then
    run_id="$(next_run_name)"
  fi
  run_id="$(normalize_run "${{run_id}}")"

  local out_dir="${{EXP_DIR}}/${{run_id}}"

  export PYTHONPATH="${{REPO_ROOT}}/src${{PYTHONPATH:+:$PYTHONPATH}}"
  if [[ -z "${{CODEEVOLVE_POST_CKPT_CMD:-}}" ]]; then
    export CODEEVOLVE_POST_CKPT_CMD="${{PYTHON_BIN}} ${{REPO_ROOT}}/scripts/extract_best_from_run.py --run-dir ${{out_dir}} --write-islands"
    export CODEEVOLVE_POST_CKPT_CWD="${{REPO_ROOT}}"
  fi

  local cmd=()
  if command -v codeevolve >/dev/null 2>&1; then
    cmd+=(codeevolve)
  else
    cmd+=("${{PYTHON_BIN}}" -m codeevolve.cli)
  fi
  cmd+=(
    --inpt_dir="${{INPT_DIR}}"
    --cfg_path="${{CFG_PATH}}"
    --out_dir="${{out_dir}}"
    --load_ckpt="${{load_ckpt}}"
    --terminal_logging
  )

  if [[ "${{use_taskset}}" -eq 1 ]] && command -v taskset >/dev/null 2>&1; then
    cmd=(taskset --cpu-list "${{cpu_list}}" "${{cmd[@]}}")
  fi

  echo "Starting ${{run_id}}"
  echo "Out dir: ${{out_dir}}"
  "${{cmd[@]}}"
}}

cmd_warmstart() {{
  local run_id
  run_id="$(normalize_run "${{1:-}}")"
  local island="${{2:-0}}"
  local src="${{EXP_DIR}}/${{run_id}}/${{island}}/best_sol.py"
  local dest="${{INPT_DIR}}/src/init_program.py"
  [[ -f "${{src}}" ]] || err "best_sol.py not found: ${{src}}"
  [[ -f "${{dest}}" ]] || err "init_program.py not found: ${{dest}}"
  cp "${{src}}" "${{dest}}"
  echo "Warm-started from ${{src}} -> ${{dest}}"
}}

cmd_analyze() {{
  ensure_paths
  "${{PYTHON_BIN}}" "${{REPO_ROOT}}/analyze_evolution.py" --all "${{EXP_DIR}}"
}}

cmd_winner() {{
  (cd "${{PROB_DIR}}" && "${{PYTHON_BIN}}" "${{PROB_DIR}}/find_winner.py" "$@")
}}

cmd_tail() {{
  local run_id
  run_id="$(normalize_run "${{1:-}}")"
  local island="${{2:-0}}"
  local log_dir="${{EXP_DIR}}/${{run_id}}/${{island}}"
  local log_path=""
  if [[ -f "${{log_dir}}/island.log" ]]; then
    log_path="${{log_dir}}/island.log"
  elif [[ -f "${{log_dir}}/results.log" ]]; then
    log_path="${{log_dir}}/results.log"
  else
    err "No log found in ${{log_dir}} (checked island.log, results.log)"
  fi
  tail -f "${{log_path}}"
}}

cmd_ls() {{
  if [[ ! -d "${{EXP_DIR}}" ]]; then
    echo "No experiments directory: ${{EXP_DIR}}"
    return 0
  fi
  ls -1 "${{EXP_DIR}}"
}}

main() {{
  local cmd="${{1:-help}}"
  shift || true

  case "${{cmd}}" in
    run) cmd_run "$@" ;;
    warmstart) cmd_warmstart "$@" ;;
    analyze) cmd_analyze "$@" ;;
    winner) cmd_winner "$@" ;;
    tail) cmd_tail "$@" ;;
    ls|list) cmd_ls ;;
    help|-h|--help) usage ;;
    *) err "Unknown command: ${{cmd}}" ;;
  esac
}}

main "$@"
'''

_FIND_WINNER_TEMPLATE = '''\
import os
import glob
import subprocess
import json
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEARCH_DIRS = [
    os.path.join(SCRIPT_DIR, "../../experiments/{name}")
]
EVALUATOR_SCRIPT = os.path.join(SCRIPT_DIR, "input/evaluate.py")

WINNER_COPY_PATH = os.path.join(SCRIPT_DIR, "FINAL_BEST_SOL.py")
WINNER_RUN_OUTPUT = os.path.join(SCRIPT_DIR, "WINNER_RUN_OUTPUT.txt")


def find_and_rank():
    candidate_files = []
    for run_dir in SEARCH_DIRS:
        pattern = os.path.join(run_dir, "*", "*", "best_sol.py")
        candidate_files.extend(glob.glob(pattern))

    if not candidate_files:
        print(f"No best_sol.py files found in {{SEARCH_DIRS}}")
        return

    print(f"Found {{len(candidate_files)}} candidates. Evaluating...")
    results = []

    for file_path in candidate_files:
        display_name = "/".join(file_path.split("/")[-3:-1])
        temp_json = f"temp_eval_{{os.path.basename(os.path.dirname(file_path))}}.json"
        try:
            cmd = [sys.executable, EVALUATOR_SCRIPT, file_path, temp_json]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(temp_json, "r") as f:
                metrics = json.load(f)
            score = metrics.get("fitness", 0.0)
            results.append((score, file_path))
            print(f"  {{display_name:<40}} | {{score:.6f}}")
        except Exception as e:
            print(f"  {{display_name:<40}} | ERROR: {{e}}")
        finally:
            if os.path.exists(temp_json):
                os.remove(temp_json)

    if not results:
        return

    results.sort(key=lambda x: x[0], reverse=True)
    winner_score, winner_path = results[0]
    print(f"\\nWINNER: {{winner_path}}  (score={{winner_score:.8f}})")
    shutil.copy(winner_path, WINNER_COPY_PATH)
    print(f"Saved to: {{WINNER_COPY_PATH}}")


if __name__ == "__main__":
    if not os.path.exists(EVALUATOR_SCRIPT):
        print(f"Error: Could not find '{{EVALUATOR_SCRIPT}}'")
    else:
        find_and_rank()
'''

_DEFAULT_EVOLVE_CONFIG: dict = {
    "fitness_key": "fitness",
    "num_epochs": 100,
    "ckpt": 10,
    "max_size": None,
    "init_pop": 8,
    "exploration_rate": 0.45,
    "selection_policy": "roulette",
    "selection_kwargs": {"roulette_by_rank": True},
    "early_stopping_rounds": 50,
    "num_islands": 4,
    "migration_topology": "ring",
    "migration_interval": 10,
    "migration_rate": 0.2,
    "meta_prompting": False,
    "use_embedding": False,
    "use_map_elites": False,
    "num_inspirations": 3,
    "max_chat_depth": 3,
    "mp_start_marker": "# PROMPT-BLOCK-START",
    "mp_end_marker": "# PROMPT-BLOCK-END",
    "evolve_start_marker": "# EVOLVE-BLOCK-START",
    "evolve_end_marker": "# EVOLVE-BLOCK-END",
    "use_scheduler": True,
    "type": "PlateauScheduler",
    "scheduler_kwargs": {
        "min_rate": 0.2,
        "max_rate": 0.5,
        "plateau_threshold": 8,
        "increase_factor": 1.1,
        "decrease_factor": 0.92,
    },
}


_AI_TEMPLATE_MODEL_DEFAULT = "lfm2.5-thinking:1.2b"
_AI_TRANSCRIBE_MODEL_DEFAULT = "gpt-4o-mini-transcribe"
_AI_NEW_PROBLEM_SYSTEM_PROMPT = """\
You generate starter files for CodeEvolve, an LLM-driven evolutionary algorithm.

HOW CODEEVOLVE WORKS:
- A population of Python programs (candidates) is evolved across parallel islands.
- Each generation: parents are selected, an LLM mutates the code between EVOLVE-BLOCK \
markers via SEARCH/REPLACE diffs, then the candidate is evaluated in a sandboxed \
environment with resource limits (time, memory).
- evaluate.py runs the candidate, computes metrics, and writes JSON with a "fitness" \
score (float, 0-1, higher=better). Fitness drives selection.
- init_program.py is the seed candidate that gets evolved. Only "import math" is \
allowed. The LLM only modifies code between # EVOLVE-BLOCK-START / # EVOLVE-BLOCK-END.
- sys_msg tells the mutator LLM what to optimize and what constraints to follow.

Return a SINGLE JSON OBJECT (no markdown fences, no prose outside the JSON) with keys:
  "suggested_problem_name", "evaluator_py", "init_program_py", "sys_msg"

EVALUATOR (evaluate.py) REQUIREMENTS:
- CLI interface: python evaluate.py <candidate.py> <results.json>
- Must have: def evaluate(candidate_path: str) -> dict  and  def main() -> int
- Use importlib.util to dynamically load the candidate module
- Call candidate's compute() (or get_params()+compute(params)) to get its output
- Compare output against target/ground truth to compute fitness
- ALWAYS return {"fitness": <float>} even on errors (use 0.0 for failures)
- Write the result dict to results.json via json.dump
- Wrap ALL candidate calls in try/except to handle crashes gracefully

INIT PROGRAM (init_program.py) REQUIREMENTS:
- First line: import math  (the ONLY allowed import)
- Evolvable code wrapped in: # EVOLVE-BLOCK-START ... # EVOLVE-BLOCK-END
- Must define: get_params() returning a dict, compute(params) returning a value
- Start with a naive/simple implementation - evolution will improve it

SYS_MSG REQUIREMENTS:
- Wrapped in: # PROMPT-BLOCK-START ... # PROMPT-BLOCK-END
- Tell the LLM what the fitness measures and what to optimize
- State constraints: only import math, ASCII only, no external libraries
- Suggest strategies: tune get_params(), restructure compute(), etc.

RULES:
1) ASCII only. Code must be immediately runnable with just "import math".
2) Output ONLY the JSON object - no explanation text before or after.
"""

_AI_EVAL_FORMAT_EXAMPLE = '''\
"""
Evaluator for sample_problem.
Interface: python evaluate.py <candidate.py> <results.json>
"""
import importlib.util
import json
import math
import sys


def evaluate(candidate_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("candidate", candidate_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        return {"fitness": 0.0, "error": f"import: {exc}"}

    try:
        value = float(mod.compute({"x": 2.0, "y": 3.0}))
    except Exception as exc:
        return {"fitness": 0.0, "error": f"compute: {exc}"}

    target = 13.0
    mae = abs(value - target)
    fitness = 1.0 / (1.0 + mae)
    return {"fitness": float(fitness), "value": float(value), "mae": float(mae)}


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <candidate.py> <results.json>")
        return 1
    metrics = evaluate(sys.argv[1])
    if "fitness" not in metrics:
        metrics["fitness"] = 0.0
    metrics["fitness"] = float(metrics["fitness"])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_AI_INIT_FORMAT_EXAMPLE = '''\
import math

# EVOLVE-BLOCK-START


def get_params():
    return {"alpha": 1.0, "beta": 0.5}


def compute(params):
    x = float(params.get("alpha", 1.0))
    y = float(params.get("beta", 0.5))
    return x * x + 3.0 * y


# EVOLVE-BLOCK-END
'''


def _apply_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")

    # Global defaults
    style.configure(".", background=C.BASE, foreground=C.TEXT, font=FONT_UI,
                    borderwidth=0, focuscolor=C.BLUE)
    style.map(".", foreground=[("disabled", C.OVERLAY0)])

    # Frames
    style.configure("TFrame", background=C.BASE)
    style.configure("Surface.TFrame", background=C.SURFACE0)
    style.configure("Header.TFrame", background=C.BASE)
    style.configure("Toolbar.TFrame", background=C.BASE)

    # Labels
    style.configure("TLabel", background=C.BASE, foreground=C.TEXT, font=FONT_UI)
    style.configure("Dim.TLabel", foreground=C.OVERLAY0)
    style.configure("Title.TLabel", font=FONT_TITLE, foreground=C.LAVENDER)
    style.configure("Ok.TLabel", foreground=C.GREEN)
    style.configure("Err.TLabel", foreground=C.RED)
    style.configure("Status.TLabel", background=C.MANTLE, foreground=C.SUBTEXT0,
                    font=FONT_MONO_SM, padding=(6, 2))
    style.configure("Pill.Ok.TLabel", background=C.SURFACE0, foreground=C.GREEN,
                    font=FONT_UI_B, padding=(8, 2))
    style.configure("Pill.Err.TLabel", background=C.SURFACE0, foreground=C.RED,
                    font=FONT_UI_B, padding=(8, 2))

    # LabelFrames
    style.configure("TLabelframe", background=C.BASE, foreground=C.SUBTEXT1,
                    font=FONT_UI_B, borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background=C.BASE, foreground=C.SUBTEXT1,
                    font=FONT_UI_B)
    style.configure("Card.TLabelframe", background=C.SURFACE0, foreground=C.SUBTEXT1,
                    font=FONT_UI_B, borderwidth=1, relief="solid")
    style.configure("Card.TLabelframe.Label", background=C.SURFACE0, foreground=C.SUBTEXT1,
                    font=FONT_UI_B)

    # Buttons
    style.configure("TButton", background=C.SURFACE1, foreground=C.TEXT, font=FONT_UI,
                    padding=(12, 6), borderwidth=0)
    style.map("TButton",
              background=[("active", C.SURFACE2), ("disabled", C.SURFACE0)],
              foreground=[("disabled", C.OVERLAY0)])

    style.configure("Accent.TButton", background=C.BLUE, foreground=C.CRUST, font=FONT_UI_B)
    style.map("Accent.TButton",
              background=[("active", C.LAVENDER), ("disabled", C.SURFACE1)],
              foreground=[("disabled", C.OVERLAY0)])

    style.configure("Danger.TButton", background=C.RED, foreground=C.CRUST, font=FONT_UI_B)
    style.map("Danger.TButton",
              background=[("active", C.FLAMINGO), ("disabled", C.SURFACE1)],
              foreground=[("disabled", C.OVERLAY0)])

    # Entries
    style.configure("TEntry", fieldbackground=C.SURFACE0, foreground=C.TEXT,
                    insertcolor=C.TEXT, borderwidth=1, padding=4)
    style.map("TEntry", fieldbackground=[("focus", C.SURFACE1)])

    # Comboboxes
    style.configure("TCombobox", fieldbackground=C.SURFACE0, foreground=C.TEXT,
                    background=C.SURFACE1, arrowcolor=C.SUBTEXT0, borderwidth=1, padding=4)
    style.map("TCombobox",
              fieldbackground=[("focus", C.SURFACE1), ("readonly", C.SURFACE0)],
              foreground=[("readonly", C.TEXT)])

    # Checkbuttons
    style.configure("TCheckbutton", background=C.BASE, foreground=C.TEXT, font=FONT_UI)
    style.map("TCheckbutton", background=[("active", C.SURFACE0)])

    # Scrollbars
    style.configure("Vertical.TScrollbar", background=C.SURFACE0, troughcolor=C.MANTLE,
                    arrowcolor=C.SUBTEXT0, borderwidth=0)
    style.map("Vertical.TScrollbar", background=[("active", C.SURFACE2)])

    # PanedWindow
    style.configure("TPanedwindow", background=C.SURFACE0)
    style.configure("Sash", sashthickness=4, gripcount=0)

    # Notebook (used for tabs)
    style.configure("TNotebook", background=C.MANTLE, borderwidth=0)
    style.configure("TNotebook.Tab", background=C.SURFACE0, foreground=C.SUBTEXT0,
                    font=FONT_UI, padding=(16, 8))
    style.map("TNotebook.Tab",
              background=[("selected", C.BASE), ("active", C.SURFACE1)],
              foreground=[("selected", C.TEXT)])

    # Separator
    style.configure("TSeparator", background=C.SURFACE1)

    # Combobox dropdown (Tk option database)
    root.option_add("*TCombobox*Listbox.background", C.SURFACE0)
    root.option_add("*TCombobox*Listbox.foreground", C.TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", C.BLUE)
    root.option_add("*TCombobox*Listbox.selectForeground", C.CRUST)
    root.option_add("*TCombobox*Listbox.font", FONT_UI)


# =============================================================================
# Data model
# =============================================================================
@dataclass(frozen=True)
class Problem:
    name: str
    prob_dir: Path
    script_path: Path
    commands: frozenset[str]
    help_text: str


@dataclass(frozen=True)
class VizNode:
    prog_id: str
    parent_id: Optional[str]
    island: int
    generation: int
    fitness: float
    metric: float
    code: Optional[str] = None
    eval_metrics: Optional[dict] = None
    metrics: Optional[dict[str, float]] = None
    features: Optional[dict[str, float]] = None


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) != 6:
        return (0, 0, 0)
    try:
        return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
    except Exception:
        return (0, 0, 0)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r = max(0, min(255, int(round(rgb[0]))))
    g = max(0, min(255, int(round(rgb[1]))))
    b = max(0, min(255, int(round(rgb[2]))))
    return f"#{r:02x}{g:02x}{b:02x}"


def _mix_color(a: str, b: str, t: float) -> str:
    t = _clamp01(t)
    ar, ag, ab = _hex_to_rgb(a)
    br, bg, bb = _hex_to_rgb(b)
    return _rgb_to_hex((
        ar + (br - ar) * t,
        ag + (bg - ag) * t,
        ab + (bb - ab) * t,
    ))


def _find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists() and (p / "problems").is_dir():
            return p
    return start


def _parse_commands_from_help(help_text: str) -> frozenset[str]:
    lines = help_text.splitlines()
    try:
        i = next(idx for idx, ln in enumerate(lines) if ln.strip() == "Commands:")
    except StopIteration:
        return frozenset()
    cmds: set[str] = set()
    for ln in lines[i + 1:]:
        if not ln.strip():
            break
        token = ln.strip().split()[0]
        if token.startswith(("-", "[")):
            continue
        for part in token.split("|"):
            part = part.strip()
            if part:
                cmds.add(part)
    return frozenset(cmds)


def _discover_problems(repo_root: Path) -> dict[str, Problem]:
    probs_dir = repo_root / "problems"
    out: dict[str, Problem] = {}
    for child in sorted(probs_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        script_path = child / f"{name}.sh"
        if not script_path.exists():
            continue
        try:
            r = subprocess.run(
                ["bash", str(script_path), "help"],
                capture_output=True, text=True, cwd=str(repo_root), timeout=3,
            )
            help_text = (r.stdout or "") + (r.stderr or "")
        except Exception:
            help_text = ""
        cmds = _parse_commands_from_help(help_text)
        out[name] = Problem(name=name, prob_dir=child, script_path=script_path,
                            commands=cmds, help_text=help_text)
    return out


def _open_path(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        messagebox.showerror("Open Failed", f"Could not open:\n{path}\n\n{e}")


def _island_log(island_dir: Path) -> Optional[Path]:
    """Return the island log path, checking v0.3 name first then legacy."""
    for name in ("island.log", "results.log"):
        p = island_dir / name
        if p.exists():
            return p
    return None


_RE_FITNESS_VAL = re.compile(r"fitness=([\d.]+)")


def _best_fitness_for_run(run_dir: Path, fitness_key: str = "combined_score") -> Optional[float]:
    """Scan island directories for the best fitness value."""
    best = None
    for island in run_dir.iterdir():
        if not island.is_dir():
            continue
        log_path = _island_log(island)
        if log_path is None:
            continue
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = _RE_FITNESS_VAL.search(line)
                    if m:
                        val = float(m.group(1))
                        if best is None or val > best:
                            best = val
        except Exception:
            continue
    return best


def _run_last_log_mtime(run_dir: Path) -> Optional[float]:
    """Return the newest island log mtime for a run."""
    newest = None
    for island in run_dir.iterdir():
        if not island.is_dir():
            continue
        log_path = _island_log(island)
        if log_path is None:
            continue
        try:
            mtime = log_path.stat().st_mtime
        except Exception:
            continue
        newest = mtime if newest is None or mtime > newest else newest
    return newest


def _sanitize_problem_name(raw: str, fallback: str = "my_problem") -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", (raw or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = fallback
    if text and text[0].isdigit():
        text = f"p_{text}"
    return text


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty response from model")
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response did not contain JSON object")
        data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _fallback_template_payload_from_text(text: str, name_hint: str) -> dict[str, Any]:
    raw = (text or "").strip()
    out: dict[str, Any] = {
        "suggested_problem_name": _sanitize_problem_name(name_hint or "my_problem"),
    }
    if not raw:
        return out

    m = re.search(
        r"(?:suggested_problem_name|problem_name|name)\s*[:=]\s*([A-Za-z0-9_ -]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        out["suggested_problem_name"] = _sanitize_problem_name(m.group(1).strip())

    # Strategy: first extract marked blocks to separate the three files,
    # then use code-fence segments only for parts not yet found.
    # This prevents assigning the same blob to all three fields.

    # 1) Extract marked blocks (most reliable separation)
    sys_seg = ""
    init_seg = ""
    evaluator_seg = ""
    remainder = raw

    prompt_block, remainder = _extract_marked_block(
        remainder, "# PROMPT-BLOCK-START", "# PROMPT-BLOCK-END"
    )
    if prompt_block:
        sys_seg = prompt_block

    evolve_block, remainder = _extract_marked_block(
        remainder, "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"
    )
    if evolve_block:
        init_seg = evolve_block

    # 2) Try code-fenced blocks for anything still missing
    code_blocks = re.findall(r"```[^\n]*\n(.*?)```", raw, flags=re.DOTALL)
    segments = [seg.strip() for seg in code_blocks if seg.strip()]

    for seg in segments:
        if not evaluator_seg and "def evaluate(" in seg and ("json.dump" in seg or "fitness" in seg):
            # Only use this segment if it's distinct from init/sys
            if not (init_seg and seg == init_seg) and not (sys_seg and seg == sys_seg):
                evaluator_seg = seg
        if not init_seg and ("def get_params(" in seg and "def compute(" in seg):
            if not (evaluator_seg and seg == evaluator_seg):
                init_seg = seg
        if not sys_seg and "# PROMPT-BLOCK-START" in seg:
            if not (evaluator_seg and seg == evaluator_seg):
                sys_seg = seg

    # 3) If evaluator not yet found, try to extract from remainder
    #    (remainder has marked blocks removed, so it's likely just the evaluator)
    if not evaluator_seg and remainder.strip():
        rem = remainder.strip()
        if "def evaluate(" in rem:
            evaluator_seg = rem
        elif "def get_params(" in rem and "def compute(" in rem and not init_seg:
            # The "evaluator" section was actually init code
            init_seg = rem

    # 4) If init found without import math, add it
    if init_seg and "import math" not in init_seg:
        init_seg = "import math\n\n" + init_seg.lstrip()

    if evaluator_seg:
        out["evaluator_py"] = evaluator_seg
    if init_seg:
        out["init_program_py"] = init_seg
    if sys_seg:
        out["sys_msg"] = sys_seg
    return out


def _extract_chat_content(chat_payload: dict[str, Any]) -> str:
    choices = chat_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Missing choices in chat response")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Invalid choices payload")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Missing message in chat response")
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        content = "\n".join(parts)
    if not isinstance(content, str):
        raise ValueError("Unsupported chat content format")
    # Strip <think>...</think> blocks from thinking models
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def _ensure_marked_block(text: str, start_marker: str, end_marker: str) -> str:
    body = (text or "").strip("\n")
    if start_marker in body and end_marker in body:
        return body + ("\n" if not body.endswith("\n") else "")
    if not body:
        body = "TODO"
    return f"{start_marker}\n{body}\n{end_marker}\n"


def _strip_code_fences(text: str) -> str:
    body = (text or "").strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    return body


def _ensure_generated_evaluator(text: str, problem_name: str) -> str:
    code = _strip_code_fences(text)
    if not code:
        return _EVAL_TEMPLATE.format(name=problem_name)

    # Core requirement: must have evaluate() and mention fitness
    if "def evaluate(" not in code:
        return _EVAL_TEMPLATE.format(name=problem_name)
    if '"fitness"' not in code and "'fitness'" not in code and "fitness" not in code:
        return _EVAL_TEMPLATE.format(name=problem_name)

    # Ensure required imports are present
    if "import importlib.util" not in code:
        code = "import importlib.util\n" + code.lstrip()
    if "import json" not in code:
        code = "import json\n" + code.lstrip()
    if "import sys" not in code:
        code = "import sys\n" + code.lstrip()

    # Add main() boilerplate if missing
    if "def main(" not in code:
        code = code.rstrip() + "\n\n\n" + _EVAL_MAIN_BOILERPLATE

    return code.strip("\n") + "\n"


_EVAL_MAIN_BOILERPLATE = '''\
def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <candidate.py> <results.json>")
        return 1
    metrics = evaluate(sys.argv[1])
    if "fitness" not in metrics:
        metrics["fitness"] = 0.0
    metrics["fitness"] = float(metrics["fitness"])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _ensure_generated_init_program(text: str) -> str:
    code = _strip_code_fences(text)
    if not code:
        return _INIT_TEMPLATE
    # Must have at least the two core functions
    if "def get_params(" not in code or "def compute(" not in code:
        return _INIT_TEMPLATE
    if "import math" not in code:
        code = "import math\n\n" + code.lstrip()
    # Strip disallowed imports (only "import math" is allowed)
    lines = code.splitlines()
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") and stripped != "import math":
            continue
        if stripped.startswith("from ") and "import" in stripped:
            continue
        filtered.append(line)
    code = "\n".join(filtered)
    # Don't require markers here - _ensure_marked_block in the caller will wrap if needed
    return code.strip("\n") + "\n"


def _extract_marked_block(text: str, start_marker: str, end_marker: str) -> tuple[str, str]:
    body = text or ""
    start = body.find(start_marker)
    if start < 0:
        return "", body
    end = body.find(end_marker, start + len(start_marker))
    if end < 0:
        return "", body
    end += len(end_marker)
    block = body[start:end].strip("\n")
    rest = (body[:start] + body[end:]).strip("\n")
    return (block + "\n"), (rest + "\n" if rest else "")


def _first_nonempty_str(data: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if body:
        return f"HTTP {exc.code}: {body[:900]}"
    return f"HTTP {exc.code}: {exc.reason}"


def _api_post_json(api_base: str, api_key: str, path: str, payload: dict[str, Any],
                   timeout_s: float = 180.0) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(_http_error_message(exc)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc
    try:
        payload_obj = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON from {path}: {raw[:500]}") from exc
    if not isinstance(payload_obj, dict):
        raise RuntimeError(f"Invalid response object from {path}")
    return payload_obj


def _encode_multipart_form(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----CodeEvolveBoundary{uuid.uuid4().hex}"
    out = bytearray()
    b = boundary.encode("utf-8")
    for key, value in fields.items():
        out.extend(b"--" + b + b"\r\n")
        out.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        out.extend(value.encode("utf-8"))
        out.extend(b"\r\n")
    for key, (filename, data, content_type) in files.items():
        out.extend(b"--" + b + b"\r\n")
        out.extend(
            f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        out.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        out.extend(data)
        out.extend(b"\r\n")
    out.extend(b"--" + b + b"--\r\n")
    return bytes(out), boundary


def _api_transcribe_audio(
    api_base: str,
    api_key: str,
    audio_path: Path,
    model: str,
    timeout_s: float = 180.0,
) -> str:
    if not audio_path.exists():
        raise RuntimeError(f"Audio file missing: {audio_path}")
    if audio_path.stat().st_size < 256:
        raise RuntimeError("Recorded audio is too short. Please speak a bit longer.")
    data = audio_path.read_bytes()
    body, boundary = _encode_multipart_form(
        fields={
            "model": model,
            "response_format": "json",
        },
        files={
            "file": (audio_path.name, data, "audio/wav"),
        },
    )
    url = f"{api_base.rstrip('/')}/audio/transcriptions"
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(_http_error_message(exc)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Transcription request failed: {exc.reason}") from exc
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Invalid transcription response: {raw[:500]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid transcription response payload")
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Transcription returned empty text")
    return text.strip()


def _build_generation_messages(description: str, problem_name_hint: str) -> list[dict[str, str]]:
    user_text = description.strip()
    if problem_name_hint:
        user_text += f"\n\nProblem name hint: {problem_name_hint}"

    # Build a single user message with structural examples and the request
    example_section = (
        "Here is a COMPLETE example of the expected JSON output for a simple problem "
        "(x^2 + 3y with target 13). Adapt the logic for the user's problem but keep "
        "the exact same structure:\n\n"
        "EXAMPLE evaluate.py (embed as JSON string in evaluator_py key):\n"
        f"{_AI_EVAL_FORMAT_EXAMPLE}\n\n"
        "EXAMPLE init_program.py (embed as JSON string in init_program_py key):\n"
        f"{_AI_INIT_FORMAT_EXAMPLE}\n\n"
        "EXAMPLE sys_msg (embed as JSON string in sys_msg key):\n"
        "# PROMPT-BLOCK-START\n"
        "You are optimizing a Python program.\n"
        "HARD CONSTRAINTS (violation = fitness 0):\n"
        "1. ONLY allowed import: \"import math\" - do NOT add numpy, scipy, etc.\n"
        "2. Do not add new \"import\" lines\n"
        "3. Only ASCII characters\n"
        "STRATEGIES:\n"
        "- Modify compute() to improve fitness\n"
        "- Tune parameters in get_params()\n"
        "# PROMPT-BLOCK-END\n\n"
        "---\n\n"
        "Now generate files for this problem:\n\n"
    )

    return [
        {"role": "system", "content": _AI_NEW_PROBLEM_SYSTEM_PROMPT},
        {"role": "user", "content": example_section + user_text},
    ]


def _coerce_generated_problem_templates(
    model_json: dict[str, Any],
    name_hint: str,
) -> dict[str, str]:
    suggested = _first_nonempty_str(
        model_json,
        (
            "suggested_problem_name",
            "problem_name",
            "name",
            "problem",
        ),
    )
    suggested = _sanitize_problem_name(suggested or name_hint or "my_problem")

    files_obj = model_json.get("files")
    files_map = files_obj if isinstance(files_obj, dict) else {}

    evaluator_raw = _first_nonempty_str(
        model_json,
        (
            "evaluator_py",
            "evaluate_py",
            "evaluator",
            "evaluate",
        ),
    ) or _first_nonempty_str(
        files_map, ("evaluate.py", "evaluator.py", "evaluate_py", "evaluator_py")
    )

    init_raw = _first_nonempty_str(
        model_json,
        (
            "init_program_py",
            "init_program",
            "init_py",
            "candidate_py",
            "solution_py",
        ),
    ) or _first_nonempty_str(
        files_map, ("init_program.py", "init_program_py", "init_program", "candidate.py")
    )

    sys_raw = _first_nonempty_str(
        model_json,
        (
            "sys_msg",
            "system_prompt",
            "system_message",
            "sys_message",
            "prompt",
            "prompt_text",
        ),
    ) or _first_nonempty_str(
        files_map, ("sys_msg", "system_prompt", "system_message", "prompt.txt", "sys_msg.txt")
    )

    # If model mixed prompt block into init code, move it to sys_msg.
    prompt_from_init, init_remainder = _extract_marked_block(
        init_raw, "# PROMPT-BLOCK-START", "# PROMPT-BLOCK-END"
    )
    if prompt_from_init and not sys_raw.strip():
        sys_raw = prompt_from_init
        init_raw = init_remainder

    # If model mixed EVOLVE block into sys_msg, move it to init.
    evolve_from_sys, sys_remainder = _extract_marked_block(
        sys_raw, "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"
    )
    if evolve_from_sys and not init_raw.strip():
        init_raw = "import math\n\n" + evolve_from_sys
        sys_raw = sys_remainder

    evaluator = _ensure_generated_evaluator(evaluator_raw, problem_name=suggested)
    init_program = _ensure_generated_init_program(init_raw)
    init_program = _ensure_marked_block(
        init_program, "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"
    )

    sys_raw = _strip_code_fences(sys_raw) if isinstance(sys_raw, str) else ""
    if not sys_raw.strip():
        sys_raw = _SYSMSG_TEMPLATE
    sys_msg = _ensure_marked_block(
        sys_raw, "# PROMPT-BLOCK-START", "# PROMPT-BLOCK-END"
    )

    return {
        "suggested_problem_name": suggested,
        "evaluator_py": evaluator.strip("\n") + "\n",
        "init_program_py": init_program,
        "sys_msg": sys_msg,
    }


def _api_generate_problem_templates(
    api_base: str,
    api_key: str,
    model: str,
    description: str,
    problem_name_hint: str,
    timeout_s: float = 240.0,
) -> dict[str, str]:
    if not description.strip():
        raise RuntimeError("Description cannot be empty")
    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_generation_messages(description, problem_name_hint),
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = _api_post_json(api_base, api_key, "/chat/completions", payload, timeout_s=timeout_s)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "response_format" not in msg:
            raise
        payload.pop("response_format", None)
        resp = _api_post_json(api_base, api_key, "/chat/completions", payload, timeout_s=timeout_s)
    content = _extract_chat_content(resp)
    try:
        parsed = _extract_json_object(content)
    except Exception:
        parsed = _fallback_template_payload_from_text(content, name_hint=problem_name_hint)
    return _coerce_generated_problem_templates(parsed, name_hint=problem_name_hint)


# =============================================================================
# Dashboard
# =============================================================================
class Dashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.repo_root = _find_repo_root(Path(__file__).resolve().parent)
        self.problems = _discover_problems(self.repo_root)
        self.experiments_dir = self.repo_root / "experiments"

        self.title("CodeEvolve Dashboard")
        self.geometry("1440x900")
        self.minsize(1000, 700)
        self.configure(bg=C.BASE)

        _apply_theme(self)

        self._proc: Optional[subprocess.Popen[str]] = None
        self._proc_lock = threading.Lock()
        self._q: queue.Queue[str] = queue.Queue()
        self._closing = False
        self._close_kill_at: Optional[float] = None
        self._status_after: Optional[str] = None
        self._models_refresh_inflight = False
        self._run_start_time: Optional[float] = None
        self._viz_after: Optional[str] = None
        self._viz_last_ckpt: Optional[int] = None
        self._viz_selected_id: Optional[str] = None
        self._viz_nodes_cache: dict[str, VizNode] = {}
        self._viz_positions: dict[str, tuple[float, float]] = {}
        self._viz_hit_radius: dict[str, float] = {}
        self._viz_metric_key: str = "combined_score"
        self._viz_metric_names: list[str] = ["combined_score", "fitness"]
        self._viz_map_feature_names: list[str] = []
        self._viz_elite_map_type: str = ""
        self._viz_map_cell_cache: dict[tuple[int, ...], tuple[str, float]] = {}
        self._viz_parent_map: dict[str, Optional[str]] = {}
        self._viz_children_map: dict[str, list[str]] = {}
        self._viz_delta_map: dict[str, Optional[float]] = {}
        self._viz_depth_map: dict[str, int] = {}
        self._viz_island_best_ids: dict[int, str] = {}
        self._viz_global_best_id: Optional[str] = None
        self._runs_auto_after: Optional[str] = None
        self._run_item_names: list[str] = []
        self._run_item_meta: list[dict[str, object]] = []
        self._run_score_cache: dict[tuple[str, str], tuple[Optional[float], Optional[float]]] = {}
        self._state_path = self.repo_root / ".dashboard_state.json"
        self._active_cmd: str = ""

        self._build_ui()
        self._load_ui_state()
        self._refresh_configs()
        self._refresh_runs()
        self._refresh_models_from_config()
        self._refresh_config_editor()
        self._refresh_models_tab()
        self._refresh_installed_models()
        self._bind_shortcuts()
        self._tick_drain()
        if self.auto_refresh_var.get():
            self._schedule_runs_auto_refresh(1200)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=0)

        # ---- Header bar ----
        header = ttk.Frame(self, padding=(14, 12, 14, 8), style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(9, weight=1)

        ttk.Label(header, text="CodeEvolve", style="Title.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 16))

        ttk.Label(header, text="Problem:").grid(row=0, column=1, sticky="w")
        self.problem_var = tk.StringVar(value=(sorted(self.problems.keys())[0] if self.problems else ""))
        self.problem_combo = ttk.Combobox(
            header, textvariable=self.problem_var,
            values=sorted(self.problems.keys()), state="readonly", width=22,
        )
        self.problem_combo.grid(row=0, column=2, sticky="w", padx=(4, 6))
        self.problem_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_problem_change())
        ttk.Button(header, text="Scan", width=5,
                   command=self._refresh_problems).grid(row=0, column=3, sticky="w", padx=(0, 4))
        ttk.Button(header, text="New", width=4,
                   command=self._new_problem_dialog).grid(row=0, column=4, sticky="w", padx=(0, 12))

        ttk.Label(header, text="API_BASE:").grid(row=0, column=5, sticky="w")
        common_api_bases = [
            "http://localhost:11434/v1",
            "http://127.0.0.1:11434/v1",
            "http://localhost:1234/v1",
            "http://localhost:8000/v1",
            "http://localhost:8080/v1",
            "http://localhost:5001/v1",
            "https://api.openai.com/v1",
            "https://openrouter.ai/api/v1",
            "https://api.groq.com/openai/v1",
            "https://api.together.xyz/v1",
            "https://api.fireworks.ai/inference/v1",
            "https://api.deepseek.com/v1",
            "https://api.mistral.ai/v1",
            "https://api.endpoints.anyscale.com/v1",
            "https://api.cerebras.ai/v1",
            "https://api.x.ai/v1",
        ]
        common_api_bases = list(dict.fromkeys(common_api_bases))
        env_api_base = os.environ.get("API_BASE", "").strip()
        if env_api_base and env_api_base not in common_api_bases:
            common_api_bases.insert(0, env_api_base)
        default_api_base = env_api_base or (common_api_bases[0] if common_api_bases else "")
        self.api_base_var = tk.StringVar(value=default_api_base)
        self.api_key_var = tk.StringVar(value=os.environ.get("API_KEY", ""))
        self.api_base_var.trace_add("write", lambda *_: self._sync_env_status())
        self.api_key_var.trace_add("write", lambda *_: self._sync_env_status())
        self.api_base_combo = ttk.Combobox(
            header, textvariable=self.api_base_var,
            values=common_api_bases, state="normal", width=30,
        )
        self.api_base_combo.grid(row=0, column=6, sticky="w", padx=(4, 14))

        ttk.Label(header, text="API_KEY:").grid(row=0, column=7, sticky="w")
        ttk.Entry(header, textvariable=self.api_key_var, width=26, show="*").grid(
            row=0, column=8, sticky="w", padx=(4, 14))

        self.env_status = ttk.Label(header, text="", style="Dim.TLabel")
        self.env_status.grid(row=0, column=9, sticky="e")

        ttk.Separator(self, orient="horizontal").grid(row=1, column=0, sticky="ew", padx=12)

        # ---- Main paned area ----
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.grid(row=2, column=0, sticky="nsew", padx=10, pady=(6, 0))

        # ---- Left sidebar ----
        sidebar = ttk.Frame(paned, width=260)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=3)
        paned.add(sidebar, weight=0)

        # Runs list
        runs_frame = ttk.LabelFrame(sidebar, text="  Runs  ", padding=8, style="Card.TLabelframe")
        runs_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        runs_frame.rowconfigure(0, weight=1)
        runs_frame.columnconfigure(0, weight=1)

        runs_scroll = ttk.Scrollbar(runs_frame, orient="vertical")
        self.runs_list = tk.Listbox(
            runs_frame, height=8, width=40, exportselection=False,
            bg=C.SURFACE0, fg=C.TEXT, selectbackground=C.BLUE, selectforeground=C.CRUST,
            highlightthickness=1, highlightbackground=C.SURFACE1,
            borderwidth=0, font=FONT_MONO, activestyle="none",
            yscrollcommand=runs_scroll.set,
        )
        runs_scroll.config(command=self.runs_list.yview)
        self.runs_list.grid(row=0, column=0, sticky="nsew")
        runs_scroll.grid(row=0, column=1, sticky="ns")
        self.runs_list.bind("<<ListboxSelect>>", lambda _e: self._on_run_select())
        self.runs_list.bind("<Double-Button-1>", lambda _e: self._open_selected_run_dir())
        self.runs_list.bind("<Control-c>", lambda _e: self._copy_selected_run_id())

        btn_row = ttk.Frame(runs_frame, style="Surface.TFrame")
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(btn_row, text="Refresh", command=self._refresh_runs).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Open Folder", command=self._open_experiments).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Open Run", command=self._open_selected_run_dir).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Copy ID", command=self._copy_selected_run_id).pack(side="left")

        self.run_filter_var = tk.StringVar(value="")
        self.auto_refresh_var = tk.BooleanVar(value=True)
        self.run_sort_var = tk.StringVar(value=RUN_SORT_OPTIONS[0])
        self.run_status_var = tk.StringVar(value=RUN_STATUS_OPTIONS[0])
        self.run_min_score_var = tk.StringVar(value="")

        util_row = ttk.Frame(runs_frame, style="Surface.TFrame")
        util_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(util_row, text="Filter:", style="Dim.TLabel").pack(side="left")
        self.run_filter_entry = ttk.Entry(util_row, textvariable=self.run_filter_var, width=14)
        self.run_filter_entry.pack(side="left", padx=(4, 8))
        ttk.Checkbutton(
            util_row, text="Auto", variable=self.auto_refresh_var,
            command=self._on_auto_refresh_toggle,
        ).pack(side="left")

        util_row2 = ttk.Frame(runs_frame, style="Surface.TFrame")
        util_row2.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(util_row2, text="Sort:", style="Dim.TLabel").pack(side="left")
        sort_combo = ttk.Combobox(
            util_row2,
            textvariable=self.run_sort_var,
            values=RUN_SORT_OPTIONS,
            state="readonly",
            width=12,
        )
        sort_combo.pack(side="left", padx=(4, 8))
        sort_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_run_sort_change())

        ttk.Label(util_row2, text="State:", style="Dim.TLabel").pack(side="left")
        state_combo = ttk.Combobox(
            util_row2,
            textvariable=self.run_status_var,
            values=RUN_STATUS_OPTIONS,
            state="readonly",
            width=6,
        )
        state_combo.pack(side="left", padx=(4, 8))
        state_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_run_status_change())

        ttk.Label(util_row2, text="Min:", style="Dim.TLabel").pack(side="left")
        self.run_min_score_entry = ttk.Entry(util_row2, textvariable=self.run_min_score_var, width=8)
        self.run_min_score_entry.pack(side="left", padx=(4, 0))

        self.runs_meta_var = tk.StringVar(value="")
        ttk.Label(
            runs_frame, textvariable=self.runs_meta_var, style="Dim.TLabel",
            font=FONT_MONO_SM,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # Run Snapshot
        snapshot_frame = ttk.LabelFrame(sidebar, text="  Run Snapshot  ", padding=8, style="Card.TLabelframe")
        snapshot_frame.grid(row=1, column=0, sticky="nsew")
        snapshot_frame.columnconfigure(0, weight=1)
        snapshot_frame.rowconfigure(0, weight=1)
        snapshot_frame.rowconfigure(1, weight=0)

        self._snapshot = ScrolledText(
            snapshot_frame, height=10, wrap="word",
            bg=C.SURFACE0, fg=C.TEXT, insertbackground=C.TEXT,
            highlightthickness=0, borderwidth=0, padx=8, pady=6,
            font=FONT_MONO_SM,
        )
        self._snapshot.grid(row=0, column=0, sticky="nsew")
        self._snapshot.configure(state="disabled")

        snap_btns = ttk.Frame(snapshot_frame, style="Surface.TFrame")
        snap_btns.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(snap_btns, text="Refresh", command=self._refresh_run_snapshot).pack(
            side="left", padx=(0, 4))
        ttk.Button(snap_btns, text="Open Run", command=self._open_run_dir).pack(
            side="left", padx=(0, 4))
        ttk.Button(snap_btns, text="Open Best Sol", command=self._open_best_sol).pack(
            side="left", padx=(0, 4))
        ttk.Button(snap_btns, text="Open Best Prompt", command=self._open_best_prompt).pack(
            side="left", padx=(0, 4))
        snap_btns2 = ttk.Frame(snapshot_frame, style="Surface.TFrame")
        snap_btns2.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(snap_btns2, text="Apply Best", style="Accent.TButton",
                   command=self._apply_best).pack(side="left")

        # Island color palette
        self._island_colors = [
            C.BLUE, C.GREEN, C.PEACH, C.MAUVE, C.TEAL, C.PINK, C.YELLOW,
            C.FLAMINGO, C.SKY, C.ROSEWATER, C.LAVENDER, C.RED,
        ]

        # Hidden storage for installed models (used by Models tab)
        self._installed_model_names: list[str] = []
        # Hidden listbox kept for backward compat with _refresh_installed_models
        self.installed_models_list = tk.Listbox(sidebar)
        self.cfg_models_list = tk.Listbox(sidebar)
        self.models_status = ttk.Label(sidebar, text="")

        # ---- Right main area ----
        right = ttk.Frame(paned)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        paned.add(right, weight=1)

        # Controls panel
        ctl = ttk.Frame(right, padding=(10, 8), style="Toolbar.TFrame")
        ctl.grid(row=0, column=0, sticky="ew")
        ctl.columnconfigure(10, weight=1)

        self.run_id_var = tk.StringVar(value="")
        self.island_var = tk.StringVar(value="0")
        self.cpu_var = tk.StringVar(value=os.environ.get("CPU_LIST", "0-7"))
        self.load_ckpt_var = tk.StringVar(value="0")
        self.winner_args_var = tk.StringVar(value="")
        self.advanced_var = tk.BooleanVar(value=False)
        self.skip_dryad_var = tk.BooleanVar(value=True)
        self.no_taskset_var = tk.BooleanVar(value=False)
        self.notify_done_var = tk.BooleanVar(value=True)
        self.cfg_var = tk.StringVar(value="")
        self.run_filter_var.trace_add("write", lambda *_: self._refresh_runs())
        self.run_min_score_var.trace_add("write", lambda *_: self._refresh_runs())

        # Row 0: run params
        r = 0
        ttk.Label(ctl, text="Run:").grid(row=r, column=0, sticky="w")
        ttk.Entry(ctl, textvariable=self.run_id_var, width=12).grid(row=r, column=1, sticky="w", padx=(4, 10))
        lbl_island = ttk.Label(ctl, text="Island:")
        lbl_island.grid(row=r, column=2, sticky="w")
        ent_island = ttk.Entry(ctl, textvariable=self.island_var, width=4)
        ent_island.grid(row=r, column=3, sticky="w", padx=(4, 10))
        lbl_cpu = ttk.Label(ctl, text="CPU:")
        lbl_cpu.grid(row=r, column=4, sticky="w")
        ent_cpu = ttk.Entry(ctl, textvariable=self.cpu_var, width=8)
        ent_cpu.grid(row=r, column=5, sticky="w", padx=(4, 10))
        lbl_ckpt = ttk.Label(ctl, text="Ckpt:")
        lbl_ckpt.grid(row=r, column=6, sticky="w")
        ent_ckpt = ttk.Entry(ctl, textvariable=self.load_ckpt_var, width=4)
        ent_ckpt.grid(row=r, column=7, sticky="w", padx=(4, 10))

        ttk.Label(ctl, text="Config:").grid(row=r, column=8, sticky="w")
        self.cfg_combo = ttk.Combobox(ctl, textvariable=self.cfg_var, values=[], state="readonly", width=22)
        self.cfg_combo.grid(row=r, column=9, sticky="w", padx=(4, 6))
        self.cfg_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_config_change())
        ttk.Button(ctl, text="Open", command=self._open_config).grid(row=r, column=10, sticky="w")
        ttk.Checkbutton(ctl, text="Advanced", variable=self.advanced_var,
                        command=self._toggle_advanced).grid(row=r, column=11, sticky="e", padx=(12, 0))

        # Row 1: checkboxes + winner args
        r = 1
        chk_frame = ttk.Frame(ctl)
        chk_frame.grid(row=r, column=0, columnspan=6, sticky="w", pady=(6, 0))
        ttk.Checkbutton(chk_frame, text="No taskset", variable=self.no_taskset_var).pack(side="left", padx=(0, 12))
        self.skip_dryad_chk = ttk.Checkbutton(chk_frame, text="Skip Dryad (elegans)", variable=self.skip_dryad_var)
        self.skip_dryad_chk.pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            chk_frame, text="Notify on finish", variable=self.notify_done_var
        ).pack(side="left", padx=(0, 12))

        lbl_winner = ttk.Label(ctl, text="Winner args:")
        lbl_winner.grid(row=r, column=6, sticky="w", pady=(6, 0))
        ent_winner = ttk.Entry(ctl, textvariable=self.winner_args_var, width=40)
        ent_winner.grid(row=r, column=7, columnspan=4, sticky="we", pady=(6, 0), padx=(4, 0))

        # Row 2: action buttons
        r = 2
        btn_bar = ttk.Frame(ctl, style="Toolbar.TFrame")
        btn_bar.grid(row=r, column=0, columnspan=11, sticky="ew", pady=(12, 6))

        self.btn_run = ttk.Button(btn_bar, text="Run", style="Accent.TButton", command=self._cmd_run)
        self.btn_run.pack(side="left", padx=(0, 4))

        self.btn_stop = ttk.Button(btn_bar, text="Stop", style="Danger.TButton",
                                   command=self._stop_proc, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 4))
        ttk.Button(btn_bar, text="Clear Log", command=self._clear_log).pack(side="left", padx=(0, 4))

        self._adv_sep = ttk.Separator(btn_bar, orient="vertical")
        self._adv_sep.pack(side="left", fill="y", padx=8)
        self._adv_btns = ttk.Frame(btn_bar)
        self._adv_btns.pack(side="left")

        self.btn_run_next = ttk.Button(self._adv_btns, text="Run Next", style="Accent.TButton", command=self._cmd_run_next)
        self.btn_run_next.pack(side="left", padx=(0, 4))
        self.btn_analyze = ttk.Button(self._adv_btns, text="Analyze", command=self._cmd_analyze)
        self.btn_analyze.pack(side="left", padx=(0, 4))
        self.btn_winner = ttk.Button(self._adv_btns, text="Winner", command=self._cmd_winner)
        self.btn_winner.pack(side="left", padx=(0, 4))
        self.btn_viz = ttk.Button(self._adv_btns, text="Visualize", command=self._cmd_viz)
        self.btn_viz.pack(side="left", padx=(0, 4))
        self.btn_tail = ttk.Button(self._adv_btns, text="Tail Log", command=self._cmd_tail)
        self.btn_tail.pack(side="left", padx=(0, 4))
        self.btn_warmstart = ttk.Button(self._adv_btns, text="Warmstart", command=self._cmd_warmstart)
        self.btn_warmstart.pack(side="left", padx=(0, 4))
        self.btn_ls = ttk.Button(self._adv_btns, text="List Runs", command=self._cmd_ls)
        self.btn_ls.pack(side="left", padx=(0, 4))

        self._adv_grid_widgets = [lbl_island, ent_island, lbl_cpu, ent_cpu, lbl_ckpt, ent_ckpt, chk_frame, lbl_winner, ent_winner]
        self._toggle_advanced()

        ttk.Separator(right, orient="horizontal").grid(row=0, column=0, sticky="sew")

        # ---- Notebook: Output + Config tabs ----
        self._notebook = ttk.Notebook(right)
        self._notebook.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

        # --- Output tab ---
        output_tab = ttk.Frame(self._notebook)
        self._notebook.add(output_tab, text="  Output  ")
        output_tab.rowconfigure(0, weight=1)
        output_tab.columnconfigure(0, weight=1)

        self.log = ScrolledText(
            output_tab, height=24, wrap="none",
            bg=C.MANTLE, fg=C.TEXT, insertbackground=C.TEXT,
            highlightthickness=0, borderwidth=0, padx=8, pady=6,
            font=FONT_MONO_SM, selectbackground=C.BLUE, selectforeground=C.CRUST,
        )
        # Horizontal scrollbar for wide output (all islands on one line)
        log_xscroll = ttk.Scrollbar(output_tab, orient="horizontal", command=self.log.xview)
        self.log.configure(xscrollcommand=log_xscroll.set)
        log_xscroll.grid(row=1, column=0, sticky="ew")
        self.log.grid(row=0, column=0, sticky="nsew")

        # Configure text tags for colored output
        self.log.tag_configure("cmd", foreground=C.SKY)
        self.log.tag_configure("error", foreground=C.RED)
        self.log.tag_configure("warning", foreground=C.YELLOW)
        self.log.tag_configure("success", foreground=C.GREEN)
        self.log.tag_configure("info", foreground=C.TEAL)
        self.log.tag_configure("dim", foreground=C.OVERLAY0)
        self.log.tag_configure("fitness", foreground=C.PEACH, font=("Ubuntu Sans Mono", 9, "bold"))
        self.log.tag_configure("exit_ok", foreground=C.GREEN)
        self.log.tag_configure("exit_fail", foreground=C.RED)

        # --- Visualizer tab ---
        self._build_visualizer_tab()

        # --- Config tab ---
        self._build_config_tab()

        # --- Models tab ---
        self._build_models_tab()

        # ---- Status bar ----
        status_bar = ttk.Frame(self, padding=(12, 6))
        status_bar.grid(row=3, column=0, sticky="ew")
        status_bar.configure(style="Surface.TFrame")
        status_bar.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_bar, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w")
        self.status_label.configure(background=C.SURFACE0)

        self.timer_var = tk.StringVar(value="")
        self.timer_label = ttk.Label(status_bar, textvariable=self.timer_var, style="Status.TLabel")
        self.timer_label.grid(row=0, column=1, sticky="e")
        self.timer_label.configure(background=C.SURFACE0)

        self.shortcut_hint = ttk.Label(
            status_bar, text="Ctrl+R Run  Ctrl+N Next  Ctrl+F Filter  Ctrl+B Best Node  Esc Stop",
            style="Status.TLabel")
        self.shortcut_hint.grid(row=0, column=2, sticky="e", padx=(12, 0))
        self.shortcut_hint.configure(background=C.SURFACE0)

        self._sync_capabilities()

    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-r>", lambda _e: self._cmd_run())
        self.bind_all("<Control-n>", lambda _e: self._cmd_run_next())
        self.bind_all("<Escape>", lambda _e: self._stop_proc())
        self.bind_all("<Control-l>", lambda _e: self._clear_log())
        self.bind_all("<Control-a>", lambda _e: self._cmd_analyze())
        self.bind_all("<Control-f>", lambda _e: self._focus_run_filter())
        self.bind_all("<Control-b>", lambda _e: self._viz_select_best())
        self.bind_all("<Control-Shift-C>", lambda _e: self._copy_selected_run_id())
        self.bind_all("<Control-e>", lambda _e: self._viz_export_snapshot())

    def _focus_run_filter(self) -> None:
        self.run_filter_entry.focus_set()
        self.run_filter_entry.select_range(0, "end")

    def _bind_mousewheel(self, widget: tk.Widget, target: tk.Widget) -> None:
        def _on_mousewheel(event: tk.Event) -> str:
            delta = 0
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            else:
                if event.delta > 0:
                    delta = -1
                elif event.delta < 0:
                    delta = 1
                else:
                    return "break"
            target.yview_scroll(delta, "units")
            return "break"

        widget.bind("<MouseWheel>", _on_mousewheel)
        widget.bind("<Button-4>", _on_mousewheel)
        widget.bind("<Button-5>", _on_mousewheel)

    # ------------------------------------------------------------------ State
    def _selected_problem(self) -> Optional[Problem]:
        name = self.problem_var.get().strip()
        return self.problems.get(name)

    def _run_activity(self, last_mtime: Optional[float]) -> tuple[str, str]:
        if last_mtime is None:
            return ("NEW", C.OVERLAY0)
        age_s = max(0.0, time.time() - last_mtime)
        if age_s <= 120.0:
            return ("LIVE", C.GREEN)
        if age_s <= 1800.0:
            return ("WARM", C.YELLOW)
        return ("IDLE", C.SUBTEXT0)

    def _format_age(self, seconds: Optional[float]) -> str:
        if seconds is None:
            return "unknown"
        total = int(max(0.0, seconds))
        mins, sec = divmod(total, 60)
        hrs, mins = divmod(mins, 60)
        days, hrs = divmod(hrs, 24)
        if days > 0:
            return f"{days}d {hrs}h ago"
        if hrs > 0:
            return f"{hrs}h {mins}m ago"
        if mins > 0:
            return f"{mins}m {sec}s ago"
        return f"{sec}s ago"

    def _load_ui_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return

        problem = data.get("problem")
        if isinstance(problem, str) and problem in self.problems:
            self.problem_var.set(problem)

        for key, var in (
            ("run_id", self.run_id_var),
            ("island", self.island_var),
            ("cpu", self.cpu_var),
            ("load_ckpt", self.load_ckpt_var),
            ("winner_args", self.winner_args_var),
            ("cfg", self.cfg_var),
            ("run_filter", self.run_filter_var),
            ("run_sort", self.run_sort_var),
            ("run_status", self.run_status_var),
            ("run_min_score", self.run_min_score_var),
            ("viz_view", self.viz_view_var),
            ("viz_metric", self.viz_metric_var),
            ("viz_highlight", self.viz_highlight_var),
            ("viz_x_metric", self.viz_x_metric_var),
            ("viz_y_metric", self.viz_y_metric_var),
        ):
            val = data.get(key)
            if isinstance(val, str):
                var.set(val)

        for key, var in (
            ("advanced", self.advanced_var),
            ("skip_dryad", self.skip_dryad_var),
            ("no_taskset", self.no_taskset_var),
            ("auto_refresh", self.auto_refresh_var),
            ("notify_done", self.notify_done_var),
            ("viz_auto", self.viz_auto_var),
            ("viz_show_islands", self.viz_show_islands_var),
        ):
            val = data.get(key)
            if isinstance(val, bool):
                var.set(val)

        if self.run_sort_var.get() not in RUN_SORT_OPTIONS:
            self.run_sort_var.set(RUN_SORT_OPTIONS[0])
        if self.run_status_var.get() not in RUN_STATUS_OPTIONS:
            self.run_status_var.set(RUN_STATUS_OPTIONS[0])
        if self.viz_view_var.get() not in VIZ_VIEW_OPTIONS:
            self.viz_view_var.set(VIZ_VIEW_OPTIONS[0])
        if self.viz_highlight_var.get() not in VIZ_HIGHLIGHT_OPTIONS:
            self.viz_highlight_var.set(VIZ_HIGHLIGHT_OPTIONS[0])
        if self.viz_metric_var.get() not in set(self.viz_metric_combo.cget("values")):
            self.viz_metric_var.set("combined_score")
        if self.viz_x_metric_var.get() not in set(self.viz_x_metric_combo.cget("values")):
            self.viz_x_metric_var.set("delta_bic")
        if self.viz_y_metric_var.get() not in set(self.viz_y_metric_combo.cget("values")):
            self.viz_y_metric_var.set("quadrupole_score")

        api_base = data.get("api_base")
        if isinstance(api_base, str) and api_base.strip():
            values = list(self.api_base_combo.cget("values"))
            if api_base not in values:
                values.insert(0, api_base)
                self.api_base_combo.configure(values=values)
            self.api_base_var.set(api_base)

        self._toggle_advanced()
        self._toggle_viz_auto()

    def _save_ui_state(self) -> None:
        def _get_str(attr: str, default: str = "") -> str:
            var = getattr(self, attr, None)
            if var is None:
                return default
            try:
                return str(var.get()).strip()
            except Exception:
                return default

        def _get_bool(attr: str, default: bool = False) -> bool:
            var = getattr(self, attr, None)
            if var is None:
                return default
            try:
                return bool(var.get())
            except Exception:
                return default

        state = {
            "problem": _get_str("problem_var"),
            "run_id": _get_str("run_id_var"),
            "island": _get_str("island_var"),
            "cpu": _get_str("cpu_var"),
            "load_ckpt": _get_str("load_ckpt_var"),
            "winner_args": _get_str("winner_args_var"),
            "cfg": _get_str("cfg_var"),
            "api_base": _get_str("api_base_var"),
            "advanced": _get_bool("advanced_var"),
            "skip_dryad": _get_bool("skip_dryad_var"),
            "no_taskset": _get_bool("no_taskset_var"),
            "auto_refresh": _get_bool("auto_refresh_var"),
            "notify_done": _get_bool("notify_done_var"),
            "run_filter": _get_str("run_filter_var"),
            "run_sort": _get_str("run_sort_var"),
            "run_status": _get_str("run_status_var"),
            "run_min_score": _get_str("run_min_score_var"),
            "viz_view": _get_str("viz_view_var"),
            "viz_metric": _get_str("viz_metric_var"),
            "viz_highlight": _get_str("viz_highlight_var"),
            "viz_x_metric": _get_str("viz_x_metric_var"),
            "viz_y_metric": _get_str("viz_y_metric_var"),
            "viz_auto": _get_bool("viz_auto_var"),
            "viz_show_islands": _get_bool("viz_show_islands_var"),
        }
        try:
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception:
            pass

    def _on_auto_refresh_toggle(self) -> None:
        enabled = self.auto_refresh_var.get()
        if enabled:
            self._schedule_runs_auto_refresh(300)
            self._set_status("Auto-refresh enabled")
        else:
            if self._runs_auto_after:
                try:
                    self.after_cancel(self._runs_auto_after)
                except Exception:
                    pass
                self._runs_auto_after = None
            self._set_status("Auto-refresh paused")
        self._save_ui_state()

    def _on_run_sort_change(self) -> None:
        self._refresh_runs()
        self._save_ui_state()

    def _on_run_status_change(self) -> None:
        self._refresh_runs()
        self._save_ui_state()

    def _schedule_runs_auto_refresh(self, delay_ms: int) -> None:
        if self._runs_auto_after:
            try:
                self.after_cancel(self._runs_auto_after)
            except Exception:
                pass
            self._runs_auto_after = None
        self._runs_auto_after = self.after(delay_ms, self._runs_auto_tick)

    def _runs_auto_tick(self) -> None:
        self._runs_auto_after = None
        if self._closing or not self.auto_refresh_var.get():
            return

        self._refresh_runs()
        if self.run_id_var.get().strip():
            self._refresh_run_snapshot()

        with self._proc_lock:
            running = self._proc is not None
        self._schedule_runs_auto_refresh(3000 if running else 7000)

    def _sync_env_status(self) -> None:
        ok = bool(self.api_base_var.get().strip()) and bool(self.api_key_var.get().strip())
        self.env_status.configure(
            text=("API OK" if ok else "Missing API_BASE / API_KEY"),
            style=("Pill.Ok.TLabel" if ok else "Pill.Err.TLabel"),
        )

    def _sync_capabilities(self) -> None:
        p = self._selected_problem()
        if not p:
            for b in [self.btn_run, self.btn_run_next, self.btn_analyze, self.btn_winner,
                      self.btn_viz, self.btn_tail, self.btn_warmstart, self.btn_ls]:
                b.configure(state="disabled")
            return

        cmds = p.commands
        self.btn_run.configure(state=("normal" if "run" in cmds else "disabled"))
        self.btn_run_next.configure(state=("normal" if "run" in cmds else "disabled"))
        self.btn_analyze.configure(state=("normal" if "analyze" in cmds else "disabled"))
        self.btn_winner.configure(state=("normal" if "winner" in cmds else "disabled"))
        self.btn_tail.configure(state=("normal" if "tail" in cmds else "disabled"))
        self.btn_warmstart.configure(state=("normal" if "warmstart" in cmds else "disabled"))
        self.btn_ls.configure(state=("normal" if ("ls" in cmds or "list" in cmds) else "disabled"))
        self.btn_viz.configure(state=("normal" if ("viz" in cmds or "visualize" in cmds) else "disabled"))

        if "--skip-dryad" in p.help_text or "--require-dryad" in p.help_text:
            self.skip_dryad_chk.state(["!disabled"])
        else:
            self.skip_dryad_chk.state(["disabled"])

        self._sync_env_status()

    def _toggle_advanced(self) -> None:
        show = self.advanced_var.get()
        for w in getattr(self, "_adv_grid_widgets", []):
            if show:
                w.grid()
            else:
                w.grid_remove()

        if show:
            self._adv_sep.pack(side="left", fill="y", padx=8)
            self._adv_btns.pack(side="left")
        else:
            self._adv_btns.pack_forget()
            self._adv_sep.pack_forget()
        if hasattr(self, "_state_path"):
            self._save_ui_state()

    # ------------------------------------------------------------------ Problem refresh
    def _refresh_problems(self) -> None:
        """Re-scan for problems on disk and update the combo."""
        self.problems = _discover_problems(self.repo_root)
        names = sorted(self.problems.keys())
        prev = self.problem_var.get()
        self.problem_combo.configure(values=names)
        if prev not in names:
            self.problem_var.set(names[0] if names else "")
        self._on_problem_change()
        self._set_status(f"Found {len(names)} problems")
        self._save_ui_state()

    # ------------------------------------------------------------------ New problem wizard

    def _new_problem_dialog(self) -> None:
        """Open a dialog to create a new problem from templates."""
        dlg = tk.Toplevel(self)
        dlg.title("New Problem")
        dlg.configure(bg=C.BASE)
        dlg.geometry("860x760")
        dlg.transient(self)
        dlg.grab_set()

        # -- Top: problem name
        top = ttk.Frame(dlg, padding=(12, 10, 12, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Problem name:").pack(side="left")
        name_var = tk.StringVar()
        name_entry = ttk.Entry(top, textvariable=name_var, width=30)
        name_entry.pack(side="left", padx=(6, 0))
        name_entry.focus_set()

        # -- Scrollable body
        body_canvas = tk.Canvas(dlg, bg=C.BASE, highlightthickness=0)
        body_sb = ttk.Scrollbar(dlg, orient="vertical", command=body_canvas.yview)
        body_frame = ttk.Frame(body_canvas)
        body_frame.bind("<Configure>",
                        lambda _e: body_canvas.configure(scrollregion=body_canvas.bbox("all")))
        body_canvas.create_window((0, 0), window=body_frame, anchor="nw")
        body_canvas.configure(yscrollcommand=body_sb.set)
        body_sb.pack(side="right", fill="y")
        body_canvas.pack(fill="both", expand=True, padx=12, pady=(0, 4))
        # Mousewheel
        self._bind_mousewheel(body_canvas, body_canvas)
        self._bind_mousewheel(body_frame, body_canvas)

        ai_model_var = tk.StringVar(value=self._suggest_new_problem_ai_model())
        ai_timeout_var = tk.StringVar(value="240")
        use_ai_model_in_config_var = tk.BooleanVar(value=True)

        def _make_editor(parent, label, template, height=10):
            lf = ttk.LabelFrame(parent, text=f"  {label}  ", padding=6)
            lf.pack(fill="x", pady=(0, 8))
            txt = tk.Text(lf, height=height, bg=C.SURFACE0, fg=C.TEXT,
                          insertbackground=C.TEXT, selectbackground=C.BLUE,
                          selectforeground=C.CRUST, font=FONT_MONO,
                          wrap="none", undo=True, borderwidth=0)
            txt.pack(fill="x", expand=True)
            txt.insert("1.0", template)
            return txt

        ai_lf = ttk.LabelFrame(body_frame, text="  AI Agent  ", padding=8)
        ai_lf.pack(fill="x", pady=(0, 8))

        ai_cfg_row = ttk.Frame(ai_lf)
        ai_cfg_row.pack(fill="x")
        ttk.Label(ai_cfg_row, text="Model:").pack(side="left")
        ttk.Entry(ai_cfg_row, textvariable=ai_model_var, width=32).pack(side="left", padx=(6, 10))
        ttk.Button(
            ai_cfg_row,
            text="Use lfm2.5-thinking:1.2b",
            command=lambda: ai_model_var.set("lfm2.5-thinking:1.2b"),
        ).pack(side="left", padx=(0, 10))
        ttk.Label(ai_cfg_row, text="Timeout:").pack(side="left")
        ttk.Entry(ai_cfg_row, textvariable=ai_timeout_var, width=6).pack(side="left", padx=(6, 2))
        ttk.Label(ai_cfg_row, text="s", style="Dim.TLabel").pack(side="left")

        ai_flags_row = ttk.Frame(ai_lf)
        ai_flags_row.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            ai_flags_row,
            text="Write this model into new config (explore/exploit/prompt agent)",
            variable=use_ai_model_in_config_var,
        ).pack(side="left")

        ttk.Label(
            ai_lf,
            text="Describe objective, constraints, and fitness metric; then generate init/evaluate/prompt files.",
            style="Dim.TLabel",
        ).pack(anchor="w", pady=(8, 4))
        ai_desc_txt = tk.Text(
            ai_lf,
            height=6,
            bg=C.SURFACE0,
            fg=C.TEXT,
            insertbackground=C.TEXT,
            selectbackground=C.BLUE,
            selectforeground=C.CRUST,
            font=FONT_MONO,
            wrap="word",
            undo=True,
            borderwidth=0,
        )
        ai_desc_txt.pack(fill="x", expand=True)
        ai_desc_txt.insert(
            "1.0",
            "Describe your idea here. Include:\n"
            "- objective and what candidate code should output\n"
            "- fitness definition (higher is better)\n"
            "- constraints and edge cases\n"
            "- quick test example(s)\n",
        )

        ai_status_row = ttk.Frame(ai_lf)
        ai_status_row.pack(fill="x", pady=(6, 0))
        ai_status_lbl = ttk.Label(ai_status_row, text="", style="Dim.TLabel")
        ai_status_lbl.pack(side="left")
        gen_btn = ttk.Button(ai_status_row, text="Generate with AI")
        gen_btn.pack(side="right")

        eval_txt = _make_editor(body_frame, "evaluate.py", _EVAL_TEMPLATE.format(name="my_problem"), 18)
        init_txt = _make_editor(body_frame, "init_program.py  (between EVOLVE-BLOCK markers)", _INIT_TEMPLATE, 12)
        sysmsg_txt = _make_editor(body_frame, "SYS_MSG  (LLM system prompt)", _SYSMSG_TEMPLATE, 12)

        def _set_editor_text(editor: tk.Text, value: str) -> None:
            editor.delete("1.0", "end")
            editor.insert("1.0", value)

        def _set_ai_status(text: str, color: str) -> None:
            ai_status_lbl.configure(text=text, foreground=color)
            try:
                dlg.update_idletasks()
            except Exception:
                pass

        def _generate_with_ai() -> None:
            api_base = self.api_base_var.get().strip()
            api_key = self.api_key_var.get().strip()
            if not api_base or not api_key:
                messagebox.showerror(
                    "Missing API Settings",
                    "Set API_BASE and API_KEY in the dashboard header before using AI generation.",
                    parent=dlg,
                )
                return

            model = ai_model_var.get().strip() or _AI_TEMPLATE_MODEL_DEFAULT
            description = ai_desc_txt.get("1.0", "end-1c").strip()
            if not description:
                _set_ai_status("Add a problem description first.", C.RED)
                messagebox.showwarning(
                    "Missing Description",
                    "Describe the objective/fitness before generating.",
                    parent=dlg,
                )
                return

            try:
                timeout_s = float(ai_timeout_var.get().strip() or "240")
            except Exception:
                _set_ai_status("Timeout must be numeric.", C.RED)
                messagebox.showwarning("Invalid Timeout", "Timeout must be numeric.", parent=dlg)
                return
            if timeout_s <= 0:
                _set_ai_status("Timeout must be > 0.", C.RED)
                messagebox.showwarning("Invalid Timeout", "Timeout must be > 0.", parent=dlg)
                return

            hint = name_var.get().strip()
            gen_btn.configure(state="disabled")
            _set_ai_status(f"Generating with {model}...", C.YELLOW)

            def _worker() -> None:
                result: Optional[dict[str, str]] = None
                err: Optional[str] = None
                try:
                    result = _api_generate_problem_templates(
                        api_base=api_base,
                        api_key=api_key,
                        model=model,
                        description=description,
                        problem_name_hint=hint,
                        timeout_s=timeout_s,
                    )
                except Exception as exc:
                    err = str(exc)

                def _apply() -> None:
                    if not dlg.winfo_exists():
                        return
                    gen_btn.configure(state="normal")
                    if err or result is None:
                        msg = err or "unknown error"
                        _set_ai_status(
                            f"AI generation failed: {msg}",
                            C.RED,
                        )
                        messagebox.showerror(
                            "AI Generation Failed",
                            f"Model: {model}\n\n{msg}",
                            parent=dlg,
                        )
                        return
                    suggested = (result.get("suggested_problem_name", "") or "").strip()
                    if suggested:
                        name_var.set(_sanitize_problem_name(suggested))
                    _set_editor_text(eval_txt, result.get("evaluator_py", ""))
                    _set_editor_text(init_txt, result.get("init_program_py", ""))
                    _set_editor_text(sysmsg_txt, result.get("sys_msg", ""))
                    _set_ai_status(f"Generated via {model}", C.GREEN)
                    self._set_status(f"AI templates generated with {model}")

                self.after(0, _apply)

            threading.Thread(target=_worker, daemon=True).start()

        gen_btn.configure(command=_generate_with_ai)

        # -- Bottom buttons
        btn_row = ttk.Frame(dlg, padding=(12, 6, 12, 10))
        btn_row.pack(fill="x")
        status_lbl = ttk.Label(btn_row, text="", style="Dim.TLabel")
        status_lbl.pack(side="left")
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(btn_row, text="Create Problem", style="Accent.TButton",
                   command=lambda: self._create_problem(
                       dlg, name_var.get().strip(),
                       eval_txt.get("1.0", "end-1c"),
                       init_txt.get("1.0", "end-1c"),
                       sysmsg_txt.get("1.0", "end-1c"),
                       ai_model_var.get().strip() if use_ai_model_in_config_var.get() else "",
                       status_lbl,
                   )).pack(side="right")

    def _create_problem(self, dlg: tk.Toplevel, name: str,
                        eval_code: str, init_code: str, sysmsg: str,
                        agent_model: str, status_lbl: ttk.Label) -> None:
        """Validate inputs and create all problem files on disk."""
        # Validate name
        if not name:
            status_lbl.configure(text="Name cannot be empty", foreground=C.RED)
            return
        if not all(c.isalnum() or c == "_" for c in name):
            status_lbl.configure(text="Name: only letters, digits, underscores", foreground=C.RED)
            return
        prob_dir = self.repo_root / "problems" / name
        if prob_dir.exists():
            status_lbl.configure(text=f"'{name}' already exists", foreground=C.RED)
            return

        prompt_from_init, init_remainder = _extract_marked_block(
            init_code, "# PROMPT-BLOCK-START", "# PROMPT-BLOCK-END"
        )
        if prompt_from_init and not (sysmsg or "").strip():
            sysmsg = prompt_from_init
            init_code = init_remainder
        evolve_from_sys, sys_remainder = _extract_marked_block(
            sysmsg, "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"
        )
        if evolve_from_sys and not (init_code or "").strip():
            init_code = "import math\n\n" + evolve_from_sys
            sysmsg = sys_remainder

        # Normalize generated/edited files into strict CodeEvolve format.
        eval_code = _ensure_generated_evaluator(eval_code, problem_name=name)
        init_code = _ensure_generated_init_program(init_code)
        init_code = _ensure_marked_block(
            init_code, "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"
        )
        sysmsg = _ensure_marked_block(
            sysmsg, "# PROMPT-BLOCK-START", "# PROMPT-BLOCK-END"
        )

        status_lbl.configure(text="Creating...", foreground=C.YELLOW)
        dlg.update_idletasks()

        try:
            # Create directory structure
            (prob_dir / "configs").mkdir(parents=True)
            (prob_dir / "input" / "src").mkdir(parents=True)

            # evaluate.py
            (prob_dir / "input" / "evaluate.py").write_text(eval_code, encoding="utf-8")

            # init_program.py
            (prob_dir / "input" / "src" / "init_program.py").write_text(init_code, encoding="utf-8")

            # config.yaml
            agent_model = agent_model.strip()
            if agent_model == _NONE_MODEL:
                agent_model = ""

            evolve_cfg = dict(_DEFAULT_EVOLVE_CONFIG)
            exploration_ensemble: list[dict[str, Any]] = []
            exploitation_ensemble: list[dict[str, Any]] = []
            sampler_aux = {
                "model_name": "",
                "temp": 0.18,
                "top_p": 0.78,
                "max_tok": 4096,
                "retries": 3,
                "request_timeout_s": 240.0,
                "weight": 1,
                "verify_ssl": False,
            }

            if agent_model:
                evolve_cfg["meta_prompting"] = True
                exploration_ensemble = [{
                    "model_name": agent_model,
                    "temp": 0.75,
                    "top_p": 0.92,
                    "max_tok": 4096,
                    "retries": 3,
                    "request_timeout_s": 240.0,
                    "weight": 1,
                    "verify_ssl": False,
                }]
                exploitation_ensemble = [{
                    "model_name": agent_model,
                    "temp": 0.22,
                    "top_p": 0.82,
                    "max_tok": 4096,
                    "retries": 3,
                    "request_timeout_s": 240.0,
                    "weight": 1,
                    "verify_ssl": False,
                }]
                sampler_aux["model_name"] = agent_model

            cfg = {
                "SYS_MSG": sysmsg + "\n",
                "CODEBASE_PATH": "src/",
                "INIT_FILE_DATA": {"filename": "init_program.py", "language": "python"},
                "EVAL_FILE_NAME": "evaluate.py",
                "EVAL_TIMEOUT": 30,
                "SEED": 42,
                "MAX_MEM_BYTES": 5_000_000_000,
                "MEM_CHECK_INTERVAL_S": 0.1,
                "EVOLVE_CONFIG": evolve_cfg,
                "EXPLORATION_ENSEMBLE": exploration_ensemble,
                "EXPLOITATION_ENSEMBLE": exploitation_ensemble,
                "SAMPLER_AUX_LM": sampler_aux,
                "EMBEDDING": {
                    "model_name": "",
                    "retries": 3,
                    "request_timeout_s": 240.0,
                    "verify_ssl": False,
                },
            }
            cfg_path = prob_dir / "configs" / "config.yaml"
            cfg_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False,
                                          allow_unicode=True, width=120),
                                encoding="utf-8")

            # Shell script
            sh_path = prob_dir / f"{name}.sh"
            sh_path.write_text(_SH_TEMPLATE.format(name=name), encoding="utf-8")
            sh_path.chmod(0o755)

            # run.sh
            run_sh = prob_dir / "run.sh"
            run_sh.write_text(
                f'#!/usr/bin/env bash\nset -euo pipefail\n'
                f'SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"\n'
                f'exec "$SCRIPT_DIR/{name}.sh" run "$@"\n',
                encoding="utf-8",
            )
            run_sh.chmod(0o755)

            # find_winner.py
            (prob_dir / "find_winner.py").write_text(
                _FIND_WINNER_TEMPLATE.format(name=name), encoding="utf-8")

        except Exception as exc:
            status_lbl.configure(text=f"Error: {exc}", foreground=C.RED)
            return

        # Refresh and select the new problem
        self._refresh_problems()
        self.problem_var.set(name)
        self._on_problem_change()
        if agent_model:
            self._set_status(f"Created problem: {name} (agent {agent_model})")
        else:
            self._set_status(f"Created problem: {name}")
        dlg.destroy()

    # ------------------------------------------------------------------ Island fitness chart
    _RE_CHILD_SOL = re.compile(
        r"fitness=([\d.]+).*?iteration_found=(\d+)")

    def _refresh_island_fitness(self) -> None:
        """Deprecated chart hook; keep for compatibility."""
        self._refresh_run_snapshot()
        return

        p = self._selected_problem()
        if not p:
            return
        run_id = self.run_id_var.get().strip()
        if not run_id:
            self._chart_label("Select a run")
            return

        run_dir = self.experiments_dir / p.name / run_id
        if not run_dir.exists():
            self._chart_label(f"Not found: {run_id}")
            return

        fitness_key = "combined_score"
        cfg = self._load_config()
        if isinstance(cfg, dict):
            ec = cfg.get("EVOLVE_CONFIG", {})
            if isinstance(ec, dict):
                fitness_key = ec.get("fitness_key", fitness_key)

        islands = sorted(
            [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )
        if not islands:
            self._chart_label("No islands found")
            return

        # Parse fitness trajectories from logs
        # all_series: list of (island_idx, [(step, fitness), ...])
        all_series: list[tuple[int, list[tuple[int, float]]]] = []
        global_best = 0.0
        global_max_step = 1

        for island_dir in islands:
            idx = int(island_dir.name)
            log_path = _island_log(island_dir)
            points: list[tuple[int, float]] = []
            if log_path is not None:
                try:
                    with log_path.open("r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            m = self._RE_CHILD_SOL.search(line)
                            if m:
                                fit = float(m.group(1))
                                step = int(m.group(2))
                                points.append((step, fit))
                except Exception:
                    pass
            if points:
                all_series.append((idx, points))
                local_max = max(f for _, f in points)
                if local_max > global_best:
                    global_best = local_max
                local_max_step = max(s for s, _ in points)
                if local_max_step > global_max_step:
                    global_max_step = local_max_step

        if not all_series:
            self._chart_label("No data yet")
            return

        self._draw_chart(all_series, global_best, global_max_step)

        # Summary text
        summaries = []
        for idx, pts in all_series:
            best = max(f for _, f in pts)
            color_name = self._island_colors[idx % len(self._island_colors)]
            summaries.append(f"I{idx}:{best:.3f}")
        self._chart_summary.configure(text="  ".join(summaries))

    def _chart_label(self, text: str) -> None:
        """Draw a centered text label on the chart canvas."""
        self._chart.update_idletasks()
        w = self._chart.winfo_width() or 240
        h = self._chart.winfo_height() or 160
        self._chart.create_text(w // 2, h // 2, text=text, fill=C.OVERLAY0,
                                font=FONT_MONO_SM, anchor="center")

    def _draw_chart(self, all_series: list[tuple[int, list[tuple[int, float]]]],
                    y_max: float, x_max: int) -> None:
        """Draw the evolution fitness chart on the canvas."""
        self._chart.update_idletasks()
        cw = max(200, self._chart.winfo_width() or 240)
        ch = max(100, self._chart.winfo_height() or 160)

        pad_l, pad_r, pad_t, pad_b = 32, 8, 12, 20
        plot_w = cw - pad_l - pad_r
        plot_h = ch - pad_t - pad_b

        if plot_w < 20 or plot_h < 20:
            return

        y_min = 0.0
        y_range = max(0.01, y_max - y_min) * 1.05
        x_range = max(1, x_max)

        def tx(step: int) -> float:
            return pad_l + (step / x_range) * plot_w

        def ty(fit: float) -> float:
            return pad_t + plot_h - ((fit - y_min) / y_range) * plot_h

        # Grid lines
        for i in range(5):
            yv = y_min + (y_range / 4) * i
            py = ty(yv)
            self._chart.create_line(pad_l, py, cw - pad_r, py, fill=C.SURFACE1, dash=(2, 4))
            if i % 2 == 0:
                self._chart.create_text(pad_l - 4, py, text=f"{yv:.2f}", fill=C.OVERLAY0,
                                        font=("Ubuntu Sans Mono", 7), anchor="e")

        # X axis label
        self._chart.create_text(pad_l + plot_w // 2, ch - 2, text="epoch",
                                fill=C.OVERLAY0, font=("Ubuntu Sans Mono", 7), anchor="s")

        # Plot each island
        for island_idx, points in all_series:
            color = self._island_colors[island_idx % len(self._island_colors)]

            # Compute running best trajectory
            running_best: list[tuple[int, float]] = []
            best_so_far = 0.0
            for step, fit in sorted(points, key=lambda p: p[0]):
                if fit > best_so_far:
                    best_so_far = fit
                    running_best.append((step, best_so_far))

            # Draw running best as line
            if len(running_best) >= 2:
                coords = []
                for step, fit in running_best:
                    coords.extend([tx(step), ty(fit)])
                self._chart.create_line(*coords, fill=color, width=2, smooth=False)

            # Draw individual evaluations as small dots
            for step, fit in points:
                if fit > 0:
                    px, py = tx(step), ty(fit)
                    r = 2
                    self._chart.create_oval(px - r, py - r, px + r, py + r,
                                            fill=color, outline="")

            # Mark current best with larger dot
            if running_best:
                last_step, last_fit = running_best[-1]
                px, py = tx(last_step), ty(last_fit)
                r = 4
                self._chart.create_oval(px - r, py - r, px + r, py + r,
                                        fill=color, outline=C.TEXT, width=1)

        # Legend (compact, top-right corner)
        lx = cw - pad_r - 4
        ly = pad_t + 2
        for island_idx, _ in all_series:
            color = self._island_colors[island_idx % len(self._island_colors)]
            self._chart.create_rectangle(lx - 16, ly, lx - 8, ly + 6, fill=color, outline="")
            self._chart.create_text(lx - 18, ly + 3, text=str(island_idx), fill=C.SUBTEXT0,
                                    font=("Ubuntu Sans Mono", 7), anchor="e")
            ly += 10

    # ------------------------------------------------------------------ Run Snapshot
    def _set_snapshot_text(self, text: str) -> None:
        self._snapshot.configure(state="normal")
        self._snapshot.delete("1.0", "end")
        self._snapshot.insert("1.0", text)
        self._snapshot.configure(state="disabled")

    def _open_run_dir(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        run_id = self.run_id_var.get().strip()
        if not run_id:
            return
        _open_path(self.experiments_dir / p.name / run_id)

    def _open_best_sol(self) -> None:
        path = self._find_best_artifact("best_sol.py")
        if path:
            _open_path(path)
        else:
            messagebox.showinfo("Best Solution", "No best_sol.py found for this run yet.")

    def _open_best_prompt(self) -> None:
        path = self._find_best_artifact("best_prompt.txt")
        if path:
            _open_path(path)
        else:
            messagebox.showinfo("Best Prompt", "No best_prompt.txt found for this run yet.")

    def _apply_best(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        run_id = self.run_id_var.get().strip()
        if not run_id:
            messagebox.showerror("Missing Run", "Set a run name before applying best artifacts.")
            return

        best_sol = self._find_best_artifact("best_sol.py")
        best_prompt = self._find_best_artifact("best_prompt.txt")
        missing = []
        if not best_sol:
            missing.append("best_sol.py")
        if not best_prompt:
            missing.append("best_prompt.txt")
        if missing:
            messagebox.showinfo(
                "Apply Best",
                "Missing artifacts for this run:\n- " + "\n- ".join(missing),
            )
            return

        init_path = p.prob_dir / "input" / "src" / "init_program.py"
        if not init_path.exists():
            messagebox.showerror("Apply Best", f"init_program.py not found:\n{init_path}")
            return

        cfg_path = self._selected_cfg_path()
        if not cfg_path or not cfg_path.exists():
            messagebox.showerror("Apply Best", "Select a config to update SYS_MSG.")
            return
        if yaml is None:
            messagebox.showerror("Apply Best", "PyYAML is required to update SYS_MSG.")
            return

        if not messagebox.askokcancel(
            "Apply Best",
            "This will overwrite:\n"
            f"- {init_path}\n"
            f"- SYS_MSG in {cfg_path.name}\n\n"
            "Continue?",
        ):
            return

        try:
            init_path.write_text(best_sol.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Apply Best", f"Failed to update init_program.py:\n{e}")
            return

        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            prompt_txt = best_prompt.read_text(encoding="utf-8")
            if not prompt_txt.endswith("\n"):
                prompt_txt += "\n"
            cfg["SYS_MSG"] = prompt_txt
            self._dump_config(cfg, cfg_path)
        except Exception as e:
            messagebox.showerror("Apply Best", f"Failed to update SYS_MSG:\n{e}")
            return

        self._refresh_config_editor()
        self._refresh_models_from_config()
        self._set_status("Applied best solution + prompt")
        messagebox.showinfo("Apply Best", "Updated init_program.py and SYS_MSG.")

    def _find_best_artifact(self, filename: str) -> Optional[Path]:
        p = self._selected_problem()
        if not p:
            return None
        run_id = self.run_id_var.get().strip()
        if not run_id:
            return None
        run_dir = self.experiments_dir / p.name / run_id
        direct = run_dir / filename
        if direct.exists():
            return direct

        best_fit = None
        best_path = None
        for island_dir in run_dir.iterdir() if run_dir.exists() else []:
            if not island_dir.is_dir() or not island_dir.name.isdigit():
                continue
            log_path = _island_log(island_dir)
            if log_path is None:
                continue
            try:
                local_best = None
                with log_path.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        m = self._RE_CHILD_SOL.search(line)
                        if m:
                            fit = float(m.group(1))
                            if local_best is None or fit > local_best:
                                local_best = fit
                if local_best is None:
                    continue
                if best_fit is None or local_best > best_fit:
                    candidate = island_dir / filename
                    if candidate.exists():
                        best_fit = local_best
                        best_path = candidate
            except Exception:
                continue
        return best_path

    def _refresh_run_snapshot(self) -> None:
        p = self._selected_problem()
        run_id = self.run_id_var.get().strip()
        if not p or not run_id:
            self._set_snapshot_text("Select a run to see summary.")
            return

        run_dir = self.experiments_dir / p.name / run_id
        if not run_dir.exists():
            self._set_snapshot_text(f"Run not found: {run_id}")
            return

        islands = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                         key=lambda d: int(d.name))
        if not islands:
            self._set_snapshot_text("No islands found yet.")
            return

        summaries = []
        global_best = None
        global_best_island = None
        latest_epoch = 0
        last_mtime = None

        for island_dir in islands:
            idx = int(island_dir.name)
            log_path = _island_log(island_dir)
            local_best = None
            local_epoch = 0
            if log_path is not None:
                try:
                    with log_path.open("r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            m = self._RE_CHILD_SOL.search(line)
                            if m:
                                fit = float(m.group(1))
                                step = int(m.group(2))
                                if local_best is None or fit > local_best:
                                    local_best = fit
                                if step > local_epoch:
                                    local_epoch = step
                    mtime = log_path.stat().st_mtime
                    last_mtime = mtime if last_mtime is None or mtime > last_mtime else last_mtime
                except Exception:
                    pass

            if local_best is not None:
                summaries.append(f"I{idx}:{local_best:.3f}")
                if global_best is None or local_best > global_best:
                    global_best = local_best
                    global_best_island = idx
            latest_epoch = max(latest_epoch, local_epoch)

        updated_str = "unknown"
        activity = "NEW"
        age_str = "unknown"
        if last_mtime is not None:
            updated_str = datetime.fromtimestamp(last_mtime).strftime("%Y-%m-%d %H:%M:%S")
            age_str = self._format_age(time.time() - last_mtime)
        activity, _ = self._run_activity(last_mtime)

        lines = [
            f"Run: {run_id}",
            f"Islands: {len(islands)}",
            f"Latest epoch: {latest_epoch}",
            f"Activity: {activity} ({age_str})",
            f"Last update: {updated_str}",
        ]
        if global_best is not None:
            lines.append(f"Global best: {global_best:.4f} (I{global_best_island})")
        if summaries:
            lines.append("")
            lines.append("Per island:")
            lines.append("  " + "  ".join(summaries))

        # Check run root and island subdirs for best artifacts
        best_sol = (run_dir / "best_sol.py").exists() or any(
            (d / "best_sol.py").exists() for d in islands)
        best_prompt = (run_dir / "best_prompt.txt").exists() or any(
            (d / "best_prompt.txt").exists() for d in islands)
        lines.append("")
        lines.append(f"best_sol.py: {'yes' if best_sol else 'no'}")
        lines.append(f"best_prompt.txt: {'yes' if best_prompt else 'no'}")

        self._set_snapshot_text("\n".join(lines))

    # ------------------------------------------------------------------ Visualizer
    _RE_CKPT = re.compile(r"ckpt_(\d+)\.pkl$")

    def _build_visualizer_tab(self) -> None:
        viz_tab = ttk.Frame(self._notebook)
        self._notebook.add(viz_tab, text="  Visualizer  ")
        viz_tab.columnconfigure(0, weight=1)
        viz_tab.rowconfigure(1, weight=1)

        ctrl = ttk.Frame(viz_tab, padding=(10, 8), style="Toolbar.TFrame")
        ctrl.grid(row=0, column=0, sticky="ew")
        ctrl.columnconfigure(18, weight=1)

        self.viz_view_var = tk.StringVar(value=VIZ_VIEW_OPTIONS[0])
        self.viz_metric_var = tk.StringVar(value="combined_score")
        self.viz_highlight_var = tk.StringVar(value=VIZ_HIGHLIGHT_OPTIONS[0])
        self.viz_x_metric_var = tk.StringVar(value="delta_bic")
        self.viz_y_metric_var = tk.StringVar(value="quadrupole_score")
        self.viz_show_islands_var = tk.BooleanVar(value=True)
        self.viz_auto_var = tk.BooleanVar(value=True)
        self.viz_epoch_var = tk.StringVar(value="")
        self.viz_find_var = tk.StringVar(value="")

        ttk.Label(ctrl, text="View:").grid(row=0, column=0, sticky="w")
        self.viz_view_combo = ttk.Combobox(
            ctrl,
            textvariable=self.viz_view_var,
            values=VIZ_VIEW_OPTIONS,
            state="readonly",
            width=12,
        )
        self.viz_view_combo.grid(row=0, column=1, sticky="w", padx=(4, 12))
        self.viz_view_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_viz_display_change())

        ttk.Label(ctrl, text="Metric:").grid(row=0, column=2, sticky="w")
        self.viz_metric_combo = ttk.Combobox(
            ctrl,
            textvariable=self.viz_metric_var,
            values=[
                "combined_score",
                "fitness",
                "delta_bic",
                "chi2_total",
                "chi2_tt",
                "chi2_ee",
                "chi2_te",
            ],
            state="readonly",
            width=18,
        )
        self.viz_metric_combo.grid(row=0, column=3, sticky="w", padx=(4, 12))
        self.viz_metric_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_viz_metric_change())

        ttk.Label(ctrl, text="Highlight:").grid(row=0, column=4, sticky="w")
        self.viz_highlight_combo = ttk.Combobox(
            ctrl,
            textvariable=self.viz_highlight_var,
            values=VIZ_HIGHLIGHT_OPTIONS,
            state="readonly",
            width=12,
        )
        self.viz_highlight_combo.grid(row=0, column=5, sticky="w", padx=(4, 12))
        self.viz_highlight_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_viz_display_change())

        ttk.Checkbutton(
            ctrl, text="Show islands", variable=self.viz_show_islands_var,
            command=self._on_viz_display_change,
        ).grid(row=0, column=6, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="X dim:").grid(row=0, column=7, sticky="w")
        self.viz_x_metric_combo = ttk.Combobox(
            ctrl,
            textvariable=self.viz_x_metric_var,
            values=[
                "delta_bic",
                "quadrupole_score",
                "chi2_total",
                "smoothness_score",
                "combined_score",
            ],
            state="readonly",
            width=16,
        )
        self.viz_x_metric_combo.grid(row=0, column=8, sticky="w", padx=(4, 8))
        self.viz_x_metric_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_viz_display_change())

        ttk.Label(ctrl, text="Y dim:").grid(row=0, column=9, sticky="w")
        self.viz_y_metric_combo = ttk.Combobox(
            ctrl,
            textvariable=self.viz_y_metric_var,
            values=[
                "quadrupole_score",
                "delta_bic",
                "chi2_total",
                "asymptotic_score",
                "combined_score",
            ],
            state="readonly",
            width=16,
        )
        self.viz_y_metric_combo.grid(row=0, column=10, sticky="w", padx=(4, 12))
        self.viz_y_metric_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_viz_display_change())

        ttk.Checkbutton(
            ctrl, text="Auto (5 epochs)", variable=self.viz_auto_var,
            command=self._on_viz_auto_toggle,
        ).grid(row=0, column=11, sticky="w", padx=(0, 12))

        ttk.Button(ctrl, text="Refresh", command=self._refresh_visualizer).grid(
            row=0, column=12, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, text="Find:").grid(row=0, column=13, sticky="w")
        viz_find_entry = ttk.Entry(ctrl, textvariable=self.viz_find_var, width=20)
        viz_find_entry.grid(row=0, column=14, sticky="w", padx=(4, 6))
        viz_find_entry.bind("<Return>", lambda _e: self._viz_find_and_select())

        ttk.Button(ctrl, text="Find Node", command=self._viz_find_and_select).grid(
            row=0, column=15, sticky="w", padx=(0, 6)
        )
        ttk.Button(ctrl, text="Focus Best", command=self._viz_select_best).grid(
            row=0, column=16, sticky="w", padx=(0, 6)
        )
        ttk.Button(ctrl, text="Copy Details", command=self._viz_copy_details).grid(
            row=0, column=17, sticky="w", padx=(0, 6)
        )
        ttk.Button(ctrl, text="Export", command=self._viz_export_snapshot).grid(
            row=0, column=18, sticky="w", padx=(0, 12)
        )

        ttk.Label(ctrl, textvariable=self.viz_epoch_var, style="Dim.TLabel",
                  font=FONT_MONO_SM).grid(row=1, column=0, columnspan=19, sticky="w", pady=(4, 0))

        body = ttk.PanedWindow(viz_tab, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        canvas_frame = ttk.Frame(body)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        body.add(canvas_frame, weight=3)

        self._viz_canvas = tk.Canvas(
            canvas_frame, bg=C.SURFACE0, highlightthickness=0, borderwidth=0,
        )
        self._viz_canvas.grid(row=0, column=0, sticky="nsew")
        self._viz_canvas.bind("<Configure>", lambda _e: self._refresh_visualizer())
        self._viz_canvas.bind("<Button-1>", self._on_viz_click)

        detail_frame = ttk.Frame(body, padding=(10, 0, 0, 0))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(1, weight=1)
        body.add(detail_frame, weight=1)

        ttk.Label(detail_frame, text="Selection", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(4, 6)
        )
        self._viz_side_notebook = ttk.Notebook(detail_frame)
        self._viz_side_notebook.grid(row=1, column=0, sticky="nsew")

        details_tab = ttk.Frame(self._viz_side_notebook)
        self._viz_program_tab = details_tab
        details_tab.columnconfigure(0, weight=1)
        details_tab.rowconfigure(0, weight=1)
        self._viz_side_notebook.add(details_tab, text="  Program  ")
        self._viz_detail = ScrolledText(
            details_tab, height=16, wrap="word",
            bg=C.MANTLE, fg=C.TEXT, insertbackground=C.TEXT,
            highlightthickness=0, borderwidth=0, padx=8, pady=6,
            font=FONT_MONO,
        )
        self._viz_detail.grid(row=0, column=0, sticky="nsew")
        self._viz_detail.configure(state="disabled")

        diff_tab = ttk.Frame(self._viz_side_notebook)
        self._viz_diff_tab = diff_tab
        diff_tab.columnconfigure(0, weight=1)
        diff_tab.rowconfigure(0, weight=1)
        self._viz_side_notebook.add(diff_tab, text="  Diff  ")
        self._viz_diff = ScrolledText(
            diff_tab, height=12, wrap="none",
            bg=C.MANTLE, fg=C.TEXT, insertbackground=C.TEXT,
            highlightthickness=0, borderwidth=0, padx=8, pady=6,
            font=FONT_MONO_SM,
        )
        self._viz_diff.grid(row=0, column=0, sticky="nsew")
        self._viz_diff.configure(state="disabled")

        map_tab = ttk.Frame(self._viz_side_notebook)
        self._viz_map_tab = map_tab
        map_tab.columnconfigure(0, weight=1)
        map_tab.rowconfigure(0, weight=1)
        self._viz_side_notebook.add(map_tab, text="  MAP-Elites  ")
        self._viz_map_detail = ScrolledText(
            map_tab, height=10, wrap="word",
            bg=C.MANTLE, fg=C.TEXT, insertbackground=C.TEXT,
            highlightthickness=0, borderwidth=0, padx=8, pady=6,
            font=FONT_MONO_SM,
        )
        self._viz_map_detail.grid(row=0, column=0, sticky="nsew")
        self._viz_map_detail.configure(state="disabled")

        self._set_viz_diff_text("Select a node to inspect mutation diff against its parent.")
        self._set_viz_map_text("MAP-Elites metadata will appear after loading checkpoints.")

        # Kick off auto-refresh loop
        self._toggle_viz_auto()

    def _toggle_viz_auto(self) -> None:
        if self._viz_after:
            try:
                self.after_cancel(self._viz_after)
            except Exception:
                pass
            self._viz_after = None
        if self.viz_auto_var.get():
            self._viz_last_ckpt = None
            self._viz_after = self.after(1500, self._maybe_refresh_visualizer)

    def _on_viz_auto_toggle(self) -> None:
        self._toggle_viz_auto()
        self._save_ui_state()

    def _on_viz_metric_change(self) -> None:
        self._refresh_visualizer()
        self._save_ui_state()

    def _on_viz_display_change(self) -> None:
        self._refresh_visualizer()
        self._save_ui_state()

    def _set_text_widget(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_viz_diff_text(self, text: str) -> None:
        self._set_text_widget(self._viz_diff, text)

    def _set_viz_map_text(self, text: str) -> None:
        self._set_text_widget(self._viz_map_detail, text)

    def _viz_active_detail_text(self) -> str:
        if not hasattr(self, "_viz_side_notebook"):
            return self._viz_detail.get("1.0", "end-1c")
        selected = self._viz_side_notebook.select()
        if hasattr(self, "_viz_diff_tab") and selected == str(self._viz_diff_tab):
            return self._viz_diff.get("1.0", "end-1c")
        if hasattr(self, "_viz_map_tab") and selected == str(self._viz_map_tab):
            return self._viz_map_detail.get("1.0", "end-1c")
        return self._viz_detail.get("1.0", "end-1c")

    def _viz_float(self, value: object) -> Optional[float]:
        try:
            x = float(value)
        except Exception:
            return None
        return x if math.isfinite(x) else None

    def _viz_metric_lookup(self, metric_map: dict[str, float], key: str) -> Optional[float]:
        if key in metric_map:
            return metric_map[key]
        lower = key.lower()
        upper = key.upper()
        if lower in metric_map:
            return metric_map[lower]
        if upper in metric_map:
            return metric_map[upper]
        for k, v in metric_map.items():
            if k.lower() == lower:
                return v
        return None

    def _viz_node_metric_value(self, node: VizNode, key: str) -> Optional[float]:
        metrics = node.metrics if isinstance(node.metrics, dict) else {}
        if metrics:
            found = self._viz_metric_lookup(metrics, key)
            if found is not None:
                return found
        if key == "fitness":
            return node.fitness
        if key == "combined_score":
            return node.metric if self._viz_metric_key == "combined_score" else node.fitness
        eval_metrics = node.eval_metrics if isinstance(node.eval_metrics, dict) else {}
        raw = eval_metrics.get(key, eval_metrics.get(key.lower(), eval_metrics.get(key.upper())))
        return self._viz_float(raw) if raw is not None else None

    def _viz_node_dimension_value(self, node: VizNode, key: str) -> Optional[float]:
        val = self._viz_node_metric_value(node, key)
        if val is not None:
            return val
        feats = node.features if isinstance(node.features, dict) else {}
        if key in feats:
            return feats[key]
        low = key.lower()
        for fk, fv in feats.items():
            if fk.lower() == low:
                return fv
        return None

    def _viz_update_metric_controls(
        self, metric_names: set[str], feature_names: set[str]
    ) -> None:
        preferred = [
            "combined_score",
            "fitness",
            "delta_bic",
            "chi2_total",
            "chi2_tt",
            "chi2_ee",
            "chi2_te",
            "quadrupole_score",
        ]
        ordered_metrics = [k for k in preferred if k in metric_names]
        ordered_metrics.extend(sorted(k for k in metric_names if k not in set(ordered_metrics)))
        if ordered_metrics:
            self._viz_metric_names = ordered_metrics
            self.viz_metric_combo.configure(values=ordered_metrics)
            if self.viz_metric_var.get() not in ordered_metrics:
                self.viz_metric_var.set(ordered_metrics[0])

        dim_names = set(feature_names) | set(metric_names)
        ordered_dims = [k for k in preferred if k in dim_names]
        ordered_dims.extend(sorted(k for k in dim_names if k not in set(ordered_dims)))
        if not ordered_dims:
            ordered_dims = list(self._viz_metric_names)
        if ordered_dims:
            self.viz_x_metric_combo.configure(values=ordered_dims)
            self.viz_y_metric_combo.configure(values=ordered_dims)
            if self.viz_x_metric_var.get() not in ordered_dims:
                self.viz_x_metric_var.set(
                    "delta_bic" if "delta_bic" in ordered_dims else ordered_dims[0]
                )
            if self.viz_y_metric_var.get() not in ordered_dims:
                self.viz_y_metric_var.set(
                    "quadrupole_score"
                    if "quadrupole_score" in ordered_dims
                    else ordered_dims[min(1, len(ordered_dims) - 1)]
                )

    def _maybe_refresh_visualizer(self) -> None:
        self._viz_after = None
        if not self.viz_auto_var.get():
            return
        p = self._selected_problem()
        run_id = self.run_id_var.get().strip()
        if not p or not run_id:
            self._viz_after = self.after(2000, self._maybe_refresh_visualizer)
            return

        run_dir = self.experiments_dir / p.name / run_id
        _, latest_epoch = self._viz_collect_ckpts(run_dir)
        if latest_epoch is not None:
            if self._viz_last_ckpt is None or latest_epoch >= self._viz_last_ckpt + 5:
                self._refresh_visualizer()
                self._viz_last_ckpt = latest_epoch

        self._viz_after = self.after(2000, self._maybe_refresh_visualizer)

    def _viz_collect_ckpts(self, run_dir: Path) -> tuple[dict[int, Path], Optional[int]]:
        ckpts: dict[int, Path] = {}
        latest_epoch: Optional[int] = None
        if not run_dir.exists():
            return ckpts, latest_epoch
        for island_dir in run_dir.iterdir():
            if not island_dir.is_dir() or not island_dir.name.isdigit():
                continue
            ckpt_dir = island_dir / "ckpt"
            if not ckpt_dir.exists():
                continue
            best_epoch = None
            best_path = None
            for p in ckpt_dir.iterdir():
                if not p.is_file():
                    continue
                m = self._RE_CKPT.match(p.name)
                if not m:
                    continue
                epoch = int(m.group(1))
                if best_epoch is None or epoch > best_epoch:
                    best_epoch = epoch
                    best_path = p
            if best_path is not None and best_epoch is not None:
                idx = int(island_dir.name)
                ckpts[idx] = best_path
                if latest_epoch is None or best_epoch > latest_epoch:
                    latest_epoch = best_epoch
        return ckpts, latest_epoch

    def _load_ckpt(self, ckpt_path: Path) -> Optional[dict]:
        # Ensure codeevolve modules are importable for unpickling
        added_path = False
        if str(self.repo_root / "src") not in sys.path:
            sys.path.insert(0, str(self.repo_root / "src"))
            added_path = True
        try:
            with ckpt_path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None
        finally:
            if added_path:
                try:
                    sys.path.remove(str(self.repo_root / "src"))
                except Exception:
                    pass

    def _viz_metric_value(self, prog: object, key: str) -> Optional[float]:
        if key == "fitness":
            val = getattr(prog, "fitness", None)
            return float(val) if val is not None else None

        eval_metrics = getattr(prog, "eval_metrics", None)
        if isinstance(eval_metrics, dict):
            if key in eval_metrics:
                val = eval_metrics.get(key)
            elif key.upper() in eval_metrics:
                val = eval_metrics.get(key.upper())
            else:
                val = eval_metrics.get(key.lower())
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    return None

        if key == "combined_score":
            val = getattr(prog, "fitness", None)
            return float(val) if val is not None else None
        return None

    def _refresh_visualizer(self) -> None:
        self._viz_canvas.delete("all")
        self.viz_epoch_var.set("")
        self._viz_nodes_cache = {}
        self._viz_positions = {}
        self._viz_hit_radius = {}
        self._viz_map_feature_names = []
        self._viz_elite_map_type = ""
        self._viz_map_cell_cache = {}
        self._viz_parent_map = {}
        self._viz_children_map = {}
        self._viz_delta_map = {}
        self._viz_depth_map = {}
        self._viz_island_best_ids = {}
        self._viz_global_best_id = None

        p = self._selected_problem()
        if not p:
            self._viz_label("Select a problem")
            self._set_viz_detail_text("Select a problem and run to inspect evolution.")
            self._set_viz_diff_text("Select a node to inspect mutation diff against its parent.")
            self._set_viz_map_text("MAP-Elites metadata will appear after loading checkpoints.")
            return
        run_id = self.run_id_var.get().strip()
        if not run_id:
            self._viz_label("Select a run")
            self._set_viz_detail_text("Pick a run from the left panel to load checkpoints.")
            self._set_viz_diff_text("Select a node to inspect mutation diff against its parent.")
            self._set_viz_map_text("MAP-Elites metadata will appear after loading checkpoints.")
            return
        run_dir = self.experiments_dir / p.name / run_id
        if not run_dir.exists():
            self._viz_label(f"Not found: {run_id}")
            self._set_viz_detail_text(f"Run directory not found:\n{run_dir}")
            self._set_viz_diff_text("Select a node to inspect mutation diff against its parent.")
            self._set_viz_map_text("MAP-Elites metadata will appear after loading checkpoints.")
            return

        ckpt_map, latest_epoch = self._viz_collect_ckpts(run_dir)
        if not ckpt_map:
            self._viz_label("No checkpoints yet")
            self._set_viz_detail_text("No ckpt files found yet. Start a run and refresh.")
            self._set_viz_diff_text("No checkpoints loaded yet.")
            self._set_viz_map_text("No checkpoints loaded yet.")
            return

        nodes: dict[str, VizNode] = {}
        metric_names: set[str] = {"fitness", "combined_score"}
        feature_names: set[str] = set()

        for island_idx, ckpt_path in ckpt_map.items():
            ckpt = self._load_ckpt(ckpt_path)
            if not isinstance(ckpt, dict):
                continue
            sol_db = ckpt.get("sol_db")
            if sol_db is None or not hasattr(sol_db, "programs"):
                continue

            elite_map_type = getattr(sol_db, "elite_map_type", None)
            if elite_map_type and not self._viz_elite_map_type:
                self._viz_elite_map_type = str(elite_map_type)
            elite_map = getattr(sol_db, "elite_map", None)
            if elite_map is not None:
                feats = getattr(elite_map, "features", None)
                if isinstance(feats, list):
                    for feat in feats:
                        fname = getattr(feat, "name", None)
                        if isinstance(fname, str) and fname:
                            feature_names.add(fname)
                raw_map = getattr(elite_map, "map", None)
                if isinstance(raw_map, dict):
                    for raw_idx, raw_entry in raw_map.items():
                        if not isinstance(raw_entry, tuple) or len(raw_entry) < 2:
                            continue
                        elite_id, elite_fit = raw_entry[0], self._viz_float(raw_entry[1])
                        if elite_id is None or elite_fit is None:
                            continue
                        if isinstance(raw_idx, tuple):
                            cell_key = tuple(int(v) for v in raw_idx if isinstance(v, (int, float)))
                        elif isinstance(raw_idx, (int, float)):
                            cell_key = (int(raw_idx),)
                        else:
                            continue
                        if not cell_key:
                            continue
                        prev = self._viz_map_cell_cache.get(cell_key)
                        if prev is None or elite_fit > prev[1]:
                            self._viz_map_cell_cache[cell_key] = (str(elite_id), elite_fit)

            try:
                programs = sol_db.programs.values()
            except Exception:
                continue

            for prog in programs:
                metric_map: dict[str, float] = {}
                eval_metrics_raw = getattr(prog, "eval_metrics", None)
                eval_metrics = eval_metrics_raw if isinstance(eval_metrics_raw, dict) else {}

                fitness_val = self._viz_float(getattr(prog, "fitness", None))
                if fitness_val is not None:
                    metric_map["fitness"] = fitness_val
                    metric_map.setdefault("combined_score", fitness_val)

                for mk, mv in eval_metrics.items():
                    if not isinstance(mk, str):
                        continue
                    fv = self._viz_float(mv)
                    if fv is None:
                        continue
                    metric_map[mk] = fv
                    metric_names.add(mk)
                if "combined_score" not in metric_map and "COMBINED_SCORE" in metric_map:
                    metric_map["combined_score"] = metric_map["COMBINED_SCORE"]
                metric_names.update(metric_map.keys())

                feature_map: dict[str, float] = {}
                raw_features = getattr(prog, "features", None)
                if isinstance(raw_features, dict):
                    for fk, fv in raw_features.items():
                        if not isinstance(fk, str):
                            continue
                        num = self._viz_float(fv)
                        if num is None:
                            continue
                        feature_map[fk] = num
                        feature_names.add(fk)

                generation = getattr(prog, "generation", None)
                if generation is None:
                    generation = getattr(prog, "iteration_found", 0)
                try:
                    generation = int(generation)
                except Exception:
                    generation = 0
                island_found = getattr(prog, "island_found", island_idx)
                try:
                    island_found = int(island_found) if island_found is not None else island_idx
                except Exception:
                    island_found = island_idx
                parent_id = getattr(prog, "parent_id", None)
                prog_id = getattr(prog, "id", None)
                if not prog_id:
                    continue
                code = getattr(prog, "code", None)
                if fitness_val is None:
                    fitness_val = self._viz_metric_lookup(metric_map, "combined_score")
                if fitness_val is None:
                    fitness_val = 0.0

                nodes[prog_id] = VizNode(
                    prog_id=str(prog_id),
                    parent_id=str(parent_id) if parent_id else None,
                    island=island_found,
                    generation=generation,
                    fitness=float(fitness_val),
                    metric=0.0,
                    code=str(code) if code is not None else None,
                    eval_metrics=eval_metrics if isinstance(eval_metrics, dict) else None,
                    metrics=metric_map,
                    features=feature_map,
                )

        self._viz_update_metric_controls(metric_names, feature_names)
        self._viz_map_feature_names = sorted(feature_names)
        metric_key = self.viz_metric_var.get().strip() or "combined_score"
        self._viz_metric_key = metric_key

        filtered_nodes: dict[str, VizNode] = {}
        for prog_id, node in nodes.items():
            metric_val = self._viz_node_metric_value(node, metric_key)
            if metric_val is None:
                continue
            if not math.isfinite(metric_val):
                continue
            filtered_nodes[prog_id] = VizNode(
                prog_id=node.prog_id,
                parent_id=node.parent_id,
                island=node.island,
                generation=node.generation,
                fitness=node.fitness,
                metric=float(metric_val),
                code=node.code,
                eval_metrics=node.eval_metrics,
                metrics=node.metrics,
                features=node.features,
            )
        nodes = filtered_nodes

        if not nodes:
            self._viz_label("No programs for selected metric")
            self._set_viz_detail_text(
                "Checkpoints loaded, but no valid programs were found for the selected metric."
            )
            self._set_viz_diff_text("Try another metric or load more checkpoints.")
            self._update_viz_map_overview(metric_key)
            return

        edges = [(n.parent_id, n.prog_id) for n in nodes.values() if n.parent_id in nodes]
        self._viz_nodes_cache = nodes
        (
            self._viz_parent_map,
            self._viz_children_map,
            self._viz_delta_map,
            self._viz_depth_map,
        ) = self._viz_build_indexes(nodes)
        for prog_id, node in nodes.items():
            best_id = self._viz_island_best_ids.get(node.island)
            if best_id is None or node.metric > nodes[best_id].metric:
                self._viz_island_best_ids[node.island] = prog_id
        self._viz_global_best_id = max(nodes, key=lambda prog_id: nodes[prog_id].metric)

        view_mode = self.viz_view_var.get().strip()
        if view_mode not in VIZ_VIEW_OPTIONS:
            view_mode = VIZ_VIEW_OPTIONS[0]
            self.viz_view_var.set(view_mode)
        if view_mode == "Performance":
            self._draw_viz_performance(nodes, metric_key)
        elif view_mode == "List":
            self._draw_viz_list(nodes, metric_key)
        elif view_mode == "MAP-Elites":
            self._draw_viz_map_elites(nodes, metric_key)
        else:
            self._draw_viz_graph(nodes, edges, metric_key)

        if self._viz_selected_id in nodes:
            self._update_viz_details(self._viz_selected_id)
        else:
            self._viz_selected_id = None
            self._update_viz_overview()

        if view_mode != "MAP-Elites":
            self._update_viz_map_overview(metric_key)
        if latest_epoch is not None:
            self.viz_epoch_var.set(f"{view_mode} view | Latest ckpt: {latest_epoch}")
            self._viz_last_ckpt = latest_epoch

    def _viz_find_and_select(self) -> None:
        query = self.viz_find_var.get().strip()
        nodes = self._viz_nodes_cache
        if not nodes:
            self._set_status("Visualizer has no loaded nodes")
            return
        if not query:
            self._set_status("Enter a program id fragment to find")
            return

        if query in nodes:
            matches = [query]
        else:
            q = query.lower()
            matches = [pid for pid in nodes if pid.lower() == q]
            if not matches:
                matches = [pid for pid in nodes if pid.lower().startswith(q)]
            if not matches:
                matches = [pid for pid in nodes if q in pid.lower()]
            if not matches:
                matches = [
                    pid
                    for pid, node in nodes.items()
                    if node.code and q in node.code.lower()
                ]

        if not matches:
            self._set_status(f"No match for '{query}'")
            return

        best_id = max(matches, key=lambda pid: (nodes[pid].metric, nodes[pid].generation, pid))
        self._viz_selected_id = best_id
        self._update_viz_details(best_id)
        self._refresh_visualizer()
        self._set_status(f"Selected {best_id} ({len(matches)} matches)")

    def _viz_select_best(self) -> None:
        if not self._viz_nodes_cache:
            self._refresh_visualizer()
        best_id = self._viz_global_best_id
        if not best_id or best_id not in self._viz_nodes_cache:
            self._set_status("No best node available yet")
            return
        self._viz_selected_id = best_id
        self._update_viz_details(best_id)
        self._refresh_visualizer()
        self._set_status(f"Focused best node: {best_id}")

    def _viz_copy_details(self) -> None:
        text = self._viz_active_detail_text()
        if not text.strip():
            self._set_status("No visualizer details to copy")
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update_idletasks()
            self._set_status("Visualizer details copied to clipboard")
        except Exception as e:
            messagebox.showerror("Copy Failed", str(e))

    def _viz_export_snapshot(self) -> None:
        p = self._selected_problem()
        run_id = self.run_id_var.get().strip()
        if not p or not run_id:
            messagebox.showerror("Export Snapshot", "Select a problem and run first.")
            return
        if not self._viz_positions:
            self._refresh_visualizer()
            if not self._viz_positions:
                messagebox.showerror("Export Snapshot", "No visualizer graph to export yet.")
                return

        run_dir = self.experiments_dir / p.name / run_id
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{p.name}_{run_id}_{self._viz_metric_key}_{stamp}.ps"
        out = filedialog.asksaveasfilename(
            parent=self,
            title="Export Visualizer Snapshot",
            defaultextension=".ps",
            filetypes=[("PostScript", "*.ps"), ("All Files", "*.*")],
            initialdir=str(run_dir if run_dir.exists() else self.repo_root),
            initialfile=base_name,
        )
        if not out:
            return

        out_path = Path(out)
        if out_path.suffix.lower() != ".ps":
            out_path = out_path.with_suffix(".ps")

        try:
            self._viz_canvas.update_idletasks()
            self._viz_canvas.postscript(file=str(out_path), colormode="color")
            detail_txt = self._viz_detail.get("1.0", "end-1c")
            diff_txt = self._viz_diff.get("1.0", "end-1c")
            map_txt = self._viz_map_detail.get("1.0", "end-1c")
            sidecar = out_path.with_suffix(".txt")
            info = [
                f"problem={p.name}",
                f"run={run_id}",
                f"metric={self._viz_metric_key}",
                f"view={self.viz_view_var.get().strip()}",
                f"exported_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"selected={self._viz_selected_id or ''}",
                f"nodes={len(self._viz_nodes_cache)}",
            ]
            sidecar.write_text(
                "\n".join(info)
                + "\n\n[PROGRAM]\n"
                + detail_txt
                + "\n\n[DIFF]\n"
                + diff_txt
                + "\n\n[MAP-ELITES]\n"
                + map_txt,
                encoding="utf-8",
            )
            self._set_status(f"Visualizer exported: {out_path.name}")
        except Exception as e:
            messagebox.showerror("Export Snapshot", f"Failed to export snapshot:\n{e}")

    def _set_viz_detail_text(self, text: str) -> None:
        self._set_text_widget(self._viz_detail, text)

    def _update_viz_overview(self) -> None:
        if not self._viz_nodes_cache:
            self._set_viz_detail_text("Select a node to inspect lineage and generated code.")
            self._set_viz_diff_text("Select a node to inspect mutation diff against its parent.")
            return
        nodes = self._viz_nodes_cache
        metrics = [n.metric for n in nodes.values()]
        generations = [n.generation for n in nodes.values()]
        islands = sorted({n.island for n in nodes.values()})
        edge_count = sum(1 for parent in self._viz_parent_map.values() if parent is not None)
        roots = sum(1 for parent in self._viz_parent_map.values() if parent is None)
        improvements = [d for d in self._viz_delta_map.values() if d is not None and d > 0]
        regressions = [d for d in self._viz_delta_map.values() if d is not None and d < 0]
        migration_edges = sum(
            1
            for child_id, parent_id in self._viz_parent_map.items()
            if parent_id is not None
            and nodes[parent_id].island != nodes[child_id].island
        )
        best = nodes.get(self._viz_global_best_id or "")
        best_line = "Global best: n/a"
        if best is not None:
            best_line = (
                f"Global best: {best.metric:.4f} at I{best.island} / G{best.generation} "
                f"(id={best.prog_id})"
            )
        x_dim = self.viz_x_metric_var.get().strip()
        y_dim = self.viz_y_metric_var.get().strip()
        view_mode = self.viz_view_var.get().strip()
        highlight_mode = self.viz_highlight_var.get().strip()
        max_depth = max(self._viz_depth_map.values()) if self._viz_depth_map else 0
        avg_gain = (sum(improvements) / len(improvements)) if improvements else 0.0
        avg_drop = (sum(regressions) / len(regressions)) if regressions else 0.0

        lines = [
            "Evolution overview",
            "",
            f"View: {view_mode}",
            f"Metric: {self._viz_metric_key}",
            f"Highlight: {highlight_mode}",
            f"Custom dimensions: X={x_dim} | Y={y_dim}",
            "",
            f"Programs: {len(nodes)}",
            f"Mutations (edges): {edge_count}",
            f"Roots: {roots}",
            f"Islands: {', '.join(str(i) for i in islands)}",
            f"Generation span: {min(generations)} -> {max(generations)}",
            f"{self._viz_metric_key} span: {min(metrics):.4f} -> {max(metrics):.4f}",
            f"Depth (longest lineage): {max_depth}",
            f"Improving mutations: {len(improvements)}",
            f"Regressing mutations: {len(regressions)}",
            f"Cross-island migrations: {migration_edges}",
            f"Avg positive delta: {avg_gain:+.4f}",
            f"Avg negative delta: {avg_drop:+.4f}",
            "",
            best_line,
            "",
            "Click a node to inspect its lineage, delta, and generated code.",
        ]
        if self._viz_metric_names:
            preview = ", ".join(self._viz_metric_names[:10])
            if len(self._viz_metric_names) > 10:
                preview += f", ... (+{len(self._viz_metric_names) - 10} more)"
            lines.extend(["", "Detected metrics:", preview])
        self._set_viz_detail_text("\n".join(lines))
        self._set_viz_diff_text("Select a node to inspect mutation diff against its parent.")

    def _update_viz_map_overview(self, metric_key: str) -> None:
        lines = [
            "MAP-Elites overview",
            "",
            f"Selected metric: {metric_key}",
            f"Custom dimensions: X={self.viz_x_metric_var.get().strip()} | "
            f"Y={self.viz_y_metric_var.get().strip()}",
        ]

        if self._viz_elite_map_type:
            lines.append(f"Checkpoint elite map type: {self._viz_elite_map_type}")
        if self._viz_map_feature_names:
            lines.append("Checkpoint MAP-Elites features: " + ", ".join(self._viz_map_feature_names))

        if self._viz_map_cell_cache:
            lines.append(f"Checkpoint cells occupied: {len(self._viz_map_cell_cache)}")
            ranked_cells = sorted(
                self._viz_map_cell_cache.items(),
                key=lambda item: item[1][1],
                reverse=True,
            )
            lines.append("")
            lines.append("Top occupied cells:")
            for cell, (pid, fit) in ranked_cells[:8]:
                lines.append(f"cell={cell}  fitness={fit:.4f}  id={pid}")
        else:
            lines.append("")
            lines.append("No explicit checkpoint elite-map cells found.")
            lines.append("MAP-Elites view will build a live grid from selected X/Y dimensions.")

        self._set_viz_map_text("\n".join(lines))

    def _viz_build_indexes(
        self, nodes: dict[str, VizNode]
    ) -> tuple[
        dict[str, Optional[str]],
        dict[str, list[str]],
        dict[str, Optional[float]],
        dict[str, int],
    ]:
        parent_map: dict[str, Optional[str]] = {}
        children_map: dict[str, list[str]] = {prog_id: [] for prog_id in nodes}
        delta_map: dict[str, Optional[float]] = {}
        depth_map: dict[str, int] = {}

        for prog_id, node in nodes.items():
            parent_id = node.parent_id if node.parent_id in nodes else None
            parent_map[prog_id] = parent_id
            if parent_id is not None:
                children_map.setdefault(parent_id, []).append(prog_id)
                delta_map[prog_id] = node.metric - nodes[parent_id].metric
            else:
                delta_map[prog_id] = None

        for child_ids in children_map.values():
            child_ids.sort(key=lambda cid: (nodes[cid].generation, -nodes[cid].metric, cid))

        for prog_id in nodes:
            if prog_id in depth_map:
                continue
            hops = 0
            cur = prog_id
            seen: set[str] = set()
            while True:
                parent_id = parent_map.get(cur)
                if parent_id is None or parent_id in seen:
                    break
                cached = depth_map.get(parent_id)
                if cached is not None:
                    hops += cached + 1
                    break
                seen.add(parent_id)
                hops += 1
                cur = parent_id
            depth_map[prog_id] = hops

        return parent_map, children_map, delta_map, depth_map

    def _viz_lineage_sets(
        self, selected_id: Optional[str]
    ) -> tuple[set[str], set[tuple[str, str]], set[str], set[tuple[str, str]]]:
        ancestors: set[str] = set()
        ancestor_edges: set[tuple[str, str]] = set()
        descendants: set[str] = set()
        descendant_edges: set[tuple[str, str]] = set()

        if not selected_id or selected_id not in self._viz_nodes_cache:
            return ancestors, ancestor_edges, descendants, descendant_edges

        cur = selected_id
        seen_ancestors = {selected_id}
        while True:
            parent_id = self._viz_parent_map.get(cur)
            if parent_id is None or parent_id in seen_ancestors:
                break
            ancestors.add(parent_id)
            ancestor_edges.add((parent_id, cur))
            seen_ancestors.add(parent_id)
            cur = parent_id

        stack = [selected_id]
        seen_descendants = {selected_id}
        while stack:
            parent = stack.pop()
            for child in self._viz_children_map.get(parent, []):
                if child in seen_descendants:
                    continue
                seen_descendants.add(child)
                descendants.add(child)
                descendant_edges.add((parent, child))
                stack.append(child)

        return ancestors, ancestor_edges, descendants, descendant_edges

    def _viz_label(self, text: str) -> None:
        self._viz_canvas.update_idletasks()
        w = self._viz_canvas.winfo_width() or 320
        h = self._viz_canvas.winfo_height() or 200
        self._viz_canvas.create_text(w // 2, h // 2, text=text, fill=C.OVERLAY0,
                                     font=FONT_MONO_SM, anchor="center")

    def _on_viz_click(self, event: tk.Event) -> None:
        if not self._viz_positions:
            return
        closest_id = None
        closest_dist = None
        for prog_id, (x, y) in self._viz_positions.items():
            dx = x - event.x
            dy = y - event.y
            dist = dx * dx + dy * dy
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_id = prog_id
        if closest_id is None or closest_dist is None:
            return
        hit_r = self._viz_hit_radius.get(closest_id, 7.0) + 5.0
        if closest_dist > hit_r * hit_r:
            if self._viz_selected_id is not None:
                self._viz_selected_id = None
                self._update_viz_overview()
                self._refresh_visualizer()
            return
        self._viz_selected_id = closest_id
        self._update_viz_details(closest_id)
        self._refresh_visualizer()

    def _update_viz_details(self, prog_id: str) -> None:
        node = self._viz_nodes_cache.get(prog_id)
        if not node:
            return
        parent_id = self._viz_parent_map.get(node.prog_id)
        parent = self._viz_nodes_cache.get(parent_id or "")
        delta = self._viz_delta_map.get(node.prog_id)
        depth = self._viz_depth_map.get(node.prog_id, 0)
        child_ids = self._viz_children_map.get(node.prog_id, [])
        is_island_best = self._viz_island_best_ids.get(node.island) == node.prog_id
        is_global_best = self._viz_global_best_id == node.prog_id
        badges: list[str] = []
        if parent_id is None:
            badges.append("ROOT")
        if is_island_best:
            badges.append("ISLAND CHAMPION")
        if is_global_best:
            badges.append("GLOBAL CHAMPION")

        lines = [
            f"Program ID: {node.prog_id}",
            f"Island: {node.island}",
            f"Generation: {node.generation}",
            f"Fitness: {node.fitness:.4f}",
            f"{self._viz_metric_key}: {node.metric:.4f}",
            f"Mutation delta: {delta:+.4f}" if delta is not None else "Mutation delta: n/a (root)",
            f"Lineage depth: {depth}",
            f"Children: {len(child_ids)}",
            "",
        ]
        if badges:
            lines.append("Role: " + ", ".join(badges))
            lines.append("")
        if parent_id:
            if parent is not None:
                lines.append(
                    f"Parent: {parent_id} (I{parent.island}, G{parent.generation}, "
                    f"{self._viz_metric_key}={parent.metric:.4f})"
                )
                if parent.island != node.island:
                    lines.append(f"Migration edge: I{parent.island} -> I{node.island}")
            else:
                lines.append(f"Parent: {parent_id}")
            lines.append("")
        if child_ids:
            child_preview = ", ".join(child_ids[:5])
            if len(child_ids) > 5:
                child_preview += f", ... (+{len(child_ids) - 5} more)"
            lines.append("Children:")
            lines.append(child_preview)
            lines.append("")

        metric_map = node.metrics if isinstance(node.metrics, dict) else {}
        metric_items = [
            (k, v) for k, v in metric_map.items() if isinstance(k, str) and isinstance(v, float)
        ]
        if metric_items:
            metric_items = sorted(metric_items, key=lambda kv: (-kv[1], kv[0]))
            top_metrics = metric_items[:8]
            abs_max = max(1e-9, max(abs(v) for _, v in top_metrics))
            lines.append("Metric bars:")
            for key, val in top_metrics:
                blocks = int(round(20 * (abs(val) / abs_max)))
                bar = "#" * max(1, min(20, blocks))
                lines.append(f"{key:<22} {val:>11.5f}  {bar}")
            lines.append("")

        if node.eval_metrics:
            try:
                metrics_text = json.dumps(node.eval_metrics, indent=2, sort_keys=True)
            except Exception:
                metrics_text = str(node.eval_metrics)
            if len(metrics_text) > 2500:
                metrics_text = metrics_text[:2500] + "\n... truncated ..."
            lines.append("Eval metrics:")
            lines.append(metrics_text)
            lines.append("")
        if node.code:
            code = node.code
            if len(code) > 2500:
                code = code[:2500] + "\n# ... truncated ..."
            lines.append("Code:")
            lines.append(code)

        self._set_viz_detail_text("\n".join(lines))

        if parent is None:
            if parent_id is None:
                self._set_viz_diff_text("Root program has no parent. No mutation diff available.")
            else:
                self._set_viz_diff_text(f"Parent program not found in loaded checkpoints: {parent_id}")
            return
        if not parent.code or not node.code:
            self._set_viz_diff_text("Missing code for parent or child; cannot render diff.")
            return

        diff_lines = list(
            difflib.unified_diff(
                parent.code.splitlines(),
                node.code.splitlines(),
                fromfile=f"parent:{parent.prog_id}",
                tofile=f"child:{node.prog_id}",
                lineterm="",
            )
        )
        if not diff_lines:
            self._set_viz_diff_text("No textual mutation detected between parent and child code.")
            return
        if len(diff_lines) > 500:
            diff_lines = diff_lines[:500] + ["... diff truncated ..."]
        self._set_viz_diff_text("\n".join(diff_lines))

    def _draw_viz_graph(self, nodes: dict[str, VizNode],
                        edges: list[tuple[str, str]], metric_key: str) -> None:
        self._viz_canvas.update_idletasks()
        cw = max(520, self._viz_canvas.winfo_width() or 600)
        ch = max(300, self._viz_canvas.winfo_height() or 360)

        pad_l, pad_r, pad_t, pad_b = 78, 24, 82, 44
        show_islands = self.viz_show_islands_var.get()
        highlight_mode = self.viz_highlight_var.get().strip()
        real_islands = sorted({n.island for n in nodes.values()})
        islands = real_islands if show_islands else [0]
        if not islands:
            self._viz_label("No islands found")
            return
        plot_w = cw - pad_l - pad_r
        plot_h = ch - pad_t - pad_b
        if plot_w < 120 or plot_h < 80:
            self._viz_label("Visualizer area too small")
            return

        generations = [n.generation for n in nodes.values()]
        metric_vals = [n.metric for n in nodes.values()]
        g_min = min(generations)
        g_max = max(generations)
        g_range = max(1, g_max - g_min)
        m_min = min(metric_vals)
        m_max = max(metric_vals)
        m_range = max(1e-6, m_max - m_min)
        self._viz_hit_radius = {}

        # Evolution-time gradient (left=early, right=late)
        grad_steps = 14
        for i in range(grad_steps):
            t0 = i / grad_steps
            t1 = (i + 1) / grad_steps
            x0 = pad_l + t0 * plot_w
            x1 = pad_l + t1 * plot_w
            tone = _mix_color(C.MANTLE, C.SKY, 0.02 + 0.22 * t0)
            tone = _mix_color(tone, C.SURFACE0, 0.55)
            self._viz_canvas.create_rectangle(x0, pad_t, x1, ch - pad_b, fill=tone, outline="")

        # Header
        self._viz_canvas.create_text(
            pad_l, 14,
            text=f"Evolution Landscape ({metric_key})",
            fill=C.LAVENDER,
            font=("Ubuntu Sans", 11, "bold"),
            anchor="nw",
        )
        self._viz_canvas.create_text(
            cw - pad_r, 16,
            text=f"gen {g_min} -> {g_max} | {'islands on' if show_islands else 'islands collapsed'}",
            fill=C.SUBTEXT0,
            font=FONT_MONO_SM,
            anchor="ne",
        )

        improved_edges = sum(1 for d in self._viz_delta_map.values() if d is not None and d > 0)
        migration_edges = sum(
            1
            for child_id, parent_id in self._viz_parent_map.items()
            if parent_id is not None and nodes[parent_id].island != nodes[child_id].island
        )
        cards = [
            ("Programs", str(len(nodes)), C.BLUE),
            ("Mutations", str(len(edges)), C.TEAL),
            ("Improving", str(improved_edges), C.GREEN),
            ("Migrations", str(migration_edges), C.SKY),
            ("Best", f"{m_max:.4f}", C.YELLOW),
        ]
        card_gap = 8
        card_count = len(cards)
        card_w = (plot_w - card_gap * (card_count - 1)) / max(1, card_count)
        card_w = max(95, min(190, card_w))
        card_y0, card_y1 = 34, 68
        card_x = pad_l
        for label, value, accent in cards:
            card_x1 = min(cw - pad_r, card_x + card_w)
            bg = _mix_color(C.SURFACE0, accent, 0.18)
            border = _mix_color(accent, C.SURFACE1, 0.45)
            self._viz_canvas.create_rectangle(
                card_x, card_y0, card_x1, card_y1, fill=bg, outline=border, width=1
            )
            self._viz_canvas.create_text(
                card_x + 7, card_y0 + 6, text=label,
                fill=C.SUBTEXT0, font=("Ubuntu Sans Mono", 8), anchor="nw",
            )
            self._viz_canvas.create_text(
                card_x + 7, card_y0 + 21, text=value,
                fill=accent, font=("Ubuntu Sans", 10, "bold"), anchor="nw",
            )
            card_x += card_w + card_gap
            if card_x >= cw - pad_r:
                break

        # Generation axis + ticks
        tick_count = 8 if g_range > 7 else max(2, g_range + 1)
        for i in range(tick_count):
            frac = i / max(1, tick_count - 1)
            gv = int(round(g_min + frac * g_range))
            x = pad_l + frac * plot_w
            grid_color = _mix_color(C.SURFACE1, C.SKY, 0.10 + 0.20 * frac)
            self._viz_canvas.create_line(x, pad_t, x, ch - pad_b, fill=grid_color, dash=(2, 6))
            self._viz_canvas.create_text(
                x, ch - pad_b + 10, text=str(gv),
                fill=C.OVERLAY0, font=("Ubuntu Sans Mono", 8), anchor="n",
            )

        self._viz_canvas.create_line(
            pad_l, ch - pad_b + 20, cw - pad_r, ch - pad_b + 20,
            fill=C.SUBTEXT0, width=1, arrow="last",
        )
        self._viz_canvas.create_text(
            pad_l + plot_w / 2, ch - pad_b + 22,
            text="Generation timeline (early -> late)",
            fill=C.SUBTEXT0, font=("Ubuntu Sans Mono", 8), anchor="n",
        )

        # Precompute lane scales and positions
        gap = min(20, max(10, int(plot_h / max(3, len(islands) * 6))))
        band_h = (plot_h - gap * (len(islands) - 1)) / max(1, len(islands))
        positions: dict[str, tuple[float, float]] = {}
        island_nodes: dict[int, list[VizNode]] = {i: [] for i in islands}
        for n in nodes.values():
            lane = n.island if show_islands else 0
            island_nodes.setdefault(lane, []).append(n)

        for idx, island in enumerate(islands):
            band_top = pad_t + idx * (band_h + gap)
            band_bot = band_top + band_h
            group = sorted(
                island_nodes.get(island, []),
                key=lambda n: (n.generation, n.metric, n.prog_id),
            )
            if not group:
                continue

            if show_islands:
                island_color = self._island_colors[island % len(self._island_colors)]
                island_label = f"I{island}"
            else:
                island_color = C.BLUE
                island_label = "All islands"
            band_fill = _mix_color(C.MANTLE, island_color, 0.11 if idx % 2 == 0 else 0.07)
            self._viz_canvas.create_rectangle(
                pad_l, band_top, cw - pad_r, band_bot, fill=band_fill, outline=""
            )
            self._viz_canvas.create_line(
                pad_l, band_top, cw - pad_r, band_top,
                fill=_mix_color(island_color, C.SURFACE2, 0.6),
            )

            local_min = min(n.metric for n in group)
            local_max = max(n.metric for n in group)
            local_range = max(1e-9, local_max - local_min)
            self._viz_canvas.create_text(
                pad_l - 10, (band_top + band_bot) / 2,
                text=island_label,
                fill=island_color, font=("Ubuntu Sans", 10, "bold"), anchor="e",
            )
            self._viz_canvas.create_text(
                pad_l + 6, band_top + 4,
                text=f"{local_min:.3f} -> {local_max:.3f}",
                fill=C.OVERLAY0, font=("Ubuntu Sans Mono", 7), anchor="nw",
            )

            def _metric_y(metric: float) -> float:
                if local_range <= 1e-9:
                    pct = 0.5
                else:
                    pct = _clamp01((metric - local_min) / local_range)
                return band_bot - (0.14 + 0.74 * pct) * band_h

            # Running-best frontier per island
            per_gen_best: dict[int, float] = {}
            for n in group:
                per_gen_best[n.generation] = max(per_gen_best.get(n.generation, -1e18), n.metric)
            best_so_far = -1e18
            frontier: list[float] = []
            for gen in sorted(per_gen_best):
                best_so_far = max(best_so_far, per_gen_best[gen])
                x = pad_l + ((gen - g_min) / g_range) * plot_w
                y = _metric_y(best_so_far)
                frontier.extend([x, y])
            if len(frontier) >= 4:
                frontier_color = _mix_color(island_color, C.TEXT, 0.38)
                self._viz_canvas.create_line(*frontier, fill=frontier_color, width=2)

            # Node placement within island lane
            for n in group:
                x = pad_l + ((n.generation - g_min) / g_range) * plot_w
                y = _metric_y(n.metric)
                seed = sum(ord(ch) for ch in n.prog_id[-8:])
                x += ((seed % 7) - 3) * 0.55
                y += (((seed // 7) % 7) - 3) * 0.75
                x = max(pad_l + 2, min(cw - pad_r - 2, x))
                y = max(band_top + 2, min(band_bot - 2, y))
                positions[n.prog_id] = (x, y)

        ancestor_nodes, ancestor_edges, descendants, descendant_edges = self._viz_lineage_sets(
            self._viz_selected_id
        )
        positive_deltas = [d for d in self._viz_delta_map.values() if d is not None and d > 0]
        negative_deltas = [abs(d) for d in self._viz_delta_map.values() if d is not None and d < 0]
        max_gain = max(positive_deltas) if positive_deltas else 0.0
        max_drop = max(negative_deltas) if negative_deltas else 0.0

        # Draw edges (improvement/migration encoded)
        for parent_id, child_id in edges:
            p = positions.get(parent_id)
            c = positions.get(child_id)
            if not p or not c:
                continue
            parent = nodes[parent_id]
            child = nodes[child_id]
            delta = self._viz_delta_map.get(child_id)
            cross_island = parent.island != child.island

            if delta is None:
                color = C.SURFACE1
                width = 1.0
                dash = (2, 5)
            elif delta >= 0:
                gain = _clamp01(delta / max(1e-9, max_gain)) if max_gain > 0 else 0.0
                color = _mix_color(C.GREEN, C.SKY, 0.30 * gain)
                color = _mix_color(color, C.SURFACE1, 0.20)
                width = 1.2 + 2.3 * gain
                dash = ()
            else:
                drop = _clamp01(abs(delta) / max(1e-9, max_drop)) if max_drop > 0 else 0.0
                color = _mix_color(C.RED, C.SURFACE1, 0.40 - 0.20 * drop)
                width = 1.0
                dash = (2, 4)

            if cross_island:
                color = _mix_color(color, C.SKY, 0.35)
                width = max(width, 1.6)
                dash = (4, 3)

            if highlight_mode == "Migration" and not cross_island:
                color = _mix_color(color, C.SURFACE0, 0.58)
                width = max(0.8, width * 0.7)
            elif highlight_mode == "Improvement" and (delta is None or delta <= 0):
                color = _mix_color(color, C.SURFACE0, 0.58)
                width = max(0.8, width * 0.7)
            elif highlight_mode == "Recent" and g_range > 0:
                recency = (child.generation - g_min) / g_range
                if recency < 0.6:
                    color = _mix_color(color, C.SURFACE0, 0.58)

            if (parent_id, child_id) in ancestor_edges:
                color = C.YELLOW
                width = max(width, 3.0)
                dash = ()
            elif (parent_id, child_id) in descendant_edges:
                color = C.LAVENDER
                width = max(width, 2.4)
                dash = ()

            if cross_island:
                bend = 10 if child.island >= parent.island else -10
                mx = (p[0] + c[0]) / 2
                my = (p[1] + c[1]) / 2 + bend
                self._viz_canvas.create_line(
                    p[0], p[1], mx, my, c[0], c[1],
                    fill=color, width=width, dash=dash, smooth=True, splinesteps=12,
                )
            else:
                self._viz_canvas.create_line(
                    p[0], p[1], c[0], c[1],
                    fill=color, width=width, dash=dash,
                )

        # Draw nodes (size=quality, fill=delta, outline=lineage/champions)
        for n in sorted(nodes.values(), key=lambda x: (x.generation, x.metric, x.prog_id)):
            pos = positions.get(n.prog_id)
            if not pos:
                continue
            island_color = self._island_colors[n.island % len(self._island_colors)]
            delta = self._viz_delta_map.get(n.prog_id)
            if delta is None:
                fill = _mix_color(island_color, C.SURFACE1, 0.55)
            elif delta >= 0:
                gain = _clamp01(delta / max(1e-9, max_gain)) if max_gain > 0 else 0.0
                fill = _mix_color(island_color, C.GREEN, 0.28 + 0.42 * gain)
            else:
                drop = _clamp01(abs(delta) / max(1e-9, max_drop)) if max_drop > 0 else 0.0
                fill = _mix_color(island_color, C.RED, 0.16 + 0.24 * drop)

            metric_norm = _clamp01((n.metric - m_min) / m_range) if m_range > 0 else 0.5
            radius = 3.2 + 3.0 * metric_norm
            if delta is not None and delta > 0 and max_gain > 0:
                radius += 1.8 * _clamp01(delta / max_gain)
            if highlight_mode == "Recent" and g_range > 0:
                recency = (n.generation - g_min) / g_range
                radius += 1.6 * recency
            if n.prog_id == self._viz_global_best_id:
                radius += 1.6
            if n.prog_id == self._viz_selected_id:
                radius += 1.4

            parent_id = self._viz_parent_map.get(n.prog_id)
            migrated = (
                parent_id is not None
                and parent_id in nodes
                and nodes[parent_id].island != n.island
            )
            if highlight_mode == "Migration" and not migrated:
                fill = _mix_color(fill, C.SURFACE0, 0.45)
            elif highlight_mode == "Improvement" and (delta is None or delta <= 0):
                fill = _mix_color(fill, C.SURFACE0, 0.45)
            elif highlight_mode == "Top score" and metric_norm < 0.82:
                fill = _mix_color(fill, C.SURFACE0, 0.35)

            outline = ""
            width = 1.0
            if self._viz_island_best_ids.get(n.island) == n.prog_id:
                outline = C.TEXT
                width = 1.4
            if n.prog_id in descendants:
                outline = C.LAVENDER
                width = max(width, 1.6)
            if n.prog_id in ancestor_nodes:
                outline = C.YELLOW
                width = max(width, 1.8)
            if n.prog_id == self._viz_selected_id:
                outline = C.YELLOW
                width = 2.2

            self._viz_canvas.create_oval(
                pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius,
                fill=fill, outline=outline, width=width,
            )
            self._viz_hit_radius[n.prog_id] = max(6.0, radius + 2.0)

        if self._viz_global_best_id and self._viz_global_best_id in positions:
            gx, gy = positions[self._viz_global_best_id]
            self._viz_canvas.create_text(
                gx + 10, gy - 11, text="BEST",
                fill=C.YELLOW, font=("Ubuntu Sans Mono", 8, "bold"), anchor="w",
            )

        # Selected node halo
        if self._viz_selected_id and self._viz_selected_id in positions:
            x, y = positions[self._viz_selected_id]
            ring_r = self._viz_hit_radius.get(self._viz_selected_id, 9.0) + 4.0
            self._viz_canvas.create_oval(
                x - ring_r, y - ring_r, x + ring_r, y + ring_r,
                outline=_mix_color(C.YELLOW, C.SKY, 0.25), width=1,
            )
            self._viz_canvas.create_oval(
                x - ring_r - 3, y - ring_r - 3, x + ring_r + 3, y + ring_r + 3,
                outline=C.YELLOW, width=2,
            )

        # Legend
        legend_w, legend_h = 194, 72
        lx = cw - pad_r - legend_w
        ly = pad_t + 6
        self._viz_canvas.create_rectangle(
            lx, ly, lx + legend_w, ly + legend_h,
            fill=_mix_color(C.MANTLE, C.SURFACE0, 0.7), outline=C.SURFACE1, width=1,
        )
        self._viz_canvas.create_text(
            lx + 8, ly + 6, text="Edge encoding",
            fill=C.SUBTEXT0, font=("Ubuntu Sans Mono", 8), anchor="nw",
        )
        self._viz_canvas.create_line(lx + 10, ly + 24, lx + 38, ly + 24, fill=C.GREEN, width=2)
        self._viz_canvas.create_text(
            lx + 44, ly + 24, text="improves", fill=C.SUBTEXT0, font=("Ubuntu Sans Mono", 8), anchor="w",
        )
        self._viz_canvas.create_line(
            lx + 10, ly + 38, lx + 38, ly + 38, fill=C.RED, width=1, dash=(2, 4)
        )
        self._viz_canvas.create_text(
            lx + 44, ly + 38, text="regresses", fill=C.SUBTEXT0, font=("Ubuntu Sans Mono", 8), anchor="w",
        )
        self._viz_canvas.create_line(
            lx + 10, ly + 52, lx + 38, ly + 52, fill=C.SKY, width=2, dash=(4, 3)
        )
        self._viz_canvas.create_text(
            lx + 44, ly + 52, text="migration", fill=C.SUBTEXT0, font=("Ubuntu Sans Mono", 8), anchor="w",
        )

        self._viz_positions = positions

    def _draw_viz_performance(self, nodes: dict[str, VizNode], metric_key: str) -> None:
        self._viz_canvas.update_idletasks()
        cw = max(520, self._viz_canvas.winfo_width() or 600)
        ch = max(300, self._viz_canvas.winfo_height() or 360)
        pad_l, pad_r, pad_t, pad_b = 72, 24, 38, 56
        plot_w = cw - pad_l - pad_r
        plot_h = ch - pad_t - pad_b
        if plot_w < 120 or plot_h < 80:
            self._viz_label("Visualizer area too small")
            return

        by_gen: dict[int, list[VizNode]] = {}
        for node in nodes.values():
            by_gen.setdefault(node.generation, []).append(node)
        gens = sorted(by_gen.keys())
        if not gens:
            self._viz_label("No generation data")
            return

        g_min = min(gens)
        g_max = max(gens)
        g_range = max(1, g_max - g_min)
        y_vals = [n.metric for n in nodes.values()]
        y_min_raw = min(y_vals)
        y_max_raw = max(y_vals)
        if abs(y_max_raw - y_min_raw) < 1e-9:
            y_min = y_min_raw - 0.5
            y_max = y_max_raw + 0.5
        else:
            pad = 0.05 * (y_max_raw - y_min_raw)
            y_min = y_min_raw - pad
            y_max = y_max_raw + pad
        y_range = max(1e-9, y_max - y_min)

        def _xy(gen: int, val: float) -> tuple[float, float]:
            x = pad_l + ((gen - g_min) / g_range) * plot_w
            y = pad_t + (1.0 - ((val - y_min) / y_range)) * plot_h
            return x, y

        self._viz_canvas.create_rectangle(
            pad_l, pad_t, cw - pad_r, ch - pad_b,
            fill=_mix_color(C.SURFACE0, C.MANTLE, 0.35),
            outline=C.SURFACE1,
        )

        y_ticks = 6
        for i in range(y_ticks):
            frac = i / max(1, y_ticks - 1)
            val = y_max - frac * (y_max - y_min)
            y = pad_t + frac * plot_h
            self._viz_canvas.create_line(pad_l, y, cw - pad_r, y, fill=C.SURFACE1, dash=(2, 5))
            self._viz_canvas.create_text(
                pad_l - 8, y, text=f"{val:.3f}",
                fill=C.OVERLAY0, font=("Ubuntu Sans Mono", 8), anchor="e",
            )

        x_tick_count = min(9, max(2, len(gens)))
        for i in range(x_tick_count):
            frac = i / max(1, x_tick_count - 1)
            gv = int(round(g_min + frac * g_range))
            x = pad_l + frac * plot_w
            self._viz_canvas.create_line(x, pad_t, x, ch - pad_b, fill=C.SURFACE1, dash=(2, 5))
            self._viz_canvas.create_text(
                x, ch - pad_b + 10, text=str(gv),
                fill=C.OVERLAY0, font=("Ubuntu Sans Mono", 8), anchor="n",
            )

        avg_line: list[float] = []
        best_line: list[float] = []
        positions: dict[str, tuple[float, float]] = {}
        hit_radius: dict[str, float] = {}
        for gen in gens:
            group = by_gen[gen]
            avg_val = sum(n.metric for n in group) / max(1, len(group))
            best_node = max(group, key=lambda n: (n.metric, n.prog_id))
            ax, ay = _xy(gen, avg_val)
            bx, by = _xy(gen, best_node.metric)
            avg_line.extend([ax, ay])
            best_line.extend([bx, by])
            positions[best_node.prog_id] = (bx, by)
            hit_radius[best_node.prog_id] = 8.0

        if len(avg_line) >= 4:
            self._viz_canvas.create_line(*avg_line, fill=C.LAVENDER, width=2, smooth=True, splinesteps=10)
        if len(best_line) >= 4:
            self._viz_canvas.create_line(*best_line, fill=C.YELLOW, width=3, smooth=True, splinesteps=10)

        per_island: dict[int, list[tuple[int, float]]] = {}
        for gen, group in by_gen.items():
            by_island: dict[int, float] = {}
            for n in group:
                by_island[n.island] = max(by_island.get(n.island, -1e18), n.metric)
            for island, best_val in by_island.items():
                per_island.setdefault(island, []).append((gen, best_val))
        for island, series in per_island.items():
            if len(series) < 2:
                continue
            color = _mix_color(self._island_colors[island % len(self._island_colors)], C.SURFACE1, 0.35)
            points: list[float] = []
            for gen, val in sorted(series):
                x, y = _xy(gen, val)
                points.extend([x, y])
            self._viz_canvas.create_line(*points, fill=color, width=1)

        highlight_mode = self.viz_highlight_var.get().strip()
        for prog_id, (x, y) in positions.items():
            node = nodes.get(prog_id)
            if node is None:
                continue
            base_r = 5.0
            if highlight_mode == "Recent" and g_range > 0:
                recency = (node.generation - g_min) / g_range
                base_r += 2.0 * recency
            is_selected = prog_id == self._viz_selected_id
            is_best = prog_id == self._viz_global_best_id
            fill = C.YELLOW if is_best else self._island_colors[node.island % len(self._island_colors)]
            outline = C.TEXT if is_selected else ""
            width = 2 if is_selected else 1
            self._viz_canvas.create_oval(
                x - base_r, y - base_r, x + base_r, y + base_r,
                fill=fill, outline=outline, width=width,
            )
            hit_radius[prog_id] = max(hit_radius.get(prog_id, 6.0), base_r + 3.0)

        self._viz_canvas.create_text(
            pad_l, 14,
            text=f"Performance by Generation ({metric_key})",
            fill=C.LAVENDER, font=("Ubuntu Sans", 11, "bold"), anchor="nw",
        )
        self._viz_canvas.create_text(
            cw - pad_r, 14,
            text=f"gens {g_min} -> {g_max} | nodes {len(nodes)}",
            fill=C.SUBTEXT0, font=FONT_MONO_SM, anchor="ne",
        )
        self._viz_canvas.create_text(
            pad_l + 8, pad_t + 8,
            text="yellow=best per generation  lavender=population average",
            fill=C.SUBTEXT0, font=("Ubuntu Sans Mono", 8), anchor="nw",
        )

        self._viz_positions = positions
        self._viz_hit_radius = hit_radius

    def _draw_viz_list(self, nodes: dict[str, VizNode], metric_key: str) -> None:
        self._viz_canvas.update_idletasks()
        cw = max(520, self._viz_canvas.winfo_width() or 600)
        ch = max(300, self._viz_canvas.winfo_height() or 360)
        pad_l, pad_r, pad_t, pad_b = 16, 16, 38, 18
        row_h = 20
        max_rows = max(1, (ch - pad_t - pad_b - 28) // row_h)
        rows = sorted(nodes.values(), key=lambda n: (n.metric, n.generation, n.prog_id), reverse=True)
        rows = rows[:max_rows]

        self._viz_canvas.create_rectangle(
            pad_l, pad_t, cw - pad_r, ch - pad_b,
            fill=_mix_color(C.SURFACE0, C.MANTLE, 0.35),
            outline=C.SURFACE1,
        )
        self._viz_canvas.create_text(
            pad_l, 14,
            text=f"Program List ({metric_key})",
            fill=C.LAVENDER, font=("Ubuntu Sans", 11, "bold"), anchor="nw",
        )
        self._viz_canvas.create_text(
            cw - pad_r, 14,
            text=f"showing top {len(rows)} of {len(nodes)}",
            fill=C.SUBTEXT0, font=FONT_MONO_SM, anchor="ne",
        )

        header = "rank  metric      fit        gen  isl  depth  delta      id"
        self._viz_canvas.create_text(
            pad_l + 10, pad_t + 10, text=header,
            fill=C.SUBTEXT0, font=("Ubuntu Sans Mono", 8, "bold"), anchor="nw",
        )

        positions: dict[str, tuple[float, float]] = {}
        hit_radius: dict[str, float] = {}
        for i, node in enumerate(rows, start=1):
            y0 = pad_t + 24 + (i - 1) * row_h
            y1 = y0 + row_h - 2
            is_selected = node.prog_id == self._viz_selected_id
            row_bg = _mix_color(C.SURFACE0, C.BLUE, 0.22) if is_selected else _mix_color(C.SURFACE0, C.MANTLE, 0.35)
            self._viz_canvas.create_rectangle(
                pad_l + 6, y0, cw - pad_r - 6, y1,
                fill=row_bg, outline=C.SURFACE1 if i % 2 == 0 else "",
            )
            delta = self._viz_delta_map.get(node.prog_id)
            delta_text = "n/a" if delta is None else f"{delta:+.4f}"
            text = (
                f"{i:>4}  {node.metric:>9.4f}  {node.fitness:>9.4f}  "
                f"{node.generation:>4}  {node.island:>3}  "
                f"{self._viz_depth_map.get(node.prog_id, 0):>5}  {delta_text:>9}  "
                f"{node.prog_id[:18]}"
            )
            self._viz_canvas.create_text(
                pad_l + 12, (y0 + y1) / 2, text=text,
                fill=C.TEXT, font=("Ubuntu Sans Mono", 8), anchor="w",
            )
            cx = (pad_l + cw - pad_r) / 2
            cy = (y0 + y1) / 2
            positions[node.prog_id] = (cx, cy)
            hit_radius[node.prog_id] = max(8.0, (cw - pad_l - pad_r) / 2)

        self._viz_positions = positions
        self._viz_hit_radius = hit_radius

    def _draw_viz_map_elites(self, nodes: dict[str, VizNode], metric_key: str) -> None:
        self._viz_canvas.update_idletasks()
        cw = max(520, self._viz_canvas.winfo_width() or 600)
        ch = max(300, self._viz_canvas.winfo_height() or 360)
        pad_l, pad_r, pad_t, pad_b = 66, 20, 40, 54
        plot_w = cw - pad_l - pad_r
        plot_h = ch - pad_t - pad_b
        if plot_w < 160 or plot_h < 120:
            self._viz_label("Visualizer area too small")
            return

        x_key = self.viz_x_metric_var.get().strip()
        y_key = self.viz_y_metric_var.get().strip()

        points: list[tuple[VizNode, float, float]] = []
        for node in nodes.values():
            xv = self._viz_node_dimension_value(node, x_key)
            yv = self._viz_node_dimension_value(node, y_key)
            if xv is None or yv is None:
                continue
            points.append((node, xv, yv))

        if not points:
            self._viz_label(f"No points for dimensions: {x_key} vs {y_key}")
            self._set_viz_map_text(
                "MAP-Elites view could not render because selected dimensions "
                "were not present in loaded metrics."
            )
            return

        x_vals = [x for _, x, _ in points]
        y_vals = [y for _, _, y in points]
        m_vals = [node.metric for node, _, _ in points]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        m_min, m_max = min(m_vals), max(m_vals)
        x_range = max(1e-9, x_max - x_min)
        y_range = max(1e-9, y_max - y_min)
        m_range = max(1e-9, m_max - m_min)

        x_bins = min(16, max(6, int(round(math.sqrt(len(points)) * 1.5))))
        y_bins = min(12, max(4, int(round(math.sqrt(len(points)) * 1.1))))

        cell_best: dict[tuple[int, int], VizNode] = {}
        cell_vals: dict[tuple[int, int], tuple[float, float]] = {}
        for node, xv, yv in points:
            ix = int(_clamp01((xv - x_min) / x_range) * (x_bins - 1))
            iy = int(_clamp01((yv - y_min) / y_range) * (y_bins - 1))
            key = (ix, iy)
            prev = cell_best.get(key)
            if prev is None or node.metric > prev.metric:
                cell_best[key] = node
                cell_vals[key] = (xv, yv)

        cell_w = plot_w / x_bins
        cell_h = plot_h / y_bins
        positions: dict[str, tuple[float, float]] = {}
        hit_radius: dict[str, float] = {}
        for ix in range(x_bins):
            for iy in range(y_bins):
                x0 = pad_l + ix * cell_w
                y0 = pad_t + (y_bins - 1 - iy) * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                node = cell_best.get((ix, iy))
                if node is None:
                    fill = _mix_color(C.MANTLE, C.SURFACE0, 0.55)
                    border = _mix_color(C.SURFACE1, C.MANTLE, 0.35)
                else:
                    score_t = _clamp01((node.metric - m_min) / m_range)
                    fill = _mix_color(C.BLUE, C.GREEN, score_t)
                    fill = _mix_color(fill, C.SURFACE0, 0.25)
                    border = _mix_color(C.TEXT, fill, 0.7)
                self._viz_canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=border, width=1)
                if node is not None:
                    cx = (x0 + x1) / 2
                    cy = (y0 + y1) / 2
                    positions[node.prog_id] = (cx, cy)
                    hit_radius[node.prog_id] = max(8.0, min(cell_w, cell_h) * 0.48)
                    if node.prog_id == self._viz_selected_id:
                        self._viz_canvas.create_rectangle(
                            x0 + 2, y0 + 2, x1 - 2, y1 - 2, outline=C.YELLOW, width=2
                        )

        tick_count = 6
        for i in range(tick_count):
            frac = i / max(1, tick_count - 1)
            xv = x_min + frac * x_range
            x = pad_l + frac * plot_w
            self._viz_canvas.create_line(x, ch - pad_b, x, ch - pad_b + 5, fill=C.SUBTEXT0)
            self._viz_canvas.create_text(
                x, ch - pad_b + 8, text=f"{xv:.3g}",
                fill=C.OVERLAY0, font=("Ubuntu Sans Mono", 8), anchor="n",
            )
        for i in range(tick_count):
            frac = i / max(1, tick_count - 1)
            yv = y_min + frac * y_range
            y = pad_t + (1.0 - frac) * plot_h
            self._viz_canvas.create_line(pad_l - 5, y, pad_l, y, fill=C.SUBTEXT0)
            self._viz_canvas.create_text(
                pad_l - 8, y, text=f"{yv:.3g}",
                fill=C.OVERLAY0, font=("Ubuntu Sans Mono", 8), anchor="e",
            )

        self._viz_canvas.create_text(
            pad_l, 14,
            text=f"MAP-Elites Grid ({metric_key})",
            fill=C.LAVENDER, font=("Ubuntu Sans", 11, "bold"), anchor="nw",
        )
        self._viz_canvas.create_text(
            cw - pad_r, 14,
            text=f"coverage {len(cell_best)}/{x_bins * y_bins}",
            fill=C.SUBTEXT0, font=FONT_MONO_SM, anchor="ne",
        )
        self._viz_canvas.create_text(
            pad_l + plot_w / 2, ch - pad_b + 28,
            text=x_key, fill=C.SUBTEXT0, font=("Ubuntu Sans Mono", 9), anchor="n",
        )
        self._viz_canvas.create_text(
            12, pad_t + plot_h / 2,
            text=y_key, fill=C.SUBTEXT0, font=("Ubuntu Sans Mono", 9), anchor="w",
        )

        top_cells = sorted(cell_best.items(), key=lambda kv: kv[1].metric, reverse=True)[:10]
        lines = [
            "Rendered MAP-Elites grid",
            "",
            f"Dimensions: X={x_key}, Y={y_key}",
            f"Bins: {x_bins} x {y_bins}",
            f"Cell coverage: {len(cell_best)}/{x_bins * y_bins} ({100.0 * len(cell_best) / max(1, x_bins * y_bins):.1f}%)",
            f"{metric_key} span: {m_min:.4f} -> {m_max:.4f}",
        ]
        if self._viz_elite_map_type:
            lines.append(f"Checkpoint elite map type: {self._viz_elite_map_type}")
        if self._viz_map_feature_names:
            lines.append("Checkpoint features: " + ", ".join(self._viz_map_feature_names))
        if top_cells:
            lines.extend(["", "Top occupied cells:"])
            for (ix, iy), node in top_cells:
                xv, yv = cell_vals.get((ix, iy), (float("nan"), float("nan")))
                lines.append(
                    f"cell=({ix},{iy}) {metric_key}={node.metric:.4f} "
                    f"{x_key}={xv:.4f} {y_key}={yv:.4f} id={node.prog_id}"
                )
        self._set_viz_map_text("\n".join(lines))

        self._viz_positions = positions
        self._viz_hit_radius = hit_radius

    # ------------------------------------------------------------------ Log
    _RE_FITNESS = re.compile(
        r'"(?:combined_score|fitness|COMBINED_SCORE)"\s*:\s*([\d.]+)', re.IGNORECASE)
    _RE_ERROR = re.compile(r'\b(?:error|exception|traceback|failed)\b', re.IGNORECASE)
    _RE_WARNING = re.compile(r'\bwarn(?:ing)?\b', re.IGNORECASE)

    def _append_log(self, text: str) -> None:
        tag = None
        if text.startswith("$ "):
            tag = "cmd"
        elif text.startswith("[exit]"):
            tag = "exit_ok" if "returncode=0" in text else "exit_fail"
        elif text.startswith("[dashboard]"):
            tag = "info"
        elif self._RE_ERROR.search(text):
            tag = "error"
        elif self._RE_WARNING.search(text):
            tag = "warning"
        else:
            m = self._RE_FITNESS.search(text)
            if m:
                tag = "fitness"

        if tag:
            self.log.insert("end", text, tag)
        else:
            self.log.insert("end", text)
        self.log.see("end")

    def _clear_log(self) -> None:
        self.log.delete("1.0", "end")

    def _set_status(self, msg: str, ttl_ms: int = 4000) -> None:
        if self._status_after:
            try:
                self.after_cancel(self._status_after)
            except Exception:
                pass
            self._status_after = None
        self.status_var.set(msg)
        if ttl_ms > 0:
            self._status_after = self.after(ttl_ms, lambda: self.status_var.set("Ready"))

    # ------------------------------------------------------------------ Runs
    def _refresh_runs(self) -> None:
        self.runs_list.delete(0, "end")
        self._run_item_names.clear()
        self._run_item_meta.clear()
        self.runs_meta_var.set("")
        p = self._selected_problem()
        if not p:
            return
        d = self.experiments_dir / p.name
        if not d.exists():
            return
        runs = [x for x in d.iterdir() if x.is_dir() and x.name.startswith("run")]

        def _key(x: Path) -> tuple[int, str]:
            name = x.name
            num = 10**9
            try:
                if name.startswith("run_") and name[4:].isdigit():
                    num = int(name[4:])
                elif name.startswith("run") and name[3:].isdigit():
                    num = int(name[3:])
            except Exception:
                pass
            return (num, name)

        def _name_key(name: str) -> tuple[int, str]:
            num = 10**9
            try:
                if name.startswith("run_") and name[4:].isdigit():
                    num = int(name[4:])
                elif name.startswith("run") and name[3:].isdigit():
                    num = int(name[3:])
            except Exception:
                pass
            return (num, name)

        # Determine fitness key from config
        fitness_key = "combined_score"
        cfg = self._load_config()
        if isinstance(cfg, dict):
            ec = cfg.get("EVOLVE_CONFIG", {})
            if isinstance(ec, dict):
                fitness_key = ec.get("fitness_key", fitness_key)

        selected_run = self.run_id_var.get().strip()
        selected_idx = None
        filt = self.run_filter_var.get().strip().lower()
        status_filter = self.run_status_var.get().strip().upper()
        if status_filter not in {s.upper() for s in RUN_STATUS_OPTIONS}:
            status_filter = "ALL"
        min_score = None
        min_score_invalid = False
        min_score_txt = self.run_min_score_var.get().strip()
        if min_score_txt:
            try:
                min_score = float(min_score_txt)
            except Exception:
                min_score_invalid = True
        status_counts = {"LIVE": 0, "WARM": 0, "IDLE": 0, "NEW": 0}
        rows: list[dict[str, object]] = []

        for run_path in sorted(runs, key=_key):
            name = run_path.name
            last_mtime = _run_last_log_mtime(run_path)
            status, color = self._run_activity(last_mtime)
            status_counts[status] += 1

            cache_key = (str(run_path), fitness_key)
            cached = self._run_score_cache.get(cache_key)
            if cached is not None and cached[0] == last_mtime:
                best = cached[1]
            else:
                best = _best_fitness_for_run(run_path, fitness_key)
                self._run_score_cache[cache_key] = (last_mtime, best)

            best_txt = f"{best:.4f}" if best is not None else "----"
            stamp = (
                datetime.fromtimestamp(last_mtime).strftime("%m-%d %H:%M")
                if last_mtime is not None
                else "--"
            )
            if filt and filt not in name.lower():
                continue
            if status_filter != "ALL" and status != status_filter:
                continue
            if min_score is not None and (best is None or best < min_score):
                continue

            label = f"{name:<16} {status:<4} {best_txt:>6}  {stamp}"
            rows.append({
                "name": name,
                "status": status,
                "color": color,
                "best": best,
                "last_mtime": last_mtime,
                "label": label,
                "path": run_path,
            })

        sort_mode = self.run_sort_var.get().strip()
        if sort_mode == "Best Score":
            rows.sort(
                key=lambda r: (
                    r["best"] is None,
                    -float(r["best"] if r["best"] is not None else -1e30),
                    -(float(r["last_mtime"]) if r["last_mtime"] is not None else 0.0),
                    str(r["name"]),
                )
            )
        elif sort_mode == "Name":
            rows.sort(key=lambda r: _name_key(str(r["name"])))
        else:
            rows.sort(
                key=lambda r: (
                    r["last_mtime"] is None,
                    -(float(r["last_mtime"]) if r["last_mtime"] is not None else 0.0),
                    _name_key(str(r["name"])),
                )
            )

        for row in rows:
            self.runs_list.insert("end", str(row["label"]))
            idx = self.runs_list.size() - 1
            try:
                self.runs_list.itemconfig(idx, foreground=str(row["color"]))
            except Exception:
                pass
            run_name = str(row["name"])
            self._run_item_names.append(run_name)
            self._run_item_meta.append(row)
            if run_name == selected_run:
                selected_idx = idx

        shown = len(self._run_item_names)
        meta = (
            f"Shown {shown}/{len(runs)}  LIVE {status_counts['LIVE']}  "
            f"WARM {status_counts['WARM']}  IDLE {status_counts['IDLE']}  NEW {status_counts['NEW']}"
        )
        filters: list[str] = [f"sort={sort_mode or RUN_SORT_OPTIONS[0]}"]
        if status_filter != "ALL":
            filters.append(f"state={status_filter}")
        if filt:
            filters.append(f"match='{filt}'")
        if min_score is not None:
            filters.append(f"min>={min_score:.4g}")
        if min_score_invalid:
            filters.append("min=invalid(ignored)")
        if filters:
            meta += "  |  " + "  ".join(filters)
        self.runs_meta_var.set(meta)
        if selected_idx is not None:
            self.runs_list.selection_set(selected_idx)
            self.runs_list.activate(selected_idx)
            self.runs_list.see(selected_idx)

    def _on_problem_change(self) -> None:
        self._sync_capabilities()
        self._run_score_cache.clear()
        self._refresh_configs()
        self._refresh_runs()
        self._refresh_models_from_config()
        self._refresh_config_editor()
        self._refresh_models_tab()
        self._viz_last_ckpt = None
        self._refresh_visualizer()
        self._refresh_run_snapshot()
        self._save_ui_state()

    def _on_run_select(self) -> None:
        sel = self.runs_list.curselection()
        if not sel:
            return
        idx = sel[0]
        run_name = (
            self._run_item_names[idx]
            if 0 <= idx < len(self._run_item_names)
            else self.runs_list.get(idx).strip().split()[0]
        )
        self.run_id_var.set(run_name)
        self._refresh_run_snapshot()
        self._viz_last_ckpt = None
        self._refresh_visualizer()
        self._save_ui_state()

    def _selected_run_name(self) -> Optional[str]:
        sel = self.runs_list.curselection()
        if sel:
            idx = sel[0]
            if 0 <= idx < len(self._run_item_names):
                return self._run_item_names[idx]
            try:
                return self.runs_list.get(idx).strip().split()[0]
            except Exception:
                return None
        run_id = self.run_id_var.get().strip()
        return run_id or None

    def _open_selected_run_dir(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        run_name = self._selected_run_name()
        if not run_name:
            self._set_status("Select a run first")
            return
        self.run_id_var.set(run_name)
        run_dir = self.experiments_dir / p.name / run_name
        if not run_dir.exists():
            messagebox.showerror("Open Run", f"Run directory not found:\n{run_dir}")
            return
        _open_path(run_dir)
        self._save_ui_state()

    def _copy_selected_run_id(self) -> None:
        run_name = self._selected_run_name()
        if not run_name:
            self._set_status("Select a run first")
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(run_name)
            self.update_idletasks()
            self._set_status(f"Copied run id: {run_name}")
        except Exception as e:
            messagebox.showerror("Copy Failed", str(e))

    def _open_experiments(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        _open_path(self.experiments_dir / p.name)

    def _discover_configs(self, p: Problem) -> list[Path]:
        cfg_dir = p.prob_dir / "configs"
        if not cfg_dir.exists():
            return []
        cfgs = [c for c in cfg_dir.iterdir() if c.is_file() and c.suffix in {".yml", ".yaml"}]
        return sorted(cfgs)

    def _refresh_configs(self) -> None:
        p = self._selected_problem()
        if not p:
            self.cfg_combo.configure(values=[], state="disabled")
            self.cfg_var.set("")
            return
        cfgs = self._discover_configs(p)
        names = [c.name for c in cfgs]
        if not names:
            self.cfg_combo.configure(values=[], state="disabled")
            self.cfg_var.set("")
            return
        preferred = "config.yaml" if "config.yaml" in names else names[0]
        if self.cfg_var.get() not in names:
            self.cfg_var.set(preferred)
        self.cfg_combo.configure(values=names, state="readonly")

    def _selected_cfg_path(self) -> Optional[Path]:
        p = self._selected_problem()
        if not p:
            return None
        name = self.cfg_var.get().strip()
        if not name:
            return None
        return p.prob_dir / "configs" / name

    def _on_config_change(self) -> None:
        self._refresh_models_from_config()
        self._refresh_config_editor()
        self._refresh_models_tab()
        self._save_ui_state()

    def _open_config(self) -> None:
        cfg = self._selected_cfg_path()
        if not cfg:
            return
        _open_path(cfg)

    # ------------------------------------------------------------------ Config editor
    def _build_config_tab(self) -> None:
        """Build the Config editor tab inside the notebook."""
        self._cfg_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._cfg_tab, text="  Config  ")
        self._cfg_tab.columnconfigure(0, weight=1)
        self._cfg_tab.rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ttk.Frame(self._cfg_tab, padding=(10, 8), style="Toolbar.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(toolbar, text="Save to Disk", style="Accent.TButton",
                   command=self._save_config_from_editor).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Reload", command=self._refresh_config_editor).pack(
            side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Open in Editor", command=self._open_config).pack(
            side="left", padx=(0, 12))
        self._cfg_editor_status = ttk.Label(toolbar, text="", style="Dim.TLabel")
        self._cfg_editor_status.pack(side="left", padx=(6, 0))

        # Scrollable area
        container = ttk.Frame(self._cfg_tab)
        container.grid(row=1, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self._cfg_canvas = tk.Canvas(container, bg=C.BASE, highlightthickness=0, borderwidth=0)
        cfg_sb = ttk.Scrollbar(container, orient="vertical", command=self._cfg_canvas.yview)
        self._cfg_scroll_frame = ttk.Frame(self._cfg_canvas)

        self._cfg_scroll_frame.bind(
            "<Configure>",
            lambda _e: self._cfg_canvas.configure(scrollregion=self._cfg_canvas.bbox("all")))

        self._cfg_canvas_win = self._cfg_canvas.create_window(
            (0, 0), window=self._cfg_scroll_frame, anchor="nw")
        self._cfg_canvas.configure(yscrollcommand=cfg_sb.set)

        # Track canvas width so scroll_frame fills it horizontally
        def _resize_scroll_frame(event):
            self._cfg_canvas.itemconfig(self._cfg_canvas_win, width=event.width)
        self._cfg_canvas.bind("<Configure>", _resize_scroll_frame)

        self._cfg_canvas.grid(row=0, column=0, sticky="nsew")
        cfg_sb.grid(row=0, column=1, sticky="ns")

        # Mouse wheel scrolling
        self._bind_mousewheel(self._cfg_canvas, self._cfg_canvas)
        self._bind_mousewheel(self._cfg_scroll_frame, self._cfg_canvas)

        # Widget storage: (key_path, widget, value_type)
        self._cfg_widgets: list[tuple[list, tk.Widget, type]] = []
        self._cfg_data: Optional[dict] = None

    def _refresh_config_editor(self) -> None:
        """Reload config from disk and rebuild all editor widgets."""
        # Clear existing widgets
        for child in self._cfg_scroll_frame.winfo_children():
            child.destroy()
        self._cfg_widgets.clear()

        cfg = self._load_config()
        if not isinstance(cfg, dict):
            ttk.Label(self._cfg_scroll_frame, text="No config loaded",
                      style="Dim.TLabel").pack(padx=12, pady=12)
            return
        self._cfg_data = cfg

        # General settings
        general_keys = ["EVAL_TIMEOUT", "SEED", "MAX_MEM_BYTES", "MEM_CHECK_INTERVAL_S"]
        general_items = [(k, cfg[k]) for k in general_keys if k in cfg]
        if general_items:
            self._add_cfg_section("General", general_items, [])

        # EVOLVE_CONFIG
        ec = cfg.get("EVOLVE_CONFIG", {})
        if isinstance(ec, dict):
            flat = [(k, v) for k, v in ec.items()
                    if k not in SKIP_KEYS and not isinstance(v, dict)]
            if flat:
                self._add_cfg_section("Evolution", flat, ["EVOLVE_CONFIG"])

            sk = ec.get("selection_kwargs", {})
            if isinstance(sk, dict) and sk:
                self._add_cfg_section("Selection", list(sk.items()),
                                      ["EVOLVE_CONFIG", "selection_kwargs"])

            sched = ec.get("scheduler_kwargs", {})
            if isinstance(sched, dict) and sched:
                self._add_cfg_section("Scheduler", list(sched.items()),
                                      ["EVOLVE_CONFIG", "scheduler_kwargs"])

        # MAP_ELITES
        me = cfg.get("MAP_ELITES", {})
        if isinstance(me, dict):
            flat = [(k, v) for k, v in me.items() if not isinstance(v, list)]
            if flat:
                self._add_cfg_section("MAP-Elites", flat, ["MAP_ELITES"])
            features = me.get("features", [])
            if isinstance(features, list):
                for i, feat in enumerate(features):
                    if isinstance(feat, dict):
                        name = feat.get("name", f"#{i}")
                        self._add_cfg_section(
                            f"Feature: {name}", list(feat.items()),
                            ["MAP_ELITES", "features", i])

        self._cfg_editor_status.configure(text="Loaded from disk")

    def _add_cfg_section(self, title: str, items: list[tuple[str, object]],
                         key_prefix: list) -> None:
        """Add a LabelFrame section with labeled parameter widgets."""
        section = ttk.LabelFrame(self._cfg_scroll_frame, text=f"  {title}  ", padding=(12, 8))
        section.pack(fill="x", padx=8, pady=(0, 8))
        section.columnconfigure(1, weight=1)

        for row_idx, (key, value) in enumerate(items):
            if key in SKIP_KEYS:
                continue
            full_path = key_prefix + [key]

            lbl = ttk.Label(section, text=key, width=26, anchor="w")
            lbl.grid(row=row_idx, column=0, sticky="w", padx=(0, 8), pady=2)

            widget = self._create_cfg_widget(section, key, value)
            widget.grid(row=row_idx, column=1, sticky="w", pady=2)

            self._cfg_widgets.append((full_path, widget, type(value)))

    def _create_cfg_widget(self, parent, key: str, value) -> tk.Widget:
        """Create the appropriate widget for a parameter value.

        IMPORTANT: StringVar is stored on the widget as ``._var`` to prevent
        garbage-collection (Tk drops the value if the Python object dies).
        """
        # Boolean -> True/False dropdown
        if isinstance(value, bool):
            var = tk.StringVar(value=str(value))
            w = ttk.Combobox(parent, textvariable=var, values=["True", "False"],
                             state="readonly", width=12)
            w._var = var  # prevent GC
            return w

        # Known categorical -> dropdown with all options
        if key in DROPDOWN_OPTIONS:
            options = list(DROPDOWN_OPTIONS[key])
            current = str(value)
            if current not in options:
                options.insert(0, current)
            var = tk.StringVar(value=current)
            w = ttk.Combobox(parent, textvariable=var, values=options,
                             state="readonly", width=28)
            w._var = var
            return w

        # Numeric -> typed entry
        if isinstance(value, (int, float)):
            var = tk.StringVar(value=str(value))
            w = ttk.Entry(parent, textvariable=var, width=16)
            w._var = var
            return w

        # None -> entry with "null" placeholder
        if value is None:
            var = tk.StringVar(value="null")
            w = ttk.Entry(parent, textvariable=var, width=16)
            w._var = var
            return w

        # String -> typed entry
        var = tk.StringVar(value=str(value))
        w = ttk.Entry(parent, textvariable=var, width=32)
        w._var = var
        return w

    def _save_config_from_editor(self) -> None:
        """Read all widget values, convert types, and overwrite the config file."""
        cfg_path = self._selected_cfg_path()
        if not cfg_path or not self._cfg_data:
            return

        cfg = self._cfg_data

        for key_path, widget, orig_type in self._cfg_widgets:
            raw = widget.get().strip() if hasattr(widget, "get") else ""
            path_str = ".".join(str(k) for k in key_path)

            # Convert to the original type
            if orig_type is bool:
                val = raw.lower() == "true"
            elif orig_type is int:
                try:
                    val = int(float(raw))
                except (ValueError, TypeError):
                    messagebox.showerror("Invalid Value",
                                         f"{path_str} must be an integer.\nGot: {raw!r}")
                    return
            elif orig_type is float:
                try:
                    val = float(raw)
                except (ValueError, TypeError):
                    messagebox.showerror("Invalid Value",
                                         f"{path_str} must be a number.\nGot: {raw!r}")
                    return
            elif orig_type is type(None):
                val = None if raw.lower() in ("null", "none", "") else raw
            else:
                val = raw

            # Navigate to the correct place in the config dict
            target = cfg
            for k in key_path[:-1]:
                target = target[k]
            target[key_path[-1]] = val

        try:
            self._dump_config(cfg, cfg_path)
            self._cfg_editor_status.configure(text="Saved!")
            self._set_status(f"Config saved: {cfg_path.name}")
            self.after(3000, lambda: self._cfg_editor_status.configure(text=""))
            # Refresh model lists in case fitness_key etc. changed
            self._refresh_models_from_config()
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    # ------------------------------------------------------------------ Models tab
    def _build_models_tab(self) -> None:
        """Build the Models editor tab inside the notebook."""
        self._models_tab = ttk.Frame(self._notebook)
        self._notebook.add(self._models_tab, text="  Models  ")
        self._models_tab.columnconfigure(0, weight=1)
        self._models_tab.rowconfigure(1, weight=1)

        # Toolbar
        toolbar = ttk.Frame(self._models_tab, padding=(10, 8), style="Toolbar.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Button(toolbar, text="Save to Disk", style="Accent.TButton",
                   command=self._save_models_tab).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Reload", command=self._refresh_models_tab).pack(
            side="left", padx=(0, 6))
        self._models_editor_status = ttk.Label(toolbar, text="", style="Dim.TLabel")
        self._models_editor_status.pack(side="left", padx=(6, 0))

        # Scrollable area
        container = ttk.Frame(self._models_tab)
        container.grid(row=1, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self._mtab_canvas = tk.Canvas(container, bg=C.BASE, highlightthickness=0, borderwidth=0)
        mtab_sb = ttk.Scrollbar(container, orient="vertical", command=self._mtab_canvas.yview)
        self._mtab_scroll_frame = ttk.Frame(self._mtab_canvas)

        self._mtab_scroll_frame.bind(
            "<Configure>",
            lambda _e: self._mtab_canvas.configure(scrollregion=self._mtab_canvas.bbox("all")))

        self._mtab_canvas_win = self._mtab_canvas.create_window(
            (0, 0), window=self._mtab_scroll_frame, anchor="nw")
        self._mtab_canvas.configure(yscrollcommand=mtab_sb.set)

        def _resize_scroll_frame(event):
            self._mtab_canvas.itemconfig(self._mtab_canvas_win, width=event.width)
        self._mtab_canvas.bind("<Configure>", _resize_scroll_frame)

        self._mtab_canvas.grid(row=0, column=0, sticky="nsew")
        mtab_sb.grid(row=0, column=1, sticky="ns")

        # Mouse wheel scrolling
        self._bind_mousewheel(self._mtab_canvas, self._mtab_canvas)
        self._bind_mousewheel(self._mtab_scroll_frame, self._mtab_canvas)

        # Widget storage: section_key -> list of slot dicts (field_name -> widget)
        self._mtab_slots: dict[str, list[dict[str, tk.Widget]]] = {}
        self._mtab_vars: list[tk.Variable] = []  # prevent GC

    def _refresh_models_tab(self) -> None:
        """Reload models from config and rebuild the models tab."""
        for child in self._mtab_scroll_frame.winfo_children():
            child.destroy()
        self._mtab_slots.clear()
        self._mtab_vars.clear()

        cfg = self._load_config()
        if not isinstance(cfg, dict):
            ttk.Label(self._mtab_scroll_frame, text="No config loaded",
                      style="Dim.TLabel").pack(padx=12, pady=12)
            return

        # Collect available model names from config + installed
        model_names: list[str] = []
        self._collect_model_names(cfg, model_names)
        model_names = list(dict.fromkeys(model_names))
        for m in getattr(self, "_installed_model_names", []):
            if m not in model_names:
                model_names.append(m)
        self._mtab_model_options = [_NONE_MODEL] + model_names

        # Exploration (3 slots)
        self._build_llm_section(
            "Exploration Ensemble", "EXPLORATION_ENSEMBLE",
            cfg.get("EXPLORATION_ENSEMBLE") or [], 3, _LLM_FIELDS)

        # Exploitation (3 slots)
        self._build_llm_section(
            "Exploitation Ensemble", "EXPLOITATION_ENSEMBLE",
            cfg.get("EXPLOITATION_ENSEMBLE") or [], 3, _LLM_FIELDS)

        # Sampler AUX LM (1 slot)
        aux = cfg.get("SAMPLER_AUX_LM")
        self._build_llm_section(
            "Sampler AUX LM", "SAMPLER_AUX_LM",
            [aux] if isinstance(aux, dict) else [], 1, _LLM_FIELDS)

        # Embedding (1 slot, fewer fields)
        emb = cfg.get("EMBEDDING")
        self._build_llm_section(
            "Embedding", "EMBEDDING",
            [emb] if isinstance(emb, dict) else [], 1, _EMBED_FIELDS)

        self._models_editor_status.configure(text="Loaded from disk")

    def _build_llm_section(self, title: str, section_key: str,
                           entries: list[dict], num_slots: int,
                           fields: tuple[str, ...]) -> None:
        """Build a LabelFrame with a grid of LLM parameter slots."""
        section = ttk.LabelFrame(self._mtab_scroll_frame, text=f"  {title}  ",
                                 padding=(12, 8))
        section.pack(fill="x", padx=8, pady=(0, 8))

        # Header row
        for col, field in enumerate(fields):
            lbl = ttk.Label(section, text=field, style="Dim.TLabel", font=FONT_MONO_SM)
            lbl.grid(row=0, column=col, sticky="w", padx=(0, 6), pady=(0, 4))

        slots: list[dict[str, tk.Widget]] = []
        for slot_idx in range(num_slots):
            entry = entries[slot_idx] if slot_idx < len(entries) else {}
            slot_widgets: dict[str, tk.Widget] = {}

            for col, field in enumerate(fields):
                default = _LLM_DEFAULTS.get(field, "")
                value = entry.get(field, default) if entry else default
                # Empty entry (no config data) -> show "(none)" for model_name
                if not entry and field == "model_name":
                    value = _NONE_MODEL

                w = self._create_llm_widget(section, field, value)
                w.grid(row=slot_idx + 1, column=col, sticky="w", padx=(0, 6), pady=2)
                slot_widgets[field] = w

            slots.append(slot_widgets)

            # Wire model_name toggle: disable other fields when "(none)"
            model_w = slot_widgets["model_name"]
            others = {k: v for k, v in slot_widgets.items() if k != "model_name"}
            self._wire_model_toggle(model_w, others)

        self._mtab_slots[section_key] = slots

    def _create_llm_widget(self, parent, field: str, value) -> tk.Widget:
        """Create a widget for a single LLM parameter field."""
        if field == "model_name":
            var = tk.StringVar(value=str(value))
            self._mtab_vars.append(var)
            w = ttk.Combobox(parent, textvariable=var,
                             values=getattr(self, "_mtab_model_options", [_NONE_MODEL]),
                             width=28)
            w._var = var
            return w

        if field == "verify_ssl":
            var = tk.StringVar(value=str(value))
            self._mtab_vars.append(var)
            w = ttk.Combobox(parent, textvariable=var, values=["True", "False"],
                             state="readonly", width=8)
            w._var = var
            return w

        # Numeric fields
        var = tk.StringVar(value=str(value))
        self._mtab_vars.append(var)
        width = 8 if field in ("retries", "max_tok") else 10
        w = ttk.Entry(parent, textvariable=var, width=width)
        w._var = var
        return w

    def _wire_model_toggle(self, model_widget: tk.Widget,
                           other_widgets: dict[str, tk.Widget]) -> None:
        """Disable/enable fields based on whether model_name is '(none)'."""
        def _on_change(*_args):
            val = model_widget.get().strip()
            is_none = (val == _NONE_MODEL)
            for w in other_widgets.values():
                if isinstance(w, ttk.Combobox):
                    w.configure(state="disabled" if is_none else "readonly")
                else:
                    w.configure(state="disabled" if is_none else "normal")

        var = getattr(model_widget, "_var", None)
        if var:
            var.trace_add("write", _on_change)
        _on_change()

    def _save_models_tab(self) -> None:
        """Read all model widgets and write back to config YAML."""
        cfg_path = self._selected_cfg_path()
        if not cfg_path:
            return
        cfg = self._load_config()
        if not isinstance(cfg, dict):
            return

        for section_key, slots in self._mtab_slots.items():
            entries: list[dict[str, object]] = []
            for slot_widgets in slots:
                model_name = slot_widgets["model_name"].get().strip()
                if model_name == _NONE_MODEL or not model_name:
                    continue  # Skip disabled slots

                entry: dict[str, object] = {}
                for field, widget in slot_widgets.items():
                    raw = widget.get().strip()
                    if field == "model_name":
                        entry[field] = raw
                    elif field == "verify_ssl":
                        entry[field] = raw.lower() == "true"
                    elif field in ("retries", "max_tok"):
                        try:
                            entry[field] = int(float(raw))
                        except (ValueError, TypeError):
                            messagebox.showerror("Invalid",
                                                 f"{field} must be an integer.\nGot: {raw!r}")
                            return
                    elif field in ("temp", "top_p", "weight", "request_timeout_s"):
                        try:
                            entry[field] = float(raw)
                        except (ValueError, TypeError):
                            messagebox.showerror("Invalid",
                                                 f"{field} must be a number.\nGot: {raw!r}")
                            return
                    else:
                        entry[field] = raw
                entries.append(entry)

            # Single-entry sections store as dict, not list
            if section_key in ("SAMPLER_AUX_LM", "EMBEDDING"):
                cfg[section_key] = entries[0] if entries else {}
            else:
                cfg[section_key] = entries

        try:
            self._dump_config(cfg, cfg_path)
            self._models_editor_status.configure(text="Saved!")
            self._set_status(f"Models saved: {cfg_path.name}")
            self.after(3000, lambda: self._models_editor_status.configure(text=""))
            self._refresh_models_from_config()
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    def _collect_model_names(self, obj: object, out: list[str]) -> None:
        if isinstance(obj, dict):
            if "model_name" in obj and isinstance(obj["model_name"], str):
                out.append(obj["model_name"])
            for v in obj.values():
                self._collect_model_names(v, out)
        elif isinstance(obj, list):
            for v in obj:
                self._collect_model_names(v, out)

    def _refresh_models_from_config(self) -> None:
        self.cfg_models_list.delete(0, "end")
        cfg = self._selected_cfg_path()
        if not cfg or not cfg.exists():
            return
        if yaml is None:
            self.cfg_models_list.insert("end", "PyYAML not available")
            return
        try:
            with cfg.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            self.cfg_models_list.insert("end", f"Failed: {e}")
            return
        models: list[str] = []
        self._collect_model_names(data, models)
        models = list(dict.fromkeys(models))
        if not models:
            self.cfg_models_list.insert("end", "No model_name entries")
            return
        for m in models:
            self.cfg_models_list.insert("end", m)

    def _load_config(self) -> Optional[dict]:
        cfg_path = self._selected_cfg_path()
        if not cfg_path or not cfg_path.exists():
            return None
        if yaml is None:
            return None
        try:
            return yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            return None

    def _dump_config(self, cfg: dict, out_path: Path) -> None:
        if yaml is None:
            raise RuntimeError("PyYAML not available")

        class _LiteralStr(str):
            pass

        class _LiteralDumper(yaml.SafeDumper):
            pass

        def _literal_str_representer(dumper, data):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

        _LiteralDumper.add_representer(_LiteralStr, _literal_str_representer)

        cfg_to_dump = cfg
        sys_msg = cfg.get("SYS_MSG")
        if isinstance(sys_msg, str):
            msg = "\n".join(line.rstrip() for line in sys_msg.splitlines())
            if not msg.endswith("\n"):
                msg += "\n"
            cfg_to_dump = dict(cfg)
            cfg_to_dump["SYS_MSG"] = _LiteralStr(msg)

        out_path.write_text(
            yaml.dump(cfg_to_dump, sort_keys=False, default_flow_style=False,
                      allow_unicode=True, Dumper=_LiteralDumper)
        )

    def _open_llm_editor(self) -> None:
        cfg_path = self._selected_cfg_path()
        if not cfg_path:
            messagebox.showerror("No Config", "Select a config first.")
            return
        if yaml is None:
            messagebox.showerror("PyYAML Missing", "PyYAML is required to edit configs.")
            return
        cfg = self._load_config()
        if not isinstance(cfg, dict):
            messagebox.showerror("Config Error", f"Failed to parse config: {cfg_path}")
            return

        self._refresh_installed_models()

        config_models: list[str] = []
        self._collect_model_names(cfg, config_models)
        installed_lines = list(self.installed_models_list.get(0, "end"))
        installed_models = []
        for ln in installed_lines:
            parts = ln.split()
            if parts:
                installed_models.append(parts[0])
        model_options = list(dict.fromkeys([*config_models, *installed_models]))
        if not model_options:
            model_options = config_models or installed_models or []

        str_options: dict[str, list[str]] = {}

        def _collect_str_opts(entry_dict: dict):
            for k, v in entry_dict.items():
                if isinstance(v, str):
                    opts = str_options.setdefault(k, [])
                    if v not in opts:
                        opts.append(v)

        for sec in ("EXPLORATION_ENSEMBLE", "EXPLOITATION_ENSEMBLE"):
            for m in cfg.get(sec, []) or []:
                if isinstance(m, dict):
                    _collect_str_opts(m)
        for sec in ("SAMPLER_AUX_LM", "EMBEDDING"):
            m = cfg.get(sec)
            if isinstance(m, dict):
                _collect_str_opts(m)

        win = tk.Toplevel(self)
        win.title("LLM Settings")
        win.geometry("820x640")
        win.configure(bg=C.BASE)

        container = ttk.Frame(win, padding=8)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, borderwidth=0, bg=C.BASE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>",
                          lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._bind_mousewheel(canvas, canvas)
        self._bind_mousewheel(scroll_frame, canvas)

        widgets: list[tuple[dict, str, tk.Widget, object]] = []

        # Store StringVars to prevent garbage-collection
        _kept_vars: list[tk.StringVar] = []

        def _add_param_row(parent, label, key, value, entry_ref):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=16).pack(side="left")

            if key == "model_name":
                var = tk.StringVar(value=str(value))
                _kept_vars.append(var)
                cb = ttk.Combobox(row, textvariable=var, values=model_options,
                                  state="readonly", width=40)
                cb.pack(side="left", fill="x", expand=True)
                widgets.append((entry_ref, key, cb, value))
                return
            if isinstance(value, bool):
                var = tk.StringVar(value="True" if value else "False")
                _kept_vars.append(var)
                cb = ttk.Combobox(row, textvariable=var, values=["True", "False"],
                                  state="readonly", width=10)
                cb.pack(side="left")
                widgets.append((entry_ref, key, cb, value))
                return
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                var = tk.StringVar(value=str(value))
                _kept_vars.append(var)
                ent = ttk.Entry(row, textvariable=var, width=12)
                ent.pack(side="left")
                widgets.append((entry_ref, key, ent, value))
                return
            if isinstance(value, str):
                opts = str_options.get(key, [])
                if value not in opts:
                    opts = [value] + opts
                var = tk.StringVar(value=value)
                _kept_vars.append(var)
                cb = ttk.Combobox(row, textvariable=var, values=opts or [value],
                                  state="readonly", width=40)
                cb.pack(side="left", fill="x", expand=True)
                widgets.append((entry_ref, key, cb, value))
                return
            var = tk.StringVar(value=str(value))
            _kept_vars.append(var)
            ent = ttk.Entry(row, textvariable=var, width=20)
            ent.pack(side="left")
            widgets.append((entry_ref, key, ent, value))

        def _section_frame(title: str) -> ttk.LabelFrame:
            lf = ttk.LabelFrame(scroll_frame, text=f"  {title}  ", padding=8)
            lf.pack(fill="x", pady=(0, 10))
            return lf

        def _add_model_block(parent, title: str, entry_dict: dict):
            blk = ttk.LabelFrame(parent, text=title, padding=8)
            blk.pack(fill="x", pady=(0, 8))
            keys = list(entry_dict.keys())
            if "model_name" in keys:
                keys.remove("model_name")
                keys = ["model_name"] + keys
            for key in keys:
                _add_param_row(blk, key, key, entry_dict[key], entry_dict)

        for sec in ("EXPLORATION_ENSEMBLE", "EXPLOITATION_ENSEMBLE"):
            entries = cfg.get(sec, []) or []
            if not entries:
                continue
            lf = _section_frame(sec)
            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                _add_model_block(lf, f"#{idx}", entry)

        for sec in ("SAMPLER_AUX_LM", "EMBEDDING"):
            entry = cfg.get(sec)
            if isinstance(entry, dict):
                lf = _section_frame(sec)
                _add_model_block(lf, sec, entry)

        def _save() -> None:
            for entry_ref, key, widget, orig in widgets:
                val = widget.get().strip() if hasattr(widget, "get") else ""
                if isinstance(orig, bool):
                    entry_ref[key] = (val.lower() == "true")
                elif isinstance(orig, int) and not isinstance(orig, bool):
                    try:
                        num = float(val)
                    except Exception:
                        messagebox.showerror("Invalid", f"{key} expects integer.")
                        return
                    if not num.is_integer():
                        messagebox.showerror("Invalid", f"{key} expects integer.")
                        return
                    entry_ref[key] = int(num)
                elif isinstance(orig, float):
                    try:
                        entry_ref[key] = float(val)
                    except Exception:
                        messagebox.showerror("Invalid", f"{key} expects number.")
                        return
                else:
                    entry_ref[key] = val
            try:
                self._dump_config(cfg, cfg_path)
            except Exception as e:
                messagebox.showerror("Save Failed", str(e))
                return
            self._refresh_configs()
            self.cfg_var.set(cfg_path.name)
            self._refresh_models_from_config()
            self._set_status("LLM settings saved")
            messagebox.showinfo("Saved", f"Saved to:\n{cfg_path}")

        # Keep var references alive for the lifetime of the window
        win._kept_vars = _kept_vars  # type: ignore[attr-defined]

        btns = ttk.Frame(scroll_frame)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Save", style="Accent.TButton", command=_save).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Close", command=win.destroy).pack(side="left")

    def _refresh_installed_models(self) -> None:
        if self._models_refresh_inflight:
            return
        self._models_refresh_inflight = True
        self.models_status.configure(text="Refreshing...")

        def _worker() -> None:
            lines: list[str] = []
            err: Optional[str] = None
            if shutil.which("ollama") is None:
                err = "Ollama not found in PATH"
            else:
                try:
                    r = subprocess.run(
                        ["ollama", "list"], capture_output=True, text=True, timeout=5)
                    if r.returncode != 0:
                        err = (r.stderr or r.stdout or "ollama list failed").strip()
                    else:
                        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
                        if lines and lines[0].lower().startswith("name"):
                            lines = lines[1:]
                except Exception as e:
                    err = str(e)

            def _apply() -> None:
                self.installed_models_list.delete(0, "end")
                self._installed_model_names.clear()
                if err:
                    self.models_status.configure(text=err)
                else:
                    self.models_status.configure(text=f"{len(lines)} installed")
                    for ln in lines:
                        self.installed_models_list.insert("end", ln)
                        parts = ln.split()
                        if parts:
                            self._installed_model_names.append(parts[0])
                self._models_refresh_inflight = False

            self.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------ Process
    def _tick_drain(self) -> None:
        try:
            while True:
                line = self._q.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass

        with self._proc_lock:
            running = self._proc is not None
        self.btn_stop.configure(state=("normal" if running else "disabled"))

        # Update timer
        if running and self._run_start_time:
            elapsed = time.monotonic() - self._run_start_time
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            if h:
                self.timer_var.set(f"Running {h}:{m:02d}:{s:02d}")
            else:
                self.timer_var.set(f"Running {m}:{s:02d}")
        elif not running and self._run_start_time:
            elapsed = time.monotonic() - self._run_start_time
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            if h:
                self.timer_var.set(f"Finished in {h}:{m:02d}:{s:02d}")
            else:
                self.timer_var.set(f"Finished in {m}:{s:02d}")
            self._run_start_time = None

        if self._closing and not running:
            self.destroy()
            return
        self.after(100, self._tick_drain)

    def _spawn(self, cmd: list[str], *, cwd: Optional[Path] = None,
               env: Optional[dict[str, str]] = None) -> None:
        with self._proc_lock:
            if self._proc is not None:
                self._append_log("[dashboard] Busy: command already running.\n")
                self._set_status("Busy: command already running")
                return

            cmd_display = shlex.join(cmd)
            self._append_log(f"$ {cmd_display}\n")
            self._run_start_time = time.monotonic()
            self._active_cmd = cmd_display

            proc_env = dict(os.environ)
            proc_env["PYTHON"] = sys.executable
            if not proc_env.get("TERM"):
                proc_env["TERM"] = "xterm-256color"
            if env:
                proc_env.update(env)

            api_base = self.api_base_var.get().strip()
            api_key = self.api_key_var.get().strip()
            if api_base:
                proc_env["API_BASE"] = api_base
            if api_key:
                proc_env["API_KEY"] = api_key

            kwargs: dict = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
                "cwd": str(cwd or self.repo_root),
                "env": proc_env,
            }

            if os.name == "posix":
                kwargs["preexec_fn"] = os.setsid

            self._proc = subprocess.Popen(cmd, **kwargs)  # type: ignore[arg-type]
            self.btn_stop.configure(state="normal")
            self._set_status(f"Running: {' '.join(cmd[-3:])}", ttl_ms=0)

            t = threading.Thread(target=self._reader_thread, args=(self._proc, cmd_display), daemon=True)
            t.start()

    def _reader_thread(self, proc: subprocess.Popen[str], cmd_display: str) -> None:
        try:
            assert proc.stdout is not None
            for ln in proc.stdout:
                self._q.put(ln)
        except Exception as e:
            self._q.put(f"[dashboard] Reader error: {e}\n")
        finally:
            code = None
            try:
                code = proc.wait(timeout=0.1)
            except Exception:
                pass
            self._q.put(f"\n[exit] returncode={code}\n")
            with self._proc_lock:
                self._proc = None
            self.after(0, lambda: self._on_command_finished(cmd_display, code))

    def _notify_desktop(self, title: str, body: str) -> None:
        if self._closing or not self.notify_done_var.get():
            return
        if shutil.which("notify-send") is None:
            return
        try:
            subprocess.Popen(
                ["notify-send", title, body],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def _on_command_finished(self, cmd_display: str, code: Optional[int]) -> None:
        self._active_cmd = ""
        if self._closing:
            return
        self._refresh_runs()
        run_id = self.run_id_var.get().strip()
        if run_id:
            self._refresh_run_snapshot()
            cur_tab = -1
            try:
                cur_tab = int(self._notebook.index("current"))
            except Exception:
                cur_tab = -1
            if cur_tab == 1:
                self._refresh_visualizer()

        rc = -1 if code is None else int(code)
        ok = (rc == 0)
        tail = " ".join(cmd_display.split()[-4:])
        status = f"Completed ({tail})" if ok else f"Failed rc={rc} ({tail})"
        self._set_status(status, ttl_ms=9000)

        note_title = "CodeEvolve Command Finished" if ok else "CodeEvolve Command Failed"
        note_body = f"rc={rc}  {tail}"
        self._notify_desktop(note_title, note_body)

    def _stop_proc(self) -> None:
        with self._proc_lock:
            proc = self._proc
        if proc is None:
            return
        self._set_status("Stopping...")
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _ensure_api(self) -> bool:
        self._sync_env_status()
        if not (self.api_base_var.get().strip() and self.api_key_var.get().strip()):
            messagebox.showerror("Missing API Settings",
                                 "Set API_BASE and API_KEY (top bar) before running.")
            return False
        return True

    def _suggest_new_problem_ai_model(self) -> str:
        preferred = _AI_TEMPLATE_MODEL_DEFAULT.strip()
        if preferred:
            return preferred

        cfg_models: list[str] = []
        cfg = self._load_config()
        if isinstance(cfg, dict):
            self._collect_model_names(cfg, cfg_models)
        for model in cfg_models:
            if isinstance(model, str) and model.strip() and model != _NONE_MODEL:
                return model.strip()
        for model in getattr(self, "_installed_model_names", []):
            if isinstance(model, str) and model.strip() and model != _NONE_MODEL:
                return model.strip()
        return _AI_TEMPLATE_MODEL_DEFAULT

    def _suggest_new_problem_transcribe_model(self) -> str:
        api_base = self.api_base_var.get().strip().lower()
        if "openai.com" in api_base:
            return _AI_TRANSCRIBE_MODEL_DEFAULT
        return "whisper-1"

    def _on_close(self) -> None:
        if self._viz_after:
            try:
                self.after_cancel(self._viz_after)
            except Exception:
                pass
            self._viz_after = None
        if self._runs_auto_after:
            try:
                self.after_cancel(self._runs_auto_after)
            except Exception:
                pass
            self._runs_auto_after = None
        with self._proc_lock:
            running = self._proc is not None
        if running:
            if not messagebox.askokcancel("Quit", "A command is still running. Stop and exit?"):
                return
            self._save_ui_state()
            self._closing = True
            self._close_kill_at = time.monotonic() + 3.0
            self._stop_proc()
            self.after(200, self._check_force_close)
            return
        self._save_ui_state()
        self.destroy()

    def _check_force_close(self) -> None:
        if not self._closing:
            return
        with self._proc_lock:
            proc = self._proc
        if proc is None:
            self.destroy()
            return
        if self._close_kill_at is not None and time.monotonic() >= self._close_kill_at:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
            self.destroy()
            return
        self.after(200, self._check_force_close)

    # ------------------------------------------------------------------ Commands
    def _cmd_run(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        if not self._ensure_api():
            return
        run_id = self.run_id_var.get().strip()
        cmd = ["bash", str(p.script_path), "run"]
        if run_id:
            cmd += ["--run", run_id]
        else:
            cmd += ["--next"]
        self._add_common_run_flags(p, cmd)
        self._save_ui_state()
        self._spawn(cmd)

    def _cmd_run_next(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        if not self._ensure_api():
            return
        cmd = ["bash", str(p.script_path), "run", "--next"]
        self._add_common_run_flags(p, cmd)
        self._save_ui_state()
        self._spawn(cmd)

    def _add_common_run_flags(self, p: Problem, cmd: list[str]) -> None:
        cpu = self.cpu_var.get().strip()
        if cpu and "--cpu" in p.help_text:
            cmd += ["--cpu", cpu]
        load_ckpt = self.load_ckpt_var.get().strip()
        if load_ckpt and "--load-ckpt" in p.help_text:
            cmd += ["--load-ckpt", load_ckpt]
        cfg_path = self._selected_cfg_path()
        if cfg_path and "--cfg-path" in p.help_text:
            cmd += ["--cfg-path", str(cfg_path)]
        if self.no_taskset_var.get() and "--no-taskset" in p.help_text:
            cmd += ["--no-taskset"]
        if ("--skip-dryad" in p.help_text) and ("--require-dryad" in p.help_text):
            cmd += ["--skip-dryad"] if self.skip_dryad_var.get() else ["--require-dryad"]
        # Dashboard runs without a TTY, so auto-confirm if the script supports it.
        if ("--y" in p.help_text) or ("--yes" in p.help_text):
            cmd += ["--y"]

    def _cmd_analyze(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        self._spawn(["bash", str(p.script_path), "analyze"])

    def _cmd_winner(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        args = self.winner_args_var.get().strip()
        extra = shlex.split(args) if args else []
        self._spawn(["bash", str(p.script_path), "winner", *extra])

    def _cmd_viz(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        run_id = self.run_id_var.get().strip()
        if not run_id:
            messagebox.showerror("Missing Run", "Set a run name before visualize.")
            return
        island = self.island_var.get().strip() or "0"
        self._spawn(["bash", str(p.script_path), "viz", run_id, island])

    def _cmd_tail(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        run_id = self.run_id_var.get().strip()
        if not run_id:
            messagebox.showerror("Missing Run", "Set a run name before tail.")
            return
        island = self.island_var.get().strip() or "0"
        self._spawn(["bash", str(p.script_path), "tail", run_id, island])

    def _cmd_warmstart(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        run_id = self.run_id_var.get().strip()
        if not run_id:
            messagebox.showerror("Missing Run", "Set a run name before warmstart.")
            return
        if not messagebox.askokcancel("Warmstart",
                                      f"This will overwrite init_program.py with best_sol.py from {run_id}.\nContinue?"):
            return
        island = self.island_var.get().strip() or "0"
        self._spawn(["bash", str(p.script_path), "warmstart", run_id, island])

    def _cmd_ls(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        cmd = "ls" if "ls" in p.commands else ("list" if "list" in p.commands else "ls")
        self._spawn(["bash", str(p.script_path), cmd])


def main() -> int:
    app = Dashboard()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
