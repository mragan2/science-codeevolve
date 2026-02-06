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
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox
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

FONT_UI    = ("Ubuntu Sans", 10)
FONT_UI_B  = ("Ubuntu Sans", 10, "bold")
FONT_MONO  = ("Ubuntu Sans Mono", 10)
FONT_MONO_SM = ("Ubuntu Sans Mono", 9)
FONT_TITLE = ("Ubuntu Sans", 12, "bold")

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
_LLM_FIELDS = ("model_name", "temp", "top_p", "max_tok", "retries", "weight", "verify_ssl")
_LLM_DEFAULTS: dict[str, object] = {
    "model_name": _NONE_MODEL,
    "temp": 0.5,
    "top_p": 0.9,
    "max_tok": 4096,
    "retries": 3,
    "weight": 0.33,
    "verify_ssl": False,
}
_EMBED_FIELDS = ("model_name", "retries", "verify_ssl")

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
    return {{}}


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
  local log_path="${{EXP_DIR}}/${{run_id}}/${{island}}/results.log"
  [[ -f "${{log_path}}" ]] || err "results.log not found: ${{log_path}}"
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

    # Labels
    style.configure("TLabel", background=C.BASE, foreground=C.TEXT, font=FONT_UI)
    style.configure("Dim.TLabel", foreground=C.OVERLAY0)
    style.configure("Title.TLabel", font=FONT_TITLE, foreground=C.LAVENDER)
    style.configure("Ok.TLabel", foreground=C.GREEN)
    style.configure("Err.TLabel", foreground=C.RED)
    style.configure("Status.TLabel", background=C.MANTLE, foreground=C.SUBTEXT0, font=FONT_MONO_SM)

    # LabelFrames
    style.configure("TLabelframe", background=C.BASE, foreground=C.SUBTEXT1,
                    font=FONT_UI_B, borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background=C.BASE, foreground=C.SUBTEXT1,
                    font=FONT_UI_B)

    # Buttons
    style.configure("TButton", background=C.SURFACE1, foreground=C.TEXT, font=FONT_UI,
                    padding=(10, 4), borderwidth=0)
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
                    insertcolor=C.TEXT, borderwidth=1, padding=3)
    style.map("TEntry", fieldbackground=[("focus", C.SURFACE1)])

    # Comboboxes
    style.configure("TCombobox", fieldbackground=C.SURFACE0, foreground=C.TEXT,
                    background=C.SURFACE1, arrowcolor=C.SUBTEXT0, borderwidth=1, padding=3)
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
                    font=FONT_UI, padding=(14, 6))
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


def _best_fitness_for_run(run_dir: Path, fitness_key: str = "combined_score") -> Optional[float]:
    """Scan island directories for the best fitness value."""
    best = None
    for island in run_dir.iterdir():
        if not island.is_dir():
            continue
        log_path = island / "results.log"
        if not log_path.exists():
            continue
        try:
            last_line = ""
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
            if not last_line:
                continue
            data = json.loads(last_line)
            val = data.get(fitness_key)
            if val is not None:
                val = float(val)
                if best is None or val > best:
                    best = val
        except Exception:
            continue
    return best


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
        self._viz_metric_key: str = "combined_score"

        self._build_ui()
        self._refresh_runs()
        self._refresh_configs()
        self._refresh_models_from_config()
        self._refresh_config_editor()
        self._refresh_models_tab()
        self._refresh_installed_models()
        self._bind_shortcuts()
        self._tick_drain()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        # ---- Header bar ----
        header = ttk.Frame(self, padding=(12, 10, 12, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(9, weight=1)

        ttk.Label(header, text="CodeEvolve", style="Title.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 16))

        ttk.Label(header, text="Problem:").grid(row=0, column=1, sticky="w")
        self.problem_var = tk.StringVar(value=(sorted(self.problems.keys())[0] if self.problems else ""))
        self.problem_combo = ttk.Combobox(
            header, textvariable=self.problem_var,
            values=sorted(self.problems.keys()), state="readonly", width=22,
        )
        self.problem_combo.grid(row=0, column=2, sticky="w", padx=(4, 2))
        self.problem_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_problem_change())
        ttk.Button(header, text="Scan", width=4,
                   command=self._refresh_problems).grid(row=0, column=3, sticky="w", padx=(0, 2))
        ttk.Button(header, text="New", width=3,
                   command=self._new_problem_dialog).grid(row=0, column=4, sticky="w", padx=(0, 10))

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
        ttk.Entry(header, textvariable=self.api_key_var, width=22, show="*").grid(
            row=0, column=8, sticky="w", padx=(4, 14))

        self.env_status = ttk.Label(header, text="", style="Dim.TLabel")
        self.env_status.grid(row=0, column=9, sticky="e")

        ttk.Separator(self, orient="horizontal").grid(row=0, column=0, sticky="sew", padx=12)

        # ---- Main paned area ----
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 0))

        # ---- Left sidebar ----
        sidebar = ttk.Frame(paned, width=260)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)
        paned.add(sidebar, weight=0)

        # Runs list
        runs_frame = ttk.LabelFrame(sidebar, text="  Runs  ", padding=6)
        runs_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        runs_frame.rowconfigure(0, weight=1)
        runs_frame.columnconfigure(0, weight=1)

        runs_scroll = ttk.Scrollbar(runs_frame, orient="vertical")
        self.runs_list = tk.Listbox(
            runs_frame, height=8, width=28, exportselection=False,
            bg=C.SURFACE0, fg=C.TEXT, selectbackground=C.BLUE, selectforeground=C.CRUST,
            highlightthickness=0, borderwidth=0, font=FONT_MONO,
            yscrollcommand=runs_scroll.set,
        )
        runs_scroll.config(command=self.runs_list.yview)
        self.runs_list.grid(row=0, column=0, sticky="nsew")
        runs_scroll.grid(row=0, column=1, sticky="ns")
        self.runs_list.bind("<<ListboxSelect>>", lambda _e: self._on_run_select())

        btn_row = ttk.Frame(runs_frame)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(btn_row, text="Refresh", command=self._refresh_runs).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Open Folder", command=self._open_experiments).pack(side="left")

        # Run Snapshot
        snapshot_frame = ttk.LabelFrame(sidebar, text="  Run Snapshot  ", padding=6)
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

        snap_btns = ttk.Frame(snapshot_frame)
        snap_btns.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(snap_btns, text="Refresh", command=self._refresh_run_snapshot).pack(
            side="left", padx=(0, 4))
        ttk.Button(snap_btns, text="Open Run", command=self._open_run_dir).pack(
            side="left", padx=(0, 4))
        ttk.Button(snap_btns, text="Open Best Sol", command=self._open_best_sol).pack(
            side="left", padx=(0, 4))
        ttk.Button(snap_btns, text="Open Best Prompt", command=self._open_best_prompt).pack(
            side="left", padx=(0, 4))

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
        ctl = ttk.Frame(right, padding=(8, 6))
        ctl.grid(row=0, column=0, sticky="ew")
        ctl.columnconfigure(10, weight=1)

        self.run_id_var = tk.StringVar(value="")
        self.island_var = tk.StringVar(value="0")
        self.cpu_var = tk.StringVar(value=os.environ.get("CPU_LIST", "0-7"))
        self.load_ckpt_var = tk.StringVar(value="0")
        self.winner_args_var = tk.StringVar(value="")
        self.skip_dryad_var = tk.BooleanVar(value=True)
        self.no_taskset_var = tk.BooleanVar(value=False)
        self.cfg_var = tk.StringVar(value="")

        # Row 0: run params
        r = 0
        ttk.Label(ctl, text="Run:").grid(row=r, column=0, sticky="w")
        ttk.Entry(ctl, textvariable=self.run_id_var, width=12).grid(row=r, column=1, sticky="w", padx=(4, 10))
        ttk.Label(ctl, text="Island:").grid(row=r, column=2, sticky="w")
        ttk.Entry(ctl, textvariable=self.island_var, width=4).grid(row=r, column=3, sticky="w", padx=(4, 10))
        ttk.Label(ctl, text="CPU:").grid(row=r, column=4, sticky="w")
        ttk.Entry(ctl, textvariable=self.cpu_var, width=8).grid(row=r, column=5, sticky="w", padx=(4, 10))
        ttk.Label(ctl, text="Ckpt:").grid(row=r, column=6, sticky="w")
        ttk.Entry(ctl, textvariable=self.load_ckpt_var, width=4).grid(row=r, column=7, sticky="w", padx=(4, 10))

        ttk.Label(ctl, text="Config:").grid(row=r, column=8, sticky="w")
        self.cfg_combo = ttk.Combobox(ctl, textvariable=self.cfg_var, values=[], state="readonly", width=22)
        self.cfg_combo.grid(row=r, column=9, sticky="w", padx=(4, 6))
        self.cfg_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_config_change())
        ttk.Button(ctl, text="Open", command=self._open_config).grid(row=r, column=10, sticky="w")

        # Row 1: checkboxes + winner args
        r = 1
        chk_frame = ttk.Frame(ctl)
        chk_frame.grid(row=r, column=0, columnspan=6, sticky="w", pady=(6, 0))
        ttk.Checkbutton(chk_frame, text="No taskset", variable=self.no_taskset_var).pack(side="left", padx=(0, 12))
        self.skip_dryad_chk = ttk.Checkbutton(chk_frame, text="Skip Dryad (elegans)", variable=self.skip_dryad_var)
        self.skip_dryad_chk.pack(side="left", padx=(0, 12))

        ttk.Label(ctl, text="Winner args:").grid(row=r, column=6, sticky="w", pady=(6, 0))
        ttk.Entry(ctl, textvariable=self.winner_args_var, width=40).grid(
            row=r, column=7, columnspan=4, sticky="we", pady=(6, 0), padx=(4, 0))

        # Row 2: action buttons
        r = 2
        btn_bar = ttk.Frame(ctl)
        btn_bar.grid(row=r, column=0, columnspan=11, sticky="ew", pady=(10, 4))

        self.btn_run = ttk.Button(btn_bar, text="Run", style="Accent.TButton", command=self._cmd_run)
        self.btn_run.pack(side="left", padx=(0, 4))
        self.btn_run_next = ttk.Button(btn_bar, text="Run Next", style="Accent.TButton", command=self._cmd_run_next)
        self.btn_run_next.pack(side="left", padx=(0, 4))

        ttk.Separator(btn_bar, orient="vertical").pack(side="left", fill="y", padx=8)

        self.btn_analyze = ttk.Button(btn_bar, text="Analyze", command=self._cmd_analyze)
        self.btn_analyze.pack(side="left", padx=(0, 4))
        self.btn_winner = ttk.Button(btn_bar, text="Winner", command=self._cmd_winner)
        self.btn_winner.pack(side="left", padx=(0, 4))
        self.btn_viz = ttk.Button(btn_bar, text="Visualize", command=self._cmd_viz)
        self.btn_viz.pack(side="left", padx=(0, 4))
        self.btn_tail = ttk.Button(btn_bar, text="Tail Log", command=self._cmd_tail)
        self.btn_tail.pack(side="left", padx=(0, 4))
        self.btn_warmstart = ttk.Button(btn_bar, text="Warmstart", command=self._cmd_warmstart)
        self.btn_warmstart.pack(side="left", padx=(0, 4))
        self.btn_ls = ttk.Button(btn_bar, text="List Runs", command=self._cmd_ls)
        self.btn_ls.pack(side="left", padx=(0, 4))

        ttk.Separator(btn_bar, orient="vertical").pack(side="left", fill="y", padx=8)

        self.btn_stop = ttk.Button(btn_bar, text="Stop", style="Danger.TButton",
                                   command=self._stop_proc, state="disabled")
        self.btn_stop.pack(side="left", padx=(0, 4))
        ttk.Button(btn_bar, text="Clear Log", command=self._clear_log).pack(side="left", padx=(0, 4))

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
        status_bar = ttk.Frame(self, padding=(12, 4))
        status_bar.grid(row=2, column=0, sticky="ew")
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
            status_bar, text="Ctrl+R Run  Ctrl+N Next  Esc Stop  Ctrl+L Clear",
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

    # ------------------------------------------------------------------ State
    def _selected_problem(self) -> Optional[Problem]:
        name = self.problem_var.get().strip()
        return self.problems.get(name)

    def _sync_env_status(self) -> None:
        ok = bool(self.api_base_var.get().strip()) and bool(self.api_key_var.get().strip())
        self.env_status.configure(
            text=("API OK" if ok else "Missing API_BASE / API_KEY"),
            style=("Ok.TLabel" if ok else "Err.TLabel"),
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

    # ------------------------------------------------------------------ New problem wizard

    def _new_problem_dialog(self) -> None:
        """Open a dialog to create a new problem from templates."""
        dlg = tk.Toplevel(self)
        dlg.title("New Problem")
        dlg.configure(bg=C.BASE)
        dlg.geometry("860x720")
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
        for seq in ("<Button-4>", "<Button-5>"):
            body_canvas.bind(seq, lambda e: body_canvas.yview_scroll(
                -1 if e.num == 4 else 1, "units"))

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

        eval_txt = _make_editor(body_frame, "evaluate.py", _EVAL_TEMPLATE.format(name="my_problem"), 18)
        init_txt = _make_editor(body_frame, "init_program.py  (between EVOLVE-BLOCK markers)", _INIT_TEMPLATE, 12)
        sysmsg_txt = _make_editor(body_frame, "SYS_MSG  (LLM system prompt)", _SYSMSG_TEMPLATE, 12)

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
                       status_lbl,
                   )).pack(side="right")

    def _create_problem(self, dlg: tk.Toplevel, name: str,
                        eval_code: str, init_code: str, sysmsg: str,
                        status_lbl: ttk.Label) -> None:
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
            cfg = {
                "SYS_MSG": sysmsg + "\n",
                "CODEBASE_PATH": "src/",
                "INIT_FILE_DATA": {"filename": "init_program.py", "language": "python"},
                "EVAL_FILE_NAME": "evaluate.py",
                "EVAL_TIMEOUT": 30,
                "SEED": 42,
                "MAX_MEM_BYTES": 5_000_000_000,
                "MEM_CHECK_INTERVAL_S": 0.1,
                "EVOLVE_CONFIG": dict(_DEFAULT_EVOLVE_CONFIG),
                "EXPLORATION_ENSEMBLE": [],
                "EXPLOITATION_ENSEMBLE": [],
                "SAMPLER_AUX_LM": {
                    "model_name": "", "temp": 0.18, "top_p": 0.78,
                    "max_tok": 4096, "retries": 3, "weight": 1, "verify_ssl": False,
                },
                "EMBEDDING": {"model_name": "", "retries": 3, "verify_ssl": False},
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
            log_path = island_dir / "results.log"
            points: list[tuple[int, float]] = []
            if log_path.exists():
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
            log_path = island_dir / "results.log"
            if not log_path.exists():
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
            log_path = island_dir / "results.log"
            local_best = None
            local_epoch = 0
            if log_path.exists():
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

        from datetime import datetime
        updated_str = "unknown"
        if last_mtime is not None:
            updated_str = datetime.fromtimestamp(last_mtime).strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"Run: {run_id}",
            f"Islands: {len(islands)}",
            f"Latest epoch: {latest_epoch}",
            f"Last update: {updated_str}",
        ]
        if global_best is not None:
            lines.append(f"Global best: {global_best:.4f} (I{global_best_island})")
        if summaries:
            lines.append("")
            lines.append("Per island:")
            lines.append("  " + "  ".join(summaries))

        best_sol = (run_dir / "best_sol.py").exists()
        best_prompt = (run_dir / "best_prompt.txt").exists()
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

        ctrl = ttk.Frame(viz_tab, padding=(8, 6))
        ctrl.grid(row=0, column=0, sticky="ew")
        ctrl.columnconfigure(6, weight=1)

        self.viz_metric_var = tk.StringVar(value="combined_score")
        self.viz_auto_var = tk.BooleanVar(value=True)
        self.viz_epoch_var = tk.StringVar(value="")

        ttk.Label(ctrl, text="Metric:").grid(row=0, column=0, sticky="w")
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
        self.viz_metric_combo.grid(row=0, column=1, sticky="w", padx=(4, 12))
        self.viz_metric_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_visualizer())

        ttk.Checkbutton(
            ctrl, text="Auto (5 epochs)", variable=self.viz_auto_var,
            command=self._toggle_viz_auto,
        ).grid(row=0, column=2, sticky="w", padx=(0, 12))

        ttk.Button(ctrl, text="Refresh", command=self._refresh_visualizer).grid(
            row=0, column=3, sticky="w", padx=(0, 12))

        ttk.Label(ctrl, textvariable=self.viz_epoch_var, style="Dim.TLabel",
                  font=FONT_MONO_SM).grid(row=0, column=4, sticky="w")

        body = ttk.PanedWindow(viz_tab, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

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

        detail_frame = ttk.Frame(body, padding=(6, 0, 0, 0))
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(1, weight=1)
        body.add(detail_frame, weight=1)

        ttk.Label(detail_frame, text="Selection", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(4, 6)
        )
        self._viz_detail = ScrolledText(
            detail_frame, height=16, wrap="word",
            bg=C.MANTLE, fg=C.TEXT, insertbackground=C.TEXT,
            highlightthickness=0, borderwidth=0, padx=8, pady=6,
            font=FONT_MONO_SM,
        )
        self._viz_detail.grid(row=1, column=0, sticky="nsew")
        self._viz_detail.configure(state="disabled")

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

        p = self._selected_problem()
        if not p:
            self._viz_label("Select a problem")
            return
        run_id = self.run_id_var.get().strip()
        if not run_id:
            self._viz_label("Select a run")
            return
        run_dir = self.experiments_dir / p.name / run_id
        if not run_dir.exists():
            self._viz_label(f"Not found: {run_id}")
            return

        ckpt_map, latest_epoch = self._viz_collect_ckpts(run_dir)
        if not ckpt_map:
            self._viz_label("No checkpoints yet")
            return

        metric_key = self.viz_metric_var.get().strip() or "combined_score"
        self._viz_metric_key = metric_key
        nodes: dict[str, VizNode] = {}
        for island_idx, ckpt_path in ckpt_map.items():
            ckpt = self._load_ckpt(ckpt_path)
            if not isinstance(ckpt, dict):
                continue
            sol_db = ckpt.get("sol_db")
            if sol_db is None or not hasattr(sol_db, "programs"):
                continue
            try:
                programs = sol_db.programs.values()
            except Exception:
                continue
            for prog in programs:
                metric_val = self._viz_metric_value(prog, metric_key)
                if metric_val is None or not math.isfinite(metric_val):
                    continue
                fitness = getattr(prog, "fitness", None)
                if fitness is None or not math.isfinite(float(fitness)):
                    fitness_val = metric_val
                else:
                    fitness_val = float(fitness)
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
                eval_metrics = getattr(prog, "eval_metrics", None)
                nodes[prog_id] = VizNode(
                    prog_id=str(prog_id),
                    parent_id=str(parent_id) if parent_id else None,
                    island=island_found,
                    generation=generation,
                    fitness=fitness_val,
                    metric=float(metric_val),
                    code=str(code) if code is not None else None,
                    eval_metrics=eval_metrics if isinstance(eval_metrics, dict) else None,
                )

        if not nodes:
            self._viz_label("No programs in checkpoints")
            return

        edges = [(n.parent_id, n.prog_id) for n in nodes.values() if n.parent_id in nodes]
        self._draw_viz_graph(nodes, edges, metric_key)
        self._viz_nodes_cache = nodes

        if self._viz_selected_id in nodes:
            self._update_viz_details(self._viz_selected_id)

        if latest_epoch is not None:
            self.viz_epoch_var.set(f"Latest ckpt: {latest_epoch}")
            self._viz_last_ckpt = latest_epoch

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
        if closest_dist > 120:  # ~11px radius
            return
        self._viz_selected_id = closest_id
        self._update_viz_details(closest_id)
        self._refresh_visualizer()

    def _update_viz_details(self, prog_id: str) -> None:
        node = self._viz_nodes_cache.get(prog_id)
        if not node:
            return
        lines = [
            f"Program ID: {node.prog_id}",
            f"Island: {node.island}",
            f"Generation: {node.generation}",
            f"Fitness: {node.fitness:.4f}",
            f"{self._viz_metric_key}: {node.metric:.4f}",
            "",
        ]
        if node.parent_id:
            lines.append(f"Parent: {node.parent_id}")
            lines.append("")
        if node.eval_metrics:
            try:
                metrics_text = json.dumps(node.eval_metrics, indent=2, sort_keys=True)
            except Exception:
                metrics_text = str(node.eval_metrics)
            lines.append("Eval metrics:")
            lines.append(metrics_text)
            lines.append("")
        if node.code:
            code = node.code
            if len(code) > 4000:
                code = code[:4000] + "\n# ... truncated ..."
            lines.append("Code:")
            lines.append(code)

        self._viz_detail.configure(state="normal")
        self._viz_detail.delete("1.0", "end")
        self._viz_detail.insert("1.0", "\n".join(lines))
        self._viz_detail.configure(state="disabled")

    def _draw_viz_graph(self, nodes: dict[str, VizNode],
                        edges: list[tuple[str, str]], metric_key: str) -> None:
        self._viz_canvas.update_idletasks()
        cw = max(400, self._viz_canvas.winfo_width() or 600)
        ch = max(240, self._viz_canvas.winfo_height() or 360)

        pad_l, pad_r, pad_t, pad_b = 48, 18, 26, 18
        islands = sorted({n.island for n in nodes.values()})
        if not islands:
            self._viz_label("No islands found")
            return

        gap = 14
        band_h = (ch - pad_t - pad_b - gap * (len(islands) - 1)) / max(1, len(islands))
        plot_w = cw - pad_l - pad_r

        metric_vals = [n.metric for n in nodes.values()]
        m_min = min(metric_vals)
        m_max = max(metric_vals)
        m_range = max(1e-6, m_max - m_min)

        # Metric axis label
        self._viz_canvas.create_text(
            pad_l + plot_w / 2, 8,
            text=f"{metric_key}",
            fill=C.SUBTEXT0,
            font=FONT_MONO_SM,
            anchor="n",
        )

        # Metric axis ticks
        for i in range(5):
            mv = m_min + (m_range / 4) * i
            x = pad_l + (i / 4) * plot_w
            self._viz_canvas.create_line(x, pad_t - 2, x, pad_t - 8, fill=C.SURFACE1)
            self._viz_canvas.create_text(
                x, pad_t - 10, text=f"{mv:.2f}",
                fill=C.OVERLAY0, font=("Ubuntu Sans Mono", 7), anchor="s",
            )

        # Precompute positions
        positions: dict[str, tuple[float, float]] = {}
        island_nodes: dict[int, list[VizNode]] = {i: [] for i in islands}
        for n in nodes.values():
            island_nodes.setdefault(n.island, []).append(n)

        for idx, island in enumerate(islands):
            band_top = pad_t + idx * (band_h + gap)
            band_bot = band_top + band_h
            group = island_nodes.get(island, [])
            if not group:
                continue
            g_min = min(n.generation for n in group)
            g_max = max(n.generation for n in group)
            g_range = max(1, g_max - g_min)

            # Band separator + label
            self._viz_canvas.create_line(pad_l, band_top, cw - pad_r, band_top, fill=C.SURFACE1)
            self._viz_canvas.create_text(
                8, band_top + 4, text=f"Island {island}",
                fill=C.SUBTEXT0, font=FONT_MONO_SM, anchor="nw",
            )

            # Generation ticks
            tick_count = 4
            for ti in range(tick_count):
                gv = g_min + (g_range / (tick_count - 1 or 1)) * ti
                y = band_top + ((gv - g_min) / max(1, g_range)) * (band_bot - band_top)
                self._viz_canvas.create_line(pad_l - 4, y, pad_l, y, fill=C.SURFACE1)
                if ti in (0, tick_count - 1):
                    self._viz_canvas.create_text(
                        pad_l - 6, y, text=str(int(gv)),
                        fill=C.OVERLAY0, font=("Ubuntu Sans Mono", 7), anchor="e",
                    )

            for n in group:
                x = pad_l + ((n.metric - m_min) / m_range) * plot_w
                y = band_top + ((n.generation - g_min) / g_range) * (band_bot - band_top)
                positions[n.prog_id] = (x, y)

        # Draw edges first
        for parent_id, child_id in edges:
            p = positions.get(parent_id)
            c = positions.get(child_id)
            if not p or not c:
                continue
            self._viz_canvas.create_line(p[0], p[1], c[0], c[1], fill=C.SURFACE1, width=1)

        # Draw nodes
        best_by_island: dict[int, float] = {}
        for n in nodes.values():
            best_by_island[n.island] = max(best_by_island.get(n.island, -1e9), n.metric)

        for n in nodes.values():
            pos = positions.get(n.prog_id)
            if not pos:
                continue
            norm = (n.metric - m_min) / m_range if m_range > 0 else 0.5
            r = 3 + int(4 * max(0.0, min(1.0, norm)))
            color = self._island_colors[n.island % len(self._island_colors)]
            outline = C.TEXT if abs(n.metric - best_by_island.get(n.island, n.metric)) < 1e-9 else ""
            self._viz_canvas.create_oval(
                pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r,
                fill=color, outline=outline, width=1,
            )

        # Selected node ring
        if self._viz_selected_id and self._viz_selected_id in positions:
            x, y = positions[self._viz_selected_id]
            self._viz_canvas.create_oval(
                x - 9, y - 9, x + 9, y + 9,
                outline=C.YELLOW, width=2,
            )

        self._viz_positions = positions

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

        # Determine fitness key from config
        fitness_key = "combined_score"
        cfg = self._load_config()
        if isinstance(cfg, dict):
            ec = cfg.get("EVOLVE_CONFIG", {})
            if isinstance(ec, dict):
                fitness_key = ec.get("fitness_key", fitness_key)

        for run_path in sorted(runs, key=_key):
            name = run_path.name
            best = _best_fitness_for_run(run_path, fitness_key)
            if best is not None:
                label = f"{name}  ({best:.4f})"
            else:
                label = name
            self.runs_list.insert("end", label)

    def _on_problem_change(self) -> None:
        self._sync_capabilities()
        self._refresh_runs()
        self._refresh_configs()
        self._refresh_models_from_config()
        self._refresh_config_editor()
        self._refresh_models_tab()
        self._viz_last_ckpt = None
        self._refresh_visualizer()
        self._refresh_run_snapshot()

    def _on_run_select(self) -> None:
        sel = self.runs_list.curselection()
        if not sel:
            return
        raw = self.runs_list.get(sel[0])
        # Strip fitness annotation: "run3  (0.9540)" -> "run3"
        run_name = raw.split("(")[0].strip() if "(" in raw else raw.strip()
        self.run_id_var.set(run_name)
        self._refresh_run_snapshot()
        self._viz_last_ckpt = None
        self._refresh_visualizer()

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
        toolbar = ttk.Frame(self._cfg_tab, padding=(8, 6))
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

        # Mouse wheel scrolling (Linux: Button-4/5)
        def _on_mousewheel(event):
            self._cfg_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        for w in (self._cfg_canvas, self._cfg_scroll_frame):
            w.bind("<Button-4>", _on_mousewheel)
            w.bind("<Button-5>", _on_mousewheel)

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
        toolbar = ttk.Frame(self._models_tab, padding=(8, 6))
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

        # Mouse wheel scrolling (Linux: Button-4/5)
        def _on_mousewheel(event):
            self._mtab_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        for w in (self._mtab_canvas, self._mtab_scroll_frame):
            w.bind("<Button-4>", _on_mousewheel)
            w.bind("<Button-5>", _on_mousewheel)

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
                    elif field in ("temp", "top_p", "weight"):
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

            self._append_log(f"$ {shlex.join(cmd)}\n")
            self._run_start_time = time.monotonic()

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

            t = threading.Thread(target=self._reader_thread, args=(self._proc,), daemon=True)
            t.start()

    def _reader_thread(self, proc: subprocess.Popen[str]) -> None:
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
            # Auto-refresh runs after process exits
            self.after(500, self._refresh_runs)

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

    def _on_close(self) -> None:
        if self._viz_after:
            try:
                self.after_cancel(self._viz_after)
            except Exception:
                pass
            self._viz_after = None
        with self._proc_lock:
            running = self._proc is not None
        if running:
            if not messagebox.askokcancel("Quit", "A command is still running. Stop and exit?"):
                return
            self._closing = True
            self._close_kill_at = time.monotonic() + 3.0
            self._stop_proc()
            self.after(200, self._check_force_close)
            return
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
        self._spawn(cmd)

    def _cmd_run_next(self) -> None:
        p = self._selected_problem()
        if not p:
            return
        if not self._ensure_api():
            return
        cmd = ["bash", str(p.script_path), "run", "--next"]
        self._add_common_run_flags(p, cmd)
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
