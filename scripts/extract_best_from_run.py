#!/usr/bin/env python3
"""
Extract best solution and best prompt from a CodeEvolve run directory.

Usage:
  python scripts/extract_best_from_run.py --run-dir /path/to/experiments/<problem>/<run>

If --run-dir is omitted, the script tries to infer it from the current working
directory by walking up until it finds an island folder containing results.log.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from codeevolve.utils.ckpt_utils import load_ckpt


def _lang_to_ext(lang: Optional[str]) -> str:
    if not lang:
        return ".py"
    l = str(lang).strip().lower()
    if l in {"python", "py"}:
        return ".py"
    if l in {"cpp", "c++"}:
        return ".cpp"
    if l in {"c"}:
        return ".c"
    if l in {"js", "javascript"}:
        return ".js"
    if l in {"ts", "typescript"}:
        return ".ts"
    return ".txt"


def _infer_run_dir(start: Path) -> Optional[Path]:
    curr = start.resolve()
    for p in [curr] + list(curr.parents):
        if (p / "results.log").exists() and (p / "ckpt").is_dir():
            # island dir -> run dir is parent
            return p.parent
        # run dir heuristic: has numeric subdirs with results.log
        island_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.isdigit()]
        if island_dirs and any((d / "results.log").exists() for d in island_dirs):
            return p
    return None


def _find_islands(run_dir: Path) -> list[Path]:
    return sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda p: int(p.name),
    )


def _find_latest_ckpt(ckpt_dir: Path) -> Optional[int]:
    if not ckpt_dir.exists():
        return None
    epochs = []
    for f in ckpt_dir.iterdir():
        m = re.match(r"ckpt_(\d+)\.pkl$", f.name)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def _parse_best_scores(results_log: Path) -> Tuple[Optional[float], Optional[float]]:
    if not results_log.exists():
        return None, None
    best_sol = None
    best_prompt = None
    with results_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Best solution: Program(id=" in line:
                m = re.search(r"combined_score[^0-9]*([0-9]+\\.[0-9]+)", line)
                if m:
                    best_sol = float(m.group(1))
            elif "Best prompt: Program(id=" in line:
                m = re.search(r"fitness=([0-9]+\\.[0-9]+)", line)
                if m:
                    best_prompt = float(m.group(1))
    return best_sol, best_prompt


def _load_best_from_ckpt(ckpt_dir: Path):
    epoch = _find_latest_ckpt(ckpt_dir)
    if epoch is None:
        return None
    prompt_db, sol_db, _state, _sched = load_ckpt(epoch, ckpt_dir)
    if sol_db is None or prompt_db is None:
        return None
    sol = sol_db.programs[sol_db.best_prog_id]
    prompt = prompt_db.programs[prompt_db.best_prog_id]
    return sol, prompt, epoch


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract best solution/prompt from a run directory.")
    parser.add_argument("--run-dir", type=str, default="", help="Path to experiments/<problem>/<run>")
    parser.add_argument("--write-islands", action="store_true", help="Write best_sol/best_prompt into island dirs if missing.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None or not run_dir.exists():
        inferred = _infer_run_dir(Path.cwd())
        if inferred is None:
            print("Could not infer run directory. Pass --run-dir explicitly.", file=sys.stderr)
            return 1
        run_dir = inferred

    islands = _find_islands(run_dir)
    if not islands:
        print(f"No island directories found in {run_dir}", file=sys.stderr)
        return 1

    best_sol_code = None
    best_sol_ext = ".py"
    best_sol_score = None
    best_prompt_code = None
    best_prompt_score = None

    for isl in islands:
        results_log = isl / "results.log"
        log_sol_score, log_prompt_score = _parse_best_scores(results_log)

        # Prefer ckpt for actual code
        ckpt_dir = isl / "ckpt"
        ckpt = _load_best_from_ckpt(ckpt_dir)
        sol_code = None
        prompt_code = None
        sol_score = log_sol_score
        prompt_score = log_prompt_score
        if ckpt:
            sol, prompt, _epoch = ckpt
            sol_code = sol.code
            prompt_code = prompt.code
            sol_score = sol.fitness
            prompt_score = prompt.fitness
            sol_ext = _lang_to_ext(sol.language)
        else:
            sol_ext = None
            # Fallback to existing best_sol/best_prompt files if present
            for cand in isl.glob("best_sol.*"):
                sol_code = cand.read_text()
                sol_ext = cand.suffix
                break
            prompt_path = isl / "best_prompt.txt"
            if prompt_path.exists():
                prompt_code = prompt_path.read_text()

        if sol_code is not None:
            if best_sol_score is None or (sol_score is not None and sol_score > best_sol_score):
                best_sol_score = sol_score
                best_sol_code = sol_code
                best_sol_ext = sol_ext or ".py"

        if prompt_code is not None:
            if best_prompt_score is None or (prompt_score is not None and prompt_score > best_prompt_score):
                best_prompt_score = prompt_score
                best_prompt_code = prompt_code

        if args.write_islands and sol_code is not None:
            out_path = isl / f"best_sol{sol_ext or '.py'}"
            if not out_path.exists():
                out_path.write_text(sol_code)
        if args.write_islands and prompt_code is not None:
            out_path = isl / "best_prompt.txt"
            if not out_path.exists():
                out_path.write_text(prompt_code)

    if best_sol_code is None and best_prompt_code is None:
        print("No best solution/prompt found. (No ckpt or best_* files present)", file=sys.stderr)
        return 2

    if best_sol_code is not None:
        best_sol_path = run_dir / f"best_sol{best_sol_ext}"
        best_sol_path.write_text(best_sol_code)
        print(f"Wrote best solution to {best_sol_path} (score={best_sol_score})")
    else:
        print("Best solution code not available (missing ckpt/best_sol).", file=sys.stderr)

    if best_prompt_code is not None:
        best_prompt_path = run_dir / "best_prompt.txt"
        best_prompt_path.write_text(best_prompt_code)
        print(f"Wrote best prompt to {best_prompt_path} (score={best_prompt_score})")
    else:
        print("Best prompt code not available (missing ckpt/best_prompt).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
