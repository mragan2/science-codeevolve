import os
import glob
import subprocess
import json
import shutil
import sys
import argparse
from pathlib import Path
import yaml

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_INIT_PATH = os.path.join(INPUT_DIR, "src", "init_program.py")
DEFAULT_CFG_PATH = os.path.join(SCRIPT_DIR, "configs", "config.yaml")

# 1. SETUP: Where to look?
SEARCH_DIRS = [
    os.path.join(SCRIPT_DIR, "../../experiments/elegans")
]
EVALUATOR_SCRIPT = os.path.join(INPUT_DIR, "evaluate.py")

WINNER_COPY_PATH = os.path.join(RESULTS_DIR, "FINAL_BEST_SOL.py")
WINNER_RUN_OUTPUT = os.path.join(RESULTS_DIR, "WINNER_RUN_OUTPUT.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Find and snapshot the best solution.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update init_program.py and SYS_MSG in config.yaml using the winner.",
    )
    parser.add_argument(
        "--apply-init",
        action="store_true",
        help="Replace input/src/init_program.py with the winning best_sol.py.",
    )
    parser.add_argument(
        "--update-sysmsg",
        action="store_true",
        help="Replace SYS_MSG in config.yaml with the winning best_prompt.txt.",
    )
    parser.add_argument(
        "--init-path",
        type=str,
        default=DEFAULT_INIT_PATH,
        help="Path to init_program.py to overwrite.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=DEFAULT_CFG_PATH,
        help="Path to config.yaml to update SYS_MSG.",
    )
    return parser.parse_args()

def find_and_rank():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Find all best_sol.py files in numbered subdirectories (0/best_sol.py, 1/best_sol.py...)
    candidate_files = []
    for run_dir in SEARCH_DIRS:
        # SEARCH_DIRS already contains absolute paths
        pattern = os.path.join(run_dir, "*", "*", "best_sol.py")
        candidate_files.extend(glob.glob(pattern))

    if not candidate_files:
        print("❌ No 'best_sol.py' files found!")
        print(f"   Checked inside: {SEARCH_DIRS}")
        return None

    print(f"🔎 Found {len(candidate_files)} candidates. Evaluating now...\n")
    print(f"{'ISLAND/PATH':<40} | {'FITNESS':<10} | {'STATUS'}")
    print("-" * 65)

    results = []

    for file_path in candidate_files:
        # Create a display name (e.g., "run_1/0")
        display_name = "/".join(file_path.split("/")[-3:-1])

        # Unique temp json per candidate folder (write inside INPUT_DIR)
        temp_json = os.path.join(
            INPUT_DIR,
            f"temp_eval_{os.path.basename(os.path.dirname(file_path))}.json",
        )

        try:
            # Run evaluation: python evaluate.py <candidate.py> <output.json>
            cmd = [sys.executable, EVALUATOR_SCRIPT, file_path, temp_json]
            env = dict(os.environ)
            if "CE_SKIP_DRYAD" not in env:
                dryad_zip = os.path.join(SCRIPT_DIR, "input", "data", "dryad2024", "dryad.zip")
                if not os.path.exists(dryad_zip):
                    env["CE_SKIP_DRYAD"] = "1"
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                cwd=INPUT_DIR,
            )

            # Read score
            with open(temp_json, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            score = metrics.get("fitness", float("-inf"))
            results.append((score, file_path))

            print(f"{display_name:<40} | {score:.6f}   | ✅ OK")

        except Exception as e:
            print(f"{display_name:<40} | {'ERROR':<10} | ❌ {str(e)[:40]}")
        finally:
            if os.path.exists(temp_json):
                os.remove(temp_json)

    print("-" * 65)
    if not results:
        print("No valid scores could be calculated.")
        return None

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)

    winner_score, winner_path = results[0]
    print(f"\n🏆 WINNER: {winner_path}")
    print(f"   SCORE : {winner_score:.8f}")

    # Copy winner to problem directory
    shutil.copy(winner_path, WINNER_COPY_PATH)
    print(f"   💾 Saved to: {WINNER_COPY_PATH}")

    # Run the winner script and save output
    print(f"\n▶ Running winner script: {winner_path}\n")
    run_cmd = [sys.executable, winner_path]

    try:
        proc = subprocess.run(
            run_cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=INPUT_DIR,
        )

        # Echo to terminal
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")

        # Save combined output to a file
        with open(WINNER_RUN_OUTPUT, "w", encoding="utf-8") as f:
            f.write("COMMAND: " + " ".join(run_cmd) + "\n")
            f.write(f"RETURN_CODE: {proc.returncode}\n\n")
            if proc.stdout:
                f.write("STDOUT:\n" + proc.stdout + "\n")
            if proc.stderr:
                f.write("STDERR:\n" + proc.stderr + "\n")

        if proc.returncode == 0:
            print(f"\n✅ Winner script finished OK. Output saved to: {WINNER_RUN_OUTPUT}")
        else:
            print(f"\n⚠️ Winner script exited with code {proc.returncode}. Output saved to: {WINNER_RUN_OUTPUT}")

    except Exception as e:
        print(f"\n❌ Failed to run winner script: {e}")

    return winner_path


def _apply_init_program(winner_path: str, init_path: str):
    if not os.path.isfile(winner_path):
        print(f"❌ Winner file not found: {winner_path}")
        return
    os.makedirs(os.path.dirname(init_path), exist_ok=True)
    shutil.copy(winner_path, init_path)
    print(f"✅ Updated init_program.py -> {init_path}")


def _apply_sys_msg(winner_path: str, cfg_path: str):
    prompt_path = os.path.join(os.path.dirname(winner_path), "best_prompt.txt")
    if not os.path.isfile(prompt_path):
        print(f"❌ best_prompt.txt not found next to winner: {prompt_path}")
        return
    if not os.path.isfile(cfg_path):
        print(f"❌ Config file not found: {cfg_path}")
        return

    prompt_text = Path(prompt_path).read_text()
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    if not isinstance(cfg, dict):
        print(f"❌ Failed to parse config: {cfg_path}")
        return

    cfg["SYS_MSG"] = prompt_text

    class _LiteralDumper(yaml.SafeDumper):
        pass

    def _str_representer(dumper, data):
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    _LiteralDumper.add_representer(str, _str_representer)
    Path(cfg_path).write_text(yaml.dump(cfg, sort_keys=False, Dumper=_LiteralDumper))
    print(f"✅ Updated SYS_MSG in config -> {cfg_path}")

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(EVALUATOR_SCRIPT):
        print(f"❌ Error: Could not find '{EVALUATOR_SCRIPT}'")
        sys.exit(1)

    winner_path = find_and_rank()
    if not winner_path:
        sys.exit(1)

    apply_init = args.apply or args.apply_init
    update_sys = args.apply or args.update_sysmsg

    if apply_init:
        _apply_init_program(winner_path, args.init_path)
    if update_sys:
        _apply_sys_msg(winner_path, args.config_path)
