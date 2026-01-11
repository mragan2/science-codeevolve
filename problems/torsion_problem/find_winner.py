import os
import glob
import subprocess
import json
import shutil
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. SETUP: Where to look?
SEARCH_DIRS = [
    os.path.join(SCRIPT_DIR, "../../experiments/torsion_problem")
]
EVALUATOR_SCRIPT = os.path.join(SCRIPT_DIR, "input/evaluate.py")

WINNER_COPY_PATH = os.path.join(SCRIPT_DIR, "FINAL_BEST_SOL.py")
WINNER_RUN_OUTPUT = os.path.join(SCRIPT_DIR, "WINNER_RUN_OUTPUT.txt")

def find_and_rank():
    # Find all best_sol.py files in numbered subdirectories (0/best_sol.py, 1/best_sol.py...)
    candidate_files = []
    for run_dir in SEARCH_DIRS:
        # SEARCH_DIRS already contains absolute paths
        pattern = os.path.join(run_dir, "*", "*", "best_sol.py")
        candidate_files.extend(glob.glob(pattern))

    if not candidate_files:
        print("❌ No 'best_sol.py' files found!")
        print(f"   Checked inside: {SEARCH_DIRS}")
        return

    print(f"🔎 Found {len(candidate_files)} candidates. Evaluating now...\n")
    print(f"{'ISLAND/PATH':<40} | {'FITNESS':<10} | {'STATUS'}")
    print("-" * 65)

    results = []

    for file_path in candidate_files:
        # Create a display name (e.g., "run_1/0")
        display_name = "/".join(file_path.split("/")[-3:-1])

        # Unique temp json per candidate folder
        temp_json = f"temp_eval_{os.path.basename(os.path.dirname(file_path))}.json"

        try:
            # Run evaluation: python evaluate.py <candidate.py> <output.json>
            cmd = [sys.executable, EVALUATOR_SCRIPT, file_path, temp_json]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Read score
            with open(temp_json, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            score = metrics.get("combined_score", 0.0)
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
        return

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
        proc = subprocess.run(run_cmd, check=False, capture_output=True, text=True)

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

if __name__ == "__main__":
    if not os.path.exists(EVALUATOR_SCRIPT):
        print(f"❌ Error: Could not find '{EVALUATOR_SCRIPT}'")
    else:
        find_and_rank()
