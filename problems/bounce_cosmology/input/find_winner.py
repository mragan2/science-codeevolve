import os
import glob
import subprocess
import json
import shutil
import sys

# --- CONFIGURATION (Hardcoded for your specific paths) ---
BASE_DIR = "/home/rag/Projects/science-codeevolve"
EVALUATOR_PATH = os.path.join(BASE_DIR, "problems/bounce_cosmology/input/evaluate.py")

# Look in both run_mr1 and run_mr2
SEARCH_DIRS = [
    os.path.join(BASE_DIR, "experiments/bounce_cosmology/run_mr1"),
    os.path.join(BASE_DIR, "experiments/bounce_cosmology/run_mr2")
]
# --------------------------------------------------

def find_and_rank():
    # 1. Verify Evaluator Exists
    if not os.path.exists(EVALUATOR_PATH):
        print(f"❌ CRITICAL ERROR: Evaluator not found at:\n   {EVALUATOR_PATH}")
        return

    # 2. Find all best_sol.py files
    candidate_files = []
    for run_dir in SEARCH_DIRS:
        # We look recursively for any 'best_sol.py' inside the experiment folders
        pattern = os.path.join(run_dir, "**", "best_sol.py")
        found = glob.glob(pattern, recursive=True)
        candidate_files.extend(found)

    if not candidate_files:
        print("❌ No 'best_sol.py' files found!")
        print(f"   Checked inside: {SEARCH_DIRS}")
        return

    print(f"🔎 Found {len(candidate_files)} candidates. Evaluating now...\n")
    print(f"{'SOURCE':<50} | {'SCORE':<10} | {'STATUS'}")
    print("-" * 75)

    results = []

    for file_path in candidate_files:
        # Create a readable label (e.g. "run_mr1/0/best_sol.py")
        display_name = file_path.replace(BASE_DIR + "/", "")[-50:]
        
        # Temp file for JSON output
        temp_json = f"temp_result_{os.path.basename(os.path.dirname(file_path))}.json"
        
        try:
            # 3. Run Evaluate.py using the full path
            cmd = [sys.executable, EVALUATOR_PATH, file_path, temp_json]
            
            # Run silently
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 4. Read the Score
            with open(temp_json, "r") as f:
                metrics = json.load(f)
            
            score = metrics.get("combined_score", 0.0)
            results.append((score, file_path))
            
            print(f"{display_name:<50} | {score:.6f}   | ✅ OK")

        except Exception as e:
            print(f"{display_name:<50} | {'ERROR':<10} | ❌ {str(e)[:15]}")
        finally:
            if os.path.exists(temp_json):
                os.remove(temp_json)

    # 5. Declare Winner
    print("-" * 75)
    if results:
        # Sort descending by score
        results.sort(key=lambda x: x[0], reverse=True)
        
        winner_score, winner_path = results[0]
        print(f"\n🏆 WINNER FOUND!")
        print(f"   Score: {winner_score:.8f}")
        print(f"   File : {winner_path}")
        
        # Save it to your current folder
        dest = "FINAL_BEST_SOL.py"
        shutil.copy(winner_path, dest)
        print(f"   💾 SAVED TO: {os.path.abspath(dest)}")
    else:
        print("No valid scores found.")

if __name__ == "__main__":
    find_and_rank()
