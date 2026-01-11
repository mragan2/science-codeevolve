import os
import glob
import subprocess
import json
import shutil
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. SETUP: Where to look?
# Note: alphaevolve_math_problems contains multiple sub-problems
# This script will look for results in the experiments directory
SEARCH_DIRS = [
    "../../experiments/alphaevolve_math_problems"
]

# Note: Since this is a collection of problems, you may need to specify
# which sub-problem's evaluator to use, or run this per sub-problem
# For now, this is a template that needs to be customized per sub-problem run

WINNER_COPY_PATH = os.path.join(SCRIPT_DIR, "FINAL_BEST_SOL.py")
WINNER_RUN_OUTPUT = os.path.join(SCRIPT_DIR, "WINNER_RUN_OUTPUT.txt")

def find_and_rank():
    print("⚠️  alphaevolve_math_problems contains multiple sub-problems.")
    print("    Please specify which sub-problem you want to find the winner for.")
    print("")
    print("    Sub-problems include:")
    print("    - autocorrelation_problems/{first,second,third}_autocorr_ineq")
    print("    - heilbronn_problems/{heilbronn_convex,heilbronn_triangle}")
    print("    - kissing_number")
    print("    - minimizing_max_min_dist")
    print("    - packing_problems/{circle_packing_square,circle_packing_rect,hexagon_packing}")
    print("")
    print("    To use this script:")
    print("    1. Set the EVALUATOR_SCRIPT to the specific sub-problem's evaluate.py")
    print("    2. Set the SEARCH_DIRS to the specific experiment directory")
    print("    3. Or create separate find_winner.py scripts for each sub-problem")
    return

if __name__ == "__main__":
    find_and_rank()
