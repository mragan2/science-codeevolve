# Scripts Directory

This directory contains template scripts and utilities for the CodeEvolve project.

## Contents

### find_winner.py

**Template script** for finding the best solution across evolutionary runs.

This is a **reference implementation**. Each problem directory has its own customized version:
- `problems/bounce_cosmology/find_winner.py`
- `problems/gami/find_winner.py`
- `problems/torsion_final/find_winner.py`
- `problems/torsion_problem/find_winner.py`
- `problems/alphaevolve_math_problems/find_winner.py` (template for sub-problems)

#### Usage

To find the best solution for a specific problem, use the problem-specific script:

```bash
cd problems/<problem_name>
python find_winner.py
```

This will:
1. Search for all `best_sol.py` files in the problem's experiments directory
2. Evaluate each candidate using the problem's `evaluate.py`
3. Rank solutions by fitness score
4. Copy the winner to `FINAL_BEST_SOL.py` in the problem directory
5. Run the winner and save output to `WINNER_RUN_OUTPUT.txt`

### FINAL_BEST_SOL.py

Example/legacy best solution file. New best solutions should be generated in their respective problem directories.

## Creating a New Problem

When creating a new problem, copy and customize the `find_winner.py` template:

1. Copy `scripts/find_winner.py` to your problem directory
2. Update `SEARCH_DIRS` to point to your experiments directory
3. Update `EVALUATOR_SCRIPT` to point to your problem's evaluator
4. Ensure paths use `os.path.join()` for cross-platform compatibility
5. Test the script with existing experiment data

## Notes

- Problem-specific scripts use relative paths and are self-contained
- They output results to the problem directory, not the repository root
- Each script is configured for its specific evaluation function and experiment structure
