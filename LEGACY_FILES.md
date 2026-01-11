# Legacy Files Note

The following files in the repository root are from previous runs and are maintained for historical reference:

- `FINAL_BEST_SOL.py` - Legacy best solution (from torsion cosmology work)
- `WINNER_RUN_OUTPUT.txt` - Legacy run output
- `final_eval.json` - Legacy evaluation results
- `Untitled Document` - Script snippet for scanning islands

## Current Organization

Going forward, all best solutions should be generated in their respective problem directories:
- `problems/<problem_name>/FINAL_BEST_SOL.py`
- `problems/<problem_name>/WINNER_RUN_OUTPUT.txt`

Use the problem-specific `find_winner.py` scripts to generate these files:

```bash
cd problems/<problem_name>
python find_winner.py
```

## Experiment Results

All experimental runs should be organized under `experiments/` following the MR_DATE naming convention:
- `experiments/<problem_name>/MR_YYYYMMDD/`

Legacy experimental runs have been preserved:
- `experiments/torsion_problem_legacy_root/` - Moved from root directory

See [experiments/README.md](experiments/README.md) for details on the organizational structure.
