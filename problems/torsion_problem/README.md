# Torsion Problem

## Overview

This problem explores torsion cosmology parameter optimization for fitting cosmic microwave background (CMB) observational data.

## Objective

Evolve parameters for torsion cosmology models to best fit CMB temperature and polarization power spectra.

## Scientific Background

Einstein-Cartan theory extends General Relativity by including spacetime torsion, which can have observable effects on the CMB power spectrum. This problem uses evolutionary algorithms to discover optimal torsion parameters that match observational data.

## Directory Structure

```
torsion_problem/
├── README.md              # This file
├── find_winner.py         # Script to find best solution from experiments
├── input/                 # Problem definition
│   ├── evaluate.py        # Evaluation script
│   └── src/              # Source code for evaluation
├── configs/              # Configuration files for CodeEvolve runs
│   └── README.md         # Configuration documentation
└── run.sh                # Run script
```

## Running the Problem

### Execute CodeEvolve

```bash
cd problems/torsion_problem
./run.sh
```

Or manually:

```bash
codeevolve \
  --inpt_dir=problems/torsion_problem/input \
  --cfg_path=problems/torsion_problem/configs/config.yaml \
  --out_dir=experiments/torsion_problem/MR_$(date +%Y%m%d) \
  --terminal_logging
```

### Find the Best Solution

After experiments complete:

```bash
cd problems/torsion_problem
python find_winner.py
```

This will:
- Evaluate all solutions in `experiments/torsion_problem/`
- Rank them by combined_score
- Copy the winner to `FINAL_BEST_SOL.py`
- Generate `WINNER_RUN_OUTPUT.txt` with execution results

## Output Files

- `FINAL_BEST_SOL.py` - The best solution found
- `WINNER_RUN_OUTPUT.txt` - Output from running the best solution

## Evaluation Metrics

Solutions are evaluated based on:
- Fit quality to CMB data
- Physical constraint satisfaction
- Model complexity (via BIC or similar metrics)

## Related Problems

See also:
- `torsion_final/` - Extended version with additional constraints and objectives
