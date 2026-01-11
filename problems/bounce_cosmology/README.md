# Bounce Cosmology Problem

## Overview

This problem explores bounce cosmology models that aim to beat ΛCDM (Lambda-Cold Dark Matter) on statistical evidence using Bayesian Information Criterion (BIC).

## Objective

Find cosmological models that **BEAT ΛCDM on statistical evidence (BIC)**.

### Key Metrics

- **ΔBIC = BIC(model) - BIC(ΛCDM)**
  - ΔBIC < -10: Strong evidence FOR model
  - ΔBIC < -2: Positive evidence FOR model
  - ΔBIC > +2: Evidence AGAINST model (Occam's razor penalty)

### Fitness Function

The combined score rewards:
1. Lower χ² than ΛCDM
2. Fewer parameters (simplicity bonus)
3. Physical constraints satisfied (S(2)≈0.2, S(∞)→1)

## Data Sources

- **Planck 2018 V** (arXiv:1907.12875)
- **Popławski bounce cosmology** (arXiv:1007.0587, 1410.3881)

## Directory Structure

```
bounce_cosmology/
├── README.md              # This file
├── find_winner.py         # Script to find best solution from experiments
├── input/                 # Problem definition
│   ├── evaluate.py        # Evaluation script
│   └── src/              # Source code for evaluation
├── configs/              # Configuration files for CodeEvolve runs
│   ├── config.yaml
│   └── config_v2_exploration.yaml
├── run.sh                # Run script
└── run_phase2.sh         # Phase 2 run script
```

## Running the Problem

### Execute CodeEvolve

```bash
cd problems/bounce_cosmology
./run.sh
```

Or manually:

```bash
codeevolve \
  --inpt_dir=problems/bounce_cosmology/input \
  --cfg_path=problems/bounce_cosmology/configs/config.yaml \
  --out_dir=experiments/bounce_cosmology/MR_$(date +%Y%m%d) \
  --terminal_logging
```

### Find the Best Solution

After experiments complete:

```bash
cd problems/bounce_cosmology
python find_winner.py
```

This will:
- Evaluate all solutions in `experiments/bounce_cosmology/`
- Rank them by combined_score
- Copy the winner to `FINAL_BEST_SOL.py`
- Generate `WINNER_RUN_OUTPUT.txt` with execution results

## Output Files

- `FINAL_BEST_SOL.py` - The best solution found
- `WINNER_RUN_OUTPUT.txt` - Output from running the best solution

## References

- Planck Collaboration, "Planck 2018 results. V. CMB power spectra and likelihoods" (arXiv:1907.12875)
- Popławski, N.J., "Cosmology with torsion" (arXiv:1007.0587, 1410.3881)
