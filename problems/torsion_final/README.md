# Torsion Final Problem

## Overview

This problem evolves Einstein-Cartan torsion parameters to optimally fit CMB (Cosmic Microwave Background) spectra while predicting tensor signatures.

## Objective

Evolve Einstein-Cartan torsion cosmology parameters to:
1. Fit TT, EE, and TE CMB spectra simultaneously
2. Predict tensor-to-scalar ratio (r)
3. Predict B-mode spectrum for CMB-S4

## Scientific Background

### Discovered Transfer Function

- **S(ℓ) = 1 - 0.804/(1+(ℓ-2))^1.455**
- Power-law index **α = 1.455 ≈ 3/2** matches Einstein-Cartan prediction

### Key Parameters to Evolve

- **κ (kappa)**: Torsion coupling strength
- **β (beta)**: Spin-torsion interaction parameter
- **r_torsion**: Tensor-to-scalar ratio from bounce
- **n_t**: Tensor spectral tilt

### Physical Constraints

From Einstein-Cartan theory:
- α = 3/2 (fixed by theory)
- S(2) ≈ 0.196 (from Planck observation)
- r < 0.06 (Planck/BICEP upper limit)
- -0.1 < n_t < 0.1 (near scale-invariant)

## Data Sources

- **Popławski (2010)**: arXiv:1007.0587
- **Planck 2018**: arXiv:1807.06209

## Directory Structure

```
torsion_final/
├── README.md              # This file
├── find_winner.py         # Script to find best solution from experiments
├── input/                 # Problem definition
│   ├── evaluate.py        # Evaluation script
│   └── src/              # Source code for evaluation
├── configs/              # Configuration files for CodeEvolve runs
└── run.sh                # Run script
```

## Running the Problem

### Execute CodeEvolve

```bash
cd problems/torsion_final
./run.sh
```

Or manually:

```bash
codeevolve \
  --inpt_dir=problems/torsion_final/input \
  --cfg_path=problems/torsion_final/configs/config.yaml \
  --out_dir=experiments/torsion_final/MR_$(date +%Y%m%d) \
  --terminal_logging
```

### Find the Best Solution

After experiments complete:

```bash
cd problems/torsion_final
python find_winner.py
```

This will:
- Evaluate all solutions in `experiments/torsion_final/`
- Rank them by combined_score
- Copy the winner to `FINAL_BEST_SOL.py`
- Generate `WINNER_RUN_OUTPUT.txt` with execution results

## Output Files

- `FINAL_BEST_SOL.py` - The best solution found
- `WINNER_RUN_OUTPUT.txt` - Output from running the best solution

## References

- Popławski, N.J., "Cosmology with torsion: An alternative to cosmic inflation" (arXiv:1007.0587)
- Planck Collaboration, "Planck 2018 results" (arXiv:1807.06209)
