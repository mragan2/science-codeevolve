# C. elegans Connectome Locomotion (elegans)

## Overview
This problem evolves a connectome-driven C. elegans locomotion simulator. The candidate code must
load the real connectome from the provided Excel workbook and generate kinematics and neural
activity that match target statistics from:
- Dryad 2024 curvature datasets (optional if the zip is missing)
- WormWideWeb whole-brain activity and behavior

The evaluator enforces real connectome loading and checks key statistics derived from the
workbook, so hardcoding or bypassing file I/O will fail.

## Key Requirements (Enforced by Evaluator)
- Load the Excel workbook via `CE_CONNECTOME_PATH` (preferred) or local fallbacks.
- Parse the SI5 sheet layout correctly:
  - Column labels at row 3, starting column 4 (1-indexed)
  - Row labels at column 3, starting row 4 (1-indexed)
  - Matrix data at row 4, column 4 (1-indexed)
- Use sheet names:
  - `hermaphrodite chemical`
  - `hermaphrodite gap jn symmetric`
- Extract neuron-neuron submatrices via intersection of row/column labels.
- All randomness must be seeded from `CE_SEED`.
- Output exactly one JSON line with required keys (`positions`, `velocities`, `curvature`,
  `neural`, `dt`, `n_neurons`, `chem_sum`, `chem_nnz`, `gap_sum`, `gap_nnz`, `source_xlsx`).

## Directory Structure
```
problems/elegans/
├── configs/                  # CodeEvolve configuration
├── input/                    # Evaluation script, connectome, and datasets
├── results/                  # Winner snapshots + logs
├── raw/                      # Raw connectome source files (ODS + extracted)
├── elegans.sh                # Unified helper commands
├── run.sh                    # Wrapper for `elegans.sh run`
├── run1.sh                   # Back-compat wrapper for `elegans.sh run`
├── find_winner.py            # Evaluate and copy best solution
└── visualize_elegans.py      # Visualization utility
```

## Quick Commands
```bash
# Required for any run
export API_KEY="..."
export API_BASE="http://localhost:11434/v1"

# Start a new run (auto-picks next runN)
./problems/elegans/elegans.sh run

# Warm-start from a previous run/island (default island = 0)
./problems/elegans/elegans.sh warmstart run3 0

# Analyze all runs
./problems/elegans/elegans.sh analyze

# Visualize a run
./problems/elegans/elegans.sh viz run3 0

# Watch live progress
./problems/elegans/elegans.sh tail run4 0
```

## Running CodeEvolve (Manual)
```bash
codeevolve \
  --inpt_dir=problems/elegans/input \
  --cfg_path=problems/elegans/configs/config.yaml \
  --out_dir=experiments/elegans/run1 \
  --terminal_logging
```

## Finding the Best Solution
```bash
cd problems/elegans
python find_winner.py
```
This will:
- Evaluate all `experiments/elegans/*/*/best_sol.py` candidates
- Rank by `fitness`
- Copy the best to `problems/elegans/results/FINAL_BEST_SOL.py`
- Save output to `problems/elegans/results/WINNER_RUN_OUTPUT.txt`

Optional: warm-start from the winner and update SYS_MSG:
```bash
python find_winner.py --apply
```

## Neuropeptide Atlases (Optional)
You can enable neuropeptide expression + signaling priors via CeNGEN and a curated
neuropeptide connectome. Download and preprocess them once:

```bash
python problems/elegans/data/download_atlases.py
```

Then export the atlas path when running CodeEvolve or direct simulations:

```bash
export CE_ATLAS_DIR="problems/elegans/data"
```

If the atlas files are present, the simulator will use them as slow modulatory
signals without changing required evaluation outputs.

## Data Notes
- WormWideWeb targets are cached automatically in `input/data/` after first evaluation.
- Dryad curvature targets require placing `dryad.zip` at:
  `problems/elegans/input/data/dryad2024/dryad.zip`
  If missing, set `CE_SKIP_DRYAD=1` (run2.sh/run3.sh already do this).
