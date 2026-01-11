# GAMI (Genetic Algorithm for Memory Integrity) Problem

## Overview

This problem focuses on evolving algorithms to solve the 12-bit Origami Signal Task with noise tolerance.

## Objective

Develop robust algorithms that can recover a target signal from noisy inputs through evolutionary optimization.

### Problem Description

- **Target Signal**: 12-bit pattern `0b111000111000`
- **Noise Model**: Random bit flips (0-2 bits per test)
- **Goal**: Maximize success rate across 100 random test cases

### Evaluation Metrics

- **Success Rate**: Percentage of correctly recovered signals
- **Robustness**: Performance under varying noise levels
- **Combined Score**: Overall fitness metric

## Directory Structure

```
gami/
├── README.md              # This file
├── find_winner.py         # Script to find best solution from experiments
├── input/                 # Problem definition
│   ├── evaluate.py        # Evaluation script
│   └── src/              # Source code for evaluation
└── configs/              # Configuration files for CodeEvolve runs
```

## Running the Problem

### Execute CodeEvolve

```bash
codeevolve \
  --inpt_dir=problems/gami/input \
  --cfg_path=problems/gami/configs/config.yaml \
  --out_dir=experiments/gami/MR_$(date +%Y%m%d) \
  --terminal_logging
```

### Find the Best Solution

After experiments complete:

```bash
cd problems/gami
python find_winner.py
```

This will:
- Evaluate all solutions in `experiments/gami/`
- Rank them by combined_score
- Copy the winner to `FINAL_BEST_SOL.py`
- Generate `WINNER_RUN_OUTPUT.txt` with execution results

## Output Files

- `FINAL_BEST_SOL.py` - The best solution found
- `WINNER_RUN_OUTPUT.txt` - Output from running the best solution

## Solution Requirements

Your evolved code must implement:
- `solve(noisy_input: int) -> int`: Function that recovers the original signal from noisy input

## Evaluation Process

1. 100 random test cases are generated
2. Each test injects 0-2 random bit flips into the target signal
3. Your `solve()` function attempts to recover the original signal
4. Success rate determines the fitness score
