# AlphaEvolve Math Problems

## Overview

This directory contains a collection of mathematical optimization problems previously used to benchmark AlphaEvolve and similar evolutionary algorithm systems.

## Problem Categories

### 1. Autocorrelation Problems

Optimize sequences with specific autocorrelation properties.

**Sub-problems:**
- `first_autocorr_ineq/` - First autocorrelation inequality
- `second_autocorr_ineq/` - Second autocorrelation inequality
- `third_autocorr_ineq/` - Third autocorrelation inequality

### 2. Heilbronn Problems

Geometric optimization problems related to point configurations.

**Sub-problems:**
- `heilbronn_convex/` - Heilbronn problem for convex polygons (N=13, 14)
- `heilbronn_triangle/` - Triangle area optimization

### 3. Kissing Number

Optimize sphere packing configurations.

**Sub-problem:**
- `kissing_number/` - Maximum number of non-overlapping unit spheres that can touch a central unit sphere

### 4. Minimizing Max-Min Distance

Optimize point distributions to minimize the maximum of minimum distances.

**Sub-problems:**
- `2/` - 2-dimensional case
- `3/` - 3-dimensional case

### 5. Packing Problems

Various geometric packing optimization challenges.

**Sub-problems:**
- `circle_packing_square/` - Pack circles into a square (N=26, 32)
- `circle_packing_rect/` - Pack circles into a rectangle
- `hexagon_packing/` - Pack hexagons (N=11, 12)

## Directory Structure

```
alphaevolve_math_problems/
├── README.md                    # This file
├── find_winner.py               # Template for finding best solutions
├── autocorrelation_problems/
│   ├── first_autocorr_ineq/
│   ├── second_autocorr_ineq/
│   └── third_autocorr_ineq/
├── heilbronn_problems/
│   ├── heilbronn_convex/
│   │   ├── 13/
│   │   └── 14/
│   └── heilbronn_triangle/
├── kissing_number/
├── minimizing_max_min_dist/
│   ├── 2/
│   └── 3/
└── packing_problems/
    ├── circle_packing_square/
    │   ├── 26/
    │   └── 32/
    ├── circle_packing_rect/
    └── hexagon_packing/
        ├── 11/
        └── 12/
```

## Running Problems

Since this is a collection of multiple problems, each sub-problem needs to be run independently:

### Example: Running an Autocorrelation Problem

```bash
codeevolve \
  --inpt_dir=problems/alphaevolve_math_problems/autocorrelation_problems/first_autocorr_ineq/input \
  --cfg_path=problems/alphaevolve_math_problems/autocorrelation_problems/first_autocorr_ineq/configs/config.yaml \
  --out_dir=experiments/alphaevolve_math_problems/first_autocorr_ineq/MR_$(date +%Y%m%d) \
  --terminal_logging
```

### Example: Running a Packing Problem

```bash
codeevolve \
  --inpt_dir=problems/alphaevolve_math_problems/packing_problems/circle_packing_square/26/input \
  --cfg_path=problems/alphaevolve_math_problems/packing_problems/circle_packing_square/26/configs/config.yaml \
  --out_dir=experiments/alphaevolve_math_problems/circle_packing_square_26/MR_$(date +%Y%m%d) \
  --terminal_logging
```

## Finding Best Solutions

The `find_winner.py` script in this directory is a **template** that needs customization for each sub-problem. To find the best solution for a specific sub-problem:

1. Create a copy of the template for your sub-problem
2. Set `EVALUATOR_SCRIPT` to the sub-problem's `evaluate.py`
3. Set `SEARCH_DIRS` to the appropriate experiment directory
4. Run the customized script

## References

These problems are based on benchmarks used in:
- AlphaEvolve (Google DeepMind)
- Various mathematical optimization literature

Each sub-problem directory may contain additional documentation specific to that problem.
