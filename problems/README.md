# Problems Directory

This directory contains problem definitions for CodeEvolve evolutionary optimization.

## Available Problems

### Scientific Discovery Problems

#### bounce_cosmology
Evolve bounce cosmology models to beat ΛCDM on statistical evidence using Bayesian Information Criterion (BIC).
- **Domain**: Cosmology, CMB Analysis
- **Objective**: Optimize cosmological models against Planck 2018 data
- **See**: [bounce_cosmology/README.md](bounce_cosmology/README.md)

#### gami
Genetic Algorithm for Memory Integrity - evolve robust algorithms to recover 12-bit signals from noisy inputs.
- **Domain**: Algorithm Design, Error Correction
- **Objective**: Maximize signal recovery success rate
- **See**: [gami/README.md](gami/README.md)

#### torsion_final
Evolve Einstein-Cartan torsion parameters to fit CMB spectra and predict tensor signatures.
- **Domain**: Theoretical Physics, Cosmology
- **Objective**: Fit TT, EE, TE spectra; predict B-mode signatures
- **See**: [torsion_final/README.md](torsion_final/README.md)

#### torsion_problem
Parameter optimization for torsion cosmology models fitting CMB observational data.
- **Domain**: Cosmology, Parameter Fitting
- **Objective**: Optimize torsion parameters for CMB fit
- **See**: [torsion_problem/README.md](torsion_problem/README.md)

### Mathematical Optimization Problems

#### alphaevolve_math_problems
Collection of mathematical optimization problems from AlphaEvolve benchmarks.
- **Domain**: Mathematical Optimization, Geometry
- **Includes**: Autocorrelation, Heilbronn, Kissing Number, Packing problems
- **See**: [alphaevolve_math_problems/README.md](alphaevolve_math_problems/README.md)

## Problem Structure

Each problem directory follows this structure:

```
problem_name/
├── README.md              # Problem documentation
├── find_winner.py         # Script to find best solution from experiments
├── input/                 # Problem definition files
│   ├── evaluate.py        # Evaluation function
│   └── src/              # Supporting code (optional)
├── configs/              # CodeEvolve configuration files
│   └── config.yaml
└── run.sh                # Run script (optional)
```

## Running a Problem

### 1. Execute CodeEvolve

```bash
codeevolve \
  --inpt_dir=problems/<problem_name>/input \
  --cfg_path=problems/<problem_name>/configs/config.yaml \
  --out_dir=experiments/<problem_name>/MR_$(date +%Y%m%d) \
  --terminal_logging
```

### 2. Find Best Solution

After experiments complete:

```bash
cd problems/<problem_name>
python find_winner.py
```

This will:
- Evaluate all solutions in the experiments directory
- Rank them by fitness
- Copy the winner to `FINAL_BEST_SOL.py`
- Generate `WINNER_RUN_OUTPUT.txt`

## Creating a New Problem

Use the `problem_template/` as a starting point:

1. Copy the template directory
2. Create your evaluation function in `input/evaluate.py`
3. Define initial code in `input/src/`
4. Configure evolution parameters in `configs/config.yaml`
5. Customize `find_winner.py` for your problem
6. Write a comprehensive README

### Evaluation Function Requirements

Your `evaluate.py` must:
- Accept two arguments: `<code_path>` and `<results_path>`
- Execute the candidate solution
- Extract fitness metrics
- Write results as JSON to `<results_path>`

Example minimal evaluator:

```python
import sys
import json
import subprocess

def evaluate(code_path, results_path):
    # Run the solution
    result = subprocess.run([sys.executable, code_path], 
                          capture_output=True, text=True, timeout=60)
    
    # Extract fitness (customize for your problem)
    fitness = extract_fitness(result.stdout)
    
    # Save results
    with open(results_path, 'w') as f:
        json.dump({'combined_score': fitness}, f)

if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2])
```

## Best Practices

1. **Documentation**: Write clear READMEs explaining the problem, objectives, and evaluation criteria
2. **Configuration**: Provide well-commented config files with sensible defaults
3. **Evaluation**: Make evaluation deterministic when possible (use seeds)
4. **Validation**: Include physical/logical constraints in your fitness function
5. **Organization**: Use the MR_DATE format for experiment outputs

## References

- Main documentation: [../README.md](../README.md)
- Experiments directory: [../experiments/README.md](../experiments/README.md)
- Scripts reference: [../scripts/README.md](../scripts/README.md)
