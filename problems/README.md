# CodeEvolve Problems Directory - Complete Guide

This comprehensive guide covers everything you need to know about creating, configuring, and running CodeEvolve experiments.

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Quick Start](#quick-start)
3. [Configuration File Reference](#configuration-file-reference)
4. [Creating Your Own Problem](#creating-your-own-problem)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

---

## Directory Structure

Each project follows a standardized structure:

```
problems/
├── PROJECT_NAME/                    # Your project name (e.g., "F_time")
│   ├── run.sh                      # ⭐ Project-specific run script (RECOMMENDED LOCATION)
│   ├── input/
│   │   ├── evaluate.py             # Evaluation script (required)
│   │   └── src/
│   │       └── init_program.py     # Initial program to evolve (required)
│   └── configs/
│       ├── config.yaml             # Standard configuration
│       ├── config_mp_insp.yaml     # Meta-prompting + Inspiration (recommended)
│       ├── config_insp.yaml        # Inspiration-based crossover only
│       ├── config_mp.yaml          # Meta-prompting only
│       ├── config_no_mp_or_insp.yaml # Basic evolution
│       └── config_no_evolve.yaml   # Baseline (no evolution)
└── run_template.sh                 # Template to copy (don't edit this directly)
```

### Required Files

1. **`input/src/init_program.py`** - Initial solution to evolve
2. **`input/evaluate.py`** - Fitness evaluation script
3. **`configs/config.yaml`** - Configuration parameters
4. **`run.sh`** (recommended) - Project-specific run script copied from template

---

## Quick Start

### Method 1: Using the Template Script (Recommended)

**Best Practice: Place `run.sh` in each project folder for self-contained, portable projects.**

```bash
# 1. Copy the template to your project folder
cp problems/run_template.sh problems/YOUR_PROJECT/run.sh

# 2. Edit run.sh and set PROJECT_NAME
cd problems/YOUR_PROJECT
nano run.sh
# Change: PROJECT_NAME="YOUR_PROJECT"

# 3. Run from the project folder
bash run.sh

# Or run from anywhere:
bash problems/YOUR_PROJECT/run.sh
```

**Why in project folder?**
- ✅ Self-contained: Everything for the project is in one place
- ✅ Portable: Easy to share or move projects
- ✅ Project-specific: Each project can have custom settings
- ✅ Parallel runs: Run multiple projects simultaneously
- ✅ Simple: Just `cd` to project and run `bash run.sh`

### API Key Configuration

The run script supports multiple ways to configure API keys:

**Option 1: Set in run.sh (Quick but less secure)**
```bash
# Edit your run.sh file
API_KEY="your-api-key-here"
API_BASE="https://api.openai.com/v1"
```
⚠️ **Warning**: Don't commit API keys to git! Add run.sh to .gitignore if it contains keys.

**Option 2: Environment Variables (Recommended for development)**
```bash
export API_KEY="your-api-key-here"
export API_BASE="https://api.openai.com/v1"
bash problems/YOUR_PROJECT/run.sh
```

**Option 3: External File (Most Secure)**
```bash
# 1. Copy the example file
cp problems/.api_keys.example problems/.api_keys

# 2. Edit with your actual keys
nano problems/.api_keys

# 3. Source it in your run.sh
# Add this line to run.sh:
source problems/.api_keys

# 4. Run normally
bash run.sh
```

The `.api_keys` file is automatically ignored by git for security.

### Method 2: Direct Command Line

```bash
codeevolve \
    --inpt_dir="problems/YOUR_PROJECT/input/" \
    --cfg_path="problems/YOUR_PROJECT/configs/config.yaml" \
    --out_dir="experiments/YOUR_PROJECT/run_001/" \
    --load_ckpt=-1 \
    --terminal_logging
```

---

## Configuration File Reference

All configuration files are in YAML format. Below is a complete reference of all parameters.

### Configuration File Structure

```yaml
# System message for the LLM
SYS_MSG: |
  # PROMPT-BLOCK-START
  Your problem description and instructions here
  # PROMPT-BLOCK-END

# File paths and names
CODEBASE_PATH: 'src/'
INIT_FILE_DATA: {filename: 'init_program.py', language: 'python'}
EVAL_FILE_NAME: 'evaluate.py'
EVAL_TIMEOUT: 180

# Resource limits
MAX_MEM_BYTES: 1000000000
MEM_CHECK_INTERVAL_S: 0.1

# Evolution configuration
EVOLVE_CONFIG: {...}

# LLM ensemble configuration
ENSEMBLE: [...]

# Auxiliary LLM for meta-prompting
SAMPLER_AUX_LM: {...}

# Embedding model (if using similarity features)
EMBEDDING: {...}

# MAP-Elites configuration (if using quality-diversity)
MAP_ELITES: {...}
```

### Top-Level Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `SYS_MSG` | string | System message for the LLM containing problem description. Must include `PROMPT-BLOCK-START` and `PROMPT-BLOCK-END` markers | See [System Message](#system-message) |
| `CODEBASE_PATH` | string | Path to source code directory relative to input directory | `'src/'` |
| `INIT_FILE_DATA` | dict | Initial program file information | `{filename: 'init_program.py', language: 'python'}` |
| `EVAL_FILE_NAME` | string | Name of the evaluation script | `'evaluate.py'` |
| `EVAL_TIMEOUT` | int | Maximum execution time in seconds for program evaluation | `180` |
| `MAX_MEM_BYTES` | int | Maximum memory usage in bytes (1GB = 1000000000) | `1000000000` |
| `MEM_CHECK_INTERVAL_S` | float | Interval for memory monitoring in seconds | `0.1` |

### EVOLVE_CONFIG Parameters

The `EVOLVE_CONFIG` section controls the evolutionary algorithm:

```yaml
EVOLVE_CONFIG:
  # Fitness and evaluation
  fitness_key: 'FITNESS_KEY'           # Key in evaluation results to use as fitness
  
  # Population management
  num_epochs: 100                      # Number of evolutionary epochs to run
  ckpt: 5                              # Save checkpoint every N epochs
  max_size: 40                         # Maximum population size per island
  init_pop: 6                          # Initial population size to generate
  
  # Evolution strategy
  exploration_rate: 0.3                # Probability of exploration vs exploitation (0.0-1.0)
  selection_policy: 'roulette'         # Parent selection method
  selection_kwargs:                    # Additional selection parameters
    roulette_by_rank: true            # Use rank-based roulette (vs fitness-based)
  
  # Termination
  early_stopping_rounds: 100           # Stop if no improvement for N epochs
  
  # Island model (distributed evolution)
  num_islands: 5                       # Number of parallel islands
  migration_topology: 'ring'           # How islands are connected
  migration_interval: 40               # Migrate solutions every N epochs
  migration_rate: 0.1                  # Fraction of population to migrate
  
  # Advanced features
  meta_prompting: true                 # Enable meta-prompting for prompt evolution
  use_embedding: false                 # Use embeddings for solution similarity
  use_map_elites: false               # Enable MAP-Elites quality-diversity algorithm
  num_inspirations: 3                  # Number of solutions to use as inspiration
  max_chat_depth: 3                    # Maximum depth of conversation history
```

#### Detailed Parameter Descriptions

**Fitness and Evaluation:**
- `fitness_key`: Must match a key returned by your `evaluate.py` script. Example: if your evaluate script returns `{"score": 0.95}`, use `fitness_key: 'score'`

**Population Management:**
- `num_epochs`: Typical range is 50-500 depending on problem complexity
- `ckpt`: Save frequency for checkpoints. Lower = more frequent saves
- `max_size`: Larger populations explore more but use more resources (20-100 typical)
- `init_pop`: Start with 5-10 diverse initial solutions

**Evolution Strategy:**
- `exploration_rate`: 0.3 = 30% exploration (meta-prompting), 70% exploitation (depth refinement)
  - Higher values (0.5-0.7): More diverse search, better for hard problems
  - Lower values (0.1-0.3): More focused refinement, better when close to optimum
  
- `selection_policy`: Choose from:
  - `'roulette'`: Probabilistic selection based on fitness/rank
  - `'tournament'`: Select best from random subsets (requires `selection_kwargs: {tournament_size: 3}`)
  - `'random'`: Uniform random selection
  - `'best'`: Always select the best (greedy)

**Island Model:**
- `num_islands`: More islands = more diverse search but higher cost (1-10 typical)
  - Single island (1): Faster, less diverse
  - Multiple islands (5-10): Slower, more diverse, better for complex problems

- `migration_topology`: How islands exchange solutions:
  - `'ring'`: Each island connects to 2 neighbors (balanced)
  - `'fully_connected'`: All islands connect to all others (maximum mixing)
  - `'star'`: Central hub with spokes (centralized)
  - `'empty'`: No migration (independent islands)

- `migration_interval`: How often to migrate (20-50 typical)
  - Too frequent: Convergence, loss of diversity
  - Too rare: Islands evolve independently

- `migration_rate`: Fraction to migrate (0.05-0.2 typical)
  - 0.1 = send top 10% of population to neighbors

**Advanced Features:**
- `meta_prompting`: 
  - `true`: LLM evolves the prompt itself for better solutions
  - `false`: Use fixed prompt throughout evolution
  - Recommended: `true` for complex problems

- `use_embedding`:
  - `true`: Use semantic embeddings to measure solution similarity
  - `false`: Use fitness only
  - Requires `EMBEDDING` configuration

- `use_map_elites`:
  - `true`: Use quality-diversity algorithm (explores behavioral space)
  - `false`: Standard evolutionary algorithm (maximizes single fitness)
  - Requires `MAP_ELITES` configuration

- `num_inspirations`: Number of high-performing solutions to show as examples (0-5 typical)
  - 0: No inspiration (pure generation from scratch)
  - 1-3: Moderate inspiration (recommended)
  - 4+: Heavy inspiration (risk of premature convergence)

- `max_chat_depth`: How many ancestor solutions to include in context (1-5 typical)
  - Higher values: More context but longer prompts
  - Lower values: Less context but faster generation

### ENSEMBLE Configuration

Define multiple LLM models with weighted selection:

```yaml
ENSEMBLE:
  - model_name: 'GOOGLE_GEMINI-2.5-FLASH'    # Model identifier
    temp: 0.7                                 # Temperature (0.0-2.0)
    top_p: 0.95                               # Nucleus sampling (0.0-1.0)
    retries: 3                                # Retry attempts on failure
    weight: 0.8                               # Probability of selecting this model
    verify_ssl: false                         # SSL certificate verification
    
  - model_name: 'GOOGLE_GEMINI-2.5-PRO'
    temp: 0.7
    top_p: 0.95
    retries: 3
    weight: 0.2                               # 20% chance vs 80% for FLASH
    verify_ssl: false
```

**Supported Model Name Formats:**
- Google Gemini: `GOOGLE_GEMINI-2.5-FLASH`, `GOOGLE_GEMINI-2.5-PRO`
- OpenAI: `OPENAI_GPT-4`, `OPENAI_GPT-4-TURBO`, `OPENAI_GPT-3.5-TURBO`
- Azure OpenAI: `AZURE_GPT-4`
- Anthropic: `ANTHROPIC_CLAUDE-3-OPUS`, `ANTHROPIC_CLAUDE-3-SONNET`

**Parameter Details:**
- `temp`: Controls randomness (0.0 = deterministic, 1.0 = balanced, 2.0 = creative)
- `top_p`: Nucleus sampling threshold (0.95 = top 95% probability mass)
- `weight`: Relative probability (weights are normalized, e.g., 0.8 and 0.2 = 80%/20% split)

### SAMPLER_AUX_LM Configuration

Auxiliary LLM for meta-prompting (evolving prompts):

```yaml
SAMPLER_AUX_LM:
  model_name: 'GOOGLE_GEMINI-2.5-FLASH'
  temp: 0.7
  top_p: 0.95
  retries: 3
  weight: 1
  verify_ssl: false
```

Only used when `meta_prompting: true` in `EVOLVE_CONFIG`.

### EMBEDDING Configuration

For computing solution similarity (semantic embeddings):

```yaml
EMBEDDING:
  model_name: 'AZURE_TEXT-EMBEDDING-3-SMALL'
  retries: 3
  verify_ssl: false
```

**Supported Embedding Models:**
- Azure: `AZURE_TEXT-EMBEDDING-3-SMALL`, `AZURE_TEXT-EMBEDDING-3-LARGE`
- OpenAI: `OPENAI_TEXT-EMBEDDING-3-SMALL`, `OPENAI_TEXT-EMBEDDING-3-LARGE`
- OpenAI (legacy): `OPENAI_TEXT-EMBEDDING-ADA-002`

Only used when `use_embedding: true` in `EVOLVE_CONFIG`.

### MAP_ELITES Configuration

Quality-diversity algorithm exploring behavioral feature space:

#### Grid-based MAP-Elites

```yaml
MAP_ELITES:
  elite_map_type: 'grid'
  features:
    - name: 'feature1'              # Feature name (must match evaluation output)
      min_val: 0.0                  # Minimum feature value
      max_val: 1.0                  # Maximum feature value
      num_bins: 10                  # Number of bins to discretize feature space
    - name: 'feature2'
      min_val: -5.0
      max_val: 5.0
      num_bins: 20
```

Creates a grid of `num_bins` × `num_bins` cells. Each cell stores the best solution with features in that range.

#### CVT-based MAP-Elites (Centroidal Voronoi Tessellation)

```yaml
MAP_ELITES:
  elite_map_type: 'cvt'
  features:
    - name: 'feature1'
      min_val: 0.0
      max_val: 1.0
    - name: 'feature2'
      min_val: -5.0
      max_val: 5.0
  elite_map_kwargs:
    num_centroids: 50               # Number of Voronoi cells
    num_init_samples: 1000          # Samples for CVT initialization
    max_iter: 300                   # Max iterations for CVT algorithm
    tolerance: 0.0001               # Convergence tolerance
```

Creates adaptive regions using Voronoi tessellation. Better for high-dimensional feature spaces.

**When to Use MAP-Elites:**
- Want diverse solutions, not just highest fitness
- Features represent meaningful behavioral characteristics
- Exploring tradeoffs between multiple objectives

Only used when `use_map_elites: true` in `EVOLVE_CONFIG`.

### System Message

The `SYS_MSG` should contain your problem description:

```yaml
SYS_MSG: |
  # PROMPT-BLOCK-START
  You are an expert Python programmer. Your task is to write efficient code
  that solves the traveling salesman problem for N cities.
  
  Requirements:
  - Implement a function 'solve_tsp(distances)' that takes a distance matrix
  - Return a tuple (tour, total_distance) where tour is a list of city indices
  - Optimize for solution quality and runtime
  - The code will be evaluated on instances with 20-100 cities
  
  Your code must be within the EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers.
  # PROMPT-BLOCK-END
```

**Best Practices:**
- Clearly state the problem and objectives
- Specify input/output format
- Mention any constraints or requirements
- Include evaluation criteria
- Keep it concise but complete

---

## Creating Your Own Problem

### Step 1: Set Up Directory Structure

```bash
# Copy the template
cp -r problems/problem_template problems/my_problem

cd problems/my_problem
```

### Step 2: Create Initial Program

Edit `input/src/init_program.py`:

```python
# EVOLVE-BLOCK-START
def solve_my_problem(input_data):
    """
    Your initial solution here.
    This is the starting point for evolution.
    """
    # Simple baseline implementation
    result = do_something_basic(input_data)
    return result
# EVOLVE-BLOCK-END
```

**Important:**
- Code must be between `EVOLVE-BLOCK-START` and `EVOLVE-BLOCK-END` markers
- Only code in this block will be evolved
- Can import standard libraries outside the block

### Step 3: Create Evaluation Script

Edit `input/evaluate.py`:

```python
import sys
import json
from importlib import __import__

def evaluate(program_path: str, results_path: str) -> None:
    """
    Evaluate the evolved program and compute fitness.
    """
    # Import the program
    module_name = os.path.splitext(os.path.basename(program_path))[0]
    program = __import__(module_name)
    
    # Run your test cases
    test_cases = load_test_cases()
    scores = []
    
    for test_input, expected_output in test_cases:
        try:
            output = program.solve_my_problem(test_input)
            score = compute_score(output, expected_output)
            scores.append(score)
        except Exception as e:
            scores.append(0.0)  # Penalize errors
    
    # Compute final fitness
    avg_score = sum(scores) / len(scores)
    
    # Save results
    results = {
        "fitness": avg_score,           # Main fitness (used by fitness_key)
        "individual_scores": scores,     # Optional: detailed breakdown
        "feature1": compute_feature1(),  # Optional: for MAP-Elites
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
```

**Key Points:**
- Must accept two arguments: `program_path` and `results_path`
- Must write JSON results to `results_path`
- JSON must include the key specified by `fitness_key` in config
- Higher fitness values should be better
- Handle exceptions gracefully (return low fitness for errors)

### Step 4: Configure Evolution

Edit `configs/config.yaml`:

```yaml
SYS_MSG: |
  # PROMPT-BLOCK-START
  <Your problem description here>
  # PROMPT-BLOCK-END

CODEBASE_PATH: 'src/'
INIT_FILE_DATA: {filename: 'init_program.py', language: 'python'}
EVAL_FILE_NAME: 'evaluate.py'
EVAL_TIMEOUT: 180

MAX_MEM_BYTES: 2000000000  # 2GB
MEM_CHECK_INTERVAL_S: 0.1

EVOLVE_CONFIG:
  fitness_key: 'fitness'  # Matches key in evaluate.py results
  num_epochs: 100
  ckpt: 5
  max_size: 40
  init_pop: 6
  exploration_rate: 0.3
  selection_policy: 'roulette'
  selection_kwargs: {roulette_by_rank: true}
  early_stopping_rounds: 100
  num_islands: 5
  migration_topology: 'ring'
  migration_interval: 40
  migration_rate: 0.1
  meta_prompting: true
  use_embedding: false
  use_map_elites: false
  num_inspirations: 3
  max_chat_depth: 3

# Add your LLM configuration
ENSEMBLE: [{model_name: 'YOUR_MODEL', temp: 0.7, top_p: 0.95, retries: 3, weight: 1, verify_ssl: false}]
SAMPLER_AUX_LM: {model_name: 'YOUR_MODEL', temp: 0.7, top_p: 0.95, retries: 3, weight: 1, verify_ssl: false}
```

### Step 5: Set Up API Keys

```bash
# Set your API keys as environment variables
export API_KEY="your-api-key-here"
export API_BASE="https://api.your-provider.com/v1"
```

### Step 6: Create Run Script

```bash
cp ../run_template.sh run.sh
nano run.sh  # Edit PROJECT_NAME to "my_problem"
```

### Step 7: Run Evolution

```bash
bash run.sh
```

---

## Advanced Features

### Using Multiple Configuration Variants

Create different configs for experiments:

- **`config_mp_insp.yaml`**: Full features (meta-prompting + inspiration)
- **`config_mp.yaml`**: Meta-prompting only (prompt evolution)
- **`config_insp.yaml`**: Inspiration only (crossover-like behavior)
- **`config_no_mp_or_insp.yaml`**: Basic evolution (depth refinement only)
- **`config_no_evolve.yaml`**: Baseline (no evolution, evaluate initial solution)

Compare performance across different evolutionary strategies.

### Resuming from Checkpoints

To resume evolution from a checkpoint:

```bash
# In run.sh, set:
LOAD_CKPT=50  # Resume from epoch 50
```

Or via command line:

```bash
codeevolve --inpt_dir=... --cfg_path=... --out_dir=... --load_ckpt=50
```

### CPU Affinity

Restrict to specific CPUs for performance isolation:

```bash
# In run.sh, set:
CPU_LIST="0-7"  # Use CPUs 0 through 7
# or
CPU_LIST="0,2,4,6"  # Use specific CPUs
```

### Quality-Diversity with MAP-Elites

For problems where you want diverse solutions exploring different behaviors:

1. Define behavioral features in your evaluate.py:
```python
results = {
    "fitness": overall_score,
    "speed": execution_time,      # Feature 1
    "memory": memory_usage,        # Feature 2
}
```

2. Enable MAP-Elites in config:
```yaml
EVOLVE_CONFIG:
  use_map_elites: true

MAP_ELITES:
  elite_map_type: 'grid'
  features:
    - {name: 'speed', min_val: 0, max_val: 10, num_bins: 10}
    - {name: 'memory', min_val: 0, max_val: 100, num_bins: 10}
```

This creates a 10×10 grid exploring the speed/memory tradeoff space.

---

## Troubleshooting

### Common Errors

**Error: "codeevolve command not found"**
```bash
pip install -e .
```

**Error: "Input directory does not exist"**
Check your directory structure matches the required format:
```bash
ls problems/YOUR_PROJECT/input/
ls problems/YOUR_PROJECT/input/src/
ls problems/YOUR_PROJECT/configs/
```

**Error: "Config file does not exist"**
```bash
# List available configs
ls problems/YOUR_PROJECT/configs/
# Use exact filename without .yaml in run.sh
```

**Error: "API key not set" or "Authentication failed"**

Three ways to fix:

1. **Environment variables:**
```bash
export API_KEY="your-key"
export API_BASE="https://api.openai.com/v1"
bash run.sh
```

2. **In run.sh file:**
```bash
# Edit run.sh and uncomment/set:
API_KEY="your-api-key-here"
API_BASE="https://api.openai.com/v1"
```

3. **External file (recommended):**
```bash
# Create .api_keys file
cp problems/.api_keys.example problems/.api_keys
nano problems/.api_keys  # Add your keys

# In run.sh, uncomment:
source problems/.api_keys
```

**Important**: Never commit API keys to version control!

**Error: "Evaluation timeout"**
Increase `EVAL_TIMEOUT` in config.yaml (seconds):
```yaml
EVAL_TIMEOUT: 300  # 5 minutes
```

**Error: "Memory exceeded"**
Increase `MAX_MEM_BYTES` in config.yaml:
```yaml
MAX_MEM_BYTES: 4000000000  # 4GB
```

### Performance Tips

1. **Start small**: Begin with `num_epochs: 20`, `max_size: 20`, `num_islands: 1` for testing
2. **Monitor progress**: Check `experiments/PROJECT/OUTPUT/logs/` for evolution progress
3. **Tune exploration**: Increase `exploration_rate` if stuck in local optima
4. **Use inspiration**: Set `num_inspirations: 3` for better solution quality
5. **Enable meta-prompting**: Set `meta_prompting: true` for complex problems

### Debug Mode

For more detailed logging:

```bash
codeevolve --inpt_dir=... --cfg_path=... --out_dir=... --terminal_logging
```

### Getting Help

- Check logs in `experiments/PROJECT/OUTPUT/logs/`
- Review the main README.md in the repository root
- See OPTIMIZATIONS.md for performance tuning
- Create an issue on GitHub for bugs or questions

---

## Configuration Examples

### Example 1: Simple Problem (TSP)

```yaml
EVOLVE_CONFIG:
  fitness_key: 'tour_length'  # Lower is better (negate in evaluate.py)
  num_epochs: 50
  max_size: 30
  exploration_rate: 0.3
  meta_prompting: true
  num_inspirations: 2
```

### Example 2: Complex Optimization

```yaml
EVOLVE_CONFIG:
  fitness_key: 'score'
  num_epochs: 200
  max_size: 50
  exploration_rate: 0.5
  num_islands: 10
  migration_interval: 20
  meta_prompting: true
  num_inspirations: 4
```

### Example 3: Quality-Diversity

```yaml
EVOLVE_CONFIG:
  fitness_key: 'performance'
  num_epochs: 150
  use_map_elites: true
  num_inspirations: 3

MAP_ELITES:
  elite_map_type: 'cvt'
  features:
    - {name: 'complexity', min_val: 0, max_val: 100}
    - {name: 'novelty', min_val: 0, max_val: 1}
  elite_map_kwargs:
    num_centroids: 100
```

---

## Output Structure

Results are saved to `experiments/PROJECT_NAME/OUTPUT_NAME/`:

```
experiments/PROJECT_NAME/run_20241212_120000/
├── checkpoints/
│   ├── epoch_5/
│   │   ├── sol_db_island_0.pkl
│   │   ├── sol_db_island_1.pkl
│   │   └── ...
│   ├── epoch_10/
│   └── ...
├── logs/
│   ├── island_0.log
│   ├── island_1.log
│   └── global.log
├── results/
│   ├── best_solutions.json
│   ├── fitness_progression.csv
│   └── final_population.json
└── config.yaml  # Copy of configuration used
```

---

## Additional Resources

- **Main README**: Project overview and installation
- **OPTIMIZATIONS.md**: Performance tuning and future improvements
- **Problem Templates**: See `problems/problem_template/` for examples
- **Research Paper**: [CodeEvolve arxiv.org/abs/2510.14150](https://arxiv.org/abs/2510.14150)

---

**Questions? Issues? Feature Requests?**

Open an issue on GitHub: https://github.com/inter-co/science-codeevolve/issues
