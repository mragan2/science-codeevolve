# Initial Run Configuration Guide

Use this guide to choose configuration values for a first run of CodeEvolve. It summarizes the available settings from the codebase and provides a conservative starter configuration.

## Required command-line and environment inputs
- CLI arguments (see `src/codeevolve/cli.py`):
  - `--inpt_dir` – path to the problem input directory containing `src/` and `evaluate.py`.
  - `--cfg_path` – path to the YAML config file (copied into the output directory at runtime).
  - `--out_dir` – directory where logs, checkpoints, and outputs are written.
  - `--load_ckpt` – checkpoint epoch to resume (0 starts fresh, -1 loads latest shared checkpoint).
  - `--terminal_logging` – stream logs from all islands in the terminal.
- Environment variables: export `API_KEY` and `API_BASE` before running; execution aborts if they are missing.

## Configuration sections and key fields
- **Top-level file paths and limits** (see `problems/README.md`):
  - `SYS_MSG`: system message block delimiting the evolution prompt.
  - `CODEBASE_PATH`: relative path to the evolving code (default `'src/'`).
  - `INIT_FILE_DATA`: `{filename, language}` for the seed program inside the codebase path.
  - `EVAL_FILE_NAME`: evaluator script name (usually `evaluate.py`).
  - `EVAL_TIMEOUT`, `MAX_MEM_BYTES`, `MEM_CHECK_INTERVAL_S`: runtime and resource safety limits.
- **EVOLVE_CONFIG** (evolution core):
  - Fitness and loop: `fitness_key`, `num_epochs`, `ckpt`, `early_stopping_rounds`.
  - Population: `max_size`, `init_pop`.
  - Search behavior: `exploration_rate`, `selection_policy`, `selection_kwargs`.
  - Islands: `num_islands`, `migration_topology` (`ring`, `fully_connected`, `star`, `empty`), `migration_interval`, `migration_rate`.
  - Advanced toggles: `meta_prompting`, `use_embedding`, `use_map_elites`, `num_inspirations`, `max_chat_depth`.
- **ENSEMBLE**: weighted list of LLMs with `model_name`, `temp`, `top_p`, `retries`, `weight`, `verify_ssl`.
- **SAMPLER_AUX_LM**: auxiliary LLM for meta-prompting (used when `meta_prompting: true`).
- **EMBEDDING**: embedding model configuration (required when `use_embedding: true`).
- **MAP_ELITES**: feature definitions and optional `elite_map_kwargs` when `use_map_elites: true`.

## Recommended starter configuration
Use this YAML as a baseline for a first run. Adjust `fitness_key` to match your evaluator output and replace model names with ones available to your account.

```yaml
SYS_MSG: |
  # PROMPT-BLOCK-START
  You are an expert developer. Improve the provided solution while following all problem constraints.
  Keep code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers.
  # PROMPT-BLOCK-END

CODEBASE_PATH: 'src/'
INIT_FILE_DATA: {filename: 'init_program.py', language: 'python'}
EVAL_FILE_NAME: 'evaluate.py'
EVAL_TIMEOUT: 180
MAX_MEM_BYTES: 1000000000
MEM_CHECK_INTERVAL_S: 0.1

EVOLVE_CONFIG:
  fitness_key: 'score'            # replace with the metric key produced by your evaluator
  num_epochs: 50                  # shorter initial run to verify setup
  ckpt: 5                         # frequent checkpoints early on
  max_size: 30
  init_pop: 6
  exploration_rate: 0.3
  selection_policy: 'roulette'
  selection_kwargs: {roulette_by_rank: true}
  early_stopping_rounds: 50
  num_islands: 2                 # lightweight parallelism for a first run
  migration_topology: 'ring'
  migration_interval: 30
  migration_rate: 0.1
  meta_prompting: true
  use_embedding: false
  use_map_elites: false
  num_inspirations: 3
  max_chat_depth: 3

ENSEMBLE:
  - model_name: 'OPENAI_GPT-4-TURBO'  # swap to a model you can access
    temp: 0.6
    top_p: 0.9
    retries: 3
    weight: 1.0
    verify_ssl: true

SAMPLER_AUX_LM:
  model_name: 'OPENAI_GPT-4-TURBO'
  temp: 0.6
  top_p: 0.9
  retries: 3
  weight: 1.0
  verify_ssl: true

# Add only if enabling embedding-based similarity
# EMBEDDING:
#   model_name: 'OPENAI_TEXT-EMBEDDING-3-SMALL'
#   retries: 3
#   verify_ssl: true

# Add only if enabling MAP-Elites
# MAP_ELITES:
#   elite_map_type: 'grid'
#   features:
#     - {name: 'feature1', min_val: 0.0, max_val: 1.0, num_bins: 10}
```

### Quick adjustments
If you want an even faster smoke test before a longer run, start from the YAML above and change only these fields:

```yaml
EVOLVE_CONFIG:
  num_epochs: 5            # finish quickly just to validate wiring
  ckpt: 1                  # checkpoint every epoch when debugging
  num_islands: 1           # single island reduces concurrency complexity
  exploration_rate: 0.5    # explore more aggressively on tiny runs
  max_chat_depth: 2        # keep conversations short while iterating

EVAL_TIMEOUT: 60           # shorten evaluator runtime limit
MAX_MEM_BYTES: 500000000   # tighter cap surfaces leaks early
```

### Rationale for starter values
- Shorter `num_epochs` with frequent checkpoints validates the pipeline before longer runs.
- Two islands in a `ring` give diversity without heavy resource load; you can scale up once stable.
- `meta_prompting` on and `num_inspirations: 3` leverage prompt evolution and inspiration examples, matching the framework defaults.
- `use_embedding` and `use_map_elites` are off initially to reduce external dependencies; enable after the core loop is stable.

### Local-friendly open models (<6B parameters)
If you want to test CodeEvolve on a single NVIDIA RTX 2070 Super (8 GB), these open models fit comfortably—especially with 4-bit or 8-bit weights—and are good starting points for coding tasks:

- **Qwen2.5-3B-Instruct** (3B) — strong generalist with robust multilingual support; handles reasoning-heavy prompts better than most small models.
- **Qwen2.5-1.5B-Instruct** (1.5B) — lightweight variant when you need maximum speed or want to pack multiple workers per GPU.
- **Gemma-2-2B** (2B) — compact Google model with efficient memory use; pair with low temperatures for deterministic evolution steps.
- **Phi-3-mini-4k-instruct** (3.8B) — excels at code and math for its size; good default for single-agent runs.
- **DeepSeek-Coder-1.3B-Instruct** (1.3B) — code-focused model that stays within strict memory limits; useful for baseline code generation.
- **SmolLM2-1.7B-Instruct** (1.7B) — very small footprint for fast iterations or multi-island experiments on one GPU.

Tips:
- Prefer 4-bit quantization (`bitsandbytes` or `GPTQ`) to leave headroom for larger batch sizes or concurrent islands.
- Keep `temp` between 0.4–0.7 and `top_p` around 0.9 for balanced exploration without excessive randomness on small models.
- Start with a single island and shorter `num_epochs` when using the smallest models, then scale islands or epochs after validating stability.
