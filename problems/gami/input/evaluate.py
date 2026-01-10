import sys
import os
import random
import importlib.util
import json

def _safe_import_module(py_path):
    spec = importlib.util.spec_from_file_location("candidate", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def evaluate(candidate_file):
    """
    Standard Evaluator for the 12-bit Origami Signal Task.
    """
    mod = _safe_import_module(candidate_file)
    
    target_signal = 0b111000111000
    total_tests = 100
    success_count = 0
    
    for _ in range(total_tests):
        # 1. Inject Noise: Randomly flip 0, 1, or 2 bits
        noise_mask = 0
        num_flips = random.randint(0, 2)
        
        # Create a unique mask for the bit flips
        flipped_indices = random.sample(range(12), num_flips)
        for idx in flipped_indices:
            noise_mask |= (1 << idx)
            
        noisy_input = target_signal ^ noise_mask
        
        # 2. Run the Evolved Code
        try:
            result = mod.solve(noisy_input)
            
            # 3. Check Result: We expect 1 because noise is <= 2 bits
            if int(result) == 1:
                success_count += 1
        except:
            pass # Fail silently on crashes

    # 4. Calculate Metrics
    accuracy = success_count / total_tests
    
    return {
        "combined_score": accuracy,
        "accuracy": accuracy,
        "robustness": accuracy
    }

import json

def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) != 3:
        print("Usage: python evaluate.py <candidate_program.py> <results.json>", file=sys.stderr)
        return 2
    program_path = argv[1]
    results_path = argv[2]
    metrics = evaluate(program_path)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

