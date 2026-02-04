#!/usr/bin/env python3
"""
Analyze CodeEvolve evolution progress and diagnose issues.
"""
import json
import re
from pathlib import Path
from collections import Counter

def analyze_log(log_path):
    """Extract key metrics from evolution log."""
    with open(log_path) as f:
        log = f.read()
    
    # Count fitness outcomes
    fitness_matches = re.findall(r'fitness=([\d\.\-inf]+)', log)
    fitnesses = []
    errors = 0
    for f in fitness_matches:
        if 'inf' in f:
            errors += 1
        else:
            try:
                fitnesses.append(float(f))
            except:
                pass
    
    # Find best fitness
    if fitnesses:
        best = max(fitnesses)
        recent_10 = fitnesses[-10:] if len(fitnesses) >= 10 else fitnesses
        recent_avg = sum(recent_10) / len(recent_10)
    else:
        best = 0.0
        recent_avg = 0.0
    
    # Count epochs
    epochs = len(re.findall(r'EPOCH (\d+)', log))
    
    # Early stopping
    stopping_match = re.findall(r'Early stopping counter.*?: (\d+)/(\d+)', log)
    if stopping_match:
        current_stop, max_stop = map(int, stopping_match[-1])
    else:
        current_stop, max_stop = 0, 50
    
    # Migration events
    migrations = len(re.findall(r'Migration finished', log))
    
    # Error types
    error_types = Counter()
    error_matches = re.findall(r"'error': '([^']+)", log)
    for err in error_matches[:20]:  # Sample first 20 errors
        if 'SyntaxError' in err:
            error_types['SyntaxError'] += 1
        elif 'ValueError' in err:
            error_types['ValueError'] += 1
        elif 'KeyError' in err:
            error_types['KeyError'] += 1
        elif 'IndexError' in err:
            error_types['IndexError'] += 1
        elif 'NameError' in err:
            error_types['NameError'] += 1
        else:
            error_types['Other'] += 1
    
    return {
        'total_evaluations': len(fitness_matches),
        'successful': len(fitnesses),
        'failed': errors,
        'success_rate': len(fitnesses) / max(len(fitness_matches), 1),
        'best_fitness': best,
        'recent_avg_fitness': recent_avg,
        'epochs_completed': epochs,
        'early_stop_progress': f"{current_stop}/{max_stop}",
        'migrations': migrations,
        'error_types': dict(error_types),
    }

def analyze_best_solution(sol_path):
    """Analyze best solution structure."""
    with open(sol_path) as f:
        code = f.read()
    
    # Count lines
    lines = code.split('\n')
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    
    # Check for key components
    components = {
        'Spiking neurons': 'V_thresh' in code,
        'Rate model': 'alpha * a' in code,
        'Gap junctions': 'Laplacian' in code or '- L @' in code,
        'Neuropeptides': 'pep_' in code,
        'Spectral norm': 'spectral' in code.lower(),
        'Adaptive gain': 'gain' in code and 'tau_gain' in code,
        'Body segments': 'segment' in code.lower(),
        'Proprioception': 'feedback' in code.lower(),
    }
    
    # Count parameters
    param_pattern = r'^\s*(\w+)\s*=\s*([\d\.]+)'
    params = re.findall(param_pattern, code, re.MULTILINE)
    
    return {
        'total_lines': len(lines),
        'code_lines': len(code_lines),
        'components': components,
        'parameter_count': len(params),
        'complexity': 'High' if len(code_lines) > 300 else 'Medium' if len(code_lines) > 150 else 'Low',
    }

def load_metrics(results_log):
    """Extract final metrics if available."""
    # Try to find last successful evaluation metrics
    pattern = r"eval_metrics=\{'fitness': ([\d\.]+), 'total_loss': ([\d\.]+), 'behavior_loss': ([\d\.]+), 'neural_loss': ([\d\.]+)"
    matches = re.findall(pattern, results_log)
    
    if matches:
        last = matches[-1]
        return {
            'fitness': float(last[0]),
            'total_loss': float(last[1]),
            'behavior_loss': float(last[2]),
            'neural_loss': float(last[3]),
        }
    return None

def main():
    log_file = Path('0_results.log')
    sol_file = Path('0_best_sol.py')
    
    if not log_file.exists():
        print("❌ Log file not found: 0_results.log")
        return
    
    print("=" * 70)
    print("C. ELEGANS SIMULATOR - EVOLUTION ANALYSIS")
    print("=" * 70)
    print()
    
    # Analyze log
    print("📊 EVOLUTION PROGRESS")
    print("-" * 70)
    
    with open(log_file) as f:
        log_content = f.read()
    
    stats = analyze_log(log_file)
    
    print(f"Epochs completed:      {stats['epochs_completed']}")
    print(f"Total evaluations:     {stats['total_evaluations']}")
    print(f"Successful:            {stats['successful']} ({stats['success_rate']:.1%})")
    print(f"Failed (errors):       {stats['failed']} ({1-stats['success_rate']:.1%})")
    print(f"Best fitness:          {stats['best_fitness']:.6f}")
    print(f"Recent avg (last 10):  {stats['recent_avg_fitness']:.6f}")
    print(f"Early stopping:        {stats['early_stop_progress']}")
    print(f"Migrations:            {stats['migrations']}")
    print()
    
    if stats['error_types']:
        print("🔧 ERROR BREAKDOWN")
        print("-" * 70)
        for err_type, count in sorted(stats['error_types'].items(), key=lambda x: -x[1]):
            print(f"  {err_type:20s}: {count}")
        print()
    
    # Get metrics
    metrics = load_metrics(log_content)
    if metrics:
        print("📈 BEST SOLUTION METRICS")
        print("-" * 70)
        print(f"Fitness:          {metrics['fitness']:.6f}")
        print(f"Total loss:       {metrics['total_loss']:.6f}")
        print(f"  Behavior loss:  {metrics['behavior_loss']:.6f} ({metrics['behavior_loss']/metrics['total_loss']*100:.1f}%)")
        print(f"  Neural loss:    {metrics['neural_loss']:.6f} ({metrics['neural_loss']/metrics['total_loss']*100:.1f}%)")
        print()
        
        # Loss breakdown detail
        behavior_pct = metrics['behavior_loss'] / metrics['total_loss']
        neural_pct = metrics['neural_loss'] / metrics['total_loss']
        
        print("🎯 OPTIMIZATION PRIORITIES")
        print("-" * 70)
        if behavior_pct > neural_pct:
            print(f"⚠️  FOCUS ON: Behavior (especially curvature statistics)")
            print(f"    - Curvature mean/std/PSD account for 80% of behavior loss")
            print(f"    - Try: adjust phase_offsets, curv_amp, wave generation")
        else:
            print(f"⚠️  FOCUS ON: Neural dynamics")
            print(f"    - Neural correlation structure is key (40% weight)")
            print(f"    - Try: adjust synaptic time constants, gap junction strength")
        print()
    
    # Analyze solution
    if sol_file.exists():
        print("🧬 BEST SOLUTION STRUCTURE")
        print("-" * 70)
        
        sol_stats = analyze_best_solution(sol_file)
        print(f"Code lines:       {sol_stats['code_lines']}")
        print(f"Complexity:       {sol_stats['complexity']}")
        print(f"Parameters:       {sol_stats['parameter_count']}")
        print()
        
        print("Components:")
        for comp, present in sol_stats['components'].items():
            status = "✓" if present else "✗"
            print(f"  {status} {comp}")
        print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 70)
    
    if stats['success_rate'] < 0.5:
        print("⚠️  High error rate! Consider:")
        print("   - Use config_improved.yaml (lower LLM temperatures)")
        print("   - Restart with init_program_simple.py (simpler baseline)")
        print()
    
    if stats['best_fitness'] < 0.35:
        print("⚠️  Low fitness. Suggestions:")
        print("   - Current model may be too far from target")
        print("   - Consider simpler starting point")
        print()
    elif stats['best_fitness'] < 0.45:
        print("✓ Reasonable progress. Next steps:")
        print("   - Focus on curvature generation (highest weight)")
        print("   - Tune phase_offsets, wave parameters")
        print("   - Add proprioceptive feedback")
        print()
    else:
        print("✓ Good fitness! Continue with:")
        print("   - Fine-tune neural correlation structure")
        print("   - Optimize velocity dynamics")
        print()
    
    if stats['epochs_completed'] > 30 and stats['recent_avg_fitness'] < stats['best_fitness'] * 0.95:
        print("⚠️  Evolution has plateaued. Consider:")
        print("   - Increase exploration_rate temporarily")
        print("   - Add focused meta-prompting hints")
        print("   - Check if error rate increased (sign of instability)")
        print()
    
    print("=" * 70)

if __name__ == '__main__':
    main()
