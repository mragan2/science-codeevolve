#!/usr/bin/env python3
"""Analyze evolution runs"""
import re, sys, argparse
from pathlib import Path

def analyze(log_path):
    log = log_path.read_text()
    fits = [float(f) for f in re.findall(r'fitness=([\d\.]+)', log) if 'inf' not in f]
    epochs = len(re.findall(r'EPOCH \d+', log))
    return {'epochs': epochs, 'evals': len(fits), 'best': max(fits) if fits else 0, 'recent': sum(fits[-10:])/max(len(fits[-10:]),1) if fits else 0}

def find_log(d):
    for n in ['results.log', '0_results.log']:
        if (d/n).exists(): return d/n
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('path', nargs='?', default='.')
    p.add_argument('--all', '-a', action='store_true')
    args = p.parse_args()
    
    base = Path(args.path)
    if args.all:
        patterns = [
            '*/*/results.log',
            '*/results.log',
            '*/*/0_results.log',
            '*/0_results.log',
        ]
        runs = []
        for pat in patterns:
            runs.extend(base.glob(pat))
        runs = sorted({p for p in runs})
        print(f"{'Run':<50} {'Epochs':>7} {'Evals':>7} {'Best':>10} {'Recent':>10}")
        print("-"*90)
        for log in runs:
            s = analyze(log)
            print(f"{str(log.parent):<50} {s['epochs']:>7} {s['evals']:>7} {s['best']:>10.4f} {s['recent']:>10.4f}")
    else:
        log = find_log(base)
        if not log: print(f"❌ No log in {base}"); return
        s = analyze(log)
        print(f"Epochs: {s['epochs']} | Evals: {s['evals']} | Best: {s['best']:.4f} | Recent: {s['recent']:.4f}")

if __name__ == '__main__': main()
