"""
Graviton Tensor Perturbation Evaluator
=======================================
Evaluates graviton equation solutions by measuring how well the
derived transfer functions S_TT, S_EE, S_TE recover standard cosmology
(S -> 1.0) across the main multipole range, with bonus for reproducing
the torsion dip structure near ell ~ 22.

Fitness = combined_score in [0, 1].

Interface: python evaluate.py <candidate.py> <results.json>
"""

import importlib.util
import sys
import json
import math
from pathlib import Path

# --------------------------------------------------------------------------
# Evaluation ranges
# --------------------------------------------------------------------------
ELL_MIN = 2
ELL_MAX = 15
FEATURE_ELLS = [18, 20, 22, 24, 26, 28, 30]


def _safe_eval(func, ell, params):
    try:
        v = func(ell, params)
        if not math.isfinite(v):
            return None
        if v <= 0:
            return None
        return float(v)
    except Exception:
        return None


def _smoothness_penalty(vals):
    """Penalize high-frequency oscillations (discrete curvature)."""
    if len(vals) < 3:
        return 0.0
    penalty = 0.0
    for i in range(1, len(vals) - 1):
        curvature = abs(vals[i + 1] - 2 * vals[i] + vals[i - 1])
        penalty += curvature
    return penalty


def evaluate(candidate_path: str) -> dict:
    """Load candidate module and evaluate its transfer functions."""

    spec = importlib.util.spec_from_file_location("candidate", candidate_path)
    program = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(program)
    except Exception as exc:
        return {"combined_score": 0.0, "error": f"import: {exc}"}

    # Required interface
    for fn in ("get_torsion_params", "S_TT", "S_EE", "S_TE"):
        if not hasattr(program, fn):
            return {"combined_score": 0.0, "error": f"missing function: {fn}"}

    params = program.get_torsion_params()

    # ------------------------------------------------------------------
    # Evaluate main region (ell = 2..15)
    # ------------------------------------------------------------------
    tt_vals, ee_vals, te_vals = [], [], []

    for ell in range(ELL_MIN, ELL_MAX + 1):
        tt = _safe_eval(program.S_TT, ell, params)
        ee = _safe_eval(program.S_EE, ell, params)
        te = _safe_eval(program.S_TE, ell, params)

        if tt is None or ee is None or te is None:
            return {"combined_score": 0.0, "error": f"invalid value at ell={ell}"}

        tt_vals.append(tt)
        ee_vals.append(ee)
        te_vals.append(te)

    # Chi-squared: deviation from unity (standard cosmology recovery)
    chi2_tt = sum((v - 1.0) ** 2 for v in tt_vals)
    chi2_ee = sum(0.6 * (v - 1.0) ** 2 for v in ee_vals)
    chi2_te = sum(0.4 * (v - 1.0) ** 2 for v in te_vals)
    chi2_total = chi2_tt + chi2_ee + chi2_te

    # ------------------------------------------------------------------
    # Feature score: dip at ell ~ 22
    # ------------------------------------------------------------------
    feature_vals = []
    for ell in FEATURE_ELLS:
        val = _safe_eval(program.S_TT, ell, params)
        if val is not None:
            feature_vals.append(val)

    dip_score = 0.0
    if len(feature_vals) >= 3:
        center = feature_vals[len(feature_vals) // 2]
        edge_avg = (feature_vals[0] + feature_vals[-1]) * 0.5
        dip_score = max(0.0, edge_avg - center)

    # ------------------------------------------------------------------
    # Smoothness penalty
    # ------------------------------------------------------------------
    smooth_pen = _smoothness_penalty(tt_vals)

    # ------------------------------------------------------------------
    # Asymptotic recovery: how close to 1.0 at high ell
    # ------------------------------------------------------------------
    asymp_dev = sum(abs(v - 1.0) for v in tt_vals[-4:]) / 4.0
    asymptotic_score = math.exp(-10.0 * asymp_dev)

    # ------------------------------------------------------------------
    # Combined score in [0, 1]
    # ------------------------------------------------------------------
    raw_score = -chi2_total + 0.5 * dip_score - 0.1 * smooth_pen
    # Transform: higher is better, mapped to [0, 1]
    combined_score = 1.0 / (1.0 + max(0.0, chi2_total))
    # Bonus for dip structure and smoothness
    combined_score = min(1.0, combined_score + 0.05 * dip_score)
    combined_score = max(0.0, combined_score)

    return {
        "combined_score": combined_score,
        "COMBINED_SCORE": combined_score,
        "chi2_tt": chi2_tt,
        "chi2_ee": chi2_ee,
        "chi2_te": chi2_te,
        "chi2_total": chi2_total,
        "raw_score": raw_score,
        "dip_score": dip_score,
        "smooth_penalty": smooth_pen,
        "asymptotic_score": asymptotic_score,
        "S_TT_2": tt_vals[0],
        "S_TT_10": tt_vals[8] if len(tt_vals) > 8 else None,
        "S_EE_2": ee_vals[0],
        "S_TE_2": te_vals[0],
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <candidate.py> <results.json>")
        return 1
    metrics = evaluate(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
