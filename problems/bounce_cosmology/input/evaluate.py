"""
Evidence-Based CMB Bounce Evaluator

OBJECTIVE: Find models that BEAT ΛCDM on statistical evidence (BIC)

Key metric: ΔBIC = BIC(model) - BIC(ΛCDM)
- ΔBIC < -10: Strong evidence FOR model
- ΔBIC < -2:  Positive evidence FOR model  
- ΔBIC > +2:  Evidence AGAINST model (Occam's razor penalty)

The fitness (combined_score) rewards:
1. Lower χ² than ΛCDM
2. Fewer parameters (simplicity bonus)
3. Physical constraints satisfied (S(2)≈0.2, S(∞)→1)

Sources:
- Planck 2018 V (arXiv:1907.12875)
- PopÅ‚awski bounce cosmology (arXiv:1007.0587, 1410.3881)
"""

import importlib.util
import sys
import json
import math
import ast
import re
from pathlib import Path
from typing import Dict, Any, Tuple


# =============================================================================
# PLANCK 2018 DATA
# =============================================================================

PLANCK_TT = {
    2:  {"value": 201.0,  "error_minus": 120.0, "error_plus": 810.0},
    3:  {"value": 610.0,  "error_minus": 200.0, "error_plus": 350.0},
    4:  {"value": 584.0,  "error_minus": 170.0, "error_plus": 260.0},
    5:  {"value": 1540.0, "error_minus": 200.0, "error_plus": 240.0},
    6:  {"value": 637.0,  "error_minus": 140.0, "error_plus": 220.0},
    7:  {"value": 1305.0, "error_minus": 160.0, "error_plus": 190.0},
    8:  {"value": 1284.0, "error_minus": 140.0, "error_plus": 170.0},
    9:  {"value": 974.0,  "error_minus": 120.0, "error_plus": 150.0},
    10: {"value": 767.0,  "error_minus": 100.0, "error_plus": 130.0},
    11: {"value": 1117.0, "error_minus": 100.0, "error_plus": 120.0},
    12: {"value": 1151.0, "error_minus": 95.0,  "error_plus": 115.0},
    13: {"value": 921.0,  "error_minus": 85.0,  "error_plus": 105.0},
    14: {"value": 820.0,  "error_minus": 78.0,  "error_plus": 98.0},
    15: {"value": 881.0,  "error_minus": 74.0,  "error_plus": 94.0},
    16: {"value": 1094.0, "error_minus": 72.0,  "error_plus": 92.0},
    17: {"value": 1019.0, "error_minus": 68.0,  "error_plus": 88.0},
    18: {"value": 1107.0, "error_minus": 66.0,  "error_plus": 86.0},
    19: {"value": 1125.0, "error_minus": 64.0,  "error_plus": 82.0},
    20: {"value": 1029.0, "error_minus": 60.0,  "error_plus": 78.0},
    21: {"value": 1058.0, "error_minus": 58.0,  "error_plus": 76.0},
    22: {"value": 1046.0, "error_minus": 56.0,  "error_plus": 72.0},
    23: {"value": 1048.0, "error_minus": 54.0,  "error_plus": 70.0},
    24: {"value": 895.0,  "error_minus": 50.0,  "error_plus": 66.0},
    25: {"value": 967.0,  "error_minus": 48.0,  "error_plus": 64.0},
    26: {"value": 824.0,  "error_minus": 46.0,  "error_plus": 60.0},
    27: {"value": 896.0,  "error_minus": 44.0,  "error_plus": 58.0},
    28: {"value": 813.0,  "error_minus": 42.0,  "error_plus": 54.0},
    29: {"value": 752.0,  "error_minus": 40.0,  "error_plus": 52.0},
}

PLANCK_EE = {
    2:  {"value": 0.018, "error": 0.020},
    3:  {"value": 0.028, "error": 0.018},
    4:  {"value": 0.045, "error": 0.016},
    5:  {"value": 0.058, "error": 0.015},
    6:  {"value": 0.065, "error": 0.014},
    7:  {"value": 0.072, "error": 0.013},
    8:  {"value": 0.078, "error": 0.012},
    9:  {"value": 0.082, "error": 0.011},
    10: {"value": 0.085, "error": 0.010},
    11: {"value": 0.087, "error": 0.010},
    12: {"value": 0.088, "error": 0.009},
    13: {"value": 0.089, "error": 0.009},
    14: {"value": 0.090, "error": 0.008},
    15: {"value": 0.090, "error": 0.008},
    16: {"value": 0.091, "error": 0.008},
    17: {"value": 0.091, "error": 0.007},
    18: {"value": 0.092, "error": 0.007},
    19: {"value": 0.092, "error": 0.007},
    20: {"value": 0.093, "error": 0.007},
    21: {"value": 0.095, "error": 0.007},
    22: {"value": 0.098, "error": 0.007},
    23: {"value": 0.102, "error": 0.007},
    24: {"value": 0.108, "error": 0.008},
    25: {"value": 0.116, "error": 0.008},
    26: {"value": 0.126, "error": 0.009},
    27: {"value": 0.138, "error": 0.010},
    28: {"value": 0.152, "error": 0.011},
    29: {"value": 0.168, "error": 0.012},
}

PLANCK_TE = {
    2:  {"value": -10.0, "error": 35.0},
    3:  {"value": -15.0, "error": 28.0},
    4:  {"value":  12.0, "error": 22.0},
    5:  {"value":  35.0, "error": 18.0},
    6:  {"value":  28.0, "error": 15.0},
    7:  {"value":  42.0, "error": 13.0},
    8:  {"value":  38.0, "error": 11.0},
    9:  {"value":  32.0, "error": 10.0},
    10: {"value":  25.0, "error":  9.0},
    11: {"value":  35.0, "error":  8.5},
    12: {"value":  42.0, "error":  8.0},
    13: {"value":  38.0, "error":  7.5},
    14: {"value":  32.0, "error":  7.0},
    15: {"value":  28.0, "error":  6.5},
    16: {"value":  35.0, "error":  6.0},
    17: {"value":  40.0, "error":  5.8},
    18: {"value":  45.0, "error":  5.5},
    19: {"value":  48.0, "error":  5.2},
    20: {"value":  52.0, "error":  5.0},
    21: {"value":  55.0, "error":  4.8},
    22: {"value":  58.0, "error":  4.6},
    23: {"value":  60.0, "error":  4.4},
    24: {"value":  62.0, "error":  4.2},
    25: {"value":  65.0, "error":  4.0},
    26: {"value":  68.0, "error":  3.8},
    27: {"value":  70.0, "error":  3.6},
    28: {"value":  72.0, "error":  3.4},
    29: {"value":  75.0, "error":  3.2},
}

LCDM_TT = {
    2: 1023.0, 3: 936.0, 4: 1020.0, 5: 1195.0, 6: 1096.0,
    7: 1217.0, 8: 1246.0, 9: 1109.0, 10: 1016.0, 11: 1141.0,
    12: 1161.0, 13: 1037.0, 14: 970.0, 15: 1002.0, 16: 1100.0,
    17: 1080.0, 18: 1107.0, 19: 1096.0, 20: 1049.0, 21: 1033.0,
    22: 1024.0, 23: 1000.0, 24: 956.0, 25: 934.0, 26: 889.0,
    27: 871.0, 28: 835.0, 29: 815.0,
}

LCDM_EE = {
    2: 0.020, 3: 0.030, 4: 0.048, 5: 0.062, 6: 0.070,
    7: 0.076, 8: 0.081, 9: 0.085, 10: 0.088, 11: 0.090,
    12: 0.091, 13: 0.092, 14: 0.092, 15: 0.093, 16: 0.093,
    17: 0.094, 18: 0.094, 19: 0.095, 20: 0.096, 21: 0.098,
    22: 0.101, 23: 0.105, 24: 0.110, 25: 0.118, 26: 0.128,
    27: 0.140, 28: 0.154, 29: 0.170,
}

LCDM_TE = {
    2: -5.0, 3: -10.0, 4: 15.0, 5: 38.0, 6: 30.0,
    7: 45.0, 8: 40.0, 9: 35.0, 10: 28.0, 11: 38.0,
    12: 45.0, 13: 40.0, 14: 35.0, 15: 30.0, 16: 38.0,
    17: 42.0, 18: 48.0, 19: 50.0, 20: 55.0, 21: 58.0,
    22: 60.0, 23: 62.0, 24: 65.0, 25: 68.0, 26: 70.0,
    27: 72.0, 28: 74.0, 29: 76.0,
}

# Precompute ΛCDM chi-squared
N_DATA = 28
CHI2_LCDM = 0.0
for ell in range(2, 30):
    pred = LCDM_TT[ell]
    obs = PLANCK_TT[ell]["value"]
    err = PLANCK_TT[ell]["error_plus"] if pred > obs else PLANCK_TT[ell]["error_minus"]
    CHI2_LCDM += ((obs - pred) / err) ** 2


# =============================================================================
# HELPERS
# =============================================================================

def _safe_import(path: Path):
    spec = importlib.util.spec_from_file_location("candidate", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _count_parameters(params: Dict) -> int:
    count = 0
    for v in params.values():
        if isinstance(v, (int, float)):
            count += 1
        elif isinstance(v, (list, tuple)):
            count += len(v)
        elif isinstance(v, dict):
            count += _count_parameters(v)
    return count


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(program_path: str) -> Dict[str, Any]:
    path = Path(program_path)
    
    try:
        src = path.read_text()
    except Exception:
        return {"combined_score": 0.0, "COMBINED_SCORE": 0.0, "error": "cannot read file"}

    try:
        mod = _safe_import(path)
    except Exception as e:
        return {"combined_score": 0.0, "COMBINED_SCORE": 0.0, "error": str(e)}

    if not hasattr(mod, "bounce_spectrum") or not hasattr(mod, "get_default_params"):
        return {"combined_score": 0.0, "COMBINED_SCORE": 0.0, "error": "missing API"}

    try:
        params = mod.get_default_params()
        if not isinstance(params, dict):
            params = {}
    except Exception:
        params = {}

    n_params = _count_parameters(params)
    
    # Check for polarization functions
    has_ee = hasattr(mod, "bounce_spectrum_EE")
    has_te = hasattr(mod, "bounce_spectrum_TE")
    has_polarization = bool(has_ee or has_te)
    
    # Compute model predictions
    mods_tt = {}
    for ell in range(2, 30):
        try:
            m = float(mod.bounce_spectrum(ell, params))
            if not math.isfinite(m) or m <= 0.0:
                return {"combined_score": 0.0, "COMBINED_SCORE": 0.0, "error": f"bad mod at ell={ell}"}
            mods_tt[ell] = m
        except Exception as e:
            return {"combined_score": 0.0, "COMBINED_SCORE": 0.0, "error": str(e)}

    # EE and TE modifications
    mods_ee = {}
    mods_te = {}
    if has_ee:
        for ell in range(2, 30):
            try:
                m = float(mod.bounce_spectrum_EE(ell, params))
                mods_ee[ell] = m if math.isfinite(m) and m > 0 else 1.0
            except:
                mods_ee[ell] = 1.0 - 0.3 * (1.0 - mods_tt[ell])
    else:
        for ell in range(2, 30):
            mods_ee[ell] = 1.0 - 0.3 * (1.0 - mods_tt[ell])
            
    if has_te:
        for ell in range(2, 30):
            try:
                m = float(mod.bounce_spectrum_TE(ell, params))
                mods_te[ell] = m if math.isfinite(m) else math.sqrt(mods_tt[ell] * mods_ee[ell])
            except:
                mods_te[ell] = math.sqrt(mods_tt[ell] * mods_ee[ell])
    else:
        for ell in range(2, 30):
            mods_te[ell] = math.sqrt(mods_tt[ell] * mods_ee[ell])

    # Compute chi-squared for bounce model (TT only for BIC comparison)
    chi2_tt = 0.0
    for ell in range(2, 30):
        pred = LCDM_TT[ell] * mods_tt[ell]
        obs = PLANCK_TT[ell]["value"]
        err = PLANCK_TT[ell]["error_plus"] if pred > obs else PLANCK_TT[ell]["error_minus"]
        chi2_tt += ((obs - pred) / err) ** 2

    # EE and TE chi-squared
    chi2_ee = sum(((PLANCK_EE[e]["value"] - LCDM_EE[e] * mods_ee[e]) / PLANCK_EE[e]["error"]) ** 2 
                  for e in range(2, 30))
    chi2_te = sum(((PLANCK_TE[e]["value"] - LCDM_TE[e] * mods_te[e]) / PLANCK_TE[e]["error"]) ** 2 
                  for e in range(2, 30))
    
    chi2_total = chi2_tt + 0.5 * chi2_ee + 0.5 * chi2_te

    # =========================================================================
    # KEY METRIC: ΔBIC (Bayesian Information Criterion)
    # =========================================================================
    bic_lcdm = CHI2_LCDM  # ΛCDM has 0 extra params
    bic_model = chi2_tt + n_params * math.log(N_DATA)
    delta_bic = bic_model - bic_lcdm
    delta_chi2 = CHI2_LCDM - chi2_tt  # Positive = model is better
    
    # =========================================================================
    # CONSTRAINTS
    # =========================================================================
    
    # 1. Quadrupole must be suppressed (S(2) near 0.196)
    target_s2 = 201.0 / 1023.0  # ~0.196
    quadrupole_penalty = abs(mods_tt[2] - target_s2) * 10.0
    quadrupole_score = math.exp(-((mods_tt[2] - target_s2) / 0.08) ** 2)
    
    # 2. High-ℓ must return to 1.0
    asymptotic_penalty = sum(abs(mods_tt[ell] - 1.0) for ell in range(25, 30)) * 2.0
    asymp_dev = sum(abs(mods_tt[ell] - 1.0) for ell in range(25, 30)) / 5.0
    asymptotic_score = math.exp(-15.0 * asymp_dev)
    
    # 3. Monotonicity (mostly increasing from ℓ=2 to ~15)
    mono_violations = 0
    for ell in range(2, 15):
        if mods_tt[ell + 1] < mods_tt[ell] - 0.02:
            mono_violations += 1
    mono_penalty = 0.0
    mono_score = math.exp(-0.4 * mono_violations)
    
    # 4. Smoothness
    mod_list = [mods_tt[e] for e in range(2, 30)]
    d2 = [abs(mod_list[i + 1] - 2.0 * mod_list[i] + mod_list[i - 1]) for i in range(1, len(mod_list) - 1)]
    smoothness = math.exp(-8.0 * sum(d2) / len(d2)) if d2 else 0.5
    
    total_penalty = quadrupole_penalty + asymptotic_penalty + mono_penalty
    
    # =========================================================================
    # FITNESS: Reward negative ΔBIC (model beats ΛCDM)
    # =========================================================================
    scale = 5.0
    raw_fitness = 1.0 / (1.0 + math.exp(delta_bic / scale))
    
    # Apply penalties
    combined_score = max(0.0, raw_fitness - 0.1 * total_penalty)
    
    # Simplicity bonus
    simplicity_bonus = 0.02 * max(0, 3 - n_params)
    combined_score = min(1.0, combined_score + simplicity_bonus)
    
    # Fit score (for compatibility with original evaluator)
    chi_reduced = chi2_total / (N_DATA + 14 + 14)
    fit_score = math.exp(-0.25 * abs(chi_reduced - 1.0))
    
    # Physics score
    physics_score = 0.5 * (quadrupole_score + asymptotic_score)
    
    # =========================================================================
    # RESULTS (compatible with original evaluator output format)
    # =========================================================================
    
    results = {
        # Primary metrics (same names as original)
        "combined_score": combined_score,
        "COMBINED_SCORE": combined_score,
        "fit_score": fit_score,
        "quadrupole_score": quadrupole_score,
        "smoothness": smoothness,
        "asymptotic_score": asymptotic_score,
        "mono_score": mono_score,
        "physics_score": physics_score,
        
        # Evidence metrics (new)
        "delta_bic": delta_bic,
        "delta_chi2": delta_chi2,
        "n_params": float(n_params),
        "beats_lcdm": delta_bic < 0,
        
        # Chi-squared breakdown
        "chi_sq_tt": chi2_tt,
        "chi_sq_ee": chi2_ee,
        "chi_sq_te": chi2_te,
        "chi_sq_total": chi2_total,
        "chi_reduced": chi_reduced,
        "chi2_lcdm": CHI2_LCDM,
        
        # Model values at key multipoles
        "mod_tt_2": mods_tt[2],
        "mod_tt_3": mods_tt[3],
        "mod_tt_5": mods_tt[5],
        "mod_tt_10": mods_tt[10],
        "mod_tt_20": mods_tt[20],
        
        # Diagnostics
        "has_polarization": has_polarization,
        "raw_fitness": raw_fitness,
        "quadrupole_penalty": quadrupole_penalty,
        "asymptotic_penalty": asymptotic_penalty,
        "mono_penalty": mono_penalty,
    }
    
    return results


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
