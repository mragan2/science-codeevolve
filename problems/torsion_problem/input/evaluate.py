"""
Torsion Cosmology Parameter Evaluator
=====================================

OBJECTIVE: Evolve Einstein-Cartan torsion parameters to optimally fit
CMB TT, EE, and TE spectra while predicting tensor (B-mode) signatures.

BACKGROUND:
- Your discovered transfer function S(ell) = 1 - 0.804/(1+(ell-2))^1.455
- The power-law index alpha = 1.455 ~ 3/2 matches Einstein-Cartan prediction
- Now we evolve the full torsion model to:
  1. Fit TT, EE, TE simultaneously
  2. Predict tensor-to-scalar ratio r
  3. Predict B-mode spectrum for CMB-S4

KEY PARAMETERS TO EVOLVE:
- kappa: Torsion coupling strength
- beta: Spin-torsion interaction parameter  
- r_torsion: Tensor-to-scalar ratio from bounce
- n_t: Tensor spectral tilt

PHYSICAL CONSTRAINTS (from Einstein-Cartan theory):
- alpha = 3/2 (fixed by theory)
- S(2) ~ 0.196 (from Planck observation)
- r < 0.06 (Planck/BICEP upper limit)
- -0.1 < n_t < 0.1 (near scale-invariant)

References:
- Poplawski (2010): arXiv:1007.0587
- Planck 2018: arXiv:1807.06209
"""

import importlib.util
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any

# =============================================================================
# PLANCK 2018 DATA (TT, EE, TE)
# =============================================================================

PLANCK_TT = {
    2: {"value": 201.0, "error_minus": 120.0, "error_plus": 810.0},
    3: {"value": 610.0, "error_minus": 200.0, "error_plus": 350.0},
    4: {"value": 584.0, "error_minus": 170.0, "error_plus": 260.0},
    5: {"value": 1540.0, "error_minus": 200.0, "error_plus": 240.0},
    6: {"value": 637.0, "error_minus": 140.0, "error_plus": 220.0},
    7: {"value": 1305.0, "error_minus": 160.0, "error_plus": 190.0},
    8: {"value": 1284.0, "error_minus": 140.0, "error_plus": 170.0},
    9: {"value": 974.0, "error_minus": 120.0, "error_plus": 150.0},
    10: {"value": 767.0, "error_minus": 100.0, "error_plus": 130.0},
    11: {"value": 1117.0, "error_minus": 100.0, "error_plus": 120.0},
    12: {"value": 1151.0, "error_minus": 95.0, "error_plus": 115.0},
    13: {"value": 921.0, "error_minus": 85.0, "error_plus": 105.0},
    14: {"value": 820.0, "error_minus": 78.0, "error_plus": 98.0},
    15: {"value": 881.0, "error_minus": 74.0, "error_plus": 94.0},
}

PLANCK_EE = {
    2: {"value": 0.018, "error": 0.020},
    3: {"value": 0.028, "error": 0.018},
    4: {"value": 0.045, "error": 0.016},
    5: {"value": 0.058, "error": 0.015},
    6: {"value": 0.065, "error": 0.014},
    7: {"value": 0.072, "error": 0.013},
    8: {"value": 0.078, "error": 0.012},
    9: {"value": 0.082, "error": 0.011},
    10: {"value": 0.085, "error": 0.010},
    11: {"value": 0.087, "error": 0.010},
    12: {"value": 0.088, "error": 0.009},
    13: {"value": 0.089, "error": 0.009},
    14: {"value": 0.090, "error": 0.008},
    15: {"value": 0.090, "error": 0.008},
}

PLANCK_TE = {
    2: {"value": -10.0, "error": 35.0},
    3: {"value": -15.0, "error": 28.0},
    4: {"value": 12.0, "error": 22.0},
    5: {"value": 35.0, "error": 18.0},
    6: {"value": 28.0, "error": 15.0},
    7: {"value": 42.0, "error": 13.0},
    8: {"value": 38.0, "error": 11.0},
    9: {"value": 32.0, "error": 10.0},
    10: {"value": 25.0, "error": 9.0},
    11: {"value": 35.0, "error": 8.5},
    12: {"value": 42.0, "error": 8.0},
    13: {"value": 38.0, "error": 7.5},
    14: {"value": 32.0, "error": 7.0},
    15: {"value": 28.0, "error": 6.5},
}

# LCDM predictions
LCDM_TT = {2: 1023.0, 3: 936.0, 4: 1020.0, 5: 1195.0, 6: 1096.0,
           7: 1217.0, 8: 1246.0, 9: 1109.0, 10: 1016.0, 11: 1141.0,
           12: 1161.0, 13: 1037.0, 14: 970.0, 15: 1002.0}

LCDM_EE = {2: 0.020, 3: 0.030, 4: 0.048, 5: 0.062, 6: 0.070,
           7: 0.076, 8: 0.081, 9: 0.085, 10: 0.088, 11: 0.090,
           12: 0.091, 13: 0.092, 14: 0.092, 15: 0.093}

LCDM_TE = {2: -5.0, 3: -10.0, 4: 15.0, 5: 38.0, 6: 30.0,
           7: 45.0, 8: 40.0, 9: 35.0, 10: 28.0, 11: 38.0,
           12: 45.0, 13: 40.0, 14: 35.0, 15: 30.0}

# BICEP/Keck tensor upper limit
R_UPPER_LIMIT = 0.06

# Number of data points
N_DATA = 14 * 3  # TT + EE + TE

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
        mod = _safe_import(path)
    except Exception as e:
        return {"combined_score": 0.0, "error": f"import: {e}"}

    # Required functions
    required = ["get_torsion_params", "S_TT", "S_EE", "S_TE"]
    for func in required:
        if not hasattr(mod, func):
            return {"combined_score": 0.0, "error": f"missing {func}"}

    try:
        params = mod.get_torsion_params()
        if not isinstance(params, dict):
            params = {}
    except Exception as e:
        return {"combined_score": 0.0, "error": f"params: {e}"}

    n_params = _count_parameters(params)
    
    # ==========================================================================
    # Compute transfer functions
    # ==========================================================================
    S_tt, S_ee, S_te = {}, {}, {}
    
    for ell in range(2, 16):
        try:
            s_tt = float(mod.S_TT(ell, params))
            s_ee = float(mod.S_EE(ell, params))
            s_te = float(mod.S_TE(ell, params))
            
            if not all(math.isfinite(x) and x > 0 for x in [s_tt, s_ee]):
                return {"combined_score": 0.0, "error": f"invalid S at ell={ell}"}
            
            S_tt[ell] = s_tt
            S_ee[ell] = s_ee
            S_te[ell] = s_te
        except Exception as e:
            return {"combined_score": 0.0, "error": f"S({ell}): {e}"}

    # ==========================================================================
    # Compute chi^2 for TT, EE, TE
    # ==========================================================================
    chi2_tt = 0.0
    for ell in range(2, 16):
        pred = LCDM_TT[ell] * S_tt[ell]
        obs = PLANCK_TT[ell]["value"]
        err = PLANCK_TT[ell]["error_plus"] if pred > obs else PLANCK_TT[ell]["error_minus"]
        chi2_tt += ((obs - pred) / err) ** 2

    chi2_ee = 0.0
    for ell in range(2, 16):
        pred = LCDM_EE[ell] * S_ee[ell]
        obs = PLANCK_EE[ell]["value"]
        err = PLANCK_EE[ell]["error"]
        chi2_ee += ((obs - pred) / err) ** 2

    chi2_te = 0.0
    for ell in range(2, 16):
        pred = LCDM_TE[ell] * S_te[ell]
        obs = PLANCK_TE[ell]["value"]
        err = PLANCK_TE[ell]["error"]
        chi2_te += ((obs - pred) / err) ** 2

    chi2_total = chi2_tt + chi2_ee + chi2_te
    chi2_reduced = chi2_total / N_DATA

    # ==========================================================================
    # LCDM baseline chi^2
    # ==========================================================================
    chi2_lcdm_tt = sum(((PLANCK_TT[e]["value"] - LCDM_TT[e]) / 
                        (PLANCK_TT[e]["error_plus"] if LCDM_TT[e] > PLANCK_TT[e]["value"] 
                         else PLANCK_TT[e]["error_minus"]))**2 for e in range(2, 16))
    chi2_lcdm_ee = sum(((PLANCK_EE[e]["value"] - LCDM_EE[e]) / PLANCK_EE[e]["error"])**2 
                       for e in range(2, 16))
    chi2_lcdm_te = sum(((PLANCK_TE[e]["value"] - LCDM_TE[e]) / PLANCK_TE[e]["error"])**2 
                       for e in range(2, 16))
    chi2_lcdm = chi2_lcdm_tt + chi2_lcdm_ee + chi2_lcdm_te

    # ==========================================================================
    # BIC comparison
    # ==========================================================================
    bic_lcdm = chi2_lcdm
    bic_model = chi2_total + n_params * math.log(N_DATA)
    delta_bic = bic_model - bic_lcdm
    delta_chi2 = chi2_lcdm - chi2_total

    # ==========================================================================
    # Physical constraints
    # ==========================================================================
    
    # 1. Quadrupole TT suppression must match observation
    target_s2 = 201.0 / 1023.0  # ~0.196
    quadrupole_score = math.exp(-((S_tt[2] - target_s2) / 0.05) ** 2)
    
    # 2. High-ell must return to 1.0
    asymp_dev = sum(abs(S_tt[ell] - 1.0) for ell in range(12, 16)) / 4.0
    asymptotic_score = math.exp(-10.0 * asymp_dev)
    
    # 3. Tensor-to-scalar ratio (if provided)
    r_torsion = params.get("r_torsion", 0.01)
    if r_torsion > R_UPPER_LIMIT:
        tensor_penalty = (r_torsion - R_UPPER_LIMIT) * 10
    else:
        tensor_penalty = 0.0
    tensor_score = math.exp(-tensor_penalty)
    
    # 4. EE-TT correlation (torsion predicts specific relationship)
    # In ECSK: S_EE ~ S_TT^(2/3) for spin-2 torsion coupling
    ee_tt_correlation = 0.0
    for ell in range(2, 16):
        expected_ee = S_tt[ell] ** (2/3)
        ee_tt_correlation += (S_ee[ell] - expected_ee) ** 2
    correlation_score = math.exp(-5.0 * ee_tt_correlation / 14)

    # ==========================================================================
    # Combined fitness
    # ==========================================================================
    
    # BIC-based fitness (reward negative Delta-BIC)
    bic_fitness = 1.0 / (1.0 + math.exp(delta_bic / 5.0))
    
    # Physics constraints
    physics_score = (quadrupole_score * asymptotic_score * tensor_score * correlation_score) ** 0.25
    
    # Combined score
    combined_score = 0.6 * bic_fitness + 0.4 * physics_score
    
    # Simplicity bonus
    if n_params <= 2:
        combined_score += 0.02 * (3 - n_params)
    
    combined_score = max(0.0, min(1.0, combined_score))

    # ==========================================================================
    # Tensor/B-mode predictions (for future experiments)
    # ==========================================================================
    n_t = params.get("n_t", 0.0)  # Tensor tilt
    
    # B-mode prediction at ell=80 (BICEP sweet spot)
    # C_ell^BB ~ r * (ell/80)^n_t for primordial tensors
    bb_amplitude = r_torsion * 0.01  # uK^2 at ell=80
    
    # ==========================================================================
    # Results
    # ==========================================================================
    return {
        "combined_score": combined_score,
        "COMBINED_SCORE": combined_score,
        
        # chi^2 breakdown
        "chi2_tt": chi2_tt,
        "chi2_ee": chi2_ee,
        "chi2_te": chi2_te,
        "chi2_total": chi2_total,
        "chi2_reduced": chi2_reduced,
        "chi2_lcdm": chi2_lcdm,
        
        # Evidence
        "delta_bic": delta_bic,
        "delta_chi2": delta_chi2,
        "n_params": float(n_params),
        "beats_lcdm": delta_bic < 0,
        
        # Physics scores
        "quadrupole_score": quadrupole_score,
        "asymptotic_score": asymptotic_score,
        "tensor_score": tensor_score,
        "correlation_score": correlation_score,
        "physics_score": physics_score,
        "bic_fitness": bic_fitness,
        
        # Transfer function values
        "S_TT_2": S_tt[2],
        "S_TT_10": S_tt[10],
        "S_EE_2": S_ee[2],
        "S_TE_2": S_te[2],
        
        # Tensor predictions (for CMB-S4)
        "r_torsion": r_torsion,
        "n_t": n_t,
        "BB_amplitude_ell80": bb_amplitude,
        
        # Torsion parameters (if present)
        "kappa": params.get("kappa", None),
        "beta": params.get("beta", None),
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
