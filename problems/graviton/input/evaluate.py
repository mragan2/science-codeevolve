"""
Graviton Tensor Perturbation Evaluator (Hardened v2)
=====================================================
Evaluates graviton equation solutions against REAL Planck CMB data.

Key changes from v1:
- Scores against Planck 2018 TT/EE/TE observations (not synthetic S->1.0)
- BIC-based fitness penalises parameter overfitting
- Multi-constraint physics score (geometric mean of independent checks)
- Functional torsion activity check (output behavior, not source keywords)
- Extended ell range: 2..29

Fitness = combined_score in [0, 1].

Interface: python evaluate.py <candidate.py> <results.json>
"""

import importlib.util
import sys
import json
import math
import ast
from pathlib import Path


# =============================================================================
# PLANCK 2018 DATA  (ell = 2..29)
# D_ell values in uK^2 for TT;  C_ell(ell+1)/(2pi) convention for EE/TE
# Sources: Planck 2018 V (arXiv:1907.12875), Poplawski (arXiv:1007.0587)
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

# LCDM baseline predictions
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

# Evaluation range
ELL_MIN = 2
ELL_MAX = 29
N_ELL = ELL_MAX - ELL_MIN + 1   # 28
N_DATA = N_ELL * 3               # TT + EE + TE = 84

# Precompute LCDM chi-squared baselines
_CHI2_LCDM_TT = 0.0
for _ell in range(ELL_MIN, ELL_MAX + 1):
    _pred = LCDM_TT[_ell]
    _obs = PLANCK_TT[_ell]["value"]
    _err = (PLANCK_TT[_ell]["error_plus"] if _pred > _obs
            else PLANCK_TT[_ell]["error_minus"])
    _CHI2_LCDM_TT += ((_obs - _pred) / _err) ** 2

_CHI2_LCDM_EE = 0.0
for _ell in range(ELL_MIN, ELL_MAX + 1):
    _pred = LCDM_EE[_ell]
    _obs = PLANCK_EE[_ell]["value"]
    _err = PLANCK_EE[_ell]["error"]
    _CHI2_LCDM_EE += ((_obs - _pred) / _err) ** 2

_CHI2_LCDM_TE = 0.0
for _ell in range(ELL_MIN, ELL_MAX + 1):
    _pred = LCDM_TE[_ell]
    _obs = PLANCK_TE[_ell]["value"]
    _err = PLANCK_TE[_ell]["error"]
    _CHI2_LCDM_TE += ((_obs - _pred) / _err) ** 2

_CHI2_LCDM_TOTAL = _CHI2_LCDM_TT + _CHI2_LCDM_EE + _CHI2_LCDM_TE

# Planck quadrupole target
TARGET_S2 = 201.0 / 1023.0   # ~0.1965


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _safe_eval(func, ell, params):
    """Call a transfer function safely, returning None on any problem."""
    try:
        v = func(ell, params)
        if not math.isfinite(v):
            return None
        if v <= 0:
            return None
        return float(v)
    except Exception:
        return None


def _count_parameters(params):
    """Recursively count scalar parameters in a dict."""
    count = 0
    for v in params.values():
        if isinstance(v, (int, float)):
            count += 1
        elif isinstance(v, (list, tuple)):
            count += len(v)
        elif isinstance(v, dict):
            count += _count_parameters(v)
    return count


def _validate_source_constraints(candidate_path: str):
    """
    Enforce hard source constraints:
    - ASCII only
    - imports: only one line "import math"
    """
    try:
        raw = Path(candidate_path).read_bytes()
    except Exception as exc:
        return f"read source: {exc}"

    if any(b > 127 for b in raw):
        return "hard constraint: non-ascii source"

    try:
        source = raw.decode("utf-8")
    except Exception:
        source = raw.decode("utf-8", errors="replace")

    try:
        tree = ast.parse(source, filename=candidate_path)
    except Exception as exc:
        return f"syntax: {exc}"

    import_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            return "hard constraint: forbidden 'from ... import ...'"
        if isinstance(node, ast.Import):
            import_count += 1
            if len(node.names) != 1:
                return "hard constraint: only 'import math' is allowed"
            alias = node.names[0]
            if alias.name != "math" or alias.asname is not None:
                return "hard constraint: only 'import math' is allowed"

    if import_count > 1:
        return "hard constraint: multiple import lines"
    return None


# --------------------------------------------------------------------------
# Main evaluator
# --------------------------------------------------------------------------

def evaluate(candidate_path: str) -> dict:
    """Load candidate module and evaluate against Planck data."""

    constraint_error = _validate_source_constraints(candidate_path)
    if constraint_error is not None:
        return {"combined_score": 0.0, "error": constraint_error}

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

    try:
        params = program.get_torsion_params()
        if not isinstance(params, dict):
            params = {}
    except Exception as exc:
        return {"combined_score": 0.0, "error": f"params: {exc}"}

    n_params = _count_parameters(params)

    # ==================================================================
    # 1. Evaluate transfer functions across ell = 2..29
    # ==================================================================
    S_tt, S_ee, S_te = {}, {}, {}

    for ell in range(ELL_MIN, ELL_MAX + 1):
        tt = _safe_eval(program.S_TT, ell, params)
        ee = _safe_eval(program.S_EE, ell, params)
        te = _safe_eval(program.S_TE, ell, params)

        if tt is None or ee is None or te is None:
            return {"combined_score": 0.0, "error": f"invalid value at ell={ell}"}

        S_tt[ell] = tt
        S_ee[ell] = ee
        S_te[ell] = te

    # ==================================================================
    # 2. Chi-squared against Planck data
    # ==================================================================

    # TT chi-squared (asymmetric errors)
    chi2_tt = 0.0
    for ell in range(ELL_MIN, ELL_MAX + 1):
        pred = LCDM_TT[ell] * S_tt[ell]
        obs = PLANCK_TT[ell]["value"]
        err = (PLANCK_TT[ell]["error_plus"] if pred > obs
               else PLANCK_TT[ell]["error_minus"])
        chi2_tt += ((obs - pred) / err) ** 2

    # EE chi-squared
    chi2_ee = 0.0
    for ell in range(ELL_MIN, ELL_MAX + 1):
        pred = LCDM_EE[ell] * S_ee[ell]
        obs = PLANCK_EE[ell]["value"]
        err = PLANCK_EE[ell]["error"]
        chi2_ee += ((obs - pred) / err) ** 2

    # TE chi-squared
    chi2_te = 0.0
    for ell in range(ELL_MIN, ELL_MAX + 1):
        pred = LCDM_TE[ell] * S_te[ell]
        obs = PLANCK_TE[ell]["value"]
        err = PLANCK_TE[ell]["error"]
        chi2_te += ((obs - pred) / err) ** 2

    chi2_total = chi2_tt + chi2_ee + chi2_te

    # ==================================================================
    # 3. BIC-based fitness  (compare model vs LCDM)
    # ==================================================================
    bic_model = chi2_total + n_params * math.log(N_DATA)
    bic_lcdm = _CHI2_LCDM_TOTAL    # LCDM has 0 extra torsion params
    delta_bic = bic_model - bic_lcdm
    delta_chi2 = _CHI2_LCDM_TOTAL - chi2_total   # positive = model better

    # Smooth mapping: keep the logistic decision boundary while preserving
    # tail sensitivity for very poor models (prevents complete score collapse).
    x = delta_bic / 5.0
    if x >= 0.0:
        ex = math.exp(-x)
        bic_sigmoid = ex / (1.0 + ex)
    else:
        ex = math.exp(x)
        bic_sigmoid = 1.0 / (1.0 + ex)

    bic_tail = 1.0 / (1.0 + max(0.0, delta_bic) / 80.0)
    bic_fitness = 0.85 * bic_sigmoid + 0.15 * bic_tail

    # ==================================================================
    # 4. Multi-constraint physics score  (geometric mean)
    # ==================================================================

    # (a) Quadrupole suppression: S_TT(2) ~ 0.196
    quadrupole_score = math.exp(-((S_tt[2] - TARGET_S2) / 0.05) ** 2)

    # (b) Asymptotic recovery: S_TT(ell > 24) -> 1.0
    asymp_dev = sum(abs(S_tt[ell] - 1.0) for ell in range(25, 30)) / 5.0
    asymptotic_score = math.exp(-15.0 * asymp_dev)

    # (c) Monotonicity: S_TT mostly increasing from ell=2 to ell=15
    mono_violations = 0
    for ell in range(ELL_MIN, 15):
        if S_tt[ell + 1] < S_tt[ell] - 0.02:
            mono_violations += 1
    monotonicity_score = math.exp(-0.4 * mono_violations)

    # (d) Smoothness: low discrete curvature across full ell range
    tt_list = [S_tt[ell] for ell in range(ELL_MIN, ELL_MAX + 1)]
    d2 = []
    for i in range(1, len(tt_list) - 1):
        d2.append(abs(tt_list[i + 1] - 2.0 * tt_list[i] + tt_list[i - 1]))
    smoothness_score = math.exp(-8.0 * sum(d2) / len(d2)) if d2 else 0.5

    # (e) EE-TT correlation: S_EE ~ S_TT^(2/3)  (ECSK theory prediction)
    ee_tt_mse = 0.0
    for ell in range(ELL_MIN, ELL_MAX + 1):
        expected_ee = S_tt[ell] ** (2.0 / 3.0)
        ee_tt_mse += (S_ee[ell] - expected_ee) ** 2
    ee_tt_mse /= N_ELL
    correlation_score = math.exp(-5.0 * ee_tt_mse)

    # Geometric mean of all five constraints
    physics_score = (
        quadrupole_score
        * asymptotic_score
        * monotonicity_score
        * smoothness_score
        * correlation_score
    ) ** (1.0 / 5.0)

    # ==================================================================
    # 5. Combined score
    # ==================================================================
    combined_score = 0.6 * bic_fitness + 0.4 * physics_score

    # Simplicity bonus (reward fewer free parameters)
    if n_params <= 2:
        combined_score += 0.02 * (3 - n_params)

    combined_score = max(0.0, min(1.0, combined_score))

    # ==================================================================
    # 6. Torsion activity check  (functional, not keyword-based)
    #    The transfer function MUST show non-trivial ell-dependence:
    #    - range(S_TT) > 0.15  across ell = 2..29
    #    - S_TT(2) < 0.5       (quadrupole must be suppressed)
    #    Failure => cap at 0.10  (below any useful score)
    # ==================================================================
    tt_range = max(tt_list) - min(tt_list)
    activity_ok = (tt_range > 0.15) and (S_tt[2] < 0.5)

    if not activity_ok:
        combined_score = min(combined_score, 0.10)

    # ==================================================================
    # Return all metrics
    # ==================================================================
    return {
        # Primary fitness
        "combined_score": combined_score,
        "COMBINED_SCORE": combined_score,

        # Chi-squared breakdown
        "chi2_tt": chi2_tt,
        "chi2_ee": chi2_ee,
        "chi2_te": chi2_te,
        "chi2_total": chi2_total,
        "chi2_lcdm": _CHI2_LCDM_TOTAL,
        "chi2_lcdm_tt": _CHI2_LCDM_TT,
        "chi2_lcdm_ee": _CHI2_LCDM_EE,
        "chi2_lcdm_te": _CHI2_LCDM_TE,

        # BIC evidence
        "delta_bic": delta_bic,
        "delta_chi2": delta_chi2,
        "n_params": float(n_params),
        "beats_lcdm": delta_bic < 0,
        "bic_fitness": bic_fitness,

        # Physics constraint scores
        "quadrupole_score": quadrupole_score,
        "asymptotic_score": asymptotic_score,
        "monotonicity_score": monotonicity_score,
        "smoothness_score": smoothness_score,
        "correlation_score": correlation_score,
        "physics_score": physics_score,

        # Torsion activity
        "tt_range": tt_range,
        "activity_ok": activity_ok,

        # Transfer function values at key multipoles
        "S_TT_2": S_tt[2],
        "S_TT_10": S_tt[10],
        "S_TT_20": S_tt.get(20, None),
        "S_TT_29": S_tt.get(29, None),
        "S_EE_2": S_ee[2],
        "S_TE_2": S_te[2],
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
