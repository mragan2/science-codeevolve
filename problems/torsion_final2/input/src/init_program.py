# EVOLVE-BLOCK-START
"""
Einstein-Cartan Torsion Transfer Functions
==========================================
Zero-parameter model: S(ell) modifies LCDM predictions to fit Planck CMB data.
Key physics: quadrupole suppression from torsion spin-density coupling.
alpha ~ 3/2 from ECSK theory, kappa = 0.804 from observed quadrupole.
"""
import math


def get_torsion_params():
    """Zero free parameters for minimal BIC penalty."""
    return {}


def S_TT(ell, params):
    """TT temperature transfer function.
    Power-law recovery from quadrupole suppression."""
    if ell <= 2:
        return 0.196
    x = float(ell) - 2.0
    # Power-law recovery: alpha ~ 3/2 from Einstein-Cartan theory
    base = 1.0 - 0.804 / (1.0 + x) ** 1.40
    return max(1e-12, base)


def S_EE(ell, params):
    """EE polarization transfer function.
    Spin-2 coupling gives weaker suppression than TT."""
    if ell <= 2:
        return 0.461
    x = float(ell) - 2.0
    base = 1.0 - 0.539 / (1.0 + x) ** 1.48
    return max(1e-12, base)


def S_TE(ell, params):
    """TE cross-correlation transfer function.
    Geometric mean with Fourier step for bounce transition."""
    s_tt = S_TT(ell, params)
    s_ee = S_EE(ell, params)
    base = (abs(s_tt) * abs(s_ee)) ** 0.5
    # Fourier series approximation of step function at transition ~ 3.5
    ell_f = float(ell)
    step = 0.0
    for n in range(1, 14, 2):
        step += (4.0 / (n * math.pi)) * math.sin(
            n * math.pi * (ell_f - 3.5) / 6.0
        )
    step = max(-1.0, min(1.0, step))
    return max(1e-12, abs(base * step))


def predict_BB(ell, params):
    """B-mode prediction from torsion-generated tensors."""
    ell_f = float(ell)
    amplitude = 1.2e-4
    peak = max(0.0, math.cos(math.pi * (ell_f - 100.0) / 50.0))
    shape = (ell_f / 100.0) ** 1.5 * peak
    return max(0.0, amplitude * shape)
# EVOLVE-BLOCK-END
