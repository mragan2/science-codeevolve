# EVOLVE-BLOCK-START
"""
Minimal bounce cosmology transfer function.
Evolution should discover the simplest model that beats ΛCDM.

OBJECTIVE: Minimize ΔBIC = BIC(model) - BIC(ΛCDM)
- Each parameter costs +3.33 in BIC
- Need Δχ² > 3.33 per parameter to justify inclusion

bounce / horizon / hubble / scale / torsion / planck inspired
"""
import math

def get_default_params():
    """
    Start with 1 parameter. Evolution can add/remove.
    Current best: alpha=1.33 achieves ΔBIC=-0.24 (beats ΛCDM)
    """
    return {
        "alpha": 1.33,  # Power-law exponent
    }

def bounce_spectrum(ell, params):
    """
    Power-law recovery model:
    S(ℓ) = 1 - 0.8 / (1 + (ℓ-2))^alpha
    
    At ℓ=2: S = 1 - 0.8 = 0.2 ✓
    As ℓ→∞: S → 1.0 ✓
    """
    alpha = float(params.get("alpha", 1.33))
    
    # Clamp alpha to physical range
    if not math.isfinite(alpha) or alpha < 0.1:
        alpha = 1.33
    if alpha > 5.0:
        alpha = 5.0
    
    x = float(ell) - 2.0
    if x <= 0:
        return 0.2  # Fixed quadrupole suppression
    
    # Power-law recovery
    suppression = 0.8 / (1.0 + x)**alpha
    S = 1.0 - suppression
    
    # Ensure valid output
    if not math.isfinite(S) or S <= 0:
        return 1e-12
    return S

def bounce_spectrum_EE(ell, params):
    """EE polarization modification (30% correlation with TT deviation)."""
    tt = bounce_spectrum(ell, params)
    ee = 1.0 - 0.3 * (1.0 - tt)
    if not math.isfinite(ee) or ee <= 0:
        return 1e-12
    return ee

def bounce_spectrum_TE(ell, params):
    """TE cross-correlation (geometric mean of TT and EE)."""
    tt = bounce_spectrum(ell, params)
    ee = bounce_spectrum_EE(ell, params)
    te = math.sqrt(max(1e-12, tt * ee))
    if not math.isfinite(te) or te <= 0:
        return 1e-12
    return te
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    p = get_default_params()
    print("Power-law model S(ℓ):")
    for e in [2, 3, 4, 5, 10, 20, 30]:
        print(f"  ℓ={e}: {bounce_spectrum(e, p):.4f}")
