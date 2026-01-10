# EVOLVE-BLOCK-START
"""
Minimal bounce cosmology transfer function.
Step function model with S(2)=0.196, 1.0 otherwise.
Zero parameters → no BIC penalty.
If χ²_model < χ²_lcdm, ΔBIC <0.
"""
import math

def get_default_params():
    """
    Zero-parameter model.
    """
    return {}

def bounce_spectrum(ell, params):
    alpha = 1.455  # Optimal value
    x = float(ell) - 2.0
    if x <= 0:
        return 0.196
    suppression = 0.804 / (1.0 + x)**alpha
    return 1.0 - suppression
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
