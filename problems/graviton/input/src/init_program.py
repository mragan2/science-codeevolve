# EVOLVE-BLOCK-START
import math


def get_torsion_params():
    return {}


# ------------------------------
# Background toy cosmology
# ------------------------------
def a_of_eta(eta):
    return eta*eta + 0.15*eta


def Hconf(eta):
    a = a_of_eta(eta)
    ap = 2.0*eta + 0.15
    return ap / max(1e-12, a)


# ------------------------------
# GRAVITON EQUATION COMPONENTS
# ------------------------------
def m_g2(eta):
    # Small effective mass for high-ell suppression control
    return 0.0005


def Pi(eta, k):
    # Positive Gaussian barrier creating adiabatic suppression (dip) at ell~22
    # Target k for ell=22: k = 22.5/150 = 0.15
    k_d = 0.15
    
    # Narrow width for sharp resonance feature
    sigma_k = 0.008
    dk = k - k_d
    
    # Gaussian profile in k-space (positive for suppression)
    profile_k = math.exp(-dk*dk/(2.0*sigma_k*sigma_k))
    
    # Time modulation centered at eta=0.5 with moderate width
    eta_c = 0.5
    sigma_eta = 0.12
    profile_eta = math.exp(-(eta-eta_c)*(eta-eta_c)/(2.0*sigma_eta*sigma_eta))
    
    # Amplitude tuned for ~15-20% dip depth
    amplitude = 0.35
    
    return amplitude * profile_k * profile_eta


def omega2(eta, k, use_torsion=True):
    a = a_of_eta(eta)
    mg2 = m_g2(eta)
    if use_torsion:
        pi_val = Pi(eta, k)
    else:
        pi_val = 0.0
    return k*k + a*a*mg2 + pi_val


# ------------------------------
# RK4 tensor solver with dual mode
# ------------------------------
def solve_tensor(k, use_torsion=True):
    eta = 0.02
    eta_f = 1.0
    steps = 600  # Increased for numerical accuracy and smoothness
    h = 1.0
    hp = 0.0
    d = (eta_f - eta)/steps

    def rhs(eta, h, hp, torsion_flag):
        om2 = omega2(eta, k, torsion_flag)
        return hp, (-2.0*Hconf(eta)*hp - om2*h)

    for _ in range(steps):
        k1h, k1hp = rhs(eta, h, hp, use_torsion)
        k2h, k2hp = rhs(eta+d/2, h+d*k1h/2, hp+d*k1hp/2, use_torsion)
        k3h, k3hp = rhs(eta+d/2, h+d*k2h/2, hp+d*k2hp/2, use_torsion)
        k4h, k4hp = rhs(eta+d, h+d*k3h, hp+d*k3hp, use_torsion)

        h += (d/6)*(k1h+2*k2h+2*k3h+k4h)
        hp += (d/6)*(k1hp+2*k2hp+2*k3hp+k4hp)
        eta += d

        if not math.isfinite(h):
            return 0.0

    return abs(h)


def k_of_ell(ell):
    return (ell+0.5)/150.0


# ------------------------------
# Observable mapping via normalization
# ------------------------------
def S_TT(ell, params):
    k = k_of_ell(ell)
    
    # Compute ratio to isolate torsion effect
    T_full = solve_tensor(k, True)
    T_ref = solve_tensor(k, False)
    
    if T_ref < 1e-15:
        return 1.0
    
    # Ratio < 1 at resonance (ell~22) creates the dip
    ratio = T_full / T_ref
    
    # Linear mapping: baseline 1.0, dip to ~0.85 at resonance
    result = 0.75 + 0.25 * ratio
    
    return max(0.8, min(1.2, result))


def S_EE(ell, params):
    k = k_of_ell(ell)
    
    T_full = solve_tensor(k, True)
    T_ref = solve_tensor(k, False)
    
    if T_ref < 1e-15:
        return 1.0
    
    ratio = T_full / T_ref
    
    # EE has stronger response to torsion (deeper dip)
    result = 0.7 + 0.3 * ratio
    
    return max(0.75, min(1.25, result))


def S_TE(ell, params):
    tt = S_TT(ell, params)
    ee = S_EE(ell, params)
    
    # Geometric mean with slight decorrelation factor
    # Ensures TE < min(TT, EE) and positive
    base = math.sqrt(tt * ee)
    result = base * 0.99
    
    return max(1e-12, min(tt, ee, result))
# EVOLVE-BLOCK-END

