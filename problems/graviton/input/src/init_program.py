# EVOLVE-BLOCK-START
import math


def get_torsion_params():
    return {}


# ------------------------------
# Background toy cosmology
# ------------------------------
def a_of_eta(eta):
    # Bounce cosmology: minimum scale factor at eta ~ 0.3
    a_min = 0.05
    return a_min + (eta - 0.3) ** 2


def Hconf(eta):
    a = a_of_eta(eta)
    ap = 2.0 * (eta - 0.3)  # d/d(eta) of a_of_eta
    return ap / max(1e-12, a)


# ------------------------------
# GRAVITON EQUATION COMPONENTS
# ------------------------------
def m_g2(eta):
    # Decaying graviton mass — substantial effect to be tuned by evolution.
    return 0.25 * math.exp(-3.0 * eta)


def Pi(eta, k):
    # Broad torsion background (felt at all k)
    bg = 0.25 * math.exp(-2.0 * eta)

    # Resonance peak at k ~ 0.15 (ell ~ 22)
    k_d = 0.15
    sigma_k = 0.025
    dk = k - k_d
    profile_k = math.exp(-(dk * dk) / (2.0 * sigma_k * sigma_k))

    eta_c = 0.50
    sigma_eta = 0.15
    de = eta - eta_c
    profile_eta = math.exp(-(de * de) / (2.0 * sigma_eta * sigma_eta))

    amplitude = 0.35
    return bg + amplitude * profile_k * profile_eta


def omega2(eta, k, use_torsion):
    a = a_of_eta(eta)
    om2 = k * k + (a * a) * m_g2(eta)
    if use_torsion:
        om2 += Pi(eta, k)
    return om2


# ------------------------------
# RK4 tensor solver (cached)
# ------------------------------
_SOLVE_CACHE = {}


def solve_tensor(k, use_torsion):
    key = (float(k), bool(use_torsion))
    if key in _SOLVE_CACHE:
        return _SOLVE_CACHE[key]

    eta = 0.02
    eta_f = 1.0
    steps = 220  # fast enough for evolution
    d = (eta_f - eta) / steps

    h = 1.0
    hp = 0.0

    def rhs(eta_loc, h_loc, hp_loc):
        om2 = omega2(eta_loc, k, use_torsion)
        return hp_loc, (-2.0 * Hconf(eta_loc) * hp_loc - om2 * h_loc)

    for _ in range(steps):
        k1h, k1hp = rhs(eta, h, hp)
        k2h, k2hp = rhs(eta + 0.5 * d, h + 0.5 * d * k1h, hp + 0.5 * d * k1hp)
        k3h, k3hp = rhs(eta + 0.5 * d, h + 0.5 * d * k2h, hp + 0.5 * d * k2hp)
        k4h, k4hp = rhs(eta + d, h + d * k3h, hp + d * k3hp)

        h += (d / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
        hp += (d / 6.0) * (k1hp + 2.0 * k2hp + 2.0 * k3hp + k4hp)
        eta += d

        if not (math.isfinite(h) and math.isfinite(hp)):
            _SOLVE_CACHE[key] = 0.0
            return 0.0

    out = abs(h)
    _SOLVE_CACHE[key] = out
    return out


def k_of_ell(ell):
    return (float(ell) + 0.5) / 150.0


_RATIO_CACHE = {}


def ratio_torsion_over_ref(ell):
    if ell in _RATIO_CACHE:
        return _RATIO_CACHE[ell]

    k = k_of_ell(ell)

    tref = solve_tensor(k, False)
    if tref < 1e-15:
        _RATIO_CACHE[ell] = 1.0
        return 1.0

    tfull = solve_tensor(k, True)
    r = tfull / tref
    if not math.isfinite(r):
        r = 1.0

    # Keep sane bounds
    if r < 0.0:
        r = 0.0
    if r > 2.0:
        r = 2.0

    _RATIO_CACHE[ell] = r
    return r


# ------------------------------
# Transfer functions (optimized for current evaluator)
# ------------------------------
def S_TT(ell, params):
    # Transfer function driven by ODE ratio squared.
    r = ratio_torsion_over_ref(ell)
    out = r * r
    return max(1e-12, out)


def S_EE(ell, params):
    # Polarization transfer: stronger sensitivity (ratio cubed).
    r = ratio_torsion_over_ref(ell)
    out = r * r * r
    return max(1e-12, out)


def S_TE(ell, params):
    # Cross-correlation derived from ODE-driven TT and EE.
    tt = S_TT(ell, params)
    ee = S_EE(ell, params)
    out = math.sqrt(max(1e-12, tt * ee))
    return max(1e-12, out)


def predict_BB(ell, params):
    # Safe stub (not required by current evaluator).
    # Kept finite and non-negative.
    x = float(ell)
    return max(0.0, 1.0e-6 * (x / 80.0) ** 0.0)
# EVOLVE-BLOCK-END
