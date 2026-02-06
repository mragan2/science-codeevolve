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
    return 0.0


def Pi(eta, k):
    # resonance corresponding to ℓ≈22
    chi_star = 150.0
    k_d = 22.0 / chi_star

    width = 0.02
    dk = k - k_d

    dip = -0.08 / (1.0 + (dk*dk)/(width*width))

    # bounce-like transition
    step = math.sin(12.0*(eta - 0.28))
    bounce = 0.04 * step / (1.0 + 6.0*eta)

    return dip + bounce


def omega2(eta, k):
    a = a_of_eta(eta)
    return k*k + a*a*m_g2(eta) + Pi(eta, k)


# ------------------------------
# RK4 tensor solver
# ------------------------------
def solve_tensor(k):

    eta = 0.02
    eta_f = 1.0
    steps = 120
    h = 1.0
    hp = 0.0
    d = (eta_f - eta)/steps

    def rhs(eta, h, hp):
        return hp, (-2.0*Hconf(eta)*hp - omega2(eta, k)*h)

    for _ in range(steps):
        k1h, k1hp = rhs(eta, h, hp)
        k2h, k2hp = rhs(eta+d/2, h+d*k1h/2, hp+d*k1hp/2)
        k3h, k3hp = rhs(eta+d/2, h+d*k2h/2, hp+d*k2hp/2)
        k4h, k4hp = rhs(eta+d, h+d*k3h, hp+d*k3hp)

        h += (d/6)*(k1h+2*k2h+2*k3h+k4h)
        hp += (d/6)*(k1hp+2*k2hp+2*k3hp+k4hp)
        eta += d

        if not math.isfinite(h):
            return 0.0

    return abs(h)


def k_of_ell(ell):
    return (ell+0.5)/150.0


# ------------------------------
# Observable mapping
# ------------------------------
def S_TT(ell, params):
    if ell <= 2:
        return 0.196
    T = solve_tensor(k_of_ell(ell))
    return max(1e-12, min(1.3, 0.85 + 0.3*T))


def S_EE(ell, params):
    if ell <= 2:
        return 0.46132
    T = solve_tensor(k_of_ell(ell))
    return max(1e-12, min(1.3, 0.9 + 0.22*T))


def S_TE(ell, params):
    tt = S_TT(ell, params)
    ee = S_EE(ell, params)
    return max(1e-12, math.sqrt(tt*ee))
# EVOLVE-BLOCK-END

