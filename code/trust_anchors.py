"""
Analytical Navier-Stokes solutions used as trust anchors.

A1: Lamb-Oseen vortex (diagnoses F2 - temporal drift)
A2: Stokes oscillating plate (diagnoses F1 - spectral bias)
A3: Burgers vortex (diagnoses F6 - conservation violation)
A4: Kovasznay flow (baseline control)
"""

import numpy as np


# ── A1: Lamb-Oseen vortex ──────────────────────────────────────────────────

class LambOseen:
    """Decaying vortex: v_theta(r,t) = Γ0/(2πr) * (1 - exp(-r²/(4νt)))"""

    def __init__(self, Gamma0=1.0, nu=0.01):
        self.Gamma0 = Gamma0
        self.nu = nu
        self.t0 = 1.0 / nu  # R0=1 => t0 = R0^2/nu = 100

    def v_theta(self, r, t):
        return (self.Gamma0 / (2 * np.pi * r)) * (1 - np.exp(-r**2 / (4 * self.nu * t)))

    def omega(self, r, t):
        """Vorticity: ω = (1/r) d(r v_θ)/dr = Γ0/(4πνt) exp(-r²/(4νt))"""
        return (self.Gamma0 / (4 * np.pi * self.nu * t)) * np.exp(-r**2 / (4 * self.nu * t))

    @property
    def domain(self):
        return {"r": (0.1, 5.0), "t": (self.t0, 50 * self.t0)}


# ── A2: Stokes oscillating plate ──────────────────────────────────────────

class StokesPlate:
    """u(z,t) = U0 exp(-z√(ω/2ν)) cos(ωt - z√(ω/2ν))"""

    def __init__(self, U0=1.0, nu=0.01, omega=1.0):
        self.U0 = U0
        self.nu = nu
        self.omega = omega

    def u_exact(self, z, t):
        k = np.sqrt(self.omega / (2 * self.nu))
        return self.U0 * np.exp(-z * k) * np.cos(self.omega * t - z * k)

    @property
    def domain(self):
        return {"z": (0.0, 5.0), "t": (0.0, 4 * np.pi / self.omega)}


# ── A3: Burgers vortex ────────────────────────────────────────────────────

class BurgersVortex:
    """Steady vortex with axial stretching.
    v_theta(r) = Γ0/(2πr) (1 - exp(-αr²/(2ν)))
    v_r(r) = -α/2 r,  v_z(z) = αz
    """

    def __init__(self, Gamma0=1.0, alpha=1.0, nu=0.01):
        self.Gamma0 = Gamma0
        self.alpha = alpha
        self.nu = nu

    def v_theta(self, r):
        return (self.Gamma0 / (2 * np.pi * r)) * (1 - np.exp(-self.alpha * r**2 / (2 * self.nu)))

    def v_r(self, r):
        return -self.alpha / 2 * r

    def v_z(self, z):
        return self.alpha * z

    def circulation(self, r):
        """Γ(r) = 2π r v_θ(r)"""
        return 2 * np.pi * r * self.v_theta(r)

    @property
    def domain(self):
        return {"r": (0.1, 5.0)}


# ── A4: Kovasznay flow ────────────────────────────────────────────────────

class KovasznayFlow:
    """Steady 2D flow at low Re behind a grid.
    λ = Re/2 - √(Re²/4 + 4π²)
    u = 1 - exp(λx)cos(2πy)
    v = λ/(2π) exp(λx) sin(2πy)
    p = (1 - exp(2λx))/2
    """

    def __init__(self, Re=20.0):
        self.Re = Re
        self.lam = Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)

    def u(self, x, y):
        return 1 - np.exp(self.lam * x) * np.cos(2 * np.pi * y)

    def v(self, x, y):
        return self.lam / (2 * np.pi) * np.exp(self.lam * x) * np.sin(2 * np.pi * y)

    def p(self, x, y):
        return (1 - np.exp(2 * self.lam * x)) / 2

    @property
    def domain(self):
        return {"x": (-0.5, 1.0), "y": (-0.5, 1.5)}
