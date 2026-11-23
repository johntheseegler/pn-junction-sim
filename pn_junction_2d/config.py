"""Physical constants, Silicon parameters, grid and solver settings.

All physical quantities use cm-based semiconductor units.
"""
from dataclasses import dataclass, replace

# --- Physical constants -------------------------------------------------------
Q = 1.602176634e-19        # elementary charge [C]
KB = 1.380649e-23          # Boltzmann constant [J/K]
EPS0 = 8.854187817e-14     # vacuum permittivity [F/cm]

UM = 1e-4                  # 1 micrometre in cm


@dataclass(frozen=True)
class DeviceConfig:
    """Material, doping and grid parameters (Silicon, 300 K defaults)."""
    T: float = 300.0             # [K]
    eps_r: float = 11.7          # relative permittivity
    ni: float = 1.0e10           # intrinsic concentration [cm^-3]
    mu_n: float = 1400.0         # [cm^2/(V s)]
    mu_p: float = 450.0          # [cm^2/(V s)]
    tau_n: float = 1.0e-6        # [s]
    tau_p: float = 1.0e-6        # [s]
    NA: float = 1.0e16           # acceptors, p-side (left) [cm^-3]
    ND: float = 1.0e16           # donors, n-side (right) [cm^-3]
    Lx: float = 4.0 * UM         # domain width [cm]
    Ly: float = 2.0 * UM         # domain height [cm]
    Nx: int = 201
    Ny: int = 81

    @property
    def eps(self) -> float:
        return self.eps_r * EPS0

    @property
    def VT(self) -> float:
        return KB * self.T / Q

    @property
    def dx(self) -> float:
        return self.Lx / (self.Nx - 1)

    @property
    def dy(self) -> float:
        return self.Ly / (self.Ny - 1)

    def with_(self, **kw) -> "DeviceConfig":
        return replace(self, **kw)


@dataclass(frozen=True)
class SolverConfig:
    """Nonlinear-solver controls (all tolerances act on scaled variables)."""
    # Equilibrium Newton
    newton_tol: float = 1e-10
    newton_max_iter: int = 50
    newton_clip: float = 2.0
    # Gummel
    gummel_max_iter: int = 200
    gummel_inner_newton: int = 2
    gummel_clip: float = 2.0
    tol_psi: float = 1e-6          # max |d psi|          (units of V_T)
    tol_phi: float = 1e-6          # max |d phi_n|, |d phi_p| (units of V_T)
    tol_phi_floor: float = 1e-4    # accepted round-off floor for phi once psi has converged
    tol_poisson_res: float = 1e-8  # max |Poisson residual| (scaled)
    enable_srh: bool = True
    # Voltage ramping
    dV_forward: float = 0.05
    dV_reverse: float = -0.2
    dV_min: float = 1e-3

    def with_(self, **kw) -> "SolverConfig":
        return replace(self, **kw)
