"""Non-dimensionalisation, Bernoulli function and 2D FVM stencil assembly."""
import numpy as np
import scipy.sparse as sp

from config import Q, DeviceConfig


# --- Scaling ------------------------------------------------------------------
class Scaling:
    """Reference scales: V_T, C0, L_D, mu0, t0 and derived J0, R0, E0."""

    def __init__(self, cfg: DeviceConfig):
        self.VT = cfg.VT
        self.C0 = max(cfg.NA, cfg.ND)
        self.LD = np.sqrt(cfg.eps * self.VT / (Q * self.C0))
        self.mu0 = cfg.mu_n
        self.t0 = self.LD**2 / (self.mu0 * self.VT)
        self.J0 = Q * self.C0 * self.mu0 * self.VT / self.LD     # [A/cm^2]
        self.R0 = self.C0 / self.t0                               # [cm^-3 s^-1]
        self.E0 = self.VT / self.LD                               # [V/cm]
        self._f = {
            "psi": self.VT, "V": self.VT, "phi": self.VT,
            "n": self.C0, "p": self.C0, "N": self.C0, "ni": self.C0,
            "x": self.LD, "y": self.LD, "L": self.LD,
            "mu": self.mu0, "tau": self.t0, "t": self.t0,
            "J": self.J0, "R": self.R0, "E": self.E0,
        }

    def scale(self, kind: str, value):
        return np.asarray(value) / self._f[kind] if np.ndim(value) else value / self._f[kind]

    def unscale(self, kind: str, value):
        return np.asarray(value) * self._f[kind] if np.ndim(value) else value * self._f[kind]


# --- Bernoulli function -------------------------------------------------------
def bernoulli(z):
    """B(z) = z / (exp(z) - 1), evaluated stably for all real z."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-4
    big = z > 500.0
    neg = z < -500.0
    mid = ~(small | big | neg)
    zs = z[small]
    out[small] = 1.0 - zs / 2.0 + zs**2 / 12.0 - zs**4 / 720.0
    out[big] = 0.0
    out[neg] = -z[neg]
    zm = z[mid]
    out[mid] = zm / np.expm1(zm)
    return out if out.ndim else float(out)


# --- Structured mesh in scaled units ----------------------------------------------
class Mesh:
    """Node-centred structured grid. Arrays are shaped (Ny, Nx); flat index k = j*Nx + i."""

    def __init__(self, cfg: DeviceConfig, sc: Scaling):
        self.Nx, self.Ny = cfg.Nx, cfg.Ny
        self.N = cfg.Nx * cfg.Ny
        self.x = np.linspace(0.0, cfg.Lx, cfg.Nx)          # physical [cm]
        self.y = np.linspace(0.0, cfg.Ly, cfg.Ny)
        self.dx_phys, self.dy_phys = cfg.dx, cfg.dy
        self.dx = cfg.dx / sc.LD                             # scaled
        self.dy = cfg.dy / sc.LD
        self.X, self.Y = np.meshgrid(self.x, self.y)         # (Ny, Nx)
        # y control-volume heights (half cells on the Neumann walls), scaled
        hy = np.full(self.Ny, self.dy)
        hy[0] = hy[-1] = 0.5 * self.dy
        self.hy = hy
        self.wy_phys = hy * sc.LD                            # trapezoid weights [cm]
        dmask = np.zeros((self.Ny, self.Nx), dtype=bool)
        dmask[:, 0] = dmask[:, -1] = True
        self.dirichlet = dmask.ravel()
        self.interior = ~self.dirichlet
        self.left = np.zeros_like(dmask); self.left[:, 0] = True
        self.right = np.zeros_like(dmask); self.right[:, -1] = True
        self.left, self.right = self.left.ravel(), self.right.ravel()

        k = np.arange(self.N).reshape(self.Ny, self.Nx)
        self.kx_a, self.kx_b = k[:, :-1].ravel(), k[:, 1:].ravel()   # x faces a->b
        self.ky_a, self.ky_b = k[:-1, :].ravel(), k[1:, :].ravel()   # y faces a->b
        # 1/(face-normal spacing * control-volume extent) for each face endpoint
        self.cy_a = 1.0 / (self.dy * np.repeat(hy[:-1], self.Nx))
        self.cy_b = 1.0 / (self.dy * np.repeat(hy[1:], self.Nx))
        self.cx = 1.0 / self.dx**2

    def face_dpsi(self, psi):
        """Potential drops psi_b - psi_a across x faces and y faces."""
        return psi[self.kx_b] - psi[self.kx_a], psi[self.ky_b] - psi[self.ky_a]


def divergence_operator(mesh: Mesh, ax, bx, ay, by) -> sp.csr_matrix:
    """FVM divergence of a two-point flux F_ab = (a*u_b - b*u_a)/h.

    Returns D with (D u)_k = (1/|Omega_k|) * sum_faces F * |face|, using half
    control volumes on the j = 0, Ny-1 walls and zero flux through those walls.
    """
    cx, cya, cyb = mesh.cx, mesh.cy_a, mesh.cy_b
    ia, ib, ja, jb = mesh.kx_a, mesh.kx_b, mesh.ky_a, mesh.ky_b
    rows = np.concatenate([ia, ia, ib, ib, ja, ja, jb, jb])
    cols = np.concatenate([ib, ia, ib, ia, jb, ja, jb, ja])
    data = np.concatenate([
        cx * ax, -cx * bx, -cx * ax, cx * bx,
        cya * ay, -cya * by, -cyb * ay, cyb * by,
    ])
    return sp.csr_matrix((data, (rows, cols)), shape=(mesh.N, mesh.N))


def apply_dirichlet_rows(A: sp.spmatrix, mask: np.ndarray) -> sp.csc_matrix:
    """Replace the rows flagged by `mask` with identity rows."""
    keep = sp.diags((~mask).astype(float))
    return (keep @ A + sp.diags(mask.astype(float))).tocsc()


def laplacian(mesh: Mesh) -> sp.csr_matrix:
    ones_x = np.ones(mesh.kx_a.size)
    ones_y = np.ones(mesh.ky_a.size)
    return divergence_operator(mesh, ones_x, ones_x, ones_y, ones_y)
