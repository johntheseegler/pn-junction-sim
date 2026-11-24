"""Rescaling, electric field, SG currents, analytical references and plots."""
import numpy as np
import matplotlib.pyplot as plt

from config import Q
from numerics import bernoulli
from solver import Device, State


# --- Fields and currents ----------------------------------------------------------
def physical(dev: Device, st: State):
    """Return psi [V], n, p [cm^-3] as (Ny, Nx) arrays."""
    sc = dev.sc
    return (dev.reshape(sc.unscale("psi", st.psi)), dev.reshape(sc.unscale("n", st.n)),
            dev.reshape(sc.unscale("p", st.p)))


def electric_field(dev: Device, st: State):
    """Face-centred FVM gradients averaged to cell centres.

    Returns (xc, yc, Ex, Ey, |E|) with fields in V/cm, shape (Ny-1, Nx-1).
    """
    m = dev.mesh
    psi = dev.reshape(st.psi) * dev.sc.VT
    Ex_f = -(psi[:, 1:] - psi[:, :-1]) / m.dx_phys        # (Ny, Nx-1)
    Ey_f = -(psi[1:, :] - psi[:-1, :]) / m.dy_phys        # (Ny-1, Nx)
    Ex = 0.5 * (Ex_f[1:, :] + Ex_f[:-1, :])
    Ey = 0.5 * (Ey_f[:, 1:] + Ey_f[:, :-1])
    xc, yc = 0.5 * (m.x[1:] + m.x[:-1]), 0.5 * (m.y[1:] + m.y[:-1])
    return xc, yc, Ex, Ey, np.hypot(Ex, Ey)


def face_currents_x(dev: Device, st: State):
    """SG current densities on x faces, J_n,x and J_p,x [A/cm^2], shape (Ny, Nx-1)."""
    psi = dev.reshape(st.psi)
    # Using dn, dp directly when available omits the zero-flux Boltzmann reference and avoids cancellation
    n = dev.reshape(st.dn if st.dn is not None else st.n)
    p = dev.reshape(st.dp if st.dp is not None else st.p)
    z = psi[:, 1:] - psi[:, :-1]
    h = dev.mesh.dx
    Jn = dev.mu_n / h * (bernoulli(z) * n[:, 1:] - bernoulli(-z) * n[:, :-1])
    Jp = -dev.mu_p / h * (bernoulli(-z) * p[:, 1:] - bernoulli(z) * p[:, :-1])
    return dev.sc.unscale("J", Jn), dev.sc.unscale("J", Jp)


def column_currents(dev: Device, st: State):
    """Total current I(x_{i+1/2}) per unit depth [A/cm], trapezoid weights in y."""
    Jn, Jp = face_currents_x(dev, st)
    return dev.mesh.wy_phys @ (Jn + Jp)


def terminal_current(dev: Device, st: State) -> float:
    return float(np.mean(column_currents(dev, st)))


def current_conservation_error(dev: Device, st: State) -> float:
    I = column_currents(dev, st)
    return float(np.max(np.abs(I - I.mean())) / max(abs(I.mean()), 1e-300))


def space_charge(dev: Device, st: State):
    """rho/q = p - n + N_net  [cm^-3], shape (Ny, Nx)."""
    _, n, p = physical(dev, st)
    return p - n + dev.Nnet_phys


def depletion_width(dev: Device, st: State, row: int = None):
    """W = distance between the points where p = NA/2 (p-side) and n = ND/2 (n-side) [cm].

    (An integrated-charge width would just restate E_max via Gauss's law, so it is not an
    independent check under tanh smoothing.)
    """
    row = dev.mesh.Ny // 2 if row is None else row
    _, n, p = physical(dev, st)
    x = dev.mesh.x
    n, p = n[row], p[row]
    i_n = np.argmax(n >= dev.cfg.ND / 2)             # first index with n >= ND/2
    xn = np.interp(dev.cfg.ND / 2, n[i_n - 1:i_n + 1], x[i_n - 1:i_n + 1])
    i_p = np.where(p >= dev.cfg.NA / 2)[0][-1]       # last index with p >= NA/2
    xp = np.interp(dev.cfg.NA / 2, p[i_p:i_p + 2][::-1], x[i_p:i_p + 2][::-1])
    return xn - xp


# --- Analytical abrupt-junction references --------------------------------------------
def built_in_potential(cfg) -> float:
    return cfg.VT * np.log(cfg.NA * cfg.ND / cfg.ni**2)


def abrupt_depletion_width(cfg, V: float = 0.0, tail_correction: bool = True) -> float:
    Veff = built_in_potential(cfg) - V - (2 * cfg.VT if tail_correction else 0.0)
    return np.sqrt(2 * cfg.eps * Veff / Q * (cfg.NA + cfg.ND) / (cfg.NA * cfg.ND))


def abrupt_peak_field(cfg, V: float = 0.0, tail_correction: bool = True) -> float:
    Veff = built_in_potential(cfg) - V - (2 * cfg.VT if tail_correction else 0.0)
    return 2 * Veff / abrupt_depletion_width(cfg, V, tail_correction)


def ideality_factor(V, I, VT, vmin=0.25, vmax=0.55):
    V, I = np.asarray(V), np.abs(np.asarray(I))
    sel = (V >= vmin) & (V <= vmax)
    slope = np.polyfit(V[sel], np.log(I[sel]), 1)[0]
    return 1.0 / (slope * VT)


# --- Plots ---------------------------------------------------------------------
def plot_dashboard(dev: Device, st: State, title: str = None, fname: str = None):
    m = dev.mesh
    um = 1e4
    psi, n, p = physical(dev, st)
    xc, yc, Ex, Ey, Emag = electric_field(dev, st)
    yy = np.linspace(0, dev.cfg.Ly, 400)
    fy = dev.curve.f(yy)
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    panels = [
        (ax[0, 0], m.x, m.y, psi, r"$\psi$ [V]", "viridis"),
        (ax[1, 0], m.x, m.y, np.log10(n), r"$\log_{10} n$ [cm$^{-3}$]", "Blues"),
        (ax[1, 1], m.x, m.y, np.log10(p), r"$\log_{10} p$ [cm$^{-3}$]", "Reds"),
    ]
    for a, x, y, Z, lab, cmap in panels:
        pc = a.pcolormesh(x * um, y * um, Z, shading="auto", cmap=cmap)
        fig.colorbar(pc, ax=a, label=lab)
        a.plot(fy * um, yy * um, "w--", lw=1.2)
        a.set_xlabel(r"$x$ [$\mu$m]"); a.set_ylabel(r"$y$ [$\mu$m]")
        a.set_aspect("equal")

    # Colored unit-vector quiver plot for E = (Ex, Ey)
    ax_E = ax[0, 1]
    nx_q = 34
    ny_q = max(4, round(nx_q * dev.cfg.Ly / dev.cfg.Lx))
    sx = max(1, round(len(xc) / nx_q))
    sy = max(1, round(len(yc) / ny_q))
    ix = slice(sx // 2, None, sx)
    iy = slice(sy // 2, None, sy)
    Xq, Yq = np.meshgrid(xc[ix] * um, yc[iy] * um)
    Ex_q, Ey_q, Em_q = Ex[iy, ix], Ey[iy, ix], Emag[iy, ix]
    Emax = max(float(Emag.max()), 1e-30)
    valid = Em_q > 1e-3 * Emax
    Ux = Ex_q[valid] / Em_q[valid]
    Uy = Ey_q[valid] / Em_q[valid]
    dx_q = (xc[1] - xc[0]) * sx * um
    dy_q = (yc[1] - yc[0]) * sy * um
    arrow_len = 0.75 * min(dx_q, dy_q)

    ax_E.set_facecolor("#11131a")
    ax_E.plot(fy * um, yy * um, "w--", lw=1.1, alpha=0.65, zorder=2)
    q = ax_E.quiver(
        Xq[valid], Yq[valid], Ux, Uy, Em_q[valid],
        cmap="plasma", angles="xy", scale_units="xy", scale=1.0 / arrow_len,
        pivot="mid", width=0.0042, headwidth=3.8, headlength=4.2, headaxislength=3.6,
        zorder=3,
    )
    q.set_clim(0.0, Emax)
    fig.colorbar(q, ax=ax_E, label=r"$|\mathbf{E}|$ [V/cm]")
    ax_E.set_xlabel(r"$x$ [$\mu$m]"); ax_E.set_ylabel(r"$y$ [$\mu$m]")
    ax_E.set_aspect("equal")
    ax_E.set_xlim(0, dev.cfg.Lx * um); ax_E.set_ylim(0, dev.cfg.Ly * um)

    fig.suptitle(title or f"{dev.curve.name}, $V_{{app}}$ = {st.V:+.2f} V")
    if fname:
        fig.savefig(fname, dpi=130)
    return fig


def plot_iv(V, I, VT=None, fname: str = None):
    V, I = np.asarray(V), np.asarray(I)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.semilogy(V, np.abs(I), "o-", ms=4)
    ax.set_xlabel(r"$V_{app}$ [V]"); ax.set_ylabel(r"$|I|$ [A/cm]")
    ax.grid(True, which="both", alpha=0.3)
    if VT is not None and np.sum(V > 0.3) >= 3:
        nid = ideality_factor(V, I, VT)
        ax.set_title(f"I-V (fitted ideality n = {nid:.3f}, 0.25-0.55 V)")
    if fname:
        fig.savefig(fname, dpi=130)
    return fig
