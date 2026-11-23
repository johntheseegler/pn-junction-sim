"""Script runner: planar I-V sweep, dashboards and curvature (field-crowding) study.

Usage:
    python main.py                # everything, figures into ./figures
    python main.py --iv           # planar I-V sweep only
    python main.py --curvature    # curvature study only
"""
import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DeviceConfig, SolverConfig, UM
from geometry import cosine
from postprocess import (abrupt_depletion_width, current_conservation_error, electric_field,
                         ideality_factor, plot_dashboard, plot_iv, terminal_current)
from solver import Device, ramp, run_voltage_sweep, solve_bias, solve_equilibrium

FIG = Path(__file__).resolve().parent / "figures"


def check_domain(cfg: DeviceConfig, V_min: float):
    W = abrupt_depletion_width(cfg, V_min)
    ok = cfg.Lx >= 3 * W
    print(f"Domain check at {V_min:+.2f} V: W_d = {W * 1e4:.3f} um, Lx/W_d = {cfg.Lx / W:.2f}"
          f" -> {'OK' if ok else 'TOO NARROW'}")
    if not ok:
        raise ValueError("Lx must be >= 3 * W_d at the most negative bias")


def iv_study(cfg=DeviceConfig(), scfg=SolverConfig()):
    V_fwd = np.round(np.arange(0.05, 0.651, 0.05), 3)
    V_rev = np.round(np.arange(-0.2, -3.001, -0.2), 3)
    check_domain(cfg, V_rev.min())
    dev = Device(cfg)
    print(f"Planar device: w = {dev.w * 1e4:.4f} um, w/dx = {dev.w / cfg.dx:.2f}, "
          f"w/W_d(0) = {dev.w / abrupt_depletion_width(cfg):.3f}")
    t = time.time()
    states = run_voltage_sweep(dev, np.concatenate([V_fwd, V_rev]), scfg=scfg)
    print(f"Sweep done in {time.time() - t:.1f} s")
    V = np.array(list(states))
    I = np.array([terminal_current(dev, s) for s in states.values()])
    print(f"{'V [V]':>8} {'I [A/cm]':>12} {'cons.err':>10} {'Gummel it':>9}")
    for v, i, s in zip(V, I, states.values()):
        print(f"{v:8.2f} {i:12.4e} {current_conservation_error(dev, s):10.2e} {s.iterations:9d}")
    print(f"Fitted ideality factor (0.25-0.55 V): {ideality_factor(V, I, cfg.VT):.3f}")
    FIG.mkdir(exist_ok=True)
    plot_iv(V, I, cfg.VT, FIG / "iv_planar.png")
    for v in (0.0, 0.6, -3.0):
        plot_dashboard(dev, states[v], fname=FIG / f"dashboard_planar_{v:+.1f}V.png")
    plt.close("all")
    return dev, states


def curvature_study(cfg=DeviceConfig(), scfg=SolverConfig(), V=-2.0,
                    amplitudes=(0.0, 0.1, 0.25, 0.4), w=0.11 * UM):
    check_domain(cfg, V)
    lam = cfg.Ly
    k = 2 * np.pi / lam
    rows = []
    FIG.mkdir(exist_ok=True)
    for A in amplitudes:
        dev = Device(cfg, cosine(cfg.Lx / 2, A * UM, cfg.Ly), w=w)
        eq = solve_equilibrium(dev, scfg)
        st = ramp(dev, solve_bias(dev, 0.0, eq, scfg), V, scfg.dV_reverse, scfg)
        xc, yc, Ex, Ey, E = electric_field(dev, st)
        j, i = np.unravel_index(np.argmax(E), E.shape)
        kappa = A * UM * k**2                      # 1/cm, at the cosine extrema
        rows.append((A, kappa, E.max(), xc[i], yc[j], terminal_current(dev, st)))
        print(f"A = {A:4.2f} um: kappa_max = {kappa * 1e-4:6.3f} 1/um "
              f"(Rc = {1 / kappa * 1e4 if kappa else np.inf:6.3f} um), "
              f"|E|max = {E.max():.4e} V/cm at (x, y) = ({xc[i] * 1e4:.3f}, {yc[j] * 1e4:.3f}) um,"
              f" I = {rows[-1][-1]:.3e} A/cm")
        plot_dashboard(dev, st, fname=FIG / f"dashboard_cos_A{A:.2f}_{V:+.1f}V.png")
        plt.close("all")
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    ax.plot(rows[:, 1] * 1e-4, rows[:, 2], "o-")
    for A, kap, Em in rows[:, :3]:
        ax.annotate(f"A={A:g}", (kap * 1e-4, Em), textcoords="offset points", xytext=(4, -10))
    ax.set_xlabel(r"$\kappa_{max}$ [1/$\mu$m]"); ax.set_ylabel(r"$|\mathbf{E}|_{max}$ [V/cm]")
    ax.set_title(f"Field crowding vs curvature at {V:+.1f} V (w = {w * 1e4:g} um)")
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "emax_vs_curvature.png", dpi=130)
    plt.close(fig)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv", action="store_true")
    ap.add_argument("--curvature", action="store_true")
    a = ap.parse_args()
    run_all = not (a.iv or a.curvature)
    if a.iv or run_all:
        iv_study()
    if a.curvature or run_all:
        curvature_study()
