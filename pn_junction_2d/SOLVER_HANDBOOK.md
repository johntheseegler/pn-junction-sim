# 2D PN Junction Drift-Diffusion Solver — Handbook & Limitations

This handbook explains how to configure, run, and post-process simulations using the modular Python solver in `pn_junction_2d/`, followed by a complete summary of the physical, geometric, and numerical limitations to keep in mind when interpreting results.

---

## 1. Repository Layout

| File | Role |
|---|---|
| `config.py` | Physical constants (`Q`, `KB`, `EPS0`, `UM`), `DeviceConfig`, and `SolverConfig` |
| `geometry.py` | `Curve` class, built-in curves (`planar`, `cosine`, `parabolic`, `circular_arc`), `min_transition_width`, `check_walls`, `net_doping` |
| `numerics.py` | Non-dimensional `Scaling`, stable `bernoulli(z)`, structured `Mesh`, FVM `divergence_operator` and `laplacian` |
| `solver.py` | `Device`, `State`, equilibrium Newton (`solve_equilibrium`), Gummel solver (`solve_bias`, `ramp`, `run_voltage_sweep`), 1D reference (`solve_equilibrium_1d`) |
| `postprocess.py` | Physical rescaling, electric field, Scharfetter-Gummel currents, depletion width, analytical references, and plotting (`plot_dashboard`, `plot_iv`) |
| `main.py` | CLI runner for the planar I–V sweep and cosine curvature study |
| `PN_junction_2D.ipynb` | Interactive notebook walkthrough (with an external-script parabolic example in Section 8) |

> **Units convention:** All physical inputs and outputs use **cm-based semiconductor units**: lengths in `cm` (use `UM = 1e-4` for micrometres), concentrations in `cm^-3`, potentials in `V`, electric fields in `V/cm`, current densities in `A/cm^2`, and integrated 2D terminal currents in `A/cm` (per unit depth).

---

## 2. Quick Start

### Option A: Run the built-in CLI studies
From `pn_junction_2d/`:
```bash
python main.py              # planar I-V sweep + cosine curvature study (saves to ./figures)
python main.py --iv         # planar I-V sweep only
python main.py --curvature  # cosine curvature study only
```

### Option B: Call the solver from a Python script or notebook cell
```python
import numpy as np
import matplotlib.pyplot as plt

from config import DeviceConfig, SolverConfig, UM
from geometry import cosine
from solver import Device, solve_equilibrium, solve_bias, ramp
from postprocess import electric_field, terminal_current, current_conservation_error, plot_dashboard

cfg = DeviceConfig()
scfg = SolverConfig()

# 1. Build a device with a junction curve
curve = cosine(x0=cfg.Lx / 2, A=0.25 * UM, Ly=cfg.Ly)
dev = Device(cfg, curve)  # w=None auto-selects the minimum admissible transition width

# 2. Solve equilibrium (V = 0 V), then ramp to target bias
eq = solve_equilibrium(dev, scfg)
st = ramp(dev, solve_bias(dev, 0.0, eq, scfg), V_target=-2.0, dV=scfg.dV_reverse, scfg=scfg)

# 3. Inspect diagnostics & plot
xc, yc, Ex, Ey, Emag = electric_field(dev, st)
print(f"|E|max = {Emag.max():.3e} V/cm, I = {terminal_current(dev, st):.3e} A/cm, "
      f"cons.err = {current_conservation_error(dev, st):.2e}")

plot_dashboard(dev, st)
plt.show()
```

---

## 3. Configuring the Device & Solver

Both `DeviceConfig` and `SolverConfig` are immutable dataclasses with a `.with_(**kw)` helper for creating modified copies.

### `DeviceConfig` (`config.py`)
| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `T` | `300.0` | `K` | Lattice temperature ($V_T = k_B T / q$) |
| `eps_r` | `11.7` | — | Relative permittivity (Silicon) |
| `ni` | `1.0e10` | `cm^-3` | Intrinsic carrier concentration |
| `mu_n`, `mu_p` | `1400.0`, `450.0` | `cm^2/(V s)` | Electron and hole mobilities |
| `tau_n`, `tau_p` | `1.0e-6`, `1.0e-6` | `s` | SRH electron and hole lifetimes |
| `NA`, `ND` | `1.0e16`, `1.0e16` | `cm^-3` | p-side (left) acceptor and n-side (right) donor doping |
| `Lx`, `Ly` | `4.0 * UM`, `2.0 * UM` | `cm` | Domain width ($x$) and height ($y$) |
| `Nx`, `Ny` | `201`, `81` | — | Grid nodes in $x$ and $y$ |

Example — finer grid and asymmetric doping:
```python
cfg = DeviceConfig().with_(NA=5e16, ND=1e16, Lx=5.0 * UM, Nx=251, Ny=101)
```

### `SolverConfig` (`config.py`)
| Parameter | Default | Meaning |
|---|---|---|
| `newton_tol`, `newton_max_iter`, `newton_clip` | `1e-10`, `50`, `2.0` | Equilibrium Newton stopping tolerance, max iterations, step clip (in $V_T$) |
| `gummel_max_iter`, `gummel_inner_newton`, `gummel_clip` | `200`, `2`, `2.0` | Gummel outer iterations, inner Poisson Newton steps, step clip (in $V_T$) |
| `tol_psi`, `tol_phi`, `tol_poisson_res` | `1e-6`, `1e-6`, `1e-8` | Convergence tolerances on $\Delta \tilde{\psi}$, $\Delta \tilde{\phi}_{n,p}$, and Poisson residual |
| `tol_phi_floor` | `1e-4` | Accepted LU round-off plateau for $\Delta\tilde{\phi}$ once $\tilde{\psi}$ has converged |
| `enable_srh` | `True` | Toggle SRH recombination-generation (`False` solves pure drift-diffusion with $R = 0$) |
| `dV_forward`, `dV_reverse`, `dV_min` | `0.05`, `-0.2`, `1e-3` | Initial voltage ramp steps and minimum adaptive step size (`V`) |

Example — disable SRH or tighten forward ramp step:
```python
scfg = SolverConfig().with_(enable_srh=False, dV_forward=0.025)
```

---

## 4. Choosing or Creating a Junction Curve

The doping profile is defined by a smooth `tanh` transition across a curve $x = f(y)$:
$$N_{\text{net}}(x, y) = \frac{N_D - N_A}{2} + \frac{N_D + N_A}{2} \tanh\!\left(\frac{x - f(y)}{w}\right)$$

### Built-in Curves (`geometry.py`)
- **`planar(x0)`** — vertical junction $x = x_0$ (default when `curve=None`).
- **`cosine(x0, A, Ly, m=1)`** — $x = x_0 + A \cos(2\pi m y / L_y)$. Has $f'(0) = f'(L_y) = 0$ (no mirror kink at walls).
- **`parabolic(x0, c, Ly)`** — $x = x_0 + c (y - L_y / 2)^2$. Meets walls at an angle unless $c = 0$.
- **`circular_arc(xc, yc, R)`** — $x = x_c - \sqrt{R^2 - (y - y_c)^2}$. Requires $|y - y_c| < R$ across $[0, L_y]$.

### Custom Curves
Provide `f(y)`, `df(y)`, and `d2f(y)` (all accepting a NumPy array `y` in `cm`) to `Curve`:
```python
import numpy as np
from geometry import Curve

def gaussian_bump(x0, A, sigma, Ly):
    yc = Ly / 2.0
    def f(y):
        u = (np.asarray(y, float) - yc) / sigma
        return x0 + A * np.exp(-(u**2))
    def df(y):
        u = (np.asarray(y, float) - yc) / sigma
        return -2.0 * A / sigma * u * np.exp(-(u**2))
    def d2f(y):
        u = (np.asarray(y, float) - yc) / sigma
        return 2.0 * A / sigma**2 * (2.0 * u**2 - 1.0) * np.exp(-(u**2))
    return Curve("gaussian_bump", f, df, d2f)

dev = Device(cfg, gaussian_bump(cfg.Lx / 2, 0.3 * UM, 0.3 * UM, cfg.Ly))
```

### Transition Width `w` and Checks
- Leave `w=None` in `Device(cfg, curve)` to automatically use `min_transition_width(cfg, curve)`:
  $$w_{\min} = 2.5 \max(\Delta x, \Delta y) \sqrt{1 + \max_y |f'(y)|^2}$$
- When comparing multiple curves at different curvatures, pass a **fixed** `w` larger than the largest $w_{\min}$ across your sweep so $w$-smoothing doesn't confound the curvature trend.
- Pass `check=False` to `Device(cfg, curve, check=False)` only if you deliberately want to bypass the `w_min` check and `check_walls` warning.

---

## 5. Running Solves & Voltage Sweeps

- **Equilibrium ($V = 0\text{ V}$):**
  ```python
  eq = solve_equilibrium(dev, scfg, verbose=True)
  ```
- **Single bias target (via adaptive continuation):**
  Always start from `solve_bias(dev, 0.0, eq, scfg)` and ramp with `ramp(...)` rather than jumping directly to a large bias:
  ```python
  s0 = solve_bias(dev, 0.0, eq, scfg)
  st = ramp(dev, s0, V_target=-2.5, dV=scfg.dV_reverse, scfg=scfg, verbose=True)
  ```
- **Full I–V sweep (forward and reverse):**
  `run_voltage_sweep` automatically sorts voltages outward from `0.0 V` in both directions and returns a `{V: State}` dict:
  ```python
  voltages = np.concatenate([np.arange(0.05, 0.651, 0.05), np.arange(-0.2, -3.001, -0.2)])
  states = run_voltage_sweep(dev, voltages, scfg=scfg)
  ```
  > **Bias sign convention:** $V$ is applied to the **left (p-side) contact** while the right (n-side) contact is grounded. Positive $V > 0$ is **forward bias**; negative $V < 0$ is **reverse bias**.

---

## 6. Post-Processing & Diagnostics (`postprocess.py`)

| Function | Returns / Effect |
|---|---|
| `physical(dev, st)` | `(psi [V], n [cm^-3], p [cm^-3])` reshaped to `(Ny, Nx)` |
| `electric_field(dev, st)` | `(xc, yc, Ex, Ey, \|E\|)` at cell centres, shape `(Ny-1, Nx-1)`, in `V/cm` |
| `terminal_current(dev, st)` | Mean integrated current $I$ across all $x$-columns `[A/cm]` |
| `current_conservation_error(dev, st)` | Relative max deviation of column currents from the mean (should be $\ll 10^{-6}$, typically $\sim 10^{-13}$ for $V \le 0.30\text{ V}$) |
| `space_charge(dev, st)` | $\rho / q = p - n + N_{\text{net}}$ in `cm^-3` |
| `depletion_width(dev, st, row=None)` | Width between $p = N_A/2$ and $n = N_D/2$ along a horizontal slice `[cm]` |
| `abrupt_depletion_width(cfg, V)` / `abrupt_peak_field(cfg, V)` | 1D analytical abrupt-junction references (with optional $2 V_T$ Debye-tail correction) |
| `ideality_factor(V, I, VT)` | Fitted diode ideality factor $n$ over `0.25–0.55 V` |
| `plot_dashboard(dev, st, title=None, fname=None)` | 2×2 figure: $\psi$, $\mathbf{E}$ quiver + magnitude, $\log_{10} n$, $\log_{10} p$ |
| `plot_iv(V, I, VT=None, fname=None)` | Semilog $|I|$ vs $V_{\text{app}}$ plot |

---

## 7. Limitations of the Solver

### A. Physical Model Limitations
1. **Steady-state only:** Solves static DC equilibrium/bias ($\partial n/\partial t = \partial p/\partial t = 0$). No transient switching, capacitance-voltage (small-signal AC), or optical generation terms.
2. **Non-degenerate Boltzmann statistics:** Uses $n = n_i e^{(\psi - \phi_n)/V_T}$ and $p = n_i e^{(\phi_p - \psi)/V_T}$. Becomes inaccurate at heavy doping ($N_A, N_D \gtrsim 10^{18}\text{–}10^{19}\text{ cm}^{-3}$) where Fermi-Dirac statistics and bandgap narrowing matter.
3. **Constant mobilities:** $\mu_n$ and $\mu_p$ are spatially uniform constants — no doping-dependent mobility (e.g., Caughey-Thomas), carrier-carrier scattering, or high-field velocity saturation.
4. **SRH recombination only:** Includes mid-gap Shockley-Read-Hall recombination-generation with constant lifetimes $\tau_n, \tau_p$. Does **not** model Auger recombination (important at high injection/doping), radiative recombination, or surface recombination.
5. **No breakdown or tunneling:** No avalanche multiplication (impact ionization) or Zener / band-to-band tunneling. Under large reverse bias, $|\mathbf{E}|_{\max}$ can exceed Silicon's critical breakdown field ($\sim 3\text{–}4 \times 10^5\text{ V/cm}$) without the solver producing breakdown current.
6. **Ideal ohmic contacts & uniform permittivity:** Left ($x=0$) and right ($x=L_x$) contacts are fixed Dirichlet boundaries at exact bulk charge neutrality (no contact resistance, Schottky barrier, or finite recombination velocity). $\varepsilon_r$ is uniform everywhere (no heterojunctions or oxide layers).
7. **Smooth $\tanh$ doping transition:** The junction has a finite transition scale $w$, which softens the peak electric field compared to a step/abrupt junction (e.g., $\sim 15\text{–}20\%$ lower at $w \approx 0.06\,\mu\text{m}$).

### B. Geometric & Boundary Limitations
1. **Single-valued curves $x = f(y)$ only:** The interface must be expressible as a single-valued function of $y$. It cannot represent closed cylindrical/spherical pockets, overhangs, multi-junctions, or sharp $90^\circ$ corners without custom modifications to `net_doping`.
2. **Neumann (mirror-symmetry) top/bottom walls:** The insulating walls at $y = 0$ and $y = L_y$ enforce $\partial \psi / \partial y = 0$ and $J_y = 0$, forcing potential and carrier contours to meet those walls perpendicularly ($90^\circ$). Any curve with $f'(0) \neq 0$ or $f'(L_y) \neq 0$ (such as `parabolic` or `circular_arc`) creates a **mirror-symmetry kink** at the walls, distorting carrier densities and electric fields near $y = 0$ and $y = L_y$.
3. **Minimum transition width couples geometry to grid resolution:** Because $w \ge 2.5 \max(\Delta x, \Delta y)\sqrt{1 + \max|f'|^2}$, steep curves force either a wider (more smeared) junction or a much finer grid.
4. **Domain width vs. depletion width:** The domain must be wide enough that the depletion region never reaches the Dirichlet contacts (recommended $L_x \ge 3 W_d(V_{\min})$, checked by `check_domain` in `main.py`). At large reverse bias or low doping, you must increase `Lx` (and `Nx` to keep $\Delta x$ small).

### C. Numerical & Solver Limitations
1. **Uniform Cartesian grid:** No local mesh refinement around the junction or high-curvature tips. Increasing `Nx` and `Ny` refines the entire domain uniformly.
2. **Direct sparse LU scaling:** `scipy.sparse.linalg.spsolve` is exact and robust for moderate grids (default $201 \times 81 \approx 16\text{k}$ nodes takes a few seconds per bias), but memory and runtime grow super-linearly for very fine grids (e.g., $> 10^5$ nodes).
3. **Decoupled Gummel convergence under high injection:** Gummel iteration converges rapidly in reverse and low-to-moderate forward bias (typically 5–10 iterations), but slows down or fails to converge (`ConvergenceError`) at strong forward bias ($V \gtrsim 0.65\text{–}0.75\text{ V}$ for default $10^{16}\text{ cm}^{-3}$ Silicon) where $n, p$ approach $N_A, N_D$ and Poisson–continuity coupling becomes stiff. A coupled Newton solver would be needed for deep forward bias.
4. **Boltzmann-reference threshold (`V <= 0.30`):** In `solver.py`, the $\delta n, \delta p$ cancellation-free formulation is enabled for $V \le 0.30\text{ V}$. Just above $0.30\text{ V}$ (e.g., $0.35\text{ V}$), current conservation error briefly rises from $\sim 10^{-13}$ to $\sim 10^{-6}$ before improving again as forward current grows.
5. **Hardcoded cosine curvature in `main.py`:** `curvature_study` in `main.py` computes `kappa = A * UM * k**2` assuming a cosine curve. For arbitrary curves, compute $\kappa_{\max}$ via `dev.curve.curvature(np.linspace(0, cfg.Ly, 2001)).max()`.

---

## 8. Troubleshooting

| Symptom / Message | Cause | Fix |
|---|---|---|
| `ValueError: w = ... < required ...` | Chosen `w` is too thin for the grid spacing or curve slope | Pass `w=None` (auto), increase `w`, or refine `Nx`/`Ny` |
| `UserWarning: \|f'\| = ... at y = 0 / Ly` | Curve meets a zero-flux Neumann wall at an angle (mirror kink) | Use a curve with $f'(0) = f'(L_y) = 0$ (e.g., `cosine`), or accept the wall kink (`check=False` silences it) |
| `ValueError: Lx must be >= 3 * W_d` | Depletion region is too wide for `Lx` at the target reverse bias | Increase `Lx` (and proportionally `Nx`) in `DeviceConfig` |
| `ConvergenceError: Gummel did not converge` | Step `dV` too large or high-injection forward bias | Reduce `dV_forward`/`dV_reverse`, increase `gummel_max_iter`, or stay below high-injection bias ($\lesssim 0.65\text{ V}$ for default doping) |
| Pale/white band in $\log_{10} n$ or $\log_{10} p$ at reverse bias | Physical carrier extraction ($n, p$ swept below $n_i^2/N_{A,D}$ in the depletion zone) | Expected physics under reverse bias; not a bug |
