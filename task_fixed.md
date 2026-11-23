# Implementation Plan: 2D Curved PN Junction Drift-Diffusion Solver

## Objective & Tech Stack

Build a modular, from-scratch 2D steady-state semiconductor solver for a Silicon PN junction with a user-configurable junction curve $x = f(y)$ under applied bias.

* **Method:** Finite Volume Method (FVM) on a structured 2D Cartesian grid + Scharfetter-Gummel flux discretization + Gummel's decoupled iteration.
* **Stack:** Python 3.10+, `numpy`, `scipy` (`scipy.sparse`, `scipy.sparse.linalg.spsolve`), `matplotlib`, `pytest`.

---

## Target Directory Structure

```text
pn_junction_2d/
├── PN_junction_2D.ipynb # Self-contained notebook: theory + full solver + all studies
├── config.py          # Physical constants, Si parameters, grid & solver settings
├── geometry.py        # 2D mesh, curve functions f(y), smooth tanh doping profile
├── numerics.py        # Non-dimensionalization, Bernoulli B(z), 2D FVM stencils
├── solver.py          # Equilibrium Newton-Raphson, SG continuity, Gummel loop
├── postprocess.py     # Rescaling, E-field, current integration, heatmaps & I-V plots
├── main.py            # CLI/script runner for bias sweeps and curvature studies
├── requirements.txt
└── tests/
    ├── conftest.py           # puts pn_junction_2d/ on sys.path
    ├── test_numerics.py      # Bernoulli limits, scaling round-trip
    ├── test_equilibrium.py   # 1D planar analytical benchmark (A = 0)
    ├── test_continuity.py    # SG matrix sign pattern, Boltzmann recovery at V = 0
    └── test_bias.py          # Gummel convergence, current conservation, I-V shape

```

---

## Phase 1: Configuration, Scaling & Geometry (`config.py`, `geometry.py`, `numerics.py`)

### Tasks

1. **Define physical parameters (`config.py`):** Use `cm`-based semiconductor units for Silicon at $T = 300\text{ K}$:
* $q = 1.602176634 \times 10^{-19}\text{ C}$, $k_B = 1.380649 \times 10^{-23}\text{ J/K}$, $\varepsilon_0 = 8.854187817 \times 10^{-14}\text{ F/cm}$.
* $\varepsilon_r = 11.7$, $n_i = 1.0 \times 10^{10}\text{ cm}^{-3}$, $\mu_n = 1400\text{ cm}^2/(\text{V}\cdot\text{s})$, $\mu_p = 450\text{ cm}^2/(\text{V}\cdot\text{s})$, $\tau_n = \tau_p = 10^{-6}\text{ s}$.
* Default domain: $L_x=4.0\,\mu\text{m}$, $L_y=2.0\,\mu\text{m}$, grid $N_x=201$, $N_y=81$, symmetric doping $N_A=N_D=10^{16}\,\text{cm}^{-3}$. This gives $\Delta x=0.02\,\mu\text{m}$ and leaves adequate margin for the default $-3$ V reverse-bias design sweep.


2. **Implement non-dimensionalization (`numerics.py`):**
* Reference concentration: $C_0 = \max(N_A, N_D)$.
* Thermal voltage: $V_T = k_B T / q$.
* Extrinsic Debye length: $L_D = \sqrt{\varepsilon V_T / (q C_0)}$.
* Reference mobility & time: $\mu_0 = \mu_n$, $t_0 = L_D^2 / (\mu_0 V_T)$.
* Provide `scale()` and `unscale()` helpers for $\psi, n, p, x, y, \mu, \tau, J, R$, with $\tilde R = R\,t_0/C_0$ and $\tilde n_i = n_i/C_0$.


3. **Implement variable junction geometry (`geometry.py`):**
* Support pluggable curve functions $x = f(y)$ (each returning $f$, $f'$, $f''$): planar (`f(y) = x0`), cosine (`x0 + A * cos(2*pi*m*y/Ly)`, chosen so $f'(0)=f'(L_y)=0$ is compatible with the mirror-symmetric Neumann walls), parabolic (`x0 + c*(y - Ly/2)**2`), and circular arc ($x_c-\sqrt{R^2-(y-y_c)^2}$). Warn if $|f'|$ at $y=0$ or $y=L_y$ is not small, because then the junction hits the zero-flux walls at an angle.
* Generate $N_{\text{net}}(x,y)$ via a smooth hyperbolic tangent transition to avoid staircasing:

$$N_{\text{net}}(x,y) = \frac{N_D - N_A}{2} + \frac{N_D + N_A}{2} \tanh\left(\frac{x - f(y)}{w}\right)$$


* The tanh is taken along $x$, so the width measured normal to the junction is $w/\sqrt{1+f'^2}$. Enforce $w\ge2.5\max(\Delta x,\Delta y)\sqrt{1+\max f'^2}$; also report $w/W_d$ so the user can verify that the doping smoothing remains small compared with the depletion width.


4. **Implement numerically stable Bernoulli function (`numerics.py`):**
```python
def bernoulli(z: np.ndarray) -> np.ndarray:
    # Handle |z| < 1e-4 via Taylor series: 1 - z/2 + z^2/12 - z^4/720
    # Handle z > 500 -> 0.0 (prevent exp overflow)
    # Handle z < -500 -> -z
    # Otherwise: z / np.expm1(z)

```



### Verification Gate

* `pytest tests/test_numerics.py`: Verify `bernoulli(0)==1.0`, `bernoulli(z)-bernoulli(-z)==-z` to `1e-12*max(1,|z|)` across `z in [-1000,1000]`, zero `NaN`/`Inf` outputs, positivity of $B$, and the `scale()`/`unscale()` round-trip. (The SG matrix sign-pattern test belongs to Phase 3.)

---

## Phase 2: Thermal Equilibrium Solver (`solver.py`, `tests/test_equilibrium.py`)

### Tasks

1. **Boundary conditions:**
* Left ($i = 0$, p-contact) and Right ($i = N_x - 1$, n-contact): Dirichlet BCs set from exact charge neutrality:

$$p_p = \frac{N_A}{2} + \sqrt{\left(\frac{N_A}{2}\right)^2 + n_i^2}, \quad \tilde{\psi}_L = -\ln(p_p / n_i)$$


$$n_n = \frac{N_D}{2} + \sqrt{\left(\frac{N_D}{2}\right)^2 + n_i^2}, \quad \tilde{\psi}_R = +\ln(n_n / n_i)$$


* Top ($j = N_y - 1$) and Bottom ($j = 0$): Homogeneous Neumann BCs ($\frac{\partial \tilde{\psi}}{\partial y} = 0$, implemented via zero-flux faces in FVM). Boundary-row control volumes have height $\Delta y/2$: drop the outer flux **and** divide the remaining $y$-flux by $\Delta y/2$ (equivalently, a mirror ghost node). The same rule applies to Poisson and to both continuity operators.


2. **Newton-Raphson nonlinear Poisson solver:**
* Initial guess: charge-neutral potential $\tilde{\psi}^{(0)} = \text{arcsinh}(\tilde{N}_{\text{net}} / (2\tilde{n}_i))$.
* Residual on interior nodes: $F(\tilde{\psi}) = L\tilde{\psi} - \tilde{n}_i e^{\tilde{\psi}} + \tilde{n}_i e^{-\tilde{\psi}} + \tilde{N}_{\text{net}} = 0$, where $L$ is the sparse 2D 5-point Laplacian (with Dirichlet rows set to identity and residual $F_k=\tilde\psi_k-\tilde\psi_{BC,k}$, so the Newton RHS is $-F_k$; this enforces the prescribed nonzero contact potential rather than accidentally forcing it to zero).
* Jacobian: $J = L - \text{diag}\left(\tilde{n}_i (e^{\tilde{\psi}} + e^{-\tilde{\psi}})\right)$.
* Solve $J \Delta \tilde{\psi} = -F$ with `spsolve`; apply update clamping (`np.clip(dpsi, -2.0, 2.0)`) if step is large; stop when `max(abs(dpsi))` is below `1e-10`.



### Verification Gate

* `pytest tests/test_equilibrium.py`: Run with a planar junction ($A = 0$) and verify:
1. Convergence in fewer than 15 Newton iterations.
2. Total built-in potential $\psi_R - \psi_L$ matches $V_T \ln(N_A N_D / n_i^2)$ within 0.1% (this is a sanity check of the contact BCs only).
3. 1D slices along different $y$ rows agree to within `1e-10` (scaled potential).
4. The 2D centre row matches an independent fine-grid 1D Newton reference solution (same tanh doping, 4× finer grid) within 1% of $V_{bi}$ in potential.
5. Peak electric field $|E_{\max}|$ and depletion width $W_d$ approach the analytical abrupt-junction formulas, using the Debye-tail-corrected $V_{bi}-V-2V_T$, as $w$ is reduced while remaining resolved. Report the finite-$w$ discrepancy rather than treating the smoothed junction as exactly abrupt.
6. The net integrated space charge is approximately zero.



---

## Phase 3: Scharfetter-Gummel Continuity Solvers (`solver.py`)

### Tasks

1. **Electron continuity matrix assembly (`assemble_electron_matrix`):**
* Discretize $\tilde{\nabla}\cdot\tilde{\mathbf J}_n=\tilde R$ using FVM face fluxes, but assemble the matrix as $A_n=-\tilde\nabla_h\cdot\tilde{\mathbf J}_n$ so that the matrix has positive diagonal and non-positive off-diagonal entries:

$$\tilde{J}_{n, i+1/2, j} = \frac{\tilde{\mu}_n}{\Delta \tilde{x}} \Big[ B(\Delta \tilde{\psi}_{x+}) \tilde{n}_{i+1,j} - B(-\Delta \tilde{\psi}_{x+}) \tilde{n}_{i,j} \Big]$$



where $\Delta \tilde{\psi}_{x+} = \tilde{\psi}_{i+1,j} - \tilde{\psi}_{i,j}$ (and analogous expressions for $x-, y+, y-$ faces).
* Enforce Dirichlet BCs at $i = 0$ and $i = N_x - 1$ (fixed equilibrium bulk $n$ values) and zero-flux Neumann BCs at $j = 0$ and $j = N_y - 1$ (omit south flux at $j = 0$ and north flux at $j = N_y - 1$).
* Assemble in `scipy.sparse.csc_matrix` using vectorized index arrays (no Python double loops over `(i, j)`).


2. **Hole continuity matrix assembly (`assemble_hole_matrix`):**
* Discretize $\tilde{\nabla}\cdot\tilde{\mathbf J}_p=-\tilde R$ and assemble $A_p=+\tilde\nabla_h\cdot\tilde{\mathbf J}_p$ so that it has the same positive-diagonal, non-positive-off-diagonal matrix convention as $A_n$:

$$\tilde{J}_{p, i+1/2, j} = -\frac{\tilde{\mu}_p}{\Delta \tilde{x}} \Big[ B(-\Delta \tilde{\psi}_{x+}) \tilde{p}_{i+1,j} - B(\Delta \tilde{\psi}_{x+}) \tilde{p}_{i,j} \Big]$$


* Apply identical Dirichlet (left/right) and zero-flux Neumann (top/bottom) boundary conditions. Solve the hole system as $A_p\tilde p=-\tilde R$ under the sign convention defined above.


3. **SRH Recombination (`compute_srh`):**

$$\tilde{R} = \frac{\tilde{n}\tilde{p} - \tilde{n}_i^2}{\tilde{\tau}_p(\tilde{n} + \tilde{n}_i) + \tilde{\tau}_n(\tilde{p} + \tilde{n}_i)}$$


* Include a boolean flag `enable_srh=True` so tests can verify the $R = 0$ case first.
* **Semi-implicit SRH (positivity):** write $R=(np-n_i^2)/D$ with $D$ lagged. In the electron solve, move $p/D$ onto the diagonal: $(A_n+\text{diag}(\tilde p/D))\,\tilde n=\tilde n_i^2/D$; in the hole solve, $(A_p+\text{diag}(\tilde n/D))\,\tilde p=\tilde n_i^2/D$. This keeps both systems M-matrices with non-negative right-hand sides, so carriers remain positive. Each node's control-volume weight (including half cells at the Neumann walls) must be applied consistently to the operator and to the source.



### Verification Gate

* `pytest tests/test_continuity.py`:
  1. Both assembled continuity operators have a positive diagonal and non-positive off-diagonal entries on random and on equilibrium potentials.
  2. At $V_{\text{app}} = 0$, feed the converged equilibrium $\tilde{\psi}_0$ into both continuity solvers with $R = 0$: verify that the solved $\tilde n,\tilde p$ match the Boltzmann concentrations $\tilde n_i e^{\pm\tilde\psi_0}$ (maximum relative error $<10^{-6}$; record the achieved value) and that all values are strictly positive.

---

## Phase 4: Gummel Iteration & Voltage Ramping (`solver.py`)

### Tasks

1. **Stabilized Poisson update (Gummel Step A):**
* Update $\tilde{\psi}$ with fixed quasi-Fermi levels by solving:

$$\left(L - \text{diag}(\tilde{n} + \tilde{p})\right) \Delta \tilde{\psi} = -\left(L\tilde{\psi} - \tilde{n} + \tilde{p} + \tilde{N}_{\text{net}}\right)$$


* Apply optional under-relaxation $\tilde{\psi} \leftarrow \tilde{\psi} + \omega \Delta \tilde{\psi}$ ($\omega \in (0, 1]$, default `1.0`, reduced to `0.5` if residual grows).


2. **Gummel outer loop (`solve_bias`):**
* Update left contact Dirichlet BC: $\tilde{\psi}_L = -\ln(p_p / n_i) + V_{\text{app}} / V_T$.
* Loop up to `max_gummel_iter = 200`:
1. Run 1–3 damped Newton iterations for $\tilde{\psi}$ (Step A). Between inner iterations keep the quasi-Fermi levels fixed by updating $\tilde n\leftarrow\tilde n\,e^{\Delta\tilde\psi}$ and $\tilde p\leftarrow\tilde p\,e^{-\Delta\tilde\psi}$.
2. Compute $\tilde{R}(\tilde{n}, \tilde{p})$ (or 0 if disabled).
3. Solve $A_n(\tilde{\psi}) \tilde{n} = -\tilde{R}$ (semi-implicit SRH form; enforce Dirichlet carrier values on contacts).
4. Solve $A_p(\tilde{\psi}) \tilde{p} = -\tilde{R}$ under the matrix sign convention defined in Phase 3.
5. Check the update norms in quasi-Fermi form so minority carriers count: `max(abs(dpsi)) < 1e-6`, $\max|\Delta\tilde\phi_n|<10^{-6}$ and $\max|\Delta\tilde\phi_p|<10^{-6}$, where $\tilde\phi_n=\tilde\psi-\ln(\tilde n/\tilde n_i)$ and $\tilde\phi_p=\tilde\psi+\ln(\tilde p/\tilde n_i)$. Also require the scaled Poisson residual $\|F\|_\infty<10^{-8}$.




3. **Voltage ramping (`run_voltage_sweep`):**
* Start from $V_{\text{app}} = 0$ equilibrium solution.
* Ramp forward (e.g., `0.0 -> +0.65 V` in `0.05 V` steps) and reverse (e.g., `0.0 -> -3.0 V` in `-0.2 V` steps), warm-starting each step from the previous converged $(\tilde\psi,\tilde n,\tilde p)$. Before the reverse sweep, verify the depletion-width/domain-margin condition for the requested maximum bias.
* If a step fails to converge, automatically halve the voltage step size and retry, down to a minimum step of `1e-3 V` (then raise an error).



### Verification Gate

* `pytest tests/test_bias.py`:
* Verify total current $I(x) = \int_0^{L_y} (J_{n,x} + J_{p,x}) \, dy$ is conserved across every vertical column $i$ of the grid to within 0.1%. Integrate with trapezoid weights ($\Delta y/2$ on the two wall rows) to match the FVM control volumes.
* Verify that $I\approx0$ at $V_{\text{app}}=0$ when solved through the Gummel path.
* Verify the planar $I\text{–}V$ curve exhibits exponential forward-bias turn-on over the range where the model is expected to be diffusion/recombination dominated. Report the fitted ideality factor rather than enforcing $1\le n\le2$ as a hard pass/fail criterion. In reverse bias, check for approximately bias-independent current at moderate reverse bias; allow SRH depletion-region generation to produce bias dependence at larger reverse voltages.

---

## Phase 5: Post-Processing, Heatmaps & Curvature Study (`postprocess.py`, `main.py`)

### Tasks

1. **Field & current extraction (`postprocess.py`):**
* Compute quantitative $E_x,E_y$ from the FVM face-centered potential gradients, then average both to cell centres (the corners of the node grid) before forming $|\mathbf E|=\sqrt{E_x^2+E_y^2}$. `np.gradient` may be used for visualization only.
* Compute face-centered current densities $J_{n,x}, J_{p,x}$ using the exact Scharfetter-Gummel formula (do **not** use finite differences on $n, p$ for post-processing current).


2. **Visualization suite (`postprocess.py`):**
* 2x2 heatmap dashboard at any bias $V_{\text{app}}$:
1. Electrostatic potential $\psi(x,y)$ (V) with junction curve $x = f(y)$ overlaid.
2. Electric field magnitude $\vert{}\mathbf{E}(x,y)\vert{}$ (V/cm) with `streamplot` field lines.
3. Electron concentration $\log_{10} n(x,y)$ ($\text{cm}^{-3}$).
4. Hole concentration $\log_{10} p(x,y)$ ($\text{cm}^{-3}$).


* Semi-log $I\text{–}V$ curve ($\log_{10}\vert{}I\vert{}$ vs. $V_{\text{app}}$).


3. **Curvature parameter sweep (`main.py`):**
* Use a fixed wavelength $\lambda=L_y$ and define $f(y)=x_0+A\cos(2\pi y/\lambda)$ (cosine so that $f'=0$ on the Neumann walls), with $\kappa_{\max}=A(2\pi/\lambda)^2$ at the extrema.
* Sweep $A$ (e.g. $0,0.1,0.25,0.4\,\mu$m) at a fixed reverse bias (e.g. $-2$ V).
* Compute $\kappa(y)=|f''(y)|/[1+(f'(y))^2]^{3/2}$ and use $\kappa_{\max}=1/R_{c,\min}$ as the curvature metric.
* Plot $|\mathbf E|_{\max}$ versus $\kappa_{\max}$ and separately record the location of the field maximum.



---

## Critical Guardrails for the Coding Agent

* **No raw SI units in linear algebra:** All matrix assembly and solves must happen strictly in dimensionless variables.
* **Vectorize sparse assembly:** Build `(rows, cols, data)` arrays with `numpy` and pass once to `scipy.sparse.csc_matrix`—avoid nested Python loops over grid nodes.
* **Watch the sign of the stabilizing Jacobian term:** Because the 5-point Laplacian $L$ has negative diagonal entries, the Jacobian diagonal must be $L_{kk} - (\tilde{n}_k + \tilde{p}_k)$ (more negative), never $+ (\tilde{n}_k + \tilde{p}_k)$.
* **Domain width check:** Ensure $L_x$ is at least $3\times$ the maximum reverse-bias depletion width so the depletion region never reaches the Dirichlet contacts.
* **Forward-bias numerical limit:** Cap the default forward sweep at `0.65 V` unless full coupled Newton is added later. Treat this as a solver/convergence guardrail, not a universal physical high-injection threshold.