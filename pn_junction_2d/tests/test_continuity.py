import numpy as np
import pytest

from config import DeviceConfig
from geometry import cosine
from solver import (Device, assemble_electron_matrix, assemble_hole_matrix, solve_electrons,
                    solve_equilibrium, solve_holes)


def _check_sign_pattern(A, interior):
    A = A.tocsr()
    diag = A.diagonal()
    off = A - __import__("scipy.sparse", fromlist=["diags"]).diags(diag)
    off = off.tocsr()[interior]
    assert np.all(diag[interior] > 0)
    assert off.data.size == 0 or off.data.max() <= 0.0


@pytest.fixture(scope="module")
def eq():
    cfg = DeviceConfig()
    dev = Device(cfg, cosine(cfg.Lx / 2, 0.25e-4, cfg.Ly))
    return dev, solve_equilibrium(dev)


def test_sign_pattern_random_and_equilibrium(eq):
    dev, st = eq
    rng = np.random.default_rng(1)
    for psi in (st.psi, rng.normal(scale=10, size=dev.mesh.N)):
        _check_sign_pattern(assemble_electron_matrix(dev, psi), dev.mesh.interior)
        _check_sign_pattern(assemble_hole_matrix(dev, psi), dev.mesh.interior)


def test_operators_annihilate_boltzmann(eq):
    """SG fluxes vanish exactly for Boltzmann carriers: A n_eq = 0 on interior rows."""
    dev, st = eq
    for A, u in ((assemble_electron_matrix(dev, st.psi), st.n),
                 (assemble_hole_matrix(dev, st.psi), st.p)):
        r = (A @ u)[dev.mesh.interior]
        scale = np.abs(A).max() * u.max()
        assert np.max(np.abs(r)) < 1e-12 * scale


def test_boltzmann_recovery_v0(eq):
    dev, st = eq
    n = solve_electrons(dev, st.psi, st.n, st.p, enable_srh=False)
    p = solve_holes(dev, st.psi, st.n, st.p, enable_srh=False)
    assert np.all(n > 0) and np.all(p > 0)
    en = np.max(np.abs(n / st.n - 1))
    ep = np.max(np.abs(p / st.p - 1))
    print(f"max rel err n = {en:.2e}, p = {ep:.2e}")
    assert en < 1e-6 and ep < 1e-6
