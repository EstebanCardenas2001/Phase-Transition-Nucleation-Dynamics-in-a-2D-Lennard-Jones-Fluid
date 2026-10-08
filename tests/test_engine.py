import numpy as np
import pytest
import torch
from scipy.sparse.csgraph import connected_components

from lj2d import LJSystem, lj_shift, largest_cluster_fraction, binder_cumulant

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def brute_force(pos, L, rc):
    """O(N^2) reference in float64: forces, shifted potential energy, virial."""
    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.round(d / L)
    r2 = (d ** 2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    m = r2 < rc ** 2
    inv6 = np.where(m, r2 ** -3, 0.0)
    w = 24.0 * (2.0 * inv6 ** 2 - inv6)
    f = ((w / np.where(m, r2, 1.0))[..., None] * d).sum(1)
    u = np.where(m, 4.0 * (inv6 ** 2 - inv6) - lj_shift(rc), 0.0)
    return f, 0.5 * u.sum(), 0.5 * w.sum()


def disordered_config(sys, steps=300):
    """Melt the initial lattice so the test sees a dense, irregular configuration."""
    sys.init_lattice(1.0)
    sys.run(steps, 0.005, T=1.0, nu=1.0)


@pytest.mark.parametrize("device", DEVICES)
def test_forces_match_brute_force(device):
    sys = LJSystem(N=600, rho=0.5, B=2, device=device, dtype=torch.float64, seed=1)
    disordered_config(sys)
    sys.build_neighbor_list()
    sys.compute_forces()
    for b in range(sys.B):
        f, pe, vir = brute_force(sys.pos[b].cpu().numpy(), sys.L, sys.rc)
        np.testing.assert_allclose(sys.forces[b].cpu().numpy(), f, rtol=1e-9, atol=1e-9)
        assert sys.pe[b].item() == pytest.approx(pe, rel=1e-10)
        assert sys.virial[b].item() == pytest.approx(vir, rel=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_newton_third_law(device):
    sys = LJSystem(N=500, rho=0.4, B=3, device=device, dtype=torch.float64, seed=2)
    disordered_config(sys, 100)
    total = sys.forces.sum(1).abs().max().item()
    assert total < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_nve_energy_conservation(device):
    """With the shifted potential and a skin-protected list, total energy must not drift."""
    sys = LJSystem(N=900, rho=0.5, B=2, device=device, dtype=torch.float64, seed=3)
    disordered_config(sys)
    e0 = (sys.pe + sys.kinetic_energy()) / sys.N
    builds = sys.n_builds
    energies = []
    for _ in range(40):
        sys.run(50, 0.002)
        energies.append(((sys.pe + sys.kinetic_energy()) / sys.N).cpu().numpy())
    energies = np.array(energies)
    assert sys.n_builds > builds  # the list really was rebuilt during the run
    # The truncated force jumps at rc, so energy is conserved only to O(dt^2) noise.
    assert np.abs(energies - e0.cpu().numpy()).max() < 2e-3


@pytest.mark.parametrize("device", DEVICES)
def test_andersen_reaches_target_temperatures(device):
    temps = [0.8, 1.5]  # both supercritical, so no latent heat interferes
    sys = LJSystem(N=1024, rho=0.3, B=2, device=device, seed=4)
    sys.init_lattice(1.0)
    sys.run(1500, 0.005, T=temps, nu=2.0)
    samples = []
    for _ in range(50):
        sys.run(20, 0.005, T=temps, nu=2.0)
        samples.append(sys.temperature().cpu().numpy())
    np.testing.assert_allclose(np.mean(samples, axis=0), temps, rtol=0.03)


@pytest.mark.parametrize("device", DEVICES)
def test_clusters_match_dense_graph(device):
    sys = LJSystem(N=700, rho=0.3, B=2, device=device, dtype=torch.float64, seed=5)
    disordered_config(sys, 200)
    m, coord = largest_cluster_fraction(sys)
    for b in range(sys.B):
        p = sys.pos[b].cpu().numpy()
        d = p[:, None] - p[None]
        d -= sys.L * np.round(d / sys.L)
        r2 = (d ** 2).sum(-1)
        np.fill_diagonal(r2, np.inf)
        adj = r2 < sys.bond_r ** 2
        _, lab = connected_components(adj, directed=False)
        assert m[b] == pytest.approx(np.bincount(lab).max() / sys.N)
        np.testing.assert_array_equal(coord[b], adj.sum(1))


def test_binder_cumulant_limits():
    rng = np.random.default_rng(0)
    assert binder_cumulant(rng.normal(size=200_000)) == pytest.approx(0.0, abs=0.02)   # disordered
    assert binder_cumulant(rng.choice([-1.0, 1.0], 200_000)) == pytest.approx(2 / 3, abs=0.01)  # two-phase
