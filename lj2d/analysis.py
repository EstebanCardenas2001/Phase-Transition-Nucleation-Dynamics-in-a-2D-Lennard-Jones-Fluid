"""Structural and statistical analysis helpers."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def largest_cluster_fraction(system):
    """Largest bonded cluster / N for every replica, plus coordination numbers.

    Returns (m [B], coord [B, N] int8). Bonds are pairs closer than system.bond_r.
    """
    coord, i, j = system.bonds()
    B, N = system.B, system.N
    i, j = i.cpu().numpy(), j.cpu().numpy()
    # Bonds never cross replicas, so one graph over all B*N particles suffices.
    graph = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(B * N, B * N))
    _, labels = connected_components(graph, directed=False)
    m = np.empty(B)
    for b in range(B):
        m[b] = np.bincount(labels[b * N:(b + 1) * N]).max() / N
    return m, coord.cpu().numpy().astype(np.int8)


def block_density_moments(pos, L, n_sub):
    """Particle counts in an n_sub x n_sub grid of sub-boxes.

    pos : array [..., N, 2]. Returns counts [..., n_sub*n_sub].
    Used for the subsystem-block method (Rovere, Heermann & Binder 1990): each
    sub-box of side L / n_sub is an open subsystem of the NVT box, so its
    density fluctuates and its distribution P(rho) carries the order parameter.
    """
    idx = np.floor(pos / (L / n_sub)).astype(np.int64)
    np.clip(idx, 0, n_sub - 1, out=idx)
    cell = idx[..., 0] * n_sub + idx[..., 1]
    flat = cell.reshape(-1, cell.shape[-1])
    out = np.stack([np.bincount(c, minlength=n_sub * n_sub) for c in flat])
    return out.reshape(*cell.shape[:-1], n_sub * n_sub)


def binder_cumulant(x):
    """U = 1 - <dx^4> / (3 <dx^2>^2) with dx = x - <x>, over all samples of x."""
    dx = np.asarray(x, dtype=np.float64) - np.mean(x)
    m2, m4 = np.mean(dx ** 2), np.mean(dx ** 4)
    return 1.0 - m4 / (3.0 * m2 ** 2)


def blocking_error(series, n_blocks=10):
    """Mean and standard error from n_blocks contiguous blocks (handles autocorrelation)."""
    s = np.asarray(series, dtype=np.float64)
    usable = len(s) - len(s) % n_blocks
    blocks = s[:usable].reshape(n_blocks, -1).mean(1)
    return s.mean(), blocks.std(ddof=1) / np.sqrt(n_blocks)


def jackknife(func, samples, n_blocks=10):
    """Jackknife estimate and error of func(samples) using contiguous blocks along axis 0."""
    samples = np.asarray(samples)
    usable = len(samples) - len(samples) % n_blocks
    blocks = np.array_split(samples[:usable], n_blocks)
    full = func(samples[:usable])
    loo = np.array([func(np.concatenate(blocks[:k] + blocks[k + 1:])) for k in range(n_blocks)])
    err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((loo - loo.mean(0)) ** 2, axis=0))
    return full, err


def drift_test(series, n_parts=4):
    """Means of n_parts consecutive segments; a monotone trend flags non-equilibrium data."""
    s = np.asarray(series, dtype=np.float64)
    usable = len(s) - len(s) % n_parts
    return s[:usable].reshape(n_parts, -1, *s.shape[1:]).mean(1)
