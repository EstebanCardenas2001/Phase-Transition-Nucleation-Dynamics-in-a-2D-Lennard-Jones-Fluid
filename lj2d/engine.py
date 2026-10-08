"""Batched GPU molecular dynamics engine for the 2D Lennard-Jones fluid.

All quantities are in reduced LJ units (sigma = epsilon = m = k_B = 1).

The engine advances B independent replicas of the same box (same N, L) in
lock-step, each with its own thermostat temperature. Forces use a Verlet
neighbour list built from a cell list, so the cost per step is O(N) rather
than the O(N^2) all-pairs evaluation of the original Phase 1/2 engines.

Tensor layout: positions and velocities are [B, N, 2]. Internally particles
are addressed by a flat index g = b * N + i, so one neighbour table covers
every replica.
"""

import math

import torch


def lj_shift(rc):
    """Potential energy at the cutoff, subtracted so that V(rc) = 0."""
    inv6 = rc ** -6
    return 4.0 * (inv6 * inv6 - inv6)


def minimum_image(d, L):
    return d - L * torch.round(d / L)


def _pair_kernel(flat, nbr, mask, L: float, rc2: float, e_shift: float):
    """Per-particle force, pair energy sum and virial sum over a padded neighbour list.

    flat [M, 2], nbr/mask [M, K]. Returns f [M, 2], u [M], w [M]; u and w count
    every pair twice (once from each side) when the list is full.
    """
    d = flat[:, None, :] - flat[nbr]
    d = d - L * torch.round(d / L)
    r2 = (d * d).sum(-1)
    m = mask & (r2 < rc2)
    inv2 = torch.where(m, 1.0 / torch.where(m, r2, torch.ones_like(r2)), torch.zeros_like(r2))
    inv6 = inv2 * inv2 * inv2
    w = 24.0 * (2.0 * inv6 * inv6 - inv6)               # r . F_ij, zero outside the cutoff
    f = ((w * inv2).unsqueeze(-1) * d).sum(1)
    u = torch.where(m, 4.0 * (inv6 * inv6 - inv6) - e_shift, torch.zeros_like(r2))
    return f, u.sum(1), w.sum(1)


def _candidate_kernel(flat, cand, r_list2: float, L: float):
    """Mask of candidate slots (cand = -1 marks an empty slot) that lie within r_list."""
    self_idx = torch.arange(flat.shape[0], device=flat.device)[:, None]
    valid = (cand >= 0) & (cand != self_idx)
    d = flat[torch.where(valid, cand, self_idx)] - flat[:, None, :]
    d = d - L * torch.round(d / L)
    return valid & ((d * d).sum(-1) < r_list2)


_compiled = {}


def _get_kernel(fn, compile_):
    """torch.compile fuses a kernel's temporaries (~8x faster on a T4); eager on CPU or on request."""
    if not compile_:
        return fn
    if fn not in _compiled:
        _compiled[fn] = torch.compile(fn, dynamic=True)
    return _compiled[fn]


class LJSystem:
    """B replicas of N Lennard-Jones particles in a periodic L x L box.

    Parameters
    ----------
    N, rho : particle count and number density (sets L = sqrt(N / rho)).
    B : number of independent replicas advanced together.
    rc : interaction cutoff. The potential is truncated and shifted, so the
        dynamics are those of the truncated potential and the energy is
        continuous at rc.
    skin : Verlet skin. The list holds pairs within rc + skin and is rebuilt
        once any particle has moved more than skin / 2 since the last build.
    bond_r : distance defining a "bond" for coordination numbers and clusters
        (1.5 sigma, as in the original analysis).
    """

    def __init__(self, N, rho, B=1, rc=2.5, skin=0.5, bond_r=1.5,
                 device=None, dtype=torch.float32, seed=None, compile=None):
        self.N, self.B, self.rho = int(N), int(B), float(rho)
        self.L = math.sqrt(N / rho)
        self.rc, self.skin, self.bond_r = float(rc), float(skin), float(bond_r)
        self.r_list = self.rc + self.skin
        if self.bond_r > self.r_list:
            raise ValueError("bond_r must not exceed rc + skin")
        self.e_shift = lj_shift(self.rc)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        use_compile = self.device.type == "cuda" if compile is None else compile
        self._kernel = _get_kernel(_pair_kernel, use_compile)
        self._candidates = _get_kernel(_candidate_kernel, use_compile)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed if seed is not None else torch.seed() % 2**63)

        # Cell grid: at least 3 cells per side, otherwise the 3x3 stencil would
        # visit the same cell twice and double-count pairs.
        self.n_cells = int(self.L // self.r_list)
        if self.n_cells < 3:
            raise ValueError(f"box L={self.L:.2f} too small for cell list with r_list={self.r_list}")
        self.cell_size = self.L / self.n_cells
        offs = torch.tensor([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], device=self.device)
        self._stencil = offs  # [9, 2]

        self.pos = torch.zeros(B, N, 2, device=self.device, dtype=dtype)
        self.vel = torch.zeros_like(self.pos)
        self.forces = torch.zeros_like(self.pos)
        self.pe = torch.zeros(B, device=self.device, dtype=dtype)       # total potential energy per replica
        self.virial = torch.zeros(B, device=self.device, dtype=dtype)   # sum over pairs of r . F
        self.nbr = None          # [B*N, K] flat neighbour indices (padded with self)
        self.nbr_mask = None     # [B*N, K] True where the slot holds a real neighbour
        self._pos_at_build = None
        self.n_builds = 0

    # ------------------------------------------------------------------ setup
    def init_lattice(self, T):
        """Square lattice positions and Maxwell-Boltzmann velocities at temperature T (scalar or [B])."""
        grid = math.ceil(math.sqrt(self.N))
        spacing = self.L / grid
        x = torch.arange(grid, device=self.device, dtype=self.dtype) * spacing + spacing / 2
        gx, gy = torch.meshgrid(x, x, indexing="ij")
        lattice = torch.stack([gx.flatten(), gy.flatten()], dim=1)[: self.N]
        self.pos = lattice.unsqueeze(0).repeat(self.B, 1, 1).contiguous()
        T = self._as_batch(T)
        self.vel = torch.randn(self.B, self.N, 2, generator=self.gen, device=self.device, dtype=self.dtype)
        self.vel *= torch.sqrt(T)[:, None, None]
        self.vel -= self.vel.mean(dim=1, keepdim=True)
        self.build_neighbor_list()
        self.compute_forces()

    def set_state(self, pos, vel):
        self.pos = pos.to(self.device, self.dtype).reshape(self.B, self.N, 2).contiguous()
        self.vel = vel.to(self.device, self.dtype).reshape(self.B, self.N, 2).contiguous()
        self.pos = torch.remainder(self.pos, self.L)
        self.build_neighbor_list()
        self.compute_forces()

    def _as_batch(self, x):
        return torch.as_tensor(x, device=self.device, dtype=self.dtype).expand(self.B).clone()

    # --------------------------------------------------------- neighbour list
    @torch.no_grad()
    def build_neighbor_list(self):
        B, N, nc = self.B, self.N, self.n_cells
        flat = self.pos.reshape(B * N, 2)
        ci = torch.floor(flat / self.cell_size).long().clamp_(0, nc - 1)            # [BN, 2]
        replica = torch.arange(B, device=self.device).repeat_interleave(N)          # [BN]
        cell = replica * nc * nc + ci[:, 0] * nc + ci[:, 1]                         # global cell id

        # Padded cell table: table[c, k] = k-th particle in cell c, or -1.
        counts = torch.bincount(cell, minlength=B * nc * nc)
        order = torch.argsort(cell)
        start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(B * N, device=self.device) - start[cell[order]]
        max_occ = int(counts.max())
        table = torch.full((B * nc * nc, max_occ), -1, dtype=torch.long, device=self.device)
        table[cell[order], rank] = order

        # Candidates from the 3x3 block of cells around each particle.
        nb = (ci.unsqueeze(1) + self._stencil.unsqueeze(0)) % nc                    # [BN, 9, 2]
        nb_cell = replica[:, None] * nc * nc + nb[..., 0] * nc + nb[..., 1]         # [BN, 9]
        cand = table[nb_cell].reshape(B * N, 9 * max_occ)                           # [BN, 9*max_occ]

        self_idx = torch.arange(B * N, device=self.device)[:, None]
        within = self._candidates(flat, cand, self.r_list ** 2, self.L)

        # Compact the hits to the left: [BN, K] with K = max neighbour count.
        n_hit = within.sum(1)
        K = max(int(n_hit.max()), 1)
        # nonzero() is row-major, so a hit's slot is its rank minus the row's first rank.
        row, col = within.nonzero(as_tuple=True)
        row_start = torch.cumsum(n_hit, 0) - n_hit
        slot = torch.arange(row.numel(), device=self.device) - row_start[row]
        self.nbr = self_idx.repeat(1, K)
        self.nbr[row, slot] = cand[row, col]
        self.nbr_mask = torch.arange(K, device=self.device)[None, :] < n_hit[:, None]
        self._pos_at_build = self.pos.clone()
        self.n_builds += 1

    def _maybe_rebuild(self):
        disp = minimum_image(self.pos - self._pos_at_build, self.L)
        if float((disp * disp).sum(-1).max()) > (0.5 * self.skin) ** 2:
            self.build_neighbor_list()

    # ------------------------------------------------------------------ forces
    def _pair_geometry(self):
        """Separation vectors d_ij = r_i - r_j and r^2 for every neighbour slot."""
        flat = self.pos.reshape(self.B * self.N, 2)
        d = minimum_image(flat[:, None, :] - flat[self.nbr], self.L)                 # [BN, K, 2]
        return d, (d * d).sum(-1)

    @torch.no_grad()
    def compute_forces(self):
        flat = self.pos.reshape(self.B * self.N, 2)
        f, u, w = self._kernel(flat, self.nbr, self.nbr_mask, self.L, self.rc ** 2, self.e_shift)
        # The list is full (i sees j and j sees i), so pair sums are halved.
        self.forces = f.reshape(self.B, self.N, 2)
        self.pe = 0.5 * u.reshape(self.B, self.N).sum(1)
        self.virial = 0.5 * w.reshape(self.B, self.N).sum(1)
        return self.forces

    # -------------------------------------------------------------- dynamics
    @torch.no_grad()
    def step(self, dt, T=None, nu=1.0):
        """One velocity-Verlet step, followed by Andersen collisions if T is given.

        T : target temperature, scalar or [B]. Each particle is redrawn from the
            Maxwell-Boltzmann distribution with probability nu * dt per step.
        """
        self.vel += 0.5 * dt * self.forces
        self.pos = torch.remainder(self.pos + dt * self.vel, self.L)
        self._maybe_rebuild()
        self.compute_forces()
        self.vel += 0.5 * dt * self.forces
        if T is not None:
            T = self._as_batch(T)
            hit = torch.rand(self.B, self.N, 1, generator=self.gen, device=self.device, dtype=self.dtype) < nu * dt
            fresh = torch.randn(self.B, self.N, 2, generator=self.gen, device=self.device, dtype=self.dtype)
            fresh *= torch.sqrt(T.clamp(min=0.0))[:, None, None]
            self.vel = torch.where(hit, fresh, self.vel)

    def run(self, n_steps, dt, T=None, nu=1.0):
        for _ in range(n_steps):
            self.step(dt, T, nu)

    # ----------------------------------------------------------- observables
    def kinetic_energy(self):
        return 0.5 * (self.vel * self.vel).sum(dim=(1, 2))

    def temperature(self):
        """Instantaneous kinetic temperature, 2 degrees of freedom per particle (momentum not removed)."""
        return self.kinetic_energy() / self.N

    def pressure(self):
        area = self.L ** 2
        return self.N * self.temperature() / area + self.virial / (2.0 * area)

    @torch.no_grad()
    def bonds(self):
        """Coordination numbers [B, N] and bonded pairs (i < j, flat indices) within bond_r."""
        _, r2 = self._pair_geometry()
        bonded = self.nbr_mask & (r2 < self.bond_r ** 2)
        coord = bonded.sum(1).reshape(self.B, self.N)
        i = torch.arange(self.B * self.N, device=self.device)[:, None].expand_as(self.nbr)
        upper = bonded & (i < self.nbr)
        return coord, i[upper], self.nbr[upper]
