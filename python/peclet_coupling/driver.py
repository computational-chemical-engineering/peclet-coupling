"""CfdDem — unresolved point-particle CFD-DEM driver.

Composes a peclet.flow.Solver (Eulerian fluid, grid units: unit spacing, origin 0, cell i centred at
i+0.5) and a peclet.dem.Simulation (Lagrangian particles, positions in the SAME grid coordinates).
Each fluid step:

  1. deposit particle volumes -> void fraction eps (trilinear, periodic-folded);
  2. gather the fluid velocity + eps at each particle, evaluate the drag law -> per-particle drag
     force F, and scatter the reaction -F/Vcell onto the fluid body-force fields (momentum feedback);
  3. apply F to the particles and advance dem `dem_substeps` sub-steps (drag held constant);
  4. advance the fluid one step (its momentum RHS now carries the feedback).

The grid fields are touched zero-copy through flow.field_view(...); the particle drag is
round-tripped through the dem host API (set_external_forces). Periodic ghost handling (fold for
deposits, fill for reads) is done here in NumPy on the padded (ex,ey,ez) buffers.
"""
import numpy as np


def _sl(axis, idx):
    s = [slice(None), slice(None), slice(None)]
    s[axis] = idx
    return tuple(s)


class CfdDem:
    def __init__(self, flow, dem, *, fluid_dt, mu, rho, radius, drag="schiller_naumann",
                 dem_substeps=20, eps_min=0.05, smooth_width=0.0, periodic=(True, True, True), h=1.0,
                 move_particles=True, implicit_drag=True, porous=False, advection=True):
        from . import (_coupling, DRAG_STOKES, DRAG_SCHILLER_NAUMANN, DRAG_ERGUN, DRAG_DI_FELICE,
                       DRAG_WEN_YU, DRAG_GIDASPOW)
        self._c = _coupling
        self.flow = flow
        self.dem = dem
        # Backend: the coupling kernels run on flow's Kokkos execution space. On a GPU build the
        # arrays they touch must be device-resident, so we array-program through CuPy (device) or
        # NumPy (host); grid fields + particle state are taken zero-copy via DLPack.
        import peclet.flow as _flowmod
        _es = str(_flowmod.execution_space).lower()
        self.device = "cuda" in _es or "hip" in _es
        if self.device:
            import cupy as _xp
        else:
            import numpy as _xp
        self.xp = _xp
        self.mu = float(mu)
        self.rho = float(rho)
        self.fluid_dt = float(fluid_dt)
        self.dem_substeps = int(dem_substeps)
        self.dt_dem = self.fluid_dt / self.dem_substeps
        # Void-fraction floor — now only a divide-by-zero guard (default 0.05), NOT a physical clamp.
        # A dense bed's true voidage falls to the random-close-packing ~0.4 and below; clamping eps UP
        # to 0.4 under-predicts the Ergun 1/eps^3 drag ~3x so a coarse bed never fluidizes (the old
        # eps_min=0.4 behaviour). We instead clip only to [eps_min, 1] and tame the discretisation
        # noise that made small clamps unstable with `smooth_width` below (the MFIX approach: physical
        # small porosities are kept; the deposit noise is smoothed, not clamped away).
        self.eps_min = float(eps_min)
        # Porosity smoothing length (grid cells), decoupled from the CFD cell size — MFIX's
        # DES_DIFFUSE_WIDTH. 0 = off (plain trilinear deposit). For a coarse cell/dp bed set it ~1 cell
        # (a few particle diameters) so the void fraction the drag sees is smooth and grid-independent;
        # converted to `nsweeps` explicit diffusion sweeps (sigma = sqrt(2*alpha*nsweeps), alpha=1/6).
        self.smooth_width = float(smooth_width)
        self._smooth_alpha = 1.0 / 6.0
        self._smooth_sweeps = (max(1, int(round(self.smooth_width ** 2 / (2.0 * self._smooth_alpha))))
                               if self.smooth_width > 0.0 else 0)
        self.move_particles = bool(move_particles)  # False: fixed bed — skip DEM dynamics entirely
        self.implicit_drag = bool(implicit_drag)    # beta on the fluid diagonal (stable for stiff beds)
        # Volume-averaged continuity d(eps)/dt+div(eps u)=0 (proper unresolved CFD-DEM). When off the
        # fluid is solved incompressible (eps only in the drag) — cheaper, fine for dilute/steady beds.
        self.porous = bool(porous)
        self.h = float(h)
        self.inv_vcell = 1.0 / (self.h ** 3)
        self.periodic = tuple(bool(p) for p in periodic)
        self.drag_kind = {"stokes": DRAG_STOKES, "schiller_naumann": DRAG_SCHILLER_NAUMANN,
                          "ergun": DRAG_ERGUN, "di_felice": DRAG_DI_FELICE,
                          "wen_yu": DRAG_WEN_YU, "gidaspow": DRAG_GIDASPOW}[drag]

        nx, ny, nz = flow.get_resolution()  # LOCAL block dims under MPI
        self.g = flow.ghost_width()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.ex, self.ey, self.ez = nx + 2 * self.g, ny + 2 * self.g, nz + 2 * self.g

        # Multi-rank co-decomposition. When flow runs distributed each rank couples its LOCAL block:
        # the deposit origin is shifted by the block origin (so particles in GLOBAL coordinates land
        # in the local block) and cross-rank + periodic ghost deposits are folded with the reverse
        # (add-reduce) halo instead of the NumPy periodic fold. Detected from the flow MPI module +
        # world size; single-rank keeps the validated NumPy fold/fill path byte-for-byte.
        self.mpi = False
        try:
            import peclet.flow as _fm
            if getattr(_fm, "has_mpi", False):
                from mpi4py import MPI
                self.mpi = MPI.COMM_WORLD.Get_size() > 1
        except Exception:
            self.mpi = False
        if self.mpi and self._smooth_sweeps:  # smoothing sweeps need inter-rank halo refresh (TODO)
            import warnings
            warnings.warn("CfdDem: smooth_width is single-rank only for now; disabled under MPI "
                          "(rank-boundary diffusion would be wrong without a halo exchange per sweep).")
            self._smooth_sweeps = 0
        bo = flow.block_origin() if self.mpi else (0, 0, 0)
        self._ox, self._oy, self._oz = bo[0] * self.h, bo[1] * self.h, bo[2] * self.h
        gnx, gny, gnz = flow.global_resolution() if self.mpi else (nx, ny, nz)
        self.gnx, self.gny, self.gnz = gnx, gny, gnz
        # The CURRENT shared decomposition, as an x-fastest per-cell weight field. Uniform => the
        # default equal-cell ORB flow's init_mpi built; rebalance() overwrites it. dem is migrated onto
        # this each moving step so its ownership tracks flow's grid partition (the deposit stays
        # in-block). None single-rank.
        self._weights = np.ones(gnx * gny * gnz, dtype=np.float64) if self.mpi else None

        # deposit / void-fraction buffers. Under MPI they are REGISTERED flow fields (so the halo can
        # fold ghost deposits + fill ghosts); single-rank they are standalone padded scratch.
        xp = self.xp
        # eps must live in a REGISTERED flow field when the halo needs it (MPI) or the flow reads it
        # for the porous continuity; otherwise a standalone padded scratch is enough.
        self._eps_is_field = self.mpi or self.porous
        if self._eps_is_field:
            flow.add_field("solidvol")
            flow.add_field("eps")
        else:
            self._solidvol = xp.zeros((self.ex, self.ey, self.ez), dtype=xp.float64, order="F")
            self._eps = xp.ones((self.ex, self.ey, self.ez), dtype=xp.float64, order="F")
        if self.porous:
            flow.set_porous_continuity(True)  # projection enforces d(eps)/dt + div(eps u) = 0
        # Gas convection ON by default: fully-implicit FOU operator + explicit deferred-correction
        # TVD (unconditionally stable at the large coupled dt on both the periodic and domain-BC
        # paths). Without it the gas momentum has no inertia and freeboard velocities overshoot.
        # advection=False restores a Stokes-like (drag+viscous+pressure only) gas.
        if advection:
            flow.set_advection(True)
            flow.set_implicit_advection(True)

        if self.implicit_drag:
            flow.enable_drag()  # drag_beta on the momentum diagonal + force_* (beta*u_p) in the RHS
        else:
            flow.enable_cell_force()  # explicit reaction force in force_x/y/z
        self.dem.set_dt(self.dt_dem)
        N = dem.num_particles()  # this rank's OWNED count under MPI
        self._N = N
        self._radius0 = float(radius) if np.isscalar(radius) else None  # scalar => resizable per-rank
        self._rad = (xp.full(N, radius, dtype=xp.float32) if np.isscalar(radius)
                     else xp.asarray(np.ascontiguousarray(radius, dtype=np.float32)))
        self._fdrag = xp.zeros((N, 3), dtype=xp.float32)
        self._ufluid = xp.zeros((N, 3), dtype=xp.float32)
        self._last_slip = None
        self.last_eps = None

    # (self._ox,_oy,_oz) shifts the deposit so global particle coords land in the local block
    # (== 0 single-rank). The grid map the coupling kernels take.
    def _gm(self):
        return (self._ox, self._oy, self._oz, self.h, self.ex, self.ey, self.ez, self.g)

    # Size the per-particle scratch to the current owned count (constant single-rank / fixed bed;
    # changes across a rebalance — needs a scalar radius to re-broadcast).
    def _resize_particles(self, n):
        if self._fdrag.shape[0] == n:
            return
        xp = self.xp
        self._fdrag = xp.zeros((n, 3), dtype=xp.float32)
        self._ufluid = xp.zeros((n, 3), dtype=xp.float32)
        if self._radius0 is not None:
            self._rad = xp.full(n, self._radius0, dtype=xp.float32)

    # Grid field as a device/host array over the SAME buffer (zero-copy): CuPy on a GPU build
    # (field_view returns a DLPack capsule), NumPy on a host build.
    def _fv(self, name):
        v = self.flow.field_view(name)
        return self.xp.from_dlpack(v) if self.device else v

    # Particle positions + velocities as C-contiguous device/host (N,3) arrays (device views are
    # zero-copy DLPack; ascontiguousarray normalises the LayoutLeft device stride to C order).
    def _particles(self):
        if self.device:
            pos = self.xp.ascontiguousarray(self.xp.from_dlpack(self.dem.get_positions_view()))
            vel = self.xp.ascontiguousarray(self.xp.from_dlpack(self.dem.get_velocities_view()))
        else:
            pos = np.ascontiguousarray(self.dem.get_positions(), dtype=np.float32)
            vel = np.ascontiguousarray(self.dem.get_velocities(), dtype=np.float32)
        return pos, vel

    # --- ghost handling on a padded (ex,ey,ez) buffer ----------------------------------------
    def _fold(self, f):
        """Fold ghost-layer deposits back onto owned cells. Periodic axis: wrap to the opposite
        inner edge. Non-periodic axis: fold into the SAME-side boundary cell — a grain resting on
        the distributor scatters part of its volume (and drag beta / feedback) one layer below
        z=0; dropping it would lose hold-up exactly where the bed is densest and leave the
        deposit's leakage in ghosts the fluid reads."""
        g = self.g
        for a, (n, per) in enumerate(zip((self.nx, self.ny, self.nz), self.periodic)):
            if per:
                f[_sl(a, n + g - 1)] += f[_sl(a, g - 1)]
                f[_sl(a, g)] += f[_sl(a, n + g)]
            else:
                f[_sl(a, g)] += f[_sl(a, g - 1)]
                f[_sl(a, n + g - 1)] += f[_sl(a, n + g)]
            f[_sl(a, g - 1)] = 0.0
            f[_sl(a, n + g)] = 0.0

    def _fold_domain(self, f):
        """MPI: same-side fold of the non-periodic GLOBAL-domain-boundary ghosts of the local
        block (the reverse halo never touches them). Call BEFORE exchange_field_add."""
        g = self.g
        lo = (int(self._ox / self.h), int(self._oy / self.h), int(self._oz / self.h))
        dims = (self.nx, self.ny, self.nz)
        gdims = (self.gnx, self.gny, self.gnz)
        for a in range(3):
            if self.periodic[a]:
                continue
            if lo[a] == 0:
                f[_sl(a, g)] += f[_sl(a, g - 1)]
                f[_sl(a, g - 1)] = 0.0
            if lo[a] + dims[a] == gdims[a]:
                f[_sl(a, dims[a] + g - 1)] += f[_sl(a, dims[a] + g)]
                f[_sl(a, dims[a] + g)] = 0.0

    def _fill(self, f):
        """Fill the one ghost layer the gather stencil reads: periodic wrap, or zero-gradient at a
        non-periodic domain face (a grain near the floor must see the LOCAL bed eps/velocity, not
        the deposit's stale ghost values)."""
        g = self.g
        for a, (n, per) in enumerate(zip((self.nx, self.ny, self.nz), self.periodic)):
            if per:
                f[_sl(a, g - 1)] = f[_sl(a, n + g - 1)]
                f[_sl(a, n + g)] = f[_sl(a, g)]
            else:
                f[_sl(a, g - 1)] = f[_sl(a, g)]
                f[_sl(a, n + g)] = f[_sl(a, n + g - 1)]

    def _fill_domain(self, f):
        """MPI: zero-gradient fill of the non-periodic global-domain-boundary ghosts (the halo
        fill never touches them). Call AFTER exchange_field."""
        g = self.g
        lo = (int(self._ox / self.h), int(self._oy / self.h), int(self._oz / self.h))
        dims = (self.nx, self.ny, self.nz)
        gdims = (self.gnx, self.gny, self.gnz)
        for a in range(3):
            if self.periodic[a]:
                continue
            if lo[a] == 0:
                f[_sl(a, g - 1)] = f[_sl(a, g)]
            if lo[a] + dims[a] == gdims[a]:
                f[_sl(a, dims[a] + g)] = f[_sl(a, dims[a] + g - 1)]

    def update_void_fraction(self, pos):
        # deposit target: a registered flow field under MPI (so the halo folds ghost deposits + fills
        # ghosts), standalone scratch single-rank.
        sv = self._fv("solidvol") if self._eps_is_field else self._solidvol
        ep = self._fv("eps") if self._eps_is_field else self._eps
        sv[...] = 0.0
        if pos.shape[0] > 0:  # a rank may own no particles under MPI; still run the halo collectives
            # sdf (flow's cell-centred signed distance, <0 in solid) makes the deposit wall-aware:
            # the hold-up goes only to fluid corners (partition of unity), never leaking into walls.
            # With no geometry set sdf is all-zero -> every corner fluid -> plain trilinear deposit.
            self._c.deposit_solid_volume(pos, self._rad, sv, self._fv("sdf"), *self._gm())
        if self.mpi:
            self._fold_domain(sv)  # non-periodic domain-boundary ghosts (halo never folds them)
            self.flow.exchange_field_add("solidvol")  # fold cross-rank + periodic ghost deposits
        else:
            self._fold(sv)
        if self._smooth_sweeps:
            # Diffusive smoothing of the deposited solid volume (MFIX DES_DIFFUSE_WIDTH): decouple the
            # porosity smoothing length from the CFD cell so a coarse cell/dp bed sees a smooth,
            # grid-independent void fraction. Volume-conserving (zero-flux at the walls). Under MPI the
            # halo would need refreshing between sweeps — smoothing is single-rank for now.
            self._c.smooth_solid_volume(sv, *self._gm(), self._smooth_sweeps, self._smooth_alpha)
        self._c.compute_void_fraction(sv, ep, self.inv_vcell, self.eps_min)
        if self.mpi:
            self.flow.exchange_field("eps")  # fill the ghosts the gather stencil reads
            self._fill_domain(ep)
        else:
            self._fill(ep)
        self._eps = ep  # compute_forces reads this at the particles

    def compute_forces(self, pos, vel):
        for name in ("u", "v", "w"):
            self.flow.exchange_field(name)
        uf, vf, wf = (self._fv(n) for n in ("u", "v", "w"))
        fx, fy, fz = (self._fv(n) for n in ("force_x", "force_y", "force_z"))
        sd = self._fv("sdf")  # wall mask: gather from / scatter to fluid corners only (partition of unity)
        gm = self._gm()
        has_p = pos.shape[0] > 0  # a rank may own no particles under MPI (skip the per-particle
        db = self._fv("drag_beta") if self.implicit_drag else None  # kernels, keep the collectives)
        # porous (volume-averaged, Model B: the fluid carries the full -grad p) converts the drag
        # closures beta_B = beta_A/eps inside the kernel (model_b flag); the incompressible mode
        # keeps the literature Model-A forms unchanged.
        if has_p and self.implicit_drag:
            self._c.compute_drag_implicit(pos, vel, self._rad, uf, vf, wf, self._eps, sd, self._fdrag,
                                          db, fx, fy, fz, *gm, self.mu, self.rho, self.inv_vcell,
                                          self.drag_kind, self.porous)
        elif has_p:
            self._c.compute_drag_feedback(pos, vel, self._rad, uf, vf, wf, self._eps, sd, self._fdrag,
                                          fx, fy, fz, *gm, self.mu, self.rho, self.inv_vcell,
                                          self.drag_kind, self.porous)
        if self.implicit_drag:
            if self.mpi:
                self._fold_domain(db)
                self.flow.exchange_field_add("drag_beta")
            else:
                self._fold(db)
        if has_p:
            self._c.interpolate_velocity(pos, uf, vf, wf, self._ufluid, *gm)
        self._last_slip = self._ufluid - vel  # slip the drag saw; kept DEVICE-side (lazy .get on access)
        # fold the reaction feedback (force_*) onto owners: reverse halo under MPI, periodic wrap else.
        for nm, f in (("force_x", fx), ("force_y", fy), ("force_z", fz)):
            if self.mpi:
                self._fold_domain(f)
                self.flow.exchange_field_add(nm)
            else:
                self._fold(f)

    def slip(self, pos, vel):
        """Interpolated fluid velocity minus particle velocity (N,3) — what the drag law sees.
        `vel` may be host or device; returns a host NumPy array for convenient inspection."""
        s = self._ufluid - self.xp.asarray(vel)
        return s.get() if self.device else s

    # Diagnostics as HOST arrays, converted lazily on access (no per-step device->host copy in the hot
    # loop). Reflect the most recent step; copy them if you keep them across steps.
    @property
    def last_slip(self):
        s = getattr(self, "_last_slip", None)
        return None if s is None else (s.get() if self.device else np.asarray(s))

    @property
    def last_drag(self):
        return self._fdrag.get() if self.device else np.asarray(self._fdrag)

    def step(self):
        # Multi-rank moving particles: migrate ownership onto flow's grid partition BEFORE depositing,
        # so every owned particle sits in this rank's block (the substeps only drift it by < a ghost
        # band, corrected at the next step's migrate). Static bed / single-rank: no migration.
        if self.mpi and self.move_particles:
            self.dem.migrate_to_weights(self._weights)
        pos, vel = self._particles()
        self._resize_particles(pos.shape[0])
        self.update_void_fraction(pos)
        if self.porous and not getattr(self, "_porous_primed", False):
            self.flow.sync_porous_prev()  # first deposit: eps^n=eps^{n+1} so no spurious d(eps)/dt
            self._porous_primed = True
        self.compute_forces(pos, vel)
        self.last_eps = self._eps
        if self.move_particles:
            # Write the drag straight into dem's external-force buffer ON DEVICE (zero-copy view) — no
            # host round-trip, so the coupled step stays device-resident and the DEM remains the cost.
            efv = self.dem.get_external_forces_view()
            ext = self.xp.from_dlpack(efv) if self.device else np.asarray(efv)
            ext[...] = self._fdrag
            if self.mpi:
                self.dem.step_mpi(self.dem_substeps)  # distributed substeps (halo exchange)
            else:
                for _ in range(self.dem_substeps):
                    self.dem.step(self.dt_dem)
        self.flow.step()

    def rebalance(self, gamma=1.0):
        """Dynamic co-rebalancing (multi-rank only). Build ONE weight field over the global grid --
        fluid work (1 per cell) + gamma * particle count -- and redistribute BOTH codes onto the same
        weighted ORB from it: the flow state via rebalance_by_weights (bit-exact migration + rebuild),
        the particles via migrate_to_weights. Because both build the SAME deterministic partition from
        the same array, they stay co-located. Call at a step boundary. No-op single-rank."""
        if not self.mpi:
            return
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        gnx, gny, gnz = self.gnx, self.gny, self.gnz  # GLOBAL grid
        # bin this rank's OWNED particles onto the global ORB grid, then sum across ranks.
        pos, _ = self._particles()
        p = pos.get() if self.device else np.asarray(pos)
        counts = np.zeros((gnx, gny, gnz), dtype=np.float64)
        idx = np.floor(p / self.h).astype(np.int64)
        np.clip(idx[:, 0], 0, gnx - 1, out=idx[:, 0])
        np.clip(idx[:, 1], 0, gny - 1, out=idx[:, 1])
        np.clip(idx[:, 2], 0, gnz - 1, out=idx[:, 2])
        np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        total = np.empty_like(counts)
        comm.Allreduce(counts, total, op=MPI.SUM)
        w = (1.0 + gamma * total).flatten(order="F")
        self._weights = w  # dem is migrated onto this each moving step; flow redistributes now
        self.flow.rebalance_by_weights(w)
        self.dem.migrate_to_weights(w)
        # flow's block moved -> refresh the deposit-origin shift + local extents.
        bo = self.flow.block_origin()
        self._ox, self._oy, self._oz = bo[0] * self.h, bo[1] * self.h, bo[2] * self.h
        self.nx, self.ny, self.nz = self.flow.get_resolution()
        self.ex, self.ey, self.ez = (self.nx + 2 * self.g, self.ny + 2 * self.g, self.nz + 2 * self.g)
