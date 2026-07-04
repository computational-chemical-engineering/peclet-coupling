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
                 dem_substeps=20, eps_min=0.2, periodic=(True, True, True), h=1.0,
                 move_particles=True, implicit_drag=True):
        from . import _coupling, DRAG_STOKES, DRAG_SCHILLER_NAUMANN, DRAG_ERGUN, DRAG_DI_FELICE
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
        self.eps_min = float(eps_min)
        self.move_particles = bool(move_particles)  # False: fixed bed — skip DEM dynamics entirely
        self.implicit_drag = bool(implicit_drag)    # beta on the fluid diagonal (stable for stiff beds)
        self.h = float(h)
        self.inv_vcell = 1.0 / (self.h ** 3)
        self.periodic = tuple(bool(p) for p in periodic)
        self.drag_kind = {"stokes": DRAG_STOKES, "schiller_naumann": DRAG_SCHILLER_NAUMANN,
                          "ergun": DRAG_ERGUN, "di_felice": DRAG_DI_FELICE}[drag]

        nx, ny, nz = flow.get_resolution()
        self.g = flow.ghost_width()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.ex, self.ey, self.ez = nx + 2 * self.g, ny + 2 * self.g, nz + 2 * self.g
        # padded x-fastest scratch (F-order so ravel == flat x-fastest, matching the kernel),
        # device- or host-resident to match the kernels.
        xp = self.xp
        self._solidvol = xp.zeros((self.ex, self.ey, self.ez), dtype=xp.float64, order="F")
        self._eps = xp.ones((self.ex, self.ey, self.ez), dtype=xp.float64, order="F")

        if self.implicit_drag:
            flow.enable_drag()  # drag_beta on the momentum diagonal + force_* (beta*u_p) in the RHS
        else:
            flow.enable_cell_force()  # explicit reaction force in force_x/y/z
        self.dem.set_dt(self.dt_dem)
        N = dem.num_particles()
        self._N = N
        self._rad = (xp.full(N, radius, dtype=xp.float32) if np.isscalar(radius)
                     else xp.asarray(np.ascontiguousarray(radius, dtype=np.float32)))
        self._fdrag = xp.zeros((N, 3), dtype=xp.float32)
        self._ufluid = xp.zeros((N, 3), dtype=xp.float32)
        self.last_eps = None
        self.last_drag = None

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

    # --- periodic ghost handling on a padded (ex,ey,ez) buffer -------------------------------
    def _fold(self, f):  # deposits that landed one cell into the ghost wrap back to the inner edge
        g = self.g
        for a, (n, per) in enumerate(zip((self.nx, self.ny, self.nz), self.periodic)):
            if not per:
                continue
            f[_sl(a, n + g - 1)] += f[_sl(a, g - 1)]
            f[_sl(a, g)] += f[_sl(a, n + g)]
            f[_sl(a, g - 1)] = 0.0
            f[_sl(a, n + g)] = 0.0

    def _fill(self, f):  # fill the one ghost layer the gather stencil reads (periodic wrap)
        g = self.g
        for a, (n, per) in enumerate(zip((self.nx, self.ny, self.nz), self.periodic)):
            if not per:
                continue
            f[_sl(a, g - 1)] = f[_sl(a, n + g - 1)]
            f[_sl(a, n + g)] = f[_sl(a, g)]

    def update_void_fraction(self, pos):
        self._solidvol[...] = 0.0
        self._c.deposit_solid_volume(pos, self._rad, self._solidvol, 0.0, 0.0, 0.0, self.h,
                                     self.ex, self.ey, self.ez, self.g)
        self._fold(self._solidvol)
        self._c.compute_void_fraction(self._solidvol, self._eps, self.inv_vcell, self.eps_min)
        self._fill(self._eps)

    def compute_forces(self, pos, vel):
        for name in ("u", "v", "w"):
            self.flow.exchange_field(name)
        uf, vf, wf = (self._fv(n) for n in ("u", "v", "w"))
        fx, fy, fz = (self._fv(n) for n in ("force_x", "force_y", "force_z"))
        gm = (0.0, 0.0, 0.0, self.h, self.ex, self.ey, self.ez, self.g)
        if self.implicit_drag:
            db = self._fv("drag_beta")
            self._c.compute_drag_implicit(pos, vel, self._rad, uf, vf, wf, self._eps, self._fdrag,
                                          db, fx, fy, fz, *gm, self.mu, self.rho, self.inv_vcell,
                                          self.drag_kind)
            self._fold(db)
        else:
            self._c.compute_drag_feedback(pos, vel, self._rad, uf, vf, wf, self._eps, self._fdrag,
                                          fx, fy, fz, *gm, self.mu, self.rho, self.inv_vcell,
                                          self.drag_kind)
        self._c.interpolate_velocity(pos, uf, vf, wf, self._ufluid, *gm)
        sl = self._ufluid - vel  # slip the drag saw (host copy for inspection)
        self.last_slip = sl.get() if self.device else sl.copy()
        self._fold(fx)
        self._fold(fy)
        self._fold(fz)

    def slip(self, pos, vel):
        """Interpolated fluid velocity minus particle velocity (N,3) — what the drag law sees.
        `vel` may be host or device; returns a host NumPy array for convenient inspection."""
        s = self._ufluid - self.xp.asarray(vel)
        return s.get() if self.device else s

    def step(self):
        pos, vel = self._particles()
        self.update_void_fraction(pos)
        self.compute_forces(pos, vel)
        self.last_drag = (self._fdrag.get() if self.device else self._fdrag.copy())
        self.last_eps = self._eps
        if self.move_particles:
            # dem's set_external_forces takes a host array; copy down on a GPU build.
            self.dem.set_external_forces(self._fdrag.get() if self.device else self._fdrag)
            for _ in range(self.dem_substeps):
                self.dem.step(self.dt_dem)
        self.flow.step()
