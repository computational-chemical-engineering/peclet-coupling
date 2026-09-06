"""ResolvedCfdDem -- resolved (geometry-resolving) CFD-DEM.

Layer 4 of suite/docs/ANALYTIC_SDF_GEOMETRY.md. Where `CfdDem` treats a grain as a point with a
drag closure, this driver makes each grain an ANALYTIC SDF INSTANCE in the flow solver's scene: the
fluid resolves the actual surface, no-slip is enforced on the moving wall by the cut-cell IBM
(Layer 3 rung 2), the projection carries the wall's own volume flux (rung 3), and the coupling is a hydrodynamic-load exchange (rung L4-R2) with no drag correlation anywhere in it:
by default the DISCRETE REACTION (route (b) of the design note's OPEN FOR REVIEW 1) -- the momentum
the fluid actually lost to each grain, exactly conservative -- with the reconstructed traction
integral available as force_method="traction" for diagnostics.

UNITS. The bridge is a PURE IDENTITY: dem's positions, quaternions, velocities and angular
velocities go straight into flow's scene, and the hydrodynamic force and torque come straight back,
because a flow solver built with `Solver(cells, extent=...)` holds its scene in the caller's own
physical coordinates and reports force and torque in the caller's units. There is no scale factor
anywhere in this file, and on a cell-unit solver (no extent) the same identity holds with lengths
in cells — which is what the pre-2026-09 driver silently assumed.

Python-composed, like `CfdDem` and for the same reason: dem and flow stay separate method codes and
nothing links them in C++. Per coupling step:

  1. pull dem state (positions / quaternions / velocities / angular velocities) and push it into
     flow's scene -- instance transforms plus rung-2 rigid-body motion;
  2. flow.rebuild_geometry() -- re-derive SDF, cut-cell overlay, apertures and pressure operator;
     the velocity and pressure fields survive it;
  3. flow.step();
  4. flow.hydro_force_torque() -> per-grain force; add gravity/buoyancy; hand to dem;
  5. dem sub-steps at the DEM timestep, holding the fluid force constant.

WEAK, EXPLICIT COUPLING (spec: L4-R3). The fluid force is lagged by one fluid step. That is the
documented v1; a strongly-coupled variant would iterate 3-5 inside the step.

FLOAT/DOUBLE BOUNDARY. dem carries float32 state, flow's scene is float64. The per-step instance
rebuild converts; "zero-copy" is not literal across that divide, and does not need to be -- the
instance array is a few hundred bytes per grain against a geometry rebuild measured in tens of ms.

UNITS. Everything is in flow's grid units: cell spacing 1, cell (i,j,k) centred at (i,j,k), so dem
positions and radii must be expressed in cells.

HYDRODYNAMIC TORQUE (validated 2026-08-31). flow's reaction torque now carries the
transposed-stress wall term (flow `16e91ec`) and is GATED: a spinning sphere reproduces the exact
Stokes torque 8*pi*mu*a^3*Omega to +3.5/+2.4/+2.2% (converging with the aperture first moment),
and THIS loop with apply_torque=True decays a freely spinning sphere at 1.039x the same box's
calibrated rotational drag (rotation_gate.py part B). Before that term the torque was a structural
-31% -- the missing traction mu*(n x Omega) integrates to zero in the FORCE, which is why no
force-based gate ever saw it.

apply_torque stays **off by default** for one remaining reason: dem assigns a DEFAULT inverse
inertia unrelated to the grain's size, so handing a torque to a grain whose inertia was never set
spins it up at an arbitrary rate (the settling gate diverged to 1e+09 that way). Set the physical
principal inertia FIRST -- (2/5) m R^2 for a sphere, or `scene_particle`'s `inv_inertia_unit` --
then enable. The torque is computed and reported through `torques()` either way.

"""
import numpy as np


class ResolvedCfdDem:
    def __init__(self, flow, dem, *, radius, mu, rho_f, fluid_dt, dem_substeps=20,
                 periodic=True, gravity=(0.0, 0.0, 0.0), rho_p=None, move=True,
                 buoyancy=True, apply_torque=False,
                 force_method="reaction"):
        self.flow = flow
        self.dem = dem
        self.mu = float(mu)
        self.rho_f = float(rho_f)
        self.fluid_dt = float(fluid_dt)
        self.dem_substeps = int(dem_substeps)
        self.dt_dem = self.fluid_dt / self.dem_substeps
        self.gravity = np.asarray(gravity, dtype=np.float64)
        self.move = bool(move)
        self.buoyancy = bool(buoyancy)
        self.radius = float(radius)
        self.rho_p = float(rho_p) if rho_p is not None else None
        self.periodic = bool(periodic)
        # Hand the reaction TORQUE to dem as well as the force (dem R2, set_external_torques).
        # The torque is VALIDATED (rotating-sphere + spin-decay gates, see the class docstring);
        # off by default only because dem's default inverse inertia is not the grain's -- set a
        # physical principal inertia first, then enable.
        self.apply_torque = bool(apply_torque)
        # "reaction" (default): the discrete-reaction force -- exactly conservative (the momentum
        # the fluid lost IS the momentum the grain gains) and as accurate as the flow solution it
        # sustains. "traction": the reconstructed surface integral, kept as a diagnostic; it
        # under-reads the drag by a resolution-independent ~29% (measured), which in this loop
        # shows up as a total-momentum leak. See suite/docs/ANALYTIC_SDF_GEOMETRY.md OPEN FOR
        # REVIEW 1.
        if force_method not in ("reaction", "traction"):
            raise ValueError("force_method must be 'reaction' or 'traction'")
        self.force_method = force_method
        self.n = int(dem.num_particles())
        self.last_force = np.zeros((self.n, 3))
        self.last_torque = np.zeros((self.n, 3))
        flow.set_dt(self.fluid_dt)
        dem.set_dt(self.dt_dem)   # kept in sync, though step(dt) below passes it explicitly
        self._install_scene()

    # --- L4-R1: the dem -> scene bridge --------------------------------------------------------
    _KN_R, _KI_I, _KI_R = 16, 2, 17

    def _install_scene(self):
        """One kSphere node, one instance per grain. Spheres first, per the spec; a shaped grain is
        the same bridge with the grain's own node tree in place of the sphere leaf."""
        node_ints = np.array([1, -1, -1], dtype=np.int32)          # kSphere
        node_reals = np.zeros(self._KN_R)
        node_reals[0] = self.radius
        node_reals[14] = 1.0                                       # quaternion w
        node_reals[15] = 1.0                                       # scale
        ii, ir = self._instance_arrays()
        self.flow.set_scene(node_ints, node_reals, ii.ravel(), ir.ravel(), periodic=self.periodic)
        self._push_motion()
        self.flow.set_solid_from_scene(True)

    def _instance_arrays(self):
        pos = np.asarray(self.dem.get_positions(), dtype=np.float64)
        quat = np.asarray(self.dem.get_quaternions(), dtype=np.float64)
        ii = np.zeros((self.n, self._KI_I), dtype=np.int32)
        ir = np.zeros((self.n, self._KI_R))
        ii[:, 0] = 0        # shapeRoot
        ii[:, 1] = -1       # materialId
        ir[:, 0:3] = pos[:, 0:3]
        ir[:, 3:7] = quat[:, 0:4]   # (x,y,z,w)
        ir[:, 7] = 1.0              # scale
        return ii, ir

    def _push_motion(self):
        """Instance transforms + rigid-body velocities, straight from dem state."""
        pos = np.asarray(self.dem.get_positions(), dtype=np.float64)
        quat = np.asarray(self.dem.get_quaternions(), dtype=np.float64)
        vel = np.asarray(self.dem.get_velocities(), dtype=np.float64)
        omg = np.asarray(self.dem.get_angular_velocities(), dtype=np.float64)
        for i in range(self.n):
            self.flow.set_instance_transform(i, pos[i, 0:3].tolist(), quat[i, 0:4].tolist())
            self.flow.set_instance_motion(i, lin_vel=vel[i, 0:3].tolist(),
                                          ang_vel=omg[i, 0:3].tolist())

    # --- L4-R3: the motion loop ----------------------------------------------------------------
    def step(self):
        if self.move:
            self._push_motion()
            self.flow.rebuild_geometry()
        self.flow.step()
        if self.force_method == "reaction":
            ft = np.asarray(self.flow.hydro_force_torque_reaction())   # (2, n, 3): F, tau
            self.last_force = np.array(ft[0][: self.n], dtype=np.float64)
            self.last_torque = np.array(ft[1][: self.n], dtype=np.float64)
            self.last_force_pressure = None   # the reaction has no pressure/viscous split
            self.last_force_viscous = None
        else:
            ft = np.asarray(self.flow.hydro_force_torque())   # (4, n, 3): F, tau, F_p, F_visc
            self.last_force = np.array(ft[0][: self.n], dtype=np.float64)
            self.last_torque = np.array(ft[1][: self.n], dtype=np.float64)
            self.last_force_pressure = np.array(ft[2][: self.n], dtype=np.float64)
            self.last_force_viscous = np.array(ft[3][: self.n], dtype=np.float64)
        if not self.move:
            return
        F = self.last_force.copy()
        if self.buoyancy and self.rho_p is not None:
            # The resolved traction already contains the hydrostatic part of the pressure field, so
            # what is added here is only the BODY force on the grain itself. Net gravity on a grain
            # of density rho_p displacing rho_f is (rho_p - rho_f) V g when the fluid carries the
            # hydrostatic gradient; when it does not (the usual periodic set-up, no gravity in the
            # fluid), the full rho_p V g applies. buoyancy=False selects the latter.
            V = 4.0 / 3.0 * np.pi * self.radius**3
            F += (self.rho_p - self.rho_f) * V * self.gravity
        elif self.rho_p is not None:
            V = 4.0 / 3.0 * np.pi * self.radius**3
            F += self.rho_p * V * self.gravity
        self.dem.set_external_forces(np.ascontiguousarray(F, dtype=np.float32))
        if self.apply_torque:
            # World-frame torque; dem rotates it into the body frame in the predictor. Held
            # constant over the sub-steps, exactly like the force.
            self.dem.set_external_torques(
                np.ascontiguousarray(self.last_torque, dtype=np.float32))
        for _ in range(self.dem_substeps):
            # dt MUST be passed explicitly. dem's step(dt=0) is a dynamics-free relaxation step
            # (overlap removal only), so step() with no argument advances nothing and the driver
            # would run happily with a frozen particle.
            self.dem.step(self.dt_dem)

    def forces(self):
        return self.last_force

    def torques(self):
        return self.last_torque
