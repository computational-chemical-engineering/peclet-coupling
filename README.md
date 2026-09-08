# peclet.coupling — CFD-DEM coupling, unresolved and resolved

Two-way coupling of `peclet.flow` (Eulerian fluid) and `peclet.dem` (Lagrangian particles). Two
drivers, one vocabulary:

- **`CfdDem`** — the **unresolved** point-particle driver for dilute-to-dense suspensions and packed
  beds: a grain is a point with a drag closure (Multiphysics Phase 6, `../docs/MULTIPHYSICS_PLAN.md`).
- **`ResolvedCfdDem`** — the **resolved** driver: each grain is an analytic SDF instance in the flow
  solver's scene, the fluid resolves its surface, and there is no drag correlation anywhere
  (see [Resolved coupling](#resolved-coupling-resolvedcfddem) below).

Both live in the caller's own PHYSICAL units: the drivers take the cell size and the lower corner
from the flow solver itself (`flow.spacing`, `flow.origin`), so a solver built with
`Solver(cells, extent=...)` couples to a DEM stated in metres with no conversion anywhere, and a
cell-unit solver (no extent) keeps spacing 1. Nothing in the coupling API is stated in cells.

## Design (`CfdDem`)

Physics-free glue. The compute kernels (particle↔grid deposition, drag laws, momentum feedback) live
in the `_coupling` nanobind extension and run **in place** on the arrays the two solvers already
expose — the fluid grid fields zero-copy through `flow.field_view(...)`, the particle drag
round-tripped through the dem host API. **There is no C++ link between flow and dem**: the Python
`CfdDem` driver (`python/peclet_coupling/driver.py`) composes them. This mirrors the suite's
architecture (Python is the composition layer).

Per fluid step (`CfdDem.step()`):
1. **Void fraction** — scatter each particle's volume onto the grid (trilinear, **wall-aware**: near
   an immersed solid the weights re-normalise over the fluid corners so no hold-up leaks into walls),
   fold the ghost deposits (periodic wrap on periodic axes; **same-side fold onto the boundary cell
   at a non-periodic domain face** — a grain resting on the distributor scatters part of its volume
   below z=0, and that hold-up belongs to the bottom cell, not to a ghost the fluid never owns), and
   `ε = clamp(1 − Vsolid/Vcell, eps_min, 1)`. The floor `eps_min` defaults to **0.25**, a physical
   regularisation rather than a guard: real voidage bottoms out near random close packing (~0.36
   monodisperse, ~0.25 for wide bidisperse mixes), and anything lower can only come from
   interpenetrated particles or deposit artefacts and must not reach the volume-averaged fluid
   (whose projection amplifies the interstitial velocity by `1/ε`). The older 0.4 floor
   under-predicted dense-bed drag ~3x; the interim 0.05 guard let interpenetration artefacts
   detonate a bed. The same 0.25 is the kernel default (`_coupling.compute_void_fraction`); the
   fixed-bed tests pass `eps_min=0.05` explicitly because their uniform lattice never clamps.
   A particle whose trilinear stencil falls **outside the
   domain by more than one ghost layer** (e.g. pushed through a DEM wall by a violent contact solve)
   is dropped from the exchange entirely — no deposit, zero drag — so a runaway escapee can never
   feed a diverging `β·u_p` source into the boundary row.
   The optional **volume filter** is a PHYSICAL width: `smooth_length` (MFIX's `DES_DIFFUSE_WIDTH`,
   Capecelatro & Desjardins' `δ_f`) is a length in the caller's units, set from the particle
   diameter — the point-particle approximation is what needs `δ_f ≫ d_p`, and it must not move
   when the mesh does. It is realised as `n` explicit diffusion sweeps with a per-axis coefficient
   `α_a = C/h_a²`, `C = 1/(2 Σ_a 1/h_a²)`, `n = round(σ² Σ_a 1/h_a²)`, so the Gaussian is a ball
   in space and not in index — on a cubic mesh that reduces to `α = 1/6`, `n = round(3 (σ/h)²)`,
   term for term and bit for bit the cubic formula it generalises. Gate:
   `tests/test_smoothing_isotropy.py` (three physical `σ` equal to 8.9e-10 on a `(1, 2, 0.5)` cell,
   the single-`α` ablation off by exactly the spacing ratios, the cubic mesh bitwise).
2. **Drag + feedback** — gather the fluid velocity and ε at each particle, evaluate the drag law
   (`"stokes"`, `"schiller_naumann"`, `"ergun"`, `"di_felice"`, `"wen_yu"`, `"gidaspow"`,
   `"beetstra"` — Beetstra–van der Hoef–Kuipers 2007, `"tang"` — Tang et al. 2015; one literature
   name each), write the drag force to the particles and deposit the reaction onto the fluid
   momentum source.
3. **Advance** — apply the drag to the particles and sub-step dem `dem_substeps` times (drag held
   constant), then advance the fluid one step (its RHS/operator now carry the feedback).

### Implicit drag (the key stability piece)

An explicit reaction force `−β(u−u_p)` in the fluid RHS **diverges** for the stiff drag coefficient
β of a dense bed (β·dt/ρ ≫ 1; local β reaches ~10³). So the default feedback is **semi-implicit**:
the coupling deposits the linear-drag *coefficient* density onto flow's `drag_beta` field (added to
the momentum diagonal by `flow.enable_drag()`) and the target `β·u_p` onto `force_*` (the RHS), so
the fluid solve becomes `(ρ/dt + β)u = … + β u_p` — unconditionally stable for any β. The particle
side stays explicit (fine for moving particles at moderate β). `implicit_drag=False` selects the
explicit `−F/Vcell` feedback (dilute only).

### Two fluid modes

- **`porous=True` — volume-averaged (use this for beds).** The fluid solves the full volume-averaged
  continuity `∂ε/∂t + ∇·(εu) = 0` (u = the **interstitial** gas velocity) with a SIMPLE-like eps- and
  drag-weighted pressure projection — scheme, defaults and validation in
  `flow/doc/porous_drag_scheme.md`. The pressure-force split is **Model B**: the gas carries the full
  `−∇p`, the particles get drag + gravity, and the literature (Model-A) drag closures are converted
  once inside the kernel, `β_B = β_A/ε` (`model_b` flag). Gas convection (implicit FOU + explicit
  deferred-correction TVD) is enabled by the driver by default (`advection=True`).
- **`porous=False` — dilute simplification.** The fluid stays incompressible (`div u = 0`); ε enters
  the drag correlation only. Cheap and validated dilute→moderate. Note this is *not* "Model B":
  Models A and B both use the full continuity and differ only in the `−ε∇p` vs `−∇p` split.

Other scope notes: deposition uses `atomic_add` ⇒ results are tolerance-, not bit-exact; the `"ergun"`
drag *kind* is the superficial-velocity form built for the incompressible mode — for porous beds use
`"gidaspow"` (its dense branch is the classic interstitial Ergun form).

## Resolved coupling (`ResolvedCfdDem`)

`ResolvedCfdDem(flow, dem, *, radius, mu, rho, fluid_dt, dem_substeps=20, periodic=(bx, by, bz),
gravity=(0, 0, 0), rho_p=None, move_particles=True, buoyancy=True, apply_torque=False,
force_method="reaction")` is Layer 4 of `../docs/ANALYTIC_SDF_GEOMETRY.md`: one `kSphere` scene
node, one instance per grain, installed in the flow solver's scene (`set_scene`). Per coupling
step it pushes dem's positions / quaternions / velocities / angular velocities into the scene as
instance transforms + rigid-body motion, `flow.rebuild_geometry()` re-derives the SDF, cut-cell
overlay, apertures and pressure operator, the fluid steps, and the hydrodynamic load comes back —
by default the **discrete reaction** (the momentum the fluid actually lost to each grain, exactly
conservative), or the reconstructed traction integral with `force_method="traction"` (a
diagnostic; it under-reads the drag by a resolution-independent ~29 %). Gravity/buoyancy is added
(`rho_p`, `buoyancy`), the force (and, with `apply_torque=True`, the torque) is handed to dem, and
dem sub-steps at the DEM timestep with the load held constant (weak, explicit coupling: the fluid
load is lagged by one fluid step). Pure Python — no compiled kernels of its own, so it imports even
without the `_coupling` extension.

The bridge is a pure identity in the caller's units (dem state in, force and torque out; no scale
factor anywhere). The hydrodynamic torque is validated (a spinning sphere reproduces the Stokes
torque `8πμa³Ω` to ~2–3 %) but `apply_torque` is **off by default**: dem's default inverse inertia is
not the grain's, so set the physical principal inertia (`(2/5) m R²` for a sphere) before enabling
it. The per-grain results of the last step are the `last_force` / `last_torque` `(N,3)` arrays
(traction mode also fills `last_force_pressure` / `last_force_viscous`). `periodic` is per-axis like
everywhere else in peclet, but the flow scene's periodic images are all-or-nothing, so a mixed
triple is refused. Gallery: `peclet-examples/examples/rotating-sphere-torque`.

## Backends

`CfdDem` runs on whatever Kokkos backend `peclet.flow` was built for. On a **CUDA/HIP** build the
coupling kernels run on-device, so the driver array-programs through **CuPy** and takes the grid
fields (`flow.field_view`) and particle state (`dem.get_*_view`) zero-copy via DLPack; on a host
build it uses NumPy over the same buffers. Detected automatically from `peclet.flow.execution_space`.

## Validation (`tests/`)

Both cases pass identically on **host-openmp and CUDA (RTX 5080)**:
- **`test_terminal_velocity.py`** — single settling sphere: the slip velocity matches Stokes to
  **0.1–0.2 %** and Schiller–Naumann to **1.4–1.6 %** (the lab-frame speed is ~2× the slip because
  the particle drags its own Stokeslet flow, so the physical comparison is the slip).
- **`test_fixed_bed_ergun.py`** — uniform fixed bed (one particle per cell, ε = 0.6): the measured
  (f_drive, U) pair lands on the Ergun curve to **0.0 %** across the viscous, transition and inertial
  (Re_p ≈ 6) regimes — validating ε deposition, both Ergun terms, and the stable two-way feedback.
- **`test_fixed_bed_ergun_porous.py`** — the same bed on the **volume-averaged (porous, Model B)**
  path with the Gidaspow closure: (f_drive, U = ε·u_interstitial) lands on the Ergun curve to ~3 %
  across all three regimes with no fitted factors — validating the eps-weighted projection, the
  interstitial kinematics and the `β_B = β_A/ε` conversion together.
- **`test_mpi_fixed_bed_ergun.py`** — the fixed-bed Ergun benchmark run **distributed** (flow
  `init_mpi`, each rank couples its ORB block; particle deposits fold across ranks + periodically via
  the reverse/add-reduce halo `exchange_field_add`, deposit origin shifted by the block origin). The
  superficial velocity U (reduced over ranks) lands on the Ergun curve to **0.0 %** and is
  **bit-identical at np=1/2/4** — the distributed deposition + fold + solve reproduce the coupled
  physics exactly.
- P2G/G2P conservation + the gather/scatter adjoint identity: `core` `test_particle_grid` (host + CUDA).

## Multi-rank coupling

`CfdDem` runs distributed when the flow solver is decomposed (`flow.init_mpi(...)`, world size > 1):
each rank couples its **local block**, the deposit grid map is shifted by the block origin (so
particles in global coordinates land locally), and cross-rank + periodic ghost deposits (void
fraction + drag reaction) fold onto their owner with the reverse halo (`exchange_field_add`) instead
of the single-rank NumPy fold. `CfdDem.rebalance(gamma)` forms one weight field
(`1 + gamma * particle_count`) and redistributes BOTH codes onto the same weighted ORB
(`flow.rebalance_by_weights` + `dem.migrate_to_weights`). Give the flow + dem the same decomposition
(matching grid dims / domain) before constructing `CfdDem`.

**Moving particles** (`move_particles=True`): each fluid step `CfdDem` first migrates dem onto flow's
grid partition (`dem.migrate_to_weights`) so every owned particle sits in its rank's block, then runs
the DISTRIBUTED DEM substeps (`dem.step_mpi`, requires `dem.init_mpi` + `dem.enable_mpi_step`). A rank
that momentarily owns no particles still runs the halo collectives (the per-particle kernels are
skipped). Validated `test_mpi_fixed_bed_ergun.py` (static, bit-identical np 1/2/4) and
`test_mpi_moving_suspension.py` (drifting cloud crossing rank boundaries: the distributed
migrate + step + deposit-fold + gather reproduce single-rank to ~2e-7, np 1/2).

Two known limitations of the underlying dem distributed step (not the coupling — every distributed
coupling op is bit-identical to single-rank in isolation): (1) a rank with **zero owned particles but
an incoming ghost** deadlocks the dem step (affects very dilute clouds / np=4 of the moving test);
(2) a *sustained* dilute settling suspension in a triply-periodic box with no buoyancy is an ill-posed,
numerically unstable configuration — at np>1 the flow solve's reduction-floor non-determinism seeds
that instability. Well-posed cases (bounded / driven flow, denser beds) are unaffected.

Note: `dem.get_velocities()` (host copy getter) has a pre-existing failure after a *periodic* DEM
step on CUDA (a Kokkos strided-subview-after-resize limitation, unrelated to the coupling); the
driver uses the zero-copy device *views* throughout and exposes `last_slip` for inspection.

## Build

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/../extern/install/host-openmp"
cmake --build build -j        # -> build/peclet/coupling/_coupling.*.so (+ the staged .py files)
# run the tests (all three build trees on PYTHONPATH); each test is a pytest function AND a script:
export PYTHONPATH="$PWD/build:$PWD/../flow/build:$PWD/../dem/build"
OMP_NUM_THREADS=4 OMP_PROC_BIND=false pytest tests -q -k "terminal or ergun or isotropy"
python tests/test_fixed_bed_ergun.py
# the test_mpi_*.py tests need an MPI build of flow + dem and mpi4py (they SKIP under pytest else):
mpirun -np 2 python tests/test_mpi_fixed_bed_ergun.py
```

CI (`.github/workflows/ci.yml`) builds Kokkos (OpenMP), flow, dem and coupling from source and runs
the single-rank pytest battery on every push.

## Follow-ups

Kernel-width (vs trilinear) deposition smoothing; the `ρε` volume-averaged inertia and
`∇·[εμ(∇u+∇uᵀ)]` viscous forms in the gas momentum (accuracy — see
`flow/doc/porous_drag_scheme.md` §6); a PEA-style implicit particle-drag substep for very stiff
*moving* beds (`m_p/β < Δt` — the fluid side is already implicit).
