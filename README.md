# peclet.coupling — unresolved point-particle CFD-DEM

Two-way coupling of `peclet.flow` (Eulerian fluid) and `peclet.dem` (Lagrangian particles) for
dilute-to-dense point-particle suspensions and packed beds. Multiphysics Phase 6 — see
`../docs/MULTIPHYSICS_PLAN.md`.

## Design

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
   `ε = clamp(1 − Vsolid/Vcell, eps_min, 1)`. The floor `eps_min` defaults to 0.4 ≈ the
   random-close-packing voidage (the drag correlations are invalid, and Ergun's `1/ε` powers
   explosive, below a physical packing). A particle whose trilinear stencil falls **outside the
   domain by more than one ghost layer** (e.g. pushed through a DEM wall by a violent contact solve)
   is dropped from the exchange entirely — no deposit, zero drag — so a runaway escapee can never
   feed a diverging `β·u_p` source into the boundary row.
2. **Drag + feedback** — gather the fluid velocity and ε at each particle, evaluate the drag law
   (Stokes / Schiller–Naumann / Ergun / Di Felice / Wen & Yu / Gidaspow), write the drag force to the
   particles and deposit the reaction onto the fluid momentum source.
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
cmake --build build -j        # -> build/peclet/coupling/_coupling.*.so
# run the tests (all three build trees on PYTHONPATH):
PYTHONPATH="$PWD/build:$PWD/../flow/build:$PWD/../dem/build" \
  python tests/test_fixed_bed_ergun.py
```

## Follow-ups

Kernel-width (vs trilinear) deposition smoothing; the `ρε` volume-averaged inertia and
`∇·[εμ(∇u+∇uᵀ)]` viscous forms in the gas momentum (accuracy — see
`flow/doc/porous_drag_scheme.md` §6); a PEA-style implicit particle-drag substep for very stiff
*moving* beds (`m_p/β < Δt` — the fluid side is already implicit).
