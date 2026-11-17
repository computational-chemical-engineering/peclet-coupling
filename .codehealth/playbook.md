# peclet.coupling playbook: user-facing round (WO-35, 2026-11-17; deltas from core's, geom's, pnm's)
- **Fixture**: flow, dem, coupling on ONE Kokkos prefix; the shared trees were Sep/Oct, so flow f5f0b87 + dem 9c36779
  were built into build_deps/ (400 s). Gate: `PECLET_COUPLING_DEPS=build_deps --behaviour-only --base b1a6fe8 --config`.
- **Checks**: `python/peclet_coupling/_checks.py` (staged and installed beside the drivers; CI copies it too).
  Constructors: `@_checks.arguments(check)` binds the call to the signature and checks by name (an in-body
  line raised CfdDem.__init__'s debt 50.4 -> 50.5); functools.wraps keeps inspect.signature (dump bitwise).
- **Step path untouched**: step() calls `_update_void_fraction`/`_compute_forces` directly; only an external call
  of the public methods pays the shape check and the finiteness reduction (a device sync on a GPU build).
- **Found**: 75 silent of 123. float()/int()/bool() conversions accepted 0/negative/NaN/Inf dt, mu, rho, radius,
  eps_min outside (0, 1), dem_substeps 2.5, NaN gravity, a 2-element periodic and string flags (bool('no') is True);
  a short radius array or pos/vel of the wrong shape reached the kernels; slip() broadcast a 1-row vel.
- **Valid input kept**: update_void_fraction with (0, 3) (a rank with no particles), out-of-block internal positions
  (deposit nothing), ResolvedCfdDem mu = 0 (unused there), numpy ints/bools, 0/1 flags. No mpirun probes (checks run
  before any communication); the six tests/test_mpi_*.py scripts ran base vs head at np 1, 2: 24/24 PASS, 38 s.
- **Wall clock (TRS2)**: deps 400 s; gate 58 s (ctest 4 tests 29 s); probes 123 cases 7 s; pytest 87 cases 0.7 s.
