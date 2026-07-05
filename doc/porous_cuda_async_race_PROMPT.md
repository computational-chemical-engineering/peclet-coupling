# Session prompt — fix the porous CFD-DEM CUDA async race

Copy the block below into a fresh session (run from `suite/`).

---

Fix a confirmed asynchronous cross-stream race that crashes the volume-averaged (porous) CFD-DEM
coupling on CUDA. Full write-up: **`coupling/doc/porous_cuda_async_race.md`** — read it first.

**One-line diagnosis (already established, don't re-derive):** `flow`, `dem`, and `coupling` are
separate nanobind `.so` modules with separate Kokkos default CUDA streams. `flow.step()` returns while
its kernels are still running; the next `cpl.step()`'s coupling deposit kernels overwrite the `eps`
flow field on a *different* stream while the flow's porous kernels are still reading it → illegal
address. Proven: `CUDA_LAUNCH_BLOCKING=1` fixes it, and a `deviceSynchronize()` *after* `flow.step()`
fixes it; a sync *before* `flow.step()` does not; swapping the pressure driver does not (so it is NOT
GraphAMG). `compute-sanitizer memcheck` hides it (serializes); `initcheck` shows no uninitialized reads.

**Reproduce (crashes ~step 3):**
```bash
export PATH=/usr/local/cuda-13.2/bin:$PATH
PP="$PWD/flow/build_cuda_mphys:$PWD/dem/build_cuda_mphys:$PWD/coupling/build_cuda_mphys"
OMP_PROC_BIND=false PYTHONPATH="$PP:$PWD/coupling" flow/.venv/bin/python -c "
from examples import fluidized_bed as fb
P=fb.Params(); P.porous=True
s,d,cpl,npart=fb.build(P)
for k in range(8): cpl.step()
print('done')"
```
(If a build is stale: `cd <module> && source ../flow/.venv/bin/activate &&
cmake --build build_cuda_mphys -j$(nproc)`. OpenMP builds are `build_mphys`.)

**Goal:** make the porous coupled run stable on CUDA **without** killing async overlap (a blanket
`deviceSynchronize()` every step is the confirmed-working but unacceptable quick hack — the flow solve
is ~1.4 s, serialising the whole pipeline behind it is not OK).

**Preferred direction** (see the doc for detail): make the modules share one Kokkos stream, or attach
CUDA-event/stream semantics to the DLPack cross-module `field_view` handoff
(`core/python/ndarray_interop.hpp`) so consumers order after producers — the general fix for all
cross-module device interop, not just porous. A targeted fence at the flow-field ownership boundary is
an acceptable fallback if a shared stream is infeasible.

**Acceptance:**
- The repro above runs ≥100 steps on CUDA with no illegal-address error and finite state.
- `scratchpad/fb_si.py 200000 6.0 60 1` (realistic 1 mm bed, porous) still runs and fluidizes.
- No measurable slowdown of the *incompressible* coupled path (fb_si with the porous flag off) or the
  single-phase flow — i.e. don't globally serialise; the fix must preserve async overlap.
- OpenMP behavior unchanged; flow regression suite (`flow/tests/regression/sdflow_regression.py`) and
  the dem/flow MPI ctests still pass.

**Do not touch** (already fixed, orthogonal): the semi-implicit-drag pressure correction + the
`e_.x`-in-`KOKKOS_LAMBDA` device-capture fix (`flow 07ea855`); the porous+drag GraphAMG default
(`flow 5816b25`); the ArborX broadphase OOM guard (`dem 382372d`).
