# Bug: porous CFD-DEM crashes on CUDA — cross-module async race

**Status: RESOLVED 2026-07-06 — the diagnosis below did NOT hold up.** There was no cross-module
async race. The illegal-address crash was a DEM broadphase pair-buffer overflow (raw candidate count
fed to the narrowphase as a loop bound; fixed by `findCollisionsGrow`, dem `d4d4093`). The subsequent
porous NaN/blow-ups were two flow defects — the GraphAMG bottom solve diverging on domain-BC
operators (force-enabled for porous+drag) and the drag diagonal never entering the momentum operator
on the all-fluid domain-BC smoother path — see `flow/doc/porous_drag_projection_plan.md` §2 for the
measured root causes and fixes. The "CUDA-only" appearance was the 50-iteration pressure cap
truncating a diverging solve at backend-dependent points. Kept for the diagnostic history below.

## Symptom

Running the coupled solver with the **volume-averaged (porous) continuity** on a CUDA build crashes
after a few coupled steps with:

```
cudaStreamSynchronize(stream) error( cudaErrorIllegalAddress): an illegal memory access was encountered
  extern/src/kokkos/core/src/Cuda/Kokkos_Cuda_Instance.cpp:158
```

- **CUDA only.** The identical run on the OpenMP build is fine.
- **Porous only.** `CfdDem(..., porous=False)` (incompressible drag-only) does **not** crash — including
  the 1 M-grain realistic case (`scratchpad/fb_si.py`, 20×20×60, 200 k, ran 60 steps clean).
- **Timing / grid-size dependent.** The small grid-unit example
  (`coupling/examples/fluidized_bed.py`, `Params.porous=True`, 8×8×18) crashes at **step ~3**; the
  larger realistic grid ran 60 steps before (by luck) not tripping it. This is the signature of a race,
  not a deterministic bug.
- **State is finite before the crash** (`pos`, `u` all finite, `umax≈0.2`, residual `≈0.05`). It is an
  out-of-range memory access, not a NaN blow-up.

## Repro

```bash
cd suite
export PATH=/usr/local/cuda-13.2/bin:$PATH
PP="$PWD/flow/build_cuda_mphys:$PWD/dem/build_cuda_mphys:$PWD/coupling/build_cuda_mphys"
OMP_PROC_BIND=false PYTHONPATH="$PP:$PWD/coupling" flow/.venv/bin/python -c "
from examples import fluidized_bed as fb
P=fb.Params(); P.porous=True
s,d,cpl,npart=fb.build(P)
for k in range(8): cpl.step()
print('done')"
# -> cudaErrorIllegalAddress around step 3
```

## Diagnosis — it is an asynchronous cross-stream race (confirmed)

Four experiments pin it down:

| experiment | result | conclusion |
|---|---|---|
| `compute-sanitizer --tool memcheck` | runs clean, "done", **0 errors** | heavy serialization hides it → not a plain OOB |
| `compute-sanitizer --tool initcheck` | **crashes**, 0 uninitialized reads | not an uninitialized-memory read |
| `CUDA_LAUNCH_BLOCKING=1` | **survives** 12 steps | serializing all launches fixes it → **it is a race** |
| swap the pressure driver (GraphAMG / V-cycle / PCG) | **all three crash** | **NOT a GraphAMG bug** (my earlier guess was wrong) |
| `deviceSynchronize()` *before* `flow.step()` | still crashes | the race is not "coupling writes not ready for flow" |
| `deviceSynchronize()` *after* `flow.step()` | **survives** 15 steps | **the race is: `flow.step()`'s async kernels vs the *next* step's coupling writes** |

### Root cause

`flow`, `dem`, and `coupling` are **three separate nanobind `.so` modules**, each with its own Kokkos
runtime and therefore its own **default CUDA stream**. `flow.step()` launches its kernels
asynchronously and returns before they finish. The Python loop then proceeds to the next
`cpl.step()`, whose `update_void_fraction` runs the **coupling** module's deposit kernels (on the
*coupling* stream) that **overwrite the `eps` flow field while the previous `flow.step()`'s porous
kernels (on the *flow* stream) are still reading it** — a read/write race across two unsynchronized
streams → the flow reads a half-updated / reallocated buffer → illegal address.

Why porous-specific: the porous path is the one that routes `eps` **through a registered flow field**
read inside `project()` (`divergOpenEps`, `buildPorousCoeff*`, the `d(eps)/dt` kernels — see
`flow/src/flow_ibm.hpp` `project()` and `flow/src/mac_pressure.hpp`). The incompressible path keeps
`eps` in a coupling-owned scratch array and doesn't read it in `flow.step()`, so its race window
doesn't hit this buffer. The cross-module fields the coupling writes and the flow reads are `eps`,
`drag_beta`, and `force_{x,y,z}` (driver `_fv(...)`, deposited in `update_void_fraction` /
`compute_forces`).

## Confirmed quick fix (but not the right one)

A `deviceSynchronize()` after `flow.step()` in `CfdDem.step()` (driver.py) removes the crash. But a
full device fence every step **kills the async overlap** the suite is built around and would serialise
the whole pipeline. Do not ship this as-is.

## The proper fix (open work)

Options, roughly in order of preference:

1. **One shared Kokkos stream across the modules.** If `flow`, `dem`, `coupling` shared a single Kokkos
   execution-space instance / CUDA stream, all kernels would be ordered on that stream and the race
   would vanish with zero fences. Investigate how each module initializes Kokkos (do they share one
   `Kokkos::initialize`? they're separate `.so`s — likely separate default streams) and whether the
   coupling kernels can be launched on the flow's stream (the fields live in the flow's memory space).
2. **A targeted fence at the field-ownership boundary**, not a blanket one: fence only the flow's
   `eps`/`drag_beta`/`force` producers/consumers. E.g. `flow.step()` fences at its end on CUDA (cheap
   relative to a 1.4 s solve), or the coupling fences the flow before it deposits into flow-owned
   fields. Weigh against #1.
3. **Make the cross-module handoff explicit in the zero-copy bridge.** The DLPack views passed between
   modules carry no stream/event, so the consumer can't wait on the producer. A correct fix would
   attach/synchronize a CUDA event at each `field_view` handoff (DLPack `dl_tensor` stream semantics),
   so cupy/Kokkos consumers order after the producer. This is the general, reusable fix for the whole
   suite's cross-module device interop, not just porous.

## Handles / where to look

- Driver + cross-module writes: `coupling/python/peclet_coupling/driver.py`
  (`step`, `update_void_fraction`, `compute_forces`, `_fv`, `_particles`).
- Porous reads in the flow: `flow/src/flow_ibm.hpp::project()` (staggered branch: `divergOpenEps`,
  `d(eps)/dt` kernel, the porous coefficient block) and `flow/src/mac_pressure.hpp`
  (`divergOpenEps`, `buildPorousCoeff`, `buildPorousCoeffDrag`, `projectCorrectPorousDrag`).
- Zero-copy bridge: `core/python/ndarray_interop.hpp` (DLPack ↔ Kokkos View) and each module's
  `*_bindings.cpp` (`field_view`, `get_*_view`).
- Realistic (harder-to-trip but same latent bug) repro: `scratchpad/fb_si.py` with the porous flag.

## Related, already-fixed context (do not re-do)

- The semi-implicit-drag pressure correction (`w_f=idt/(idt+beta)`) + a real device-capture bug in the
  porous kernels (`e_.x` inside a `KOKKOS_LAMBDA` → host-pointer read) were fixed in `flow 07ea855`.
- The porous+drag solver default is GraphAMG+PCG (`flow 5816b25`) — orthogonal to this race.
- The ArborX broadphase OOM guard (`dem 382372d`) is orthogonal.
