"""Multi-rank void-fraction SMOOTHING: distributed diffusive smoothing must reproduce the
single-rank result exactly.

The MFIX-style diffusive smoothing (smooth_width) was single-rank-only: the sweep treated every
local block face as a zero-flux wall, so a rank boundary acted as a spurious internal wall. Now
interior rank faces read the halo ghost (open_faces mask) and the driver refreshes the solidvol
halo before every Jacobi sweep, which makes the multi-rank sweep arithmetic identical to the
single-rank closed-box sweep (global faces stay zero-flux, matching the validated single-rank
path byte-for-byte).

This test deposits a deterministic particle cloud, smooths, and checks:
  1. global solid-volume conservation (smoothing must not create/destroy hold-up), and
  2. the gathered global eps field matches the np=1 reference to ~machine precision.

Run:  mpirun -np 1 python test_mpi_smoothing.py   (writes the reference)
      mpirun -np {2,4} python test_mpi_smoothing.py
"""
import os
import numpy as np
import peclet.flow
import peclet.dem
from peclet.coupling import CfdDem
from mpi4py import MPI

REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoothing_ref_eps.npy")
N = 16          # global grid N^3, h = 1
R = 0.3         # particle radius
SMOOTH_W = 2.0  # smoothing length in cells
EPS_MIN = 0.05


def global_particles():
    rng = np.random.default_rng(42)
    # keep a margin off the domain faces so the trilinear deposit never lands in a non-periodic
    # corner case; the box is fully periodic anyway.
    return (rng.uniform(0.5, N - 0.5, size=(200, 3))).astype(np.float32)


def run(comm):
    rank, size = comm.Get_rank(), comm.Get_size()
    gpos = global_particles()

    if size > 1:
        (ox, oy, oz), (lnx, lny, lnz) = peclet.flow.mpi_block(N, N, N)
    else:
        (ox, oy, oz), (lnx, lny, lnz) = (0, 0, 0), (N, N, N)
    cell = np.floor(gpos).astype(int)
    keep = ((cell[:, 0] >= ox) & (cell[:, 0] < ox + lnx) &
            (cell[:, 1] >= oy) & (cell[:, 1] < oy + lny) &
            (cell[:, 2] >= oz) & (cell[:, 2] < oz + lnz))
    mine = gpos[keep]
    Np = mine.shape[0]

    s = peclet.flow.Solver(lnx, lny, lnz)
    s.set_rho(1.0); s.set_mu(1.0); s.set_dt(0.1)
    if size > 1:
        s.init_mpi(N, N, N)
    s.set_pressure_geometry(np.asfortranarray(np.full((lnx, lny, lnz), 10.0)))

    d = peclet.dem.Simulation(max(Np, 1))
    d.initialize(shape_type=1, radius=R)
    d.set_domain((0, 0, 0), (N, N, N))
    d.enable_periodicity(True, True, True)
    posw = np.concatenate([mine, np.zeros((Np, 1), dtype=np.float32)], axis=1)  # invMass 0: fixed
    d.set_positions(posw)
    d.set_velocities(np.zeros((Np, 3), dtype=np.float32))

    cpl = CfdDem(s, d, fluid_dt=0.1, mu=1.0, rho=1.0, radius=R, drag="stokes",
                 eps_min=EPS_MIN, smooth_width=SMOOTH_W, move_particles=False)
    assert cpl._smooth_sweeps > 0, "smoothing must be active under MPI now"
    cpl._resize_particles(Np)
    cpl.update_void_fraction(cpl.xp.asarray(mine))

    g = cpl.g
    sv = cpl._fv("solidvol") if cpl._eps_is_field else cpl._solidvol
    ep = cpl._eps
    if cpl.device:
        sv, ep = sv.get(), ep.get()
    sv_in = np.asarray(sv)[g:g + lnx, g:g + lny, g:g + lnz]
    ep_in = np.asarray(ep)[g:g + lnx, g:g + lny, g:g + lnz]

    # 1) conservation: smoothing must preserve the total deposited solid volume.
    vol = comm.allreduce(float(sv_in.sum()), op=MPI.SUM)
    r32 = float(np.float32(R))  # the kernel deposits with the float32-cast radius
    vol_exact = gpos.shape[0] * (4.0 / 3.0) * np.pi * r32 ** 3
    cons_err = abs(vol - vol_exact) / vol_exact

    # 2) gather the global eps and compare with the np=1 reference.
    blocks = comm.gather(((ox, oy, oz), np.ascontiguousarray(ep_in)), root=0)
    ok = cons_err < 1e-12
    if rank == 0:
        geps = np.zeros((N, N, N))
        for (bx, by, bz), b in blocks:
            geps[bx:bx + b.shape[0], by:by + b.shape[1], bz:bz + b.shape[2]] = b
        if size == 1:
            np.save(REF_FILE, geps)
            tag = "reference written"
        elif os.path.exists(REF_FILE):
            ref = np.load(REF_FILE)
            err = float(np.max(np.abs(geps - ref)))
            ok = ok and err < 1e-12
            tag = f"vs np=1 max|deps|={err:.3e}"
        else:
            tag = "NO REFERENCE (run np=1 first)"
            ok = False
        print(f"[np={size}] conservation rel-err={cons_err:.3e}  {tag}")
        print(f"MPI SMOOTHING (np={size}): {'PASS' if ok else 'FAIL'}")
    ok = comm.bcast(ok if rank == 0 else None, root=0)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run(MPI.COMM_WORLD)
