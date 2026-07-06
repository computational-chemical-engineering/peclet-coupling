"""Fixed-bed pressure drop vs Ergun — the volume-averaged (porous, Model B) path.

The same uniform fixed bed as test_fixed_bed_ergun.py, but solved with the volume-averaged
continuity (porous=True): the projection enforces div(eps u) = -d(eps)/dt with the eps- and
drag-weighted SIMPLE-like coefficient, the gas velocity inside the bed is INTERSTITIAL
(u_i = U/eps), and the drag closures are converted to Model B inside the kernel
(beta_B = beta_A/eps — the fluid carries the full -grad p, no -eps grad p split).

A periodic box fully packed at uniform eps is driven by a body force f_drive. At steady state
f_drive balances the bed resistance, so (f_drive, U) must lie on the Ergun curve

    dP/L = 150 mu (1-eps)^2 / (eps^3 d^2) U  +  1.75 rho (1-eps) / (eps^3 d) U^2 ,

with U = eps * <u> the superficial velocity (the porous solver's <u> is interstitial — the factor
eps is exactly what distinguishes this test from the incompressible one). Uses the GIDASPOW closure
(dense branch = classic Model-A Gidaspow, the literature form): the Model-B conversion + interstitial
kinematics must land on Ergun with no fitted factors. Atomic deposition => tolerance-based.
"""
import numpy as np
import peclet.flow, peclet.dem
from peclet.coupling import CfdDem


def ergun(U, eps, mu, rho, d):
    om = 1.0 - eps
    return (150.0 * mu * om * om / (eps ** 3 * d * d) * U
            + 1.75 * rho * om / (eps ** 3 * d) * U * U)


def run_bed(f_drive, eps_target=0.6, N=16, mu=1.0, rho=1.0, dt=0.5, steps=120):
    # One FIXED particle per cell centre with V_p = (1-eps)*Vcell -> uniform deposited eps.
    Vp = (1.0 - eps_target)
    r = (Vp * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    xs = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    Np = pos.shape[0]
    posw = np.concatenate([pos, np.zeros((Np, 1), dtype=np.float32)], axis=1)  # invMass 0 -> fixed

    s = peclet.flow.Solver(N, N, N)
    s.set_rho(rho); s.set_mu(mu); s.set_dt(dt)
    s.set_body_force(0.0, 0.0, f_drive)
    s.set_pressure_geometry(np.asfortranarray(np.full((N, N, N), 10.0)))
    d = peclet.dem.Simulation(Np)
    d.initialize(shape_type=1, radius=r)
    d.set_domain((0, 0, 0), (N, N, N))
    d.enable_periodicity(True, True, True)
    d.set_positions(posw)
    d.set_velocities(np.zeros((Np, 3), dtype=np.float32))
    cpl = CfdDem(s, d, fluid_dt=dt, mu=mu, rho=rho, radius=r, drag="gidaspow", eps_min=0.05,
                 move_particles=False, porous=True)
    for _ in range(steps):
        cpl.step()
    ui = s.get_w().mean()  # interstitial velocity (uniform bed)
    eps_mean = float(cpl.last_eps[cpl.g:cpl.g + N, cpl.g:cpl.g + N, cpl.g:cpl.g + N].mean())
    U = eps_mean * ui      # superficial
    return U, ui, eps_mean, ergun(U, eps_mean, mu, rho, 2 * r)


if __name__ == "__main__":
    print("f_drive     U(superf)   u_i       eps    Ergun(U,eps)  rel-err")
    ok = True
    for f_drive in (0.2, 20.0, 1000.0):
        U, ui, eps, dP = run_bed(f_drive)
        err = abs(dP - f_drive) / f_drive
        print(f"{f_drive:8.2f}  {U:.4e}  {ui:.4e}  {eps:.3f}  {dP:.4e}   {err*100:5.1f}%")
        ok = ok and err < 0.10
    assert ok, "porous fixed-bed drag does not reproduce Ergun within 10%"
    print("POROUS FIXED-BED ERGUN: PASS")
