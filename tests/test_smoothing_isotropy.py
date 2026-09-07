"""The porosity filter's width is a PHYSICAL length, on a mesh whose cells are boxes.

`smoothField` runs `n` sweeps of the explicit Laplacian with a per-axis coefficient `alpha_a`,
which gives a Gaussian of variance `sigma_a^2 = 2 alpha_a n` CELLS along axis `a`, i.e. a physical
`2 alpha_a n h_a^2`. The volume-filtering literature (Capecelatro & Desjardins 2013; MFIX's
DES_DIFFUSE_WIDTH) defines the filter by ONE physical width set from the particle diameter, so on a
box mesh the coefficients must be `alpha_a = C / h_a^2` — one C, three alphas — and the
explicit-diffusion stability bound `sum_a 2 alpha_a <= 1` caps `C = 1/(2 sum_a 1/h_a^2)`.

This gate measures that directly: deposit a unit spike, smooth, and read the SECOND MOMENTS of the
result in physical length. Three checks:

  A. per-axis alpha  -> the three physical sigmas agree, and each equals sqrt(2 C n).
  B. one alpha (the ablation, i.e. what a cubic-mesh formula does on a box mesh) -> the physical
     sigmas are in the ratio of the spacings, which is the defect.
  C. a CUBIC mesh with the per-axis coefficients is BITWISE the single-alpha kernel — the isotropic
     path is untouched, which is what `alpha_y`/`alpha_z < 0` (the "same as alpha" sentinel) buys.

Run: PYTHONPATH=<coupling build> python tests/test_smoothing_isotropy.py
"""
import numpy as np
from peclet.coupling import _coupling as C

N, G = 64, 2   # N large enough that the widest axis (h=0.5) stays clear of the walls
E = N + 2 * G
H = (1.0, 2.0, 0.5)          # a deliberately box-shaped cell
NSWEEP = 40


def smooth(alpha, ayz):
    f = np.zeros((E, E, E), order="F")
    f[G + N // 2, G + N // 2, G + N // 2] = 1.0
    C.smooth_solid_volume(f, 0.0, 0.0, 0.0, 1.0, E, E, E, G, NSWEEP, alpha, 0, *ayz)
    return f


def sigmas(f, h):
    inner = np.asarray(f)[G:G + N, G:G + N, G:G + N]
    tot = inner.sum()
    out = []
    for a in range(3):
        m = inner.sum(axis=tuple(i for i in range(3) if i != a))
        x = (np.arange(N) - (N // 2 - 0)) * h[a]
        mean = (m * x).sum() / tot
        out.append(np.sqrt((m * (x - mean) ** 2).sum() / tot))
    return tot, out


ok = True
inv2 = sum(1.0 / (x * x) for x in H)
Cc = 1.0 / (2.0 * inv2)
alpha = tuple(Cc / (x * x) for x in H)
sig_exact = np.sqrt(2.0 * Cc * NSWEEP)

tot, s = sigmas(smooth(alpha[0], alpha[1:]), H)
spread = (max(s) - min(s)) / np.mean(s)
err = max(abs(v / sig_exact - 1.0) for v in s)
print(f"A per-axis alpha={tuple(round(v, 6) for v in alpha)}  sigma_phys="
      f"{[round(v, 6) for v in s]}  target {sig_exact:.6f}")
print(f"  anisotropy {spread:.3e} (gate 1e-6)   error vs sqrt(2 C n) {err:.3e} (gate 1e-6)"
      f"   mass {tot:.15f}")
ok &= spread < 1e-6 and err < 1e-6 and abs(tot - 1.0) < 1e-12

_, s1 = sigmas(smooth(1.0 / 6.0, (-1.0, -1.0)), H)
ratio = [s1[a] / s1[0] for a in range(3)]
want = [H[a] / H[0] for a in range(3)]
print(f"B one alpha (ABLATION)      sigma_phys={[round(v, 6) for v in s1]}")
print(f"  ratios {[round(v, 4) for v in ratio]} vs the spacings {want} -> the filter is a box, "
      f"not a ball")
ok &= max(abs(ratio[a] - want[a]) for a in range(3)) < 1e-2

cub = (1.0, 1.0, 1.0)
a_c = tuple(1.0 / (2.0 * 3.0) / 1.0 for _ in cub)
f_per = smooth(a_c[0], a_c[1:])
f_one = smooth(1.0 / 6.0, (-1.0, -1.0))
bit = np.array_equal(np.asarray(f_per).view(np.uint64), np.asarray(f_one).view(np.uint64))
print(f"C cubic mesh: per-axis == single-alpha BITWISE: {bit}  "
      f"max|d| {np.max(np.abs(np.asarray(f_per) - np.asarray(f_one))):.3e}")
ok &= bit

print("SMOOTHING ISOTROPY:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
