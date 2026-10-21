"""
Compressible Euler equations — Sod shock tube: EXACT Riemann solution
======================================================================

Projects the 1-D exact Riemann solution onto the same 2-D DOLFINx mesh
used in euler_sod.py.  No time-stepping: we simply evaluate the analytical
solution at each DOF coordinate x at every snapshot time and write it as a
DOLFINx Function — so the output is perfectly compatible with euler_sod.pvd.

PDE:  ∂_t U + ∂_x F(U) = 0,   U = (ρ, ρu, ρv, E)ᵀ
EOS:  p = (γ−1)(E − ½ρ(u²+v²))

Solution structure (left → right, diaphragm at x = 0.5):
    undisturbed L | rarefaction fan | ★L   contact   ★R | shock | undisturbed R

Output: results/euler_sod_exact.pvd  (open in ParaView alongside euler_sod.pvd).
"""

import os
from mpi4py import MPI
import numpy as np
from scipy.optimize import brentq
from dolfinx import mesh, fem, io
from petsc4py import PETSc

# ── parameters ───────────────────────────────────────────────────────────────
GAMMA  = 1.4    # must match euler_sod.py
T_END  = 0.2
N_SNAP = 20

# ── mesh: identical to euler_sod.py ──────────────────────────────────────────
nx, ny = 600, 20
msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [1.0, 0.2]],
    [nx, ny],
    mesh.CellType.triangle,
)

# ── function space: identical to euler_sod.py ─────────────────────────────────
V   = fem.functionspace(msh, ("Lagrange", 1, (4,)))   # U = (ρ, ρu, ρv, E)
U_h = fem.Function(V, name="U")

# ── DOF x-coordinates (used to evaluate the 1-D exact solution) ──────────────
x_dof = V.tabulate_dof_coordinates()[:, 0]   # shape (n_nodes,)

# ── Sod initial states ────────────────────────────────────────────────────────
rho_L, u_L, p_L = 1.0,   0.0, 1.0
rho_R, u_R, p_R = 0.125, 0.0, 0.1

# ── exact Riemann solver (solve once; reuse star state at every t) ────────────
GM1 = GAMMA - 1.0
GP1 = GAMMA + 1.0
c_L = np.sqrt(GAMMA * p_L / rho_L)    # left  sound speed
c_R = np.sqrt(GAMMA * p_R / rho_R)    # right sound speed

def _f(p, rho_K, p_K, c_K):
    """Pressure function (rarefaction or shock contribution)."""
    if p <= p_K:
        return (2*c_K / GM1) * ((p / p_K)**(GM1 / (2*GAMMA)) - 1.0)
    else:
        A = 2.0 / (GP1 * rho_K)
        B = GM1 / GP1 * p_K
        return (p - p_K) * np.sqrt(A / (p + B))

p_star = brentq(
    lambda p: _f(p, rho_L, p_L, c_L) + _f(p, rho_R, p_R, c_R) + (u_R - u_L),
    1e-10, max(p_L, p_R) * 10, xtol=1e-12)
u_star = u_L - _f(p_star, rho_L, p_L, c_L)

# densities in the two star sub-regions
rho_sL = rho_L * (p_star / p_L)**(1.0 / GAMMA)             # through rarefaction
rho_sR = rho_R * (p_star/p_R + GM1/GP1) / (GM1/GP1 * p_star/p_R + 1.0)  # shock

# left star sound speed
c_sL = c_L * (p_star / p_L)**(GM1 / (2.0 * GAMMA))

# wave speeds (in similarity variable ξ = (x − 0.5)/t)
S_HL = u_L - c_L                                          # rarefaction head
S_TL = u_star - c_sL                                      # rarefaction tail
S_C  = u_star                                             # contact
S_R  = u_R + c_R * np.sqrt(GP1/(2*GAMMA)*p_star/p_R + GM1/(2*GAMMA))  # shock

print("─── Exact Riemann star state ─────────────────────────────")
print(f"  p★={p_star:.6f}  u★={u_star:.6f}")
print(f"  ρ★L={rho_sL:.6f}  ρ★R={rho_sR:.6f}")
print(f"  S_HL={S_HL:.4f}  S_TL={S_TL:.4f}  S_C={S_C:.4f}  S_R={S_R:.4f}")
print("──────────────────────────────────────────────────────────")

def exact_cons(t, x_arr):
    """
    Return (ρ, ρu, ρv, E) arrays evaluated from the exact 1-D Riemann
    solution at time t on x-coordinates x_arr.  v = 0 everywhere.
    """
    xi   = (x_arr - 0.5) / t
    rho  = np.empty_like(xi)
    u    = np.zeros_like(xi)
    p    = np.empty_like(xi)

    # vectorised assignment using masks
    fan  = (xi > S_HL) & (xi <= S_TL)
    sL   = (xi > S_TL) & (xi <= S_C)
    sR   = (xi > S_C)  & (xi <= S_R)
    rgt  = xi > S_R

    # undisturbed left (default)
    rho[:] = rho_L;  p[:] = p_L

    # inside rarefaction fan
    u_f         = 2.0/GP1 * (c_L + GM1/2.0 * u_L + xi[fan])
    c_f         = u_f - xi[fan]
    rho[fan]    = rho_L * (c_f / c_L)**(2.0/GM1)
    p[fan]      = p_L   * (c_f / c_L)**(2.0*GAMMA/GM1)
    u[fan]      = u_f

    # left star
    rho[sL] = rho_sL;  p[sL] = p_star;  u[sL] = u_star

    # right star
    rho[sR] = rho_sR;  p[sR] = p_star;  u[sR] = u_star

    # undisturbed right
    rho[rgt] = rho_R;  p[rgt] = p_R  # u already 0

    # conservative variables (v = 0)
    E = p / GM1 + 0.5 * rho * u**2
    return rho, rho*u, np.zeros_like(u), E

# ── write snapshots ───────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
vtk = io.VTKFile(msh.comm, "results/euler_sod_exact.pvd", "w")

# t = 0: initial condition
arr = U_h.x.array.reshape(-1, 4)
L0  = x_dof < 0.5
arr[L0,  0] = rho_L;  arr[L0,  1] = rho_L*u_L;  arr[L0,  2] = 0.0
arr[L0,  3] = p_L/GM1 + 0.5*rho_L*u_L**2
arr[~L0, 0] = rho_R;  arr[~L0, 1] = rho_R*u_R;  arr[~L0, 2] = 0.0
arr[~L0, 3] = p_R/GM1 + 0.5*rho_R*u_R**2
U_h.x.scatter_forward()
vtk.write_function(U_h, 0.0)

snap_interval = T_END / N_SNAP
print(f"{'snap':>6}  {'t':>8}  {'min ρ':>10}  {'max u':>10}")
for k in range(1, N_SNAP + 1):
    t = k * snap_interval
    rho_e, rhou_e, rhov_e, E_e = exact_cons(t, x_dof)
    arr[:, 0] = rho_e
    arr[:, 1] = rhou_e
    arr[:, 2] = rhov_e
    arr[:, 3] = E_e
    U_h.x.scatter_forward()
    vtk.write_function(U_h, t)
    u_max = np.abs(rhou_e / np.maximum(rho_e, 1e-14)).max()
    print(f"{k:6d}  {t:8.4f}  {rho_e.min():10.4f}  {u_max:10.4f}")

vtk.close()
print("Done — open results/euler_sod_exact.pvd in ParaView.")
