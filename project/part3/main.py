import os  # noqa: I001
from mpi4py import MPI

import time
import dolfinx
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem import petsc
from petsc4py.PETSc import KSP, ScalarType

# Physical and Numerical Parameters
GAMMA = 1.4  # Ideal gas
T_END = 0.2  # Final simulation time
CFL = 0.05
C_visc = 0.5
save_interval = 5  # Save solution to VTX every N time steps

# Mesh and Geometry Setup
nx, ny = 200, 40
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 0.2)),
    n=(nx, ny),
    cell_type=mesh.CellType.triangle,
)

tdim = msh.topology.dim
fdim = tdim - 1

# Element diameters h_K and global minimum element size h_min
imap = msh.topology.index_map(tdim)
num_cells = imap.size_local + imap.num_ghosts
cell_entities = np.arange(num_cells, dtype=np.int32)
h_K = dolfinx.cpp.mesh.h(msh._cpp_object, tdim, cell_entities)
h_min = msh.comm.allreduce(h_K[:imap.size_local].min(), op=MPI.MIN)

# Cell-to-vertex connectivity for element-wise wave speed and viscosity
msh.topology.create_connectivity(tdim, 0)
c2v = msh.topology.connectivity(tdim, 0).array.reshape(-1, 3)

# Function Spaces
W = fem.functionspace(msh, ("Lagrange", 1, (4,)))          # State space: [rho, rhou, rhov, E]
W_tensor = fem.functionspace(msh, ("Lagrange", 1, (4, 2)))  # Flux tensor space: (4 x 2)
V_scalar = fem.functionspace(msh, ("Lagrange", 1))          # Output scalar space
V_vector = fem.functionspace(msh, ("Lagrange", 1, (2,)))    # Output velocity space
V_dg0 = fem.functionspace(msh, ("DG", 0))                   # Piecewise-constant viscosity space

# Boundary Conditions (Slip Wall: v = 0 on y = 0 and y = 0.2)
facets_y = mesh.locate_entities_boundary(
    msh, dim=fdim, marker=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 0.2)
)
dofs_y = fem.locate_dofs_topological(W.sub(2), fdim, facets_y)
bc = fem.dirichletbc(value=ScalarType(0.0), dofs=dofs_y, V=W.sub(2))

# Pre-assembled Discrete Operators & Linear Solver
U = ufl.TrialFunction(W)
V = ufl.TestFunction(W)
T_tensor = ufl.TrialFunction(W_tensor)

# Rectangular divergence matrix C: maps (4 x 2) flux tensor to (4,) state space
C_form = fem.form(ufl.inner(ufl.div(T_tensor), V) * ufl.dx)
C = petsc.assemble_matrix(C_form)
C.assemble()

# Consistent mass matrix M: maps (4,) to (4,)
M_form = fem.form(ufl.inner(U, V) * ufl.dx)
M = petsc.assemble_matrix(M_form)
M.assemble()

# Pre-factorize M with LU solver for fast solves at each Runge-Kutta stage
ksp = KSP().create(MPI.COMM_WORLD)
ksp.setOperators(M)
ksp.setType("preonly")
ksp.getPC().setType("lu")

# State and Residual Functions
U_func = fem.Function(W, name="Conservative_Variables")
F_func = fem.Function(W_tensor, name="Flux_Tensor")
R = fem.Function(W)
R_visc = fem.Function(W)
dU = fem.Function(W)
eps = fem.Function(V_dg0, name="ArtificialViscosity")

# Initial Conditions: Sod Shock-Tube Problem
def initial_condition(x):
    values = np.zeros((4, x.shape[1]), dtype=ScalarType)
    left_mask = x[0] < 0.5

    # Left state: rho = 1.0, u = 0, v = 0, p = 1.0 -> E = p / (gamma - 1) = 2.5
    values[0, left_mask] = 1.0
    values[3, left_mask] = 1.0 / (GAMMA - 1.0)

    # Right state: rho = 0.125, u = 0, v = 0, p = 0.1 -> E = 0.1 / (gamma - 1) = 0.25
    values[0, ~left_mask] = 0.125
    values[3, ~left_mask] = 0.1 / (GAMMA - 1.0)

    return values

U_func.interpolate(initial_condition)
fem.set_bc(U_func.x.petsc_vec, [bc])

# UFL Expressions for Flux, Wave Speed, and Stabilization
rho, rhou, rhov, E = U_func[0], U_func[1], U_func[2], U_func[3]
u_vel = rhou / rho
v_vel = rhov / rho
p = (GAMMA - 1.0) * (E - 0.5 * rho * (u_vel**2 + v_vel**2))

# 4 x 2 convective Euler flux tensor
F_expr = ufl.as_matrix([
    [rhou,             rhov],
    [rhou * u_vel + p, rhou * v_vel],
    [rhov * u_vel,     rhov * v_vel + p],
    [(E + p) * u_vel,  (E + p) * v_vel],
])
expr_F = fem.Expression(F_expr, W_tensor.element.interpolation_points)

# Wave speed: |u| + c
c_sound = ufl.sqrt(GAMMA * p / rho)
vel_mag = ufl.sqrt(u_vel**2 + v_vel**2)
lambda_expr = fem.Expression(vel_mag + c_sound, V_scalar.element.interpolation_points)
lambda_func = fem.Function(V_scalar, name="WaveSpeed")

# First-order artificial viscosity weak form: int eps * grad(U) : grad(V) dx
visc_form = fem.form(ufl.inner(eps * ufl.grad(U_func), ufl.grad(V)) * ufl.dx)

# Output field functions and expressions for ParaView post-processing
rho_out = fem.Function(V_scalar, name="Density")
vel_out = fem.Function(V_vector, name="Velocity")
p_out   = fem.Function(V_scalar, name="Pressure")
E_out   = fem.Function(V_scalar, name="Energy")

expr_rho = fem.Expression(rho, V_scalar.element.interpolation_points)
expr_vel = fem.Expression(ufl.as_vector([u_vel, v_vel]), V_vector.element.interpolation_points)
expr_p   = fem.Expression(p, V_scalar.element.interpolation_points)
expr_E   = fem.Expression(E, V_scalar.element.interpolation_points)

# Helper Functions
def update_wave_speed_and_viscosity():
    """Interpolates wave speed, updates element viscosity eps, and returns lambda_max."""
    lambda_func.interpolate(lambda_expr)
    _, max_speed = lambda_func.x.petsc_vec.max()
    
    # Compute element-wise maximum wave speed lambda_K across 3 vertices
    lambda_nodal = lambda_func.x.array
    lambda_K = np.max(lambda_nodal[c2v], axis=1)
    
    # eps_K = C_visc * h_K * lambda_K
    eps.x.array[:] = C_visc * h_K * lambda_K
    return max_speed

def dUdt():
    """Evaluates the semi-discrete rate of change: dU/dt = M^{-1} (-C * F - R_visc)."""
    # 1. Convective term: R = -C * F
    F_func.interpolate(expr_F)
    C.mult(F_func.x.petsc_vec, R.x.petsc_vec)
    R.x.petsc_vec.scale(-1.0)

    # 2. Viscous diffusion term: R -= R_visc
    with R_visc.x.petsc_vec.localForm() as loc:
        loc.set(0.0)
    petsc.assemble_vector(R_visc.x.petsc_vec, visc_form)
    R_visc.x.petsc_vec.ghostUpdate()
    R.x.petsc_vec.axpy(-1.0, R_visc.x.petsc_vec)

    # 3. Mass matrix solve
    ksp.solve(R.x.petsc_vec, dU.x.petsc_vec)
    return dU

def update_output_fields():
    """Interpolates current solution into physical diagnostic fields for ParaView."""
    rho_out.interpolate(expr_rho)
    vel_out.interpolate(expr_vel)
    p_out.interpolate(expr_p)
    E_out.interpolate(expr_E)

# VTX Writer Setup
os.makedirs("output", exist_ok=True)
vtx = io.VTXWriter(
    msh.comm,
    "output/euler_sod.bp",
    [rho_out, vel_out, p_out, E_out, lambda_func, eps],
    engine="BP4",
)

update_output_fields()
vtx.write(0.0)

# Time-Stepping Loop (SSP-RK3)
u = U_func.x.petsc_vec
du = dU.x.petsc_vec
u0 = u.copy()  # Vector buffer for RK stage linear combinations

T = 0.0
step = 0

if msh.comm.rank == 0:
    print(f"Starting simulation on {nx}x{ny} mesh (h_min = {h_min:.4e})...")
    
t0 = time.perf_counter()

while T < T_END:
    # 1. Update wave speeds, element artificial viscosity, and adaptive time step
    max_speed = update_wave_speed_and_viscosity()
    dt = CFL * h_min / max_speed
    if T + dt > T_END:
        dt = T_END - T

    # 2. Store base state u0 = u^n
    u.copy(u0)

    # --- Stage 1 ---
    dUdt()
    u0.copy(u)
    u.axpy(dt, du)
    fem.set_bc(u, [bc])
    U_func.x.scatter_forward()

    # --- Stage 2 ---
    dUdt()
    u.axpy(dt, du)
    u.scale(0.25)
    u.axpy(0.75, u0)
    fem.set_bc(u, [bc])
    U_func.x.scatter_forward()

    # --- Stage 3 ---
    dUdt()
    u.axpy(dt, du)
    u.scale(2.0 / 3.0)
    u.axpy(1.0 / 3.0, u0)
    fem.set_bc(u, [bc])
    U_func.x.scatter_forward()

    T += dt
    step += 1

    # Save to VTX output
    if step % save_interval == 0 or T >= T_END:
        update_output_fields()
        vtx.write(T)

    if step % 20 == 0 or T >= T_END:
        U_arr = U_func.x.array.reshape(-1, 4)
        rho_min_loc = np.min(U_arr[:, 0])
        u_comp = U_arr[:, 1] / U_arr[:, 0]
        v_comp = U_arr[:, 2] / U_arr[:, 0]
        p_arr = (GAMMA - 1.0) * (U_arr[:, 3] - 0.5 * U_arr[:, 0] * (u_comp**2 + v_comp**2))
        p_min_loc = np.min(p_arr)
        
        rho_min = msh.comm.allreduce(rho_min_loc, op=MPI.MIN)
        p_min = msh.comm.allreduce(p_min_loc, op=MPI.MIN)
        if msh.comm.rank == 0:
            print(f"Step {step:4d} | T = {T:.4f}/{T_END:.2f} | dt = {dt:.4e} | min(rho) = {rho_min:.4f} | min(p) = {p_min:.4f}")
            
t1 = time.perf_counter()

max_time = msh.comm.allreduce(t1 - t0, op=MPI.MAX)

vtx.close()
if msh.comm.rank == 0:
    print(f"Simulation successfully completed in {max_time:.2f} seconds.")