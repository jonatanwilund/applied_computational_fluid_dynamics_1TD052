from mpi4py import MPI  # noqa: I001
from petsc4py.PETSc import ScalarType, KSP

import os
import numpy as np

import ufl
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem import petsc

GAMMA = 1.4
T_END = 0.2

# Create mesh
nx, ny = 50, 10
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 0.2)),
    n=(nx, ny),
    cell_type=mesh.CellType.triangle,
)

# Define function space
degree = 1  # Degree of the finite element space
W = fem.functionspace(msh, ("Lagrange", degree, (4,)))  # State space W with 4 components
W_tensor = fem.functionspace(msh, ("Lagrange", 1, (4, 2)))  # Flux tensor space

tdim = msh.topology.dim

# Compute diameter h_K for every cell K in the mesh
n_cells = msh.topology.index_map(tdim).size_local
cell_entities = np.arange(n_cells, dtype=np.int32)

h_K = dolfinx.cpp.mesh.h(msh._cpp_object, tdim, cell_entities)
h_min = msh.comm.allreduce(h_K.min(), op=MPI.MIN)

# Boundary conditions
fdim = tdim - 1

# Get horizontal facets and dofs for v ()
facets_y = mesh.locate_entities_boundary(
    msh, 
    dim=fdim, 
    marker=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 0.2)
)
dofs_y = fem.locate_dofs_topological(W.sub(2), fdim, facets_y)

# Set boundary conditions for v
bc = fem.dirichletbc(value=ScalarType(0.0), dofs=dofs_y, V=W.sub(2))

# Define variational problem
U = ufl.TrialFunction(W)
V = ufl.TestFunction(W)
T_tensor = ufl.TrialFunction(W_tensor)

# Assemble divergence matrix C once
C_form = fem.form(ufl.inner(ufl.div(T_tensor), V) * ufl.dx)
C = petsc.assemble_matrix(C_form)
C.assemble()

# Assemble mass matrix M once
M_form = fem.form(ufl.inner(U, V) * ufl.dx)
M = petsc.assemble_matrix(M_form)
M.assemble()

# Allocation of PETSc vectors
U_func = fem.Function(W)
U_func.name = "Conservative_Variables"
F_func = fem.Function(W_tensor)
R = fem.Function(W)
dU = fem.Function(W)

# Initial conditions
def initial_condition(x):
    values = np.zeros((4, x.shape[1]), dtype=ScalarType)
    
    # Left state (x < 0.5)
    left_mask = x[0] < 0.5
    values[0, left_mask] = 1.0  # rho
    values[1, left_mask] = 0.0  # rho * u
    values[2, left_mask] = 0.0  # rho * v
    values[3, left_mask] = 1.0 / (GAMMA - 1.0)  # E (2.5 for gamma=1.4)
    
    # Right state (x >= 0.5)
    right_mask = ~left_mask
    values[0, right_mask] = 0.125  # rho
    values[1, right_mask] = 0.0    # rho * u
    values[2, right_mask] = 0.0    # rho * v
    values[3, right_mask] = 0.1 / (GAMMA - 1.0)  # E (0.25 for gamma=1.4)
    
    return values

U_func.interpolate(initial_condition)
fem.set_bc(U_func.x.petsc_vec, [bc])  # Apply boundary conditions to initial state

rho, rhou, rhov, E = U_func[0], U_func[1], U_func[2], U_func[3]
u_vel = rhou / rho
v_vel = rhov / rho
p = (GAMMA - 1.0) * (E - 0.5 * rho * (u_vel**2 + v_vel**2))

F_expr = ufl.as_matrix([
    [rhou, rhov],
    [rhou * u_vel + p, rhou * v_vel],
    [rhov * u_vel, rhov * v_vel + p],
    [(E + p) * u_vel, (E + p) * v_vel],
])

expr = fem.Expression(F_expr, W_tensor.element.interpolation_points)

def update_flux_interpolant(U_func, F_func):
    """Evaluates the (4 x 2) Euler flux at all nodes in-place"""
    F_func.interpolate(expr)
    
# Set up linear solver
ksp = KSP().create(MPI.COMM_WORLD)
ksp.setOperators(M)
ksp.setType("preonly")
ksp.getPC().setType("lu")  # Pre-factorize M with LU
    
def dUdt(U_func):
    """Compute the time derivative dU/dt = -M^{-1} C F(U)"""
    update_flux_interpolant(U_func, F_func)
    
    # Compute R = -C F(U)
    C.mult(F_func.x.petsc_vec, R.x.petsc_vec)
    R.x.petsc_vec.scale(-1.0)  # R = -C F(U)
    
    ksp.solve(R.x.petsc_vec, dU.x.petsc_vec)  # Solve M dU/dt = R
    return dU

# Output directory
os.makedirs("output", exist_ok=True)

# Create helper scalar and vector functions for output
V_scalar = fem.functionspace(msh, ("Lagrange", 1))
V_vector = fem.functionspace(msh, ("Lagrange", 1, (2,)))

rho_out = fem.Function(V_scalar, name="Density")
vel_out = fem.Function(V_vector, name="Velocity")
p_out = fem.Function(V_scalar, name="Pressure")
E_out = fem.Function(V_scalar, name="Energy")
lambda_func = fem.Function(V_scalar, name="WaveSpeed")

# Define expressions for physical variables
expr_rho = fem.Expression(rho, V_scalar.element.interpolation_points)
expr_vel = fem.Expression(ufl.as_vector([u_vel, v_vel]), V_vector.element.interpolation_points)
expr_p = fem.Expression(p, V_scalar.element.interpolation_points)
expr_E = fem.Expression(E, V_scalar.element.interpolation_points)

c = ufl.sqrt(GAMMA * p / rho)
vel_mag = ufl.sqrt(u_vel**2 + v_vel**2)
lambda_expr_ufl = vel_mag + c
lambda_expr = fem.Expression(lambda_expr_ufl, V_scalar.element.interpolation_points)

def update_output_fields():
    """Update the output fields for visualization."""
    rho_out.interpolate(expr_rho)
    vel_out.interpolate(expr_vel)
    p_out.interpolate(expr_p)
    E_out.interpolate(expr_E)
    lambda_func.interpolate(lambda_expr)
# VTXWriter for output
vtx = io.VTXWriter(
    msh.comm, 
    "output/euler_sod.bp", 
    [rho_out, vel_out, p_out, E_out, lambda_func]
)

# Save initial state
update_output_fields()
vtx.write(0.0)

# References to the underlying PETSc vectors
u = U_func.x.petsc_vec
du = dU.x.petsc_vec

u0 = u.copy()  # Store the initial state for time-stepping

# Calculate time step size based on CFL condition
def update_max_wave_speed():
    """Compute the maximum wave speed in the domain."""
    lambda_func.interpolate(lambda_expr)
    _, max_speed = lambda_func.x.petsc_vec.max()
    return max_speed
    
CFL = 0.5  # CFL number
save_interval = 5  # Save output every N steps

# Time-stepping loop with SSP-RK3 
T = 0.0
step = 0
while T < T_END:
    max_speed = update_max_wave_speed()
    dt = CFL * h_min / max_speed
    if T + dt > T_END:
        dt = T_END - T  # Adjust last time step to reach T_END
    
    # SSP-RK3 stages
    u.copy(u0)
    
    # Stage 1
    dUdt(U_func)  # Stored directly in dU
    u0.copy(u)
    u.axpy(dt, du)
    fem.set_bc(u, [bc])
    
    # Stage 2
    dUdt(U_func)
    u.axpy(dt, du)
    u.scale(0.25)
    u.axpy(0.75, u0)
    fem.set_bc(u, [bc])
    
    # Stage 3
    dUdt(U_func)
    u.axpy(dt, du)
    u.scale(2.0 / 3.0)
    u.axpy(1.0 / 3.0, u0)
    fem.set_bc(u, [bc])
    
    T += dt
    step += 1
    
    if step % save_interval == 0 or T >= T_END:
        update_output_fields()
        vtx.write(T)
    
vtx.close()
print(f"Simulation completed in {step} steps, final time T = {T:.4f}")