from mpi4py import MPI
from dolfinx import mesh

from tqdm import tqdm
import numpy as np
from scipy.sparse import linalg
import matplotlib.pyplot as plt

from common.fem_assemblers import (
    nonlinear_domain_residual_assembler_2d,
    nonlinear_boundary_residual_assembler_2d,
    load_assembler_2d,
    mass_assembler_2d,
)
from common.mesh import dolfinx_to_pet


def get_mesh(N: int) -> mesh.Mesh:
    """
    Returns a 2D mesh of the square [-2, 2] x [-2.5, 1.5] with uniform triangular elements.
    The mesh is generated using DOLFINx's built-in mesh generation capabilities.
    """
    # Create a uniform rectangular mesh of the unit square
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        points=((-2, -2.5), (2, 1.5)),
        n=(N, N),
        cell_type=mesh.CellType.triangle
    )
    return domain


def u0(x: np.ndarray) -> np.ndarray:
    """Initial condition function u0(x)"""
    mask = np.sqrt(x[0] ** 2 + x[1] ** 2) <= 1
    return np.where(mask, 14*np.pi/4, np.pi/4)  # Return 14π/4 inside the unit disk, π/4 outside


def f(u: np.ndarray) -> np.ndarray:
    """Nonlinear flux term function f(u)"""
    return np.array([np.sin(u[0]), np.cos(u[1])])


def R(p, e, t, U):
    """
    Compute the nonlinear residual vector R(U) = R_domain(U) + R_boundary(U)
    for the given mesh (p, e, t) and solution vector U.
    """
    R_domain = nonlinear_domain_residual_assembler_2d(p, t, U)
    R_boundary = nonlinear_boundary_residual_assembler_2d(p, e, U)
    return R_domain - R_boundary


def SPP_RK3_step(p, e, t, mass_solve, U_n, dt):
    """
    Perform a single time step of the SSP-RK3 method for the nonlinear PDE.
    
    Parameters:
      p: Node coordinate array, shape (2, npnt)
      e: Boundary edge connectivity, shape (2, N_bound)
      t: Triangle connectivity, shape (3, nt)
            mass_solve: Function that solves a mass-matrix system
      U_n: Current solution vector at time step n, shape (npnt,)
      dt: Time step size
    Returns:
      U_np1: Updated solution vector at time step n+1, shape (npnt,)
    """
    # Stage 1
    U1 = U_n + dt * mass_solve(R(p, e, t, U_n))
    
    # Stage 2
    U2 = (3/4) * U_n + (1/4) * (U1 + dt * mass_solve(R(p, e, t, U1)))
    
    # Stage 3
    return (1/3) * U_n + (2/3) * (U2 + dt * mass_solve(R(p, e, t, U2)))


def save_solution_plot(p, t, U, T, time):
    plt.figure(figsize=(8, 6))
    plt.tripcolor(p[0, :], p[1, :], t.T, U, shading='gouraud', cmap='viridis')
    plt.colorbar(label='Solution u')
    plt.title(f'Solution at T={T}, t={time:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.savefig(f"project/part2/solution_T{T}_t{time:.2f}.png", dpi=300)


def main():
    # Create mesh
    h = 0.1
    N = round(4 / h)  # 4 is the domain width in both directions
    domain = get_mesh(N)
    p, e, t = dolfinx_to_pet(domain)
    
    # Assemble the mass matrix and factorize it once for repeated solves.
    M = mass_assembler_2d(p, t)
    mass_solve = linalg.factorized(M.tocsc())
    
    b0 = load_assembler_2d(p, t, lambda x, y: u0(np.array([x, y])))
    U0 = mass_solve(b0)
    
    # Time integration using SSP-RK3
    dt = 0.01
    T = 1.0
    num_steps = int(T / dt)
    U_n = U0.copy()
    for step in tqdm(range(num_steps), desc="Time-stepping"):
        U_n = SPP_RK3_step(p, e, t, mass_solve, U_n, dt)
        if step % 10 == 0:  # Save every 10 steps
            save_solution_plot(p, t, U_n, T, step * dt)
        
        
if __name__ == "__main__":
    main()