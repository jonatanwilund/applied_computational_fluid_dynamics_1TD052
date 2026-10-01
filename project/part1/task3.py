import gmsh
import numpy as np
from dolfinx import io, mesh
from mpi4py import MPI
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from common.fem_assemblers import (
    convection_assembler_2d,
    load_assembler_2d,
    mass_assembler_2d,
)
from common.mesh import dolfinx_to_pet


def get_unit_circle(max_length: float = 0.1) -> mesh.Mesh:
    # Initialize Gmsh
    gmsh.initialize()

    # Create model and set disk geometry (center x=0, y=0, z=0, radius=1)
    gmsh.model.add("unit_circle")
    gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()

    # Define physical group for the surface (tag 1)
    gmsh.model.addPhysicalGroup(2, [1], tag=1)

    # Generate 2D mesh with a target mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_length)
    gmsh.model.mesh.generate(2)

    # Convert Gmsh model to DOLFINx mesh
    msh = io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    # Finalize Gmsh session
    gmsh.finalize()

    return msh.mesh


def u0(x: np.ndarray, x0: np.ndarray | None = None, r0: float = 0.25) -> np.ndarray:
    if x0 is None:
        x0 = np.array([0.3, 0])
    return 0.5 * (1 - np.tanh(((x[0] - x0[0]) ** 2 + (x[1] - x0[1]) ** 2) / r0**2 - 1))


def beta(x: np.ndarray) -> np.ndarray:
    """
    Returns the convection velocity vector at point x.
    For this problem, the convection velocity is constant and given by:
        beta(x) = [1, 2]
    """
    return 2 * np.pi * np.array([-x[1], x[0]])  # Rotational velocity field


def explicit_euler(
    M: csr_matrix, C: csr_matrix, b: np.ndarray, u0: np.ndarray, dt: float, T: float
) -> np.ndarray:
    """
    Returns the nodal values of the solution at time T
    Uses explicit euler time stepping
    xi(n+1) = xi(n) + dt * M^{-1} * (b - C * xi(n))
    """
    t = 0
    u = u0.copy()
    solve_M = splu(M.tocsc()).solve  # Precompute LU factorization of M for efficiency
    while t < T:
        u += dt * solve_M(b - C @ u)
        t += dt
    return u


def explicit_rk4(
    M: csr_matrix, C: csr_matrix, b: np.ndarray, u0: np.ndarray, dt: float, T: float
) -> np.ndarray:
    """
    Returns the nodal values of the solution at time T
    Uses explicit Runge-Kutta 4th order time stepping
    """
    t = 0
    u = u0.copy()
    solve_M = splu(M.tocsc()).solve  # Precompute LU factorization of M for efficiency
    while t < T:
        k1 = solve_M(b - C @ u)
        k2 = solve_M(b - C @ (u + 0.5 * dt * k1))
        k3 = solve_M(b - C @ (u + 0.5 * dt * k2))
        k4 = solve_M(b - C @ (u + dt * k3))
        u += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return u


def error_analysis(M, C, b, u0_vals, h, CFL):
    dt = CFL * h / (2 * np.pi)  # ||beta||_inf = 2*pi at radius 1
    T = 1
    u_T = explicit_euler(M, C, b, u0_vals, dt, T)
    err_euler = np.linalg.norm(u_T - u0_vals)

    u_T_rk4 = explicit_rk4(M, C, b, u0_vals, dt, T)
    err_rk4 = np.linalg.norm(u_T_rk4 - u0_vals)
    return err_euler, err_rk4


def main():
    h = 0.05
    msh = get_unit_circle(max_length=h)
    p, e, t = dolfinx_to_pet(msh)  # noqa: RUF059

    M = mass_assembler_2d(p, t)
    C = convection_assembler_2d(p, t, beta(p)[0], beta(p)[1])
    b = load_assembler_2d(p, t, lambda x, y: 0.0)
    beta_vals = beta(p)
    bx = beta_vals[0]
    by = beta_vals[1]
    C = convection_assembler_2d(p, t, bx, by)

    print("M:", M.shape, "nnz =", M.nnz)
    print("C:", C.shape, "nnz =", C.nnz)

    # Initial condition: u0 =
    u0_vals = u0(p)

    cfl_list = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 1.1]

    errors_euler = []
    errors_rk4 = []
    for cfl in cfl_list:
        err_euler, err_rk4 = error_analysis(M, C, b, u0_vals, h, cfl)
        errors_euler.append(err_euler)
        errors_rk4.append(err_rk4)

    # Print the errors for each CFL number in a formatted table
    print("\nError Analysis:")
    print(f"{'CFL':<10}{'Error (Euler)':<20}{'Error (RK4)':<20}")
    print("-" * 50)
    for cfl, err_euler, err_rk4 in zip(cfl_list, errors_euler, errors_rk4):
        print(f"{cfl:<10}{err_euler:<20.6e}{err_rk4:<20.6e}")


if __name__ == "__main__":
    main()
