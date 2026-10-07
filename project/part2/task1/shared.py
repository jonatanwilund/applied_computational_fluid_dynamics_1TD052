import numpy as np
from dolfinx import mesh
from mpi4py import MPI
from scipy.sparse import csr_array

from common.fem_assemblers import convection_assembler_2d_vec


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
        cell_type=mesh.CellType.triangle,
    )
    return domain


def compute_element_diameters(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute the diameters of each triangular element in the mesh.

    Parameters:
      p: Node coordinate array, shape (2, npnt)
      t: Triangle connectivity, shape (3, nt)

    Returns:
      h_K: Array containing the diameter h_K of element K, shape (nt,)
    """
    p0 = p[:, t[0, :]]  # Coordinates of the first vertex of each triangle
    p1 = p[:, t[1, :]]  # Coordinates of the second vertex
    p2 = p[:, t[2, :]]  # Coordinates of the third vertex

    # Compute length of the 3 edges for all triangles
    len0 = np.hypot(p1[0] - p0[0], p1[1] - p0[1])  # Edge from vertex 0 to vertex 1
    len1 = np.hypot(p2[0] - p1[0], p2[1] - p1[1])  # Edge from vertex 1 to vertex 2
    len2 = np.hypot(p0[0] - p2[0], p0[1] - p2[1])  # Edge from vertex 2 to vertex 0

    # h_K is the longest edge of element K
    return np.maximum(np.maximum(len0, len1), len2)


def u0(x: np.ndarray) -> np.ndarray:
    """Initial condition function u0(x)"""
    mask = np.sqrt(x[0] ** 2 + x[1] ** 2) <= 1
    return np.where(
        mask, 14 * np.pi / 4, np.pi / 4
    )  # Return 14π/4 inside the unit disk, π/4 outside


def f(u: np.ndarray) -> np.ndarray:
    """Nonlinear flux term function f(u)"""
    return np.array([np.sin(u), np.cos(u)])


def SPP_RK3_step(
    p,
    t,
    mass_solve,
    U_n,
    dt,
    viscosity_matrix,
    geometry,
    artificial_viscosity=True,
):
    """
    Perform a single time step of the SSP-RK3 method for the nonlinear PDE,
    stabilized with artificial viscosity.
    """
    # Stage 1
    convection_matrix = convection_assembler_2d_vec(
        p, t, np.cos(U_n), -np.sin(U_n), geometry
    )
    operator = convection_matrix + viscosity_matrix if artificial_viscosity else convection_matrix
    R_0 = -(operator @ U_n)
    U1 = U_n + dt * mass_solve(R_0)

    # Stage 2
    convection_matrix = convection_assembler_2d_vec(
        p, t, np.cos(U1), -np.sin(U1), geometry
    )
    operator = convection_matrix + viscosity_matrix if artificial_viscosity else convection_matrix
    R_1 = -(operator @ U1)
    U2 = (3 / 4) * U_n + (1 / 4) * (U1 + dt * mass_solve(R_1))

    # Stage 3
    convection_matrix = convection_assembler_2d_vec(
        p, t, np.cos(U2), -np.sin(U2), geometry
    )
    operator = convection_matrix + viscosity_matrix if artificial_viscosity else convection_matrix
    R_2 = -(operator @ U2)
    U_np1 = (1 / 3) * U_n + (2 / 3) * (U2 + dt * mass_solve(R_2))

    return U_np1


def total_mass(M: csr_array, U: np.ndarray) -> float:
    """
    Compute the total mass of the solution vector U using the mass matrix M.

    Parameters:
      M: Mass matrix (sparse), shape (npnt, npnt)
      U: Solution vector, shape (npnt,)
    Returns:
      Total mass as a scalar value.
    """
    return np.ones_like(U) @ (M @ U)
