
import numpy as np
from mpi4py import MPI
from dolfinx import mesh

from mesh import dolfinx_to_pet
from fem_assemblers import (
    mass_assembler_2d,
    load_assembler_2d,
    stiffness_assembler_2d,
    convection_assembler_2d,
)

msh = mesh.create_unit_square(
    MPI.COMM_SELF, 4, 4,
    cell_type=mesh.CellType.triangle
)

p, e, t = dolfinx_to_pet(msh)

M = mass_assembler_2d(p, t)
b = load_assembler_2d(p, t, lambda x, y: 1.0)
A = stiffness_assembler_2d(p, t)

bx = np.ones(p.shape[1])
by = 2.0 * np.ones(p.shape[1])
C = convection_assembler_2d(p, t, bx, by)

one = np.ones(p.shape[1])

print("1^T M 1           =", one @ (M @ one))
print("sum(b), f=1       =", np.sum(b))
print("||M1-b||_inf      =", np.linalg.norm(M @ one - b, np.inf))
print("||A1||_inf        =", np.linalg.norm(A @ one, np.inf))
print("||C1||_inf        =", np.linalg.norm(C @ one, np.inf))

print("M:", M.shape, "nnz =", M.nnz)
print("A:", A.shape, "nnz =", A.nnz)
print("C:", C.shape, "nnz =", C.nnz)
