"""
Pre-assembled discrete finite element operators and linear solvers.
"""

import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py.PETSc import KSP

from .spaces import EulerSpaces


class EulerOperators:
    """Pre-assembled PETSc matrices and linear solvers for the semi-discrete system."""

    def __init__(self, spaces: EulerSpaces):
        self.spaces = spaces
        msh = spaces.mesh

        # Trial and Test functions
        U_trial = ufl.TrialFunction(spaces.W)
        V_test = ufl.TestFunction(spaces.W)
        T_tensor = ufl.TrialFunction(spaces.W_tensor)

        # 1. Divergence matrix C: maps (4 x 2) flux tensor space to (4,) state space
        # (div(F), V) = int div(T_tensor) : V dx
        self.C_form = fem.form(ufl.inner(ufl.div(T_tensor), V_test) * ufl.dx)
        self.C = petsc.assemble_matrix(self.C_form)
        self.C.assemble()

        # 2. Consistent mass matrix M: maps (4,) to (4,)
        self.M_form = fem.form(ufl.inner(U_trial, V_test) * ufl.dx)
        self.M = petsc.assemble_matrix(self.M_form)
        self.M.assemble()

        # 3. Factorized LU linear solver for fast M^{-1} solves at each stage
        self.ksp = KSP().create(msh.comm)
        self.ksp.setOperators(self.M)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")

    def apply_convective_divergence(self, F_vec, R_vec):
        """
        Computes R = -C * F.
        """
        self.C.mult(F_vec, R_vec)
        R_vec.scale(-1.0)

    def solve_mass(self, rhs_vec, out_vec):
        """
        Solves M * out = rhs for explicit Runge-Kutta updates.
        """
        self.ksp.solve(rhs_vec, out_vec)
