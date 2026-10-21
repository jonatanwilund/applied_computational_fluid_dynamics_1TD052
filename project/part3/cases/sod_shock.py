"""
Problem 3.2: 2D Sod Shock-Tube Riemann Problem.
"""

from dolfinx import fem, mesh
import numpy as np
from petsc4py.PETSc import ScalarType

from euler.spaces import EulerSpaces
from .base import BaseProblem


class SodShockProblem(BaseProblem):
    """
    Sod shock-tube problem in 2D domain [0, 1] x [0, 0.2].
    Discontinuity is located at x = 0.5.
    Top and bottom boundaries (y = 0, y = 0.2) have slip wall conditions (v = 0).
    """

    def build_mesh(self) -> mesh.Mesh:
        return mesh.create_rectangle(
            comm=self.comm,
            points=((0.0, 0.0), (1.0, 0.2)),
            n=(self.config.nx, self.config.ny),
            cell_type=mesh.CellType.triangle,
        )

    def initial_condition(self, x):
        values = np.zeros((4, x.shape[1]), dtype=ScalarType)
        left_mask = x[0] < 0.5
        gamma = self.config.gamma

        # Left state: rho = 1.0, u = 0, v = 0, p = 1.0 -> E = p / (gamma - 1)
        values[0, left_mask] = 1.0
        values[3, left_mask] = 1.0 / (gamma - 1.0)

        # Right state: rho = 0.125, u = 0, v = 0, p = 0.1 -> E = 0.1 / (gamma - 1)
        values[0, ~left_mask] = 0.125
        values[3, ~left_mask] = 0.1 / (gamma - 1.0)

        return values

    def build_bcs(self, spaces: EulerSpaces) -> list[fem.DirichletBC]:
        """Slip wall boundary condition: rhov = 0 (subspace 2) at y = 0.0 and y = 0.2."""
        facets_y = mesh.locate_entities_boundary(
            self.mesh,
            dim=spaces.fdim,
            marker=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 0.2),
        )
        dofs_y = fem.locate_dofs_topological(spaces.W.sub(2), spaces.fdim, facets_y)
        bc_y = fem.dirichletbc(value=ScalarType(0.0), dofs=dofs_y, V=spaces.W.sub(2))
        return [bc_y]
