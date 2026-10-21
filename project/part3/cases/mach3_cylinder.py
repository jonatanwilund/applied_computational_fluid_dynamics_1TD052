"""
Problem 3.4: Supersonic Mach 3 flow past a circular cylinder (stub/placeholder).
"""

from dolfinx import fem, mesh
import numpy as np

from euler.spaces import EulerSpaces
from .base import BaseProblem


class Mach3CylinderProblem(BaseProblem):
    """
    Placeholder test case for Problem 3.4: Supersonic Mach 3 flow past a cylinder.
    To be populated with unstructured Gmsh grid and inflow/outflow/wall boundary conditions.
    """

    def build_mesh(self) -> mesh.Mesh:
        raise NotImplementedError("Problem 3.4 mesh will be configured in subsequent task.")

    def initial_condition(self, x):
        raise NotImplementedError("Problem 3.4 initial conditions will be configured in subsequent task.")

    def build_bcs(self, spaces: EulerSpaces) -> list[fem.DirichletBC]:
        raise NotImplementedError("Problem 3.4 boundary conditions will be configured in subsequent task.")
