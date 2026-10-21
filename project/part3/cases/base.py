"""
Abstract base class for fluid problem cases (geometry, initial conditions, boundary conditions).
"""

from abc import ABC, abstractmethod

from dolfinx import fem, mesh
from mpi4py import MPI

from config import SimulationConfig
from euler.spaces import EulerSpaces


class BaseProblem(ABC):
    """Abstract base class defining a CFD test case."""

    def __init__(self, comm: MPI.Comm, config: SimulationConfig):
        self.comm = comm
        self.config = config
        self.mesh = self.build_mesh()

    @abstractmethod
    def build_mesh(self) -> mesh.Mesh:
        """Constructs and returns the computational mesh."""
        ...

    @abstractmethod
    def initial_condition(self, x):
        """Evaluates initial condition array of shape (4, N) at points x of shape (3, N)."""
        ...

    @abstractmethod
    def build_bcs(self, spaces: EulerSpaces) -> list[fem.DirichletBC]:
        """Constructs Dirichlet boundary conditions."""
        ...
