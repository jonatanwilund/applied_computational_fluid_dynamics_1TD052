"""
Artificial viscosity and shock-capturing stabilization schemes for the Euler equations.
"""
from mpi4py import MPI  # noqa: I001

from abc import ABC, abstractmethod

import dolfinx
from dolfinx import fem
from dolfinx.fem import petsc
import numpy as np
import ufl

from config import SimulationConfig
from .physics import max_wave_speed
from .spaces import EulerSpaces


class BaseStabilization(ABC):
    """Abstract base class for Euler equation stabilization."""

    @abstractmethod
    def update(self, U_func: fem.Function) -> float:
        """
        Updates stabilization fields based on current solution U.
        Returns the maximum wave speed (lambda_max) across the domain.
        """
        ...

    @abstractmethod
    def assemble_viscous_residual(self, R_visc_vec):
        """Assembles the viscous stabilization contribution into the given PETSc vector."""
        ...


class FirstOrderViscosity(BaseStabilization):
    """
    First-order artificial viscosity stabilization:
    eps_K = C_visc * h_K * lambda_K
    where lambda_K = max_{vertices in K} (|u| + c).
    """

    def __init__(self, spaces: EulerSpaces, config: SimulationConfig):
        self.spaces = spaces
        self.config = config
        self.c_visc = config.c_visc
        msh = spaces.mesh
        tdim = spaces.tdim

        # Element diameters h_K and global minimum element size
        self.imap = msh.topology.index_map(tdim)
        self.num_local_cells = self.imap.size_local
        num_cells = self.num_local_cells + self.imap.num_ghosts
        cell_entities = np.arange(num_cells, dtype=np.int32)
        self.h_K = dolfinx.cpp.mesh.h(msh._cpp_object, tdim, cell_entities)
        self.h_min = msh.comm.allreduce(self.h_K[: self.num_local_cells].min(), op=MPI.MIN)

        # Cell-to-vertex connectivity
        msh.topology.create_connectivity(tdim, 0)
        c2v_array = msh.topology.connectivity(tdim, 0).array
        self.c2v = c2v_array.reshape(-1, 3)

        # Wave speed function and expression
        self.lambda_func = fem.Function(spaces.V_scalar, name="WaveSpeed")
        # Piecewise-constant artificial viscosity field eps
        self.eps = fem.Function(spaces.V_dg0, name="ArtificialViscosity")

        # Weak form placeholder (constructed once state function U is registered)
        self.visc_form = None

    def setup_forms(self, U_func: fem.Function):
        """Initializes UFL expressions and weak forms linked to the solver's state function."""
        # Wave speed symbolic expression: |u| + c
        lambda_sym = max_wave_speed(U_func, self.config.gamma)
        self.lambda_expr = fem.Expression(lambda_sym, self.spaces.V_scalar.element.interpolation_points)

        # Viscous diffusion weak form: int eps * grad(U) : grad(V) dx
        V_test = ufl.TestFunction(self.spaces.W)
        self.visc_form = fem.form(ufl.inner(self.eps * ufl.grad(U_func), ufl.grad(V_test)) * ufl.dx)

    def update(self, U_func: fem.Function) -> float:
        """
        Interpolates wave speed, evaluates element-wise viscosity eps_K,
        and returns the global maximum wave speed.
        """
        self.lambda_func.interpolate(self.lambda_expr)

        # Global maximum wave speed across all MPI processes
        _, local_max = self.lambda_func.x.petsc_vec.max()
        global_max_speed = self.spaces.mesh.comm.allreduce(local_max, op=MPI.MAX)

        # Compute element-wise maximum wave speed lambda_K across triangle vertices
        lambda_nodal = self.lambda_func.x.array
        lambda_K = np.max(lambda_nodal[self.c2v], axis=1)

        # Update local element viscosities eps_K = C_visc * h_K * lambda_K
        num_local = self.num_local_cells
        self.eps.x.array[:num_local] = self.c_visc * self.h_K[:num_local] * lambda_K[:num_local]
        self.eps.x.scatter_forward()

        return global_max_speed

    def assemble_viscous_residual(self, R_visc_vec):
        """Assembles R_visc = int eps * grad(U) : grad(V) dx."""
        with R_visc_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(R_visc_vec, self.visc_form)
        R_visc_vec.ghostUpdate()
