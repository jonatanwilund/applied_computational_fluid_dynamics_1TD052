"""
Finite element function spaces for the 2D compressible Euler equations.
"""

from dolfinx import fem, mesh


class EulerSpaces:
    """Manages all finite element function spaces for the Euler solver."""

    def __init__(self, msh: mesh.Mesh):
        self.mesh = msh
        self.tdim = msh.topology.dim
        self.fdim = self.tdim - 1

        # State space: U = [rho, rhou, rhov, E]
        self.W = fem.functionspace(msh, ("Lagrange", 1, (4,)))

        # Flux tensor space: F has shape (4, 2)
        self.W_tensor = fem.functionspace(msh, ("Lagrange", 1, (4, 2)))

        # Diagnostic spaces for ParaView and scalar quantities
        self.V_scalar = fem.functionspace(msh, ("Lagrange", 1))
        self.V_vector = fem.functionspace(msh, ("Lagrange", 1, (2,)))

        # Discontinuous piecewise-constant space for artificial viscosity eps_K
        self.V_dg0 = fem.functionspace(msh, ("DG", 0))
