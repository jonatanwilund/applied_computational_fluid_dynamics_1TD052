"""
ADIOS2 VTXWriter output utilities for ParaView visualization.
"""

import os
from dolfinx import fem, io
import ufl

from config import SimulationConfig
from euler.physics import pressure, velocity
from euler.spaces import EulerSpaces
from euler.stabilization import FirstOrderViscosity


class SolutionWriter:
    """Handles field evaluation and VTXWriter ADIOS2 output export."""

    def __init__(
        self,
        spaces: EulerSpaces,
        stabilization: FirstOrderViscosity,
        config: SimulationConfig,
    ):
        self.spaces = spaces
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        filepath = os.path.join(config.output_dir, config.output_name)

        # Output fields for visualization in ParaView
        self.rho_out = fem.Function(spaces.V_scalar, name="Density")
        self.vel_out = fem.Function(spaces.V_vector, name="Velocity")
        self.p_out = fem.Function(spaces.V_scalar, name="Pressure")
        self.E_out = fem.Function(spaces.V_scalar, name="Energy")

        # Weak symbolic expressions linked to arbitrary state function
        self.expr_rho = None
        self.expr_vel = None
        self.expr_p = None
        self.expr_E = None

        # VTX output fields
        self.output_fields = [
            self.rho_out,
            self.vel_out,
            self.p_out,
            self.E_out,
            stabilization.lambda_func,
            stabilization.eps,
        ]

        self.vtx = io.VTXWriter(
            spaces.mesh.comm,
            filepath,
            self.output_fields,
            engine="BP4",
        )

    def _setup_expressions(self, U_func: fem.Function):
        """Constructs interpolation expressions for the state U."""
        rho, rhou, rhov, E = U_func[0], U_func[1], U_func[2], U_func[3]
        u_vec = velocity(U_func)
        p = pressure(rho, rhou, rhov, E, self.config.gamma)

        self.expr_rho = fem.Expression(rho, self.spaces.V_scalar.element.interpolation_points)
        self.expr_vel = fem.Expression(u_vec, self.spaces.V_vector.element.interpolation_points)
        self.expr_p = fem.Expression(p, self.spaces.V_scalar.element.interpolation_points)
        self.expr_E = fem.Expression(E, self.spaces.V_scalar.element.interpolation_points)

    def write(self, t: float, U_func: fem.Function):
        """Interpolates output fields and writes current timestep to VTX file."""
        if self.expr_rho is None:
            self._setup_expressions(U_func)

        self.rho_out.interpolate(self.expr_rho)
        self.vel_out.interpolate(self.expr_vel)
        self.p_out.interpolate(self.expr_p)
        self.E_out.interpolate(self.expr_E)

        self.vtx.write(t)

    def close(self):
        """Flushes and closes the VTXWriter."""
        self.vtx.close()
