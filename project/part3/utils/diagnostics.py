"""
Simulation stability monitoring and runtime diagnostic reporting.
"""

from dolfinx import fem
from mpi4py import MPI
import numpy as np

from config import SimulationConfig
from euler.spaces import EulerSpaces


class DiagnosticTracker:
    """Monitors simulation integrity (min density, min pressure) and prints progress."""

    def __init__(self, spaces: EulerSpaces, config: SimulationConfig):
        self.spaces = spaces
        self.config = config
        self.comm = spaces.mesh.comm
        self.rank = self.comm.rank

    def compute_extrema(self, U_func: fem.Function) -> tuple[float, float]:
        """
        Computes domain-wide minimum density and pressure using MPI allreduce.
        """
        # Reshape local array to (N_dofs, 4)
        U_arr = U_func.x.array.reshape(-1, 4)
        rho_arr = U_arr[:, 0]
        u_comp = U_arr[:, 1] / rho_arr
        v_comp = U_arr[:, 2] / rho_arr
        p_arr = (self.config.gamma - 1.0) * (
            U_arr[:, 3] - 0.5 * rho_arr * (u_comp**2 + v_comp**2)
        )

        rho_min_local = float(np.min(rho_arr))
        p_min_local = float(np.min(p_arr))

        rho_min = self.comm.allreduce(rho_min_local, op=MPI.MIN)
        p_min = self.comm.allreduce(p_min_local, op=MPI.MIN)

        return rho_min, p_min

    def report(self, step: int, t: float, dt: float, U_func: fem.Function):
        """Prints a single line progress report on rank 0."""
        rho_min, p_min = self.compute_extrema(U_func)

        if rho_min <= 0.0 or p_min <= 0.0:
            if self.rank == 0:
                print(f"WARNING: Negative physical quantities at step {step}: min(rho)={rho_min:.3e}, min(p)={p_min:.3e}")

        if self.rank == 0:
            t_end = self.config.t_end
            print(
                f"Step {step:4d} | T = {t:.4f}/{t_end:.2f} | dt = {dt:.4e} | "
                f"min(rho) = {rho_min:.4f} | min(p) = {p_min:.4f}"
            )
