"""
Explicit Runge-Kutta time marching algorithms and adaptive CFL controllers for Euler equations.
"""
from mpi4py import MPI  # noqa: I001

import time
from dolfinx import fem

from config import SimulationConfig
from .operators import EulerOperators
from .physics import euler_flux_tensor
from .spaces import EulerSpaces
from .stabilization import BaseStabilization


class SSPRK3Integrator:
    """
    Third-order Strong Stability Preserving Runge-Kutta (SSP-RK3) integrator.

    Solves the semi-discrete system:
        dU/dt = M^{-1} (-C * F(U) - R_visc(U))
    """

    def __init__(
        self,
        spaces: EulerSpaces,
        operators: EulerOperators,
        stabilization: BaseStabilization,
        bcs: list[fem.DirichletBC],
        config: SimulationConfig,
    ):
        self.spaces = spaces
        self.operators = operators
        self.stabilization = stabilization
        self.bcs = bcs
        self.config = config
        self.comm = spaces.mesh.comm

        # State and flux functions
        self.U_func = fem.Function(spaces.W, name="Conservative_Variables")
        self.F_func = fem.Function(spaces.W_tensor, name="Flux_Tensor")

        # Work residual functions and vectors
        self.R = fem.Function(spaces.W)
        self.R_visc = fem.Function(spaces.W)
        self.dU = fem.Function(spaces.W)

        # UFL expression for the convective flux tensor F(U)
        F_sym = euler_flux_tensor(self.U_func, config.gamma)
        self.expr_F = fem.Expression(F_sym, spaces.W_tensor.element.interpolation_points)

        # Connect stabilization weak forms to state function U_func
        self.stabilization.setup_forms(self.U_func)

        # PETSc vector references
        self.u = self.U_func.x.petsc_vec
        self.du = self.dU.x.petsc_vec
        self.u0 = self.u.copy()

    def dUdt(self):
        """Evaluates dU/dt = M^{-1} (-C * F - R_visc)."""
        # 1. Convective term: R = -C * F
        self.F_func.interpolate(self.expr_F)
        self.operators.apply_convective_divergence(self.F_func.x.petsc_vec, self.R.x.petsc_vec)

        # 2. Viscous diffusion term: R -= R_visc
        self.stabilization.assemble_viscous_residual(self.R_visc.x.petsc_vec)
        self.R.x.petsc_vec.axpy(-1.0, self.R_visc.x.petsc_vec)

        # 3. Mass matrix solve: dU = M^{-1} * R
        self.operators.solve_mass(self.R.x.petsc_vec, self.du)

    def solve(self, initial_condition, writer=None, diagnostics=None):
        """
        Executes the time-stepping simulation loop until T_END.
        """
        # Initialize state
        self.U_func.interpolate(initial_condition)
        fem.set_bc(self.u, self.bcs)
        self.U_func.x.scatter_forward()

        # Initial output write
        if writer is not None:
            writer.write(0.0, self.U_func)

        T = 0.0
        step = 0
        h_min = self.stabilization.h_min
        t_start = time.perf_counter()

        if self.comm.rank == 0:
            print(f"Starting simulation on mesh (h_min = {h_min:.4e})...")

        while T < self.config.t_end:
            # 1. Update wave speed, element viscosity, and adaptive time step
            max_speed = self.stabilization.update(self.U_func)
            dt = self.config.cfl * h_min / max_speed
            if T + dt > self.config.t_end:
                dt = self.config.t_end - T

            # 2. Store base state u0 = u^n
            self.u.copy(self.u0)

            # --- Stage 1 ---
            # u^(1) = u^n + dt * L(u^n)
            self.dUdt()
            self.u0.copy(self.u)
            self.u.axpy(dt, self.du)
            fem.set_bc(self.u, self.bcs)
            self.U_func.x.scatter_forward()

            # --- Stage 2 ---
            # u^(2) = 3/4 * u^n + 1/4 * (u^(1) + dt * L(u^(1)))
            self.dUdt()
            self.u.axpy(dt, self.du)
            self.u.scale(0.25)
            self.u.axpy(0.75, self.u0)
            fem.set_bc(self.u, self.bcs)
            self.U_func.x.scatter_forward()

            # --- Stage 3 ---
            # u^(n+1) = 1/3 * u^n + 2/3 * (u^(2) + dt * L(u^(2)))
            self.dUdt()
            self.u.axpy(dt, self.du)
            self.u.scale(2.0 / 3.0)
            self.u.axpy(1.0 / 3.0, self.u0)
            fem.set_bc(self.u, self.bcs)
            self.U_func.x.scatter_forward()

            T += dt
            step += 1

            # Save solution to VTX output
            if writer is not None and (step % self.config.save_interval == 0 or T >= self.config.t_end):
                writer.write(T, self.U_func)

            # Diagnostic logging
            if diagnostics is not None and (step % self.config.print_interval == 0 or T >= self.config.t_end):
                diagnostics.report(step, T, dt, self.U_func)

        t_end_wall = time.perf_counter()
        elapsed = self.comm.allreduce(t_end_wall - t_start, op=MPI.MAX)

        if writer is not None:
            writer.close()

        if self.comm.rank == 0:
            print(f"Simulation successfully completed in {elapsed:.2f} seconds ({step} steps).")

        return elapsed
