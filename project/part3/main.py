"""
Main driver script for 2D compressible Euler flow simulations.
"""

from mpi4py import MPI

from cases.sod_shock import SodShockProblem
from config import SimulationConfig
from euler.operators import EulerOperators
from euler.spaces import EulerSpaces
from euler.stabilization import FirstOrderViscosity
from euler.timestepping import SSPRK3Integrator
from utils.diagnostics import DiagnosticTracker
from utils.io import SolutionWriter


def main():
    comm = MPI.COMM_WORLD
    config = SimulationConfig()

    # 1. Setup problem geometry, initial condition, and boundary conditions
    problem = SodShockProblem(comm, config)

    # 2. Setup finite element spaces and pre-assembled operators
    spaces = EulerSpaces(problem.mesh)
    operators = EulerOperators(spaces)

    # 3. Setup stabilization (First-order artificial viscosity)
    stabilization = FirstOrderViscosity(spaces, config)

    # 4. Construct Dirichlet boundary conditions
    bcs = problem.build_bcs(spaces)

    # 5. Initialize time-marching integrator
    integrator = SSPRK3Integrator(spaces, operators, stabilization, bcs, config)

    # 6. Initialize I/O writer and runtime diagnostic tracker
    writer = SolutionWriter(spaces, stabilization, config)
    diagnostics = DiagnosticTracker(spaces, config)

    # 7. Execute simulation
    integrator.solve(
        initial_condition=problem.initial_condition,
        writer=writer,
        diagnostics=diagnostics,
    )


if __name__ == "__main__":
    main()
