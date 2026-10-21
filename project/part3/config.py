"""
Simulation configuration parameters for the compressible Euler solver.
"""

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Numerical and physical parameters for the Euler simulation."""

    gamma: float = 1.4  # Ratio of specific heats (ideal gas)
    t_end: float = 0.2  # Final simulation time
    cfl: float = 0.05  # CFL number for adaptive time stepping
    c_visc: float = 0.1  # Artificial viscosity tuning constant
    nx: int = 200  # Number of grid subdivisions in x
    ny: int = 40  # Number of grid subdivisions in y
    save_interval: int = 5  # Write VTX output every N steps
    print_interval: int = 20  # Log diagnostic line every N steps
    output_dir: str = "output"  # Directory for simulation outputs
    output_name: str = "euler_sod.bp"  # ADIOS2 VTX output folder name
