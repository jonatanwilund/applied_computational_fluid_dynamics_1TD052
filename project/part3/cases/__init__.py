"""
Test case problem setups for compressible Euler equations.
"""

from .base import BaseProblem
from .mach3_cylinder import Mach3CylinderProblem
from .sod_shock import SodShockProblem

__all__ = ["BaseProblem", "SodShockProblem", "Mach3CylinderProblem"]
