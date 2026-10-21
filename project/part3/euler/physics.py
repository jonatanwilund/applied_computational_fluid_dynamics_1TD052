"""
Thermodynamic and aerodynamic constitutive relations for the 2D compressible Euler equations.
"""

import ufl


def pressure(rho, rhou, rhov, E, gamma: float = 1.4):
    """
    Computes thermodynamic pressure from conservative variables using the ideal gas law:
    p = (gamma - 1) * (E - 0.5 * rho * |u|^2)
    """
    kinetic_energy = 0.5 * (rhou**2 + rhov**2) / rho
    return (gamma - 1.0) * (E - kinetic_energy)


def pressure_from_state(U, gamma: float = 1.4):
    """Computes pressure directly from state vector U = [rho, rhou, rhov, E]."""
    return pressure(U[0], U[1], U[2], U[3], gamma)


def velocity(U):
    """Computes fluid velocity vector u = [u, v] = [rhou/rho, rhov/rho]."""
    return ufl.as_vector([U[1] / U[0], U[2] / U[0]])


def sound_speed(rho, p, gamma: float = 1.4):
    """Computes acoustic wave speed c = sqrt(gamma * p / rho)."""
    return ufl.sqrt(gamma * p / rho)


def max_wave_speed(U, gamma: float = 1.4):
    """
    Computes the maximum convective wave speed lambda = |u| + c.
    Used for CFL time step estimation and artificial viscosity scaling.
    """
    u_vec = velocity(U)
    vel_mag = ufl.sqrt(u_vec[0] ** 2 + u_vec[1] ** 2)
    p = pressure_from_state(U, gamma)
    c = sound_speed(U[0], p, gamma)
    return vel_mag + c


def euler_flux_tensor(U, gamma: float = 1.4):
    """
    Constructs the 4 x 2 convective Euler flux tensor F(U)
    """
    rho, rhou, rhov, E = U[0], U[1], U[2], U[3]
    u = rhou / rho
    v = rhov / rho
    p = pressure(rho, rhou, rhov, E, gamma)

    return ufl.as_matrix([
        [rhou, rhov],
        [rhou * u + p, rhou * v],
        [rhov * u, rhov * v + p],
        [(E + p) * u, (E + p) * v],
    ])
