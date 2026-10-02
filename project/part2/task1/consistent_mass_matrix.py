import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import linalg
from shared import (
    SPP_RK3_step,
    compute_element_diameters,
    get_mesh,
    save_solution_plot,
    total_mass,
    u0,
)
from tqdm import tqdm

from common.fem_assemblers import (
    load_assembler_2d,
    mass_assembler_2d,
    stiffness_assembler_2d_vec,
)
from common.mesh import dolfinx_to_pet


def main():
    # Create mesh
    h = 0.1
    N = round(4 / h)  # 4 is the domain width in both directions
    domain = get_mesh(N)
    p, e, t = dolfinx_to_pet(domain)

    # Assemble the mass matrix and factorize it once for repeated solves.
    M = mass_assembler_2d(p, t)
    mass_solve = linalg.factorized(M.tocsc())
    
    # Mesh dependent artificial viscosity
    # ||f'(u)||_inf = 1
    h_K = compute_element_diameters(p, t)
    a = 0.5 * h_K  # C_vel = 0.5
    
    # Assemble stiffness matrix for artificial viscosity
    A = stiffness_assembler_2d_vec(p, t, a)

    b0 = load_assembler_2d(p, t, lambda x, y: u0(np.array([x, y])))
    U0 = mass_solve(b0)

    # Time integration using SSP-RK3
    dt = 0.01
    T = 1.0
    num_steps = int(T / dt)
    U_n = U0.copy()
    U_min_vals = []
    U_max_vals = []
    total_mass_vals = []
    for step in tqdm(range(num_steps), desc="Time-stepping"):
        U_n = SPP_RK3_step(p, e, t, mass_solve, U_n, dt)
        U_min_vals.append(U_n.min())
        U_max_vals.append(U_n.max())
        total_mass_vals.append(total_mass(M, U_n))
        if step % 10 == 0:  # Save every 10 steps
            save_solution_plot(p, t, U_n, T, step * dt)

    # Save a plot of the min, max, and total mass over time
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(num_steps) * dt, U_min_vals, label="Min U", color="blue")
    plt.plot(np.arange(num_steps) * dt, U_max_vals, label="Max U", color="orange")
    plt.plot(
        np.arange(num_steps) * dt, total_mass_vals, label="Total Mass", color="green"
    )
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Min, Max, and Total Mass over Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"project/part2/min_max_mass_T{T}.png", dpi=300)


if __name__ == "__main__":
    main()
