import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import linalg
from shared import (
    SPP_RK3_step,
    compute_element_diameters,
    get_mesh,
    total_mass,
    u0,
)
from tqdm import tqdm

from common.fem_assemblers import (
    load_assembler_2d,
    mass_assembler_2d,
    precompute_triangle_geometry,
    stiffness_assembler_2d_vec,
)
from common.mesh import dolfinx_to_pet


def solve(CFL, p, t, M, mass_solve, viscosity_matrix, geometry, U0, h_min, T=1.0):
    dt = CFL * h_min
    num_steps = int(T / dt)

    U_n = U0.copy()
    U_min_vals = [U_n.min()]
    U_max_vals = [U_n.max()]
    total_mass_vals = [total_mass(M, U_n)]
    
    for step in range(num_steps):
        # Pass the stiffness matrix A into the step function
        U_n = SPP_RK3_step(
            p, t, mass_solve, U_n, dt, viscosity_matrix, geometry
        )
        
        U_min_vals.append(U_n.min())
        U_max_vals.append(U_n.max())
        total_mass_vals.append(total_mass(M, U_n))
        
    times = np.arange(num_steps + 1) * dt
    return times, U_min_vals, U_max_vals, total_mass_vals


def main():
    # Create mesh
    h = 0.05
    N = round(4 / h)  # 4 is the domain width in both directions
    
    domain = get_mesh(N)
    p, _, t = dolfinx_to_pet(domain)
    geometry = precompute_triangle_geometry(p, t)

    M = mass_assembler_2d(p, t)
    mass_solve = linalg.factorized(M.tocsc())

    h_K = compute_element_diameters(p, t)
    epsilon_K = 0.5 * h_K
    viscosity_matrix = stiffness_assembler_2d_vec(p, t, epsilon_K)

    b0 = load_assembler_2d(p, t, lambda x, y: u0(np.array([x, y])))
    U0 = mass_solve(b0)
    
    CFL_list = [
        0.05,
        0.1, 
        0.2, 
        0.5, 
        0.8, 
    ]
    times_list = []
    U_min_vals_list = []
    U_max_vals_list = []
    total_mass_vals_list = []


    for cfl in tqdm(CFL_list, desc="CFL Loop"):
        times, U_min_vals, U_max_vals, total_mass_vals = solve(
            cfl,
            p,
            t,
            M,
            mass_solve,
            viscosity_matrix,
            geometry,
            U0,
            np.min(h_K),
        )
        times_list.append(times)
        U_min_vals_list.append(U_min_vals)
        U_max_vals_list.append(U_max_vals)
        total_mass_vals_list.append(total_mass_vals)

    colors = plt.cm.viridis(np.linspace(0, 1, len(CFL_list)))

    for cfl, times, u_min, u_max, mass, color in zip(
        CFL_list,
        times_list,
        U_min_vals_list,
        U_max_vals_list,
        total_mass_vals_list,
        colors,
    ):
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        axes[0].plot(times, u_min, color=color, label=f"CFL = {cfl}")
        axes[0].plot(times, u_max, color=color, linestyle="--")

        axes[1].plot(times, mass, color=color)

        relative_mass_change = (np.asarray(mass) - mass[0]) / abs(mass[0])
        axes[2].plot(times, relative_mass_change, color=color)

        axes[0].set_ylabel("Solution extrema")
        axes[0].set_title("Minimum and maximum solution values")
        axes[0].legend()

        axes[1].set_ylabel(r"$U^T M U$")
        axes[1].set_title("Mass-matrix quantity")

        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Relative change")
        axes[2].set_title("Relative change in mass-matrix quantity")
        axes[2].axhline(0, color="black", linewidth=0.8)

        for axis in axes:
            axis.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"project/part2/task1/plots/consistent_mass_matrix_cfl_{cfl}_comparison.png", dpi=300)



if __name__ == "__main__":
    main()
