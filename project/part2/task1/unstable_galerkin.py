from pathlib import Path

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

    U_n = U0.copy()
    U_min_vals = [U_n.min()]
    U_max_vals = [U_n.max()]
    total_mass_vals = [total_mass(M, U_n)]
    
    times = [0.0]
    time = 0.0
    while time < T:
        dt_step = min(dt, T - time)
        U_n = SPP_RK3_step(
            p,
            t,
            mass_solve,
            U_n,
            dt_step,
            viscosity_matrix,
            geometry,
            artificial_viscosity=False,
        )
        
        U_min_vals.append(U_n.min())
        U_max_vals.append(U_n.max())
        total_mass_vals.append(total_mass(M, U_n))
        time += dt_step
        times.append(time)

    return times, U_min_vals, U_max_vals, total_mass_vals


def main():
    output_dir = Path("project/part2/task1/plots/unstable")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create mesh
    h = 0.1
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

        relative_mass_change = np.abs(np.asarray(mass) - mass[0]) / abs(mass[0])
        axes[2].plot(times, relative_mass_change, color=color)

        axes[0].set_ylabel("Solution extrema")
        axes[0].set_title("Minimum and maximum solution values")
        axes[0].legend()

        axes[1].set_ylabel(r"$M_h = 1^T M U$")
        axes[1].set_title("Total mass")

        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Relative change")
        axes[2].set_title("Relative change in mass-matrix quantity")
        axes[2].axhline(0, color="black", linewidth=0.8)

        for axis in axes:
            axis.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"unstable_galerkin_cfl_{cfl}_comparison.png", dpi=300)
        plt.close(fig)



if __name__ == "__main__":
    main()
