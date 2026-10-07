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
    mass_assembler_2d,
    precompute_triangle_geometry,
    stiffness_assembler_2d_vec,
)
from common.mesh import dolfinx_to_pet


def solve(CFL, p, t, M, mass_solve, viscosity_matrix, geometry, h_min, T=1.0):
    dt = CFL * h_min

    U_n = u0(p)
    U_min_vals = [U_n.min()]
    U_max_vals = [U_n.max()]
    total_mass_vals = [total_mass(M, U_n)]

    times = [0.0]
    time = 0.0
    while time < T:
        dt_step = min(dt, T - time)
        U_n = SPP_RK3_step(p, t, mass_solve, U_n, dt_step, viscosity_matrix, geometry)

        U_min_vals.append(U_n.min())
        U_max_vals.append(U_n.max())
        total_mass_vals.append(total_mass(M, U_n))
        time += dt_step
        times.append(time)

    return times, U_min_vals, U_max_vals, total_mass_vals


def main(h: float = 0.1, CFL_list: tuple[float] = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]):
    output_dir = Path("project/part2/task1/plots/consistent")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create mesh
    N = round(4 / h)  # 4 is the domain width in both directions
    domain = get_mesh(N)
    p, _, t = dolfinx_to_pet(domain)
    geometry = precompute_triangle_geometry(p, t)

    M = mass_assembler_2d(p, t)
    mass_solve = linalg.factorized(M.tocsc())

    h_K = compute_element_diameters(p, t)

    Cvel_list = [0.1, 0.5, 1.0]
    for Cvel in Cvel_list:
        viscosity_matrix = stiffness_assembler_2d_vec(p, t, Cvel * h_K)
        results = []

        for cfl in tqdm(CFL_list, desc=f"Cvel = {Cvel}"):
            results.append(
                solve(
                    cfl,
                    p,
                    t,
                    M,
                    mass_solve,
                    viscosity_matrix,
                    geometry,
                    np.min(h_K),
                )
            )

        colors = plt.cm.viridis(np.linspace(0, 1, len(CFL_list)))
        for cfl, result, color in zip(CFL_list, results, colors):
            times, u_min, u_max, mass = result
            fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
            axes[0].plot(times, u_min, color=color, label=f"CFL = {cfl}")
            axes[0].plot(times, u_max, color=color, linestyle="--")

            axes[1].plot(times, mass, color=color)

            relative_mass_change = np.abs(np.asarray(mass) - mass[0]) / abs(mass[0])
            axes[2].plot(times, relative_mass_change, color=color)

            axes[0].set_ylabel("Solution extrema")
            axes[0].set_title(f"Minimum and maximum solution values (Cvel = {Cvel})")
            axes[0].legend()

            axes[1].set_ylabel(r"$M_h = 1^T M U$")
            axes[1].set_title("Total mass")

            axes[2].set_xlabel("Time")
            axes[2].set_ylabel("Relative change")
            axes[2].set_title("Relative change in total mass")
            axes[2].axhline(0, color="black", linewidth=0.8)

            for axis in axes:
                axis.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                output_dir / f"consistent_mass_matrix_cvel_{Cvel}_cfl_{cfl}.png",
                dpi=300,
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
