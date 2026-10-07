from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
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
    output_dir = Path("project/part2/task1/plots/lumped")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create mesh
    N = round(4 / h)  # 4 is the domain width in both directions

    domain = get_mesh(N)
    p, _, t = dolfinx_to_pet(domain)
    geometry = precompute_triangle_geometry(p, t)

    consistent_M = mass_assembler_2d(p, t)
    lumped_diagonal = np.asarray(consistent_M.sum(axis=1)).ravel()
    M = diags(lumped_diagonal, format="csr")
    mass_solve = lambda rhs: rhs / lumped_diagonal

    h_K = compute_element_diameters(p, t)

    Cvel_list = [0.1, 0.5, 1.0]
    results_by_cvel = {}
    for Cvel in Cvel_list:
        viscosity_matrix = stiffness_assembler_2d_vec(p, t, Cvel * h_K)
        results_by_cvel[Cvel] = []

        for cfl in tqdm(CFL_list, desc=f"Cvel = {Cvel}"):
            results_by_cvel[Cvel].append(
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

    colors = plt.cm.viridis(np.linspace(0, 1, len(Cvel_list)))
    for cfl_index, cfl in enumerate(CFL_list):
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        relative_axis = axes[1].twinx()
        for Cvel, color in zip(Cvel_list, colors):
            times, u_min, u_max, mass = results_by_cvel[Cvel][cfl_index]
            axes[0].plot(times, u_min, color=color, label=f"Cvel = {Cvel}")
            axes[0].plot(times, u_max, color=color, linestyle="--")
            axes[1].plot(
                times,
                mass,
                color=color,
                label=f"Cvel = {Cvel} total mass",
            )

            relative_mass_change = np.abs(np.asarray(mass) - mass[0]) / abs(mass[0])
            relative_axis.plot(
                times,
                relative_mass_change,
                color=color,
                linestyle=":",
                label=f"Cvel = {Cvel} relative change",
            )

        axes[0].set_ylabel("Solution extrema")
        axes[0].set_title(
            f"Lumped mass matrix: solution extrema (CFL = {cfl})"
        )
        axes[0].legend()

        axes[1].set_ylabel(r"$M_h = 1^T M U$")
        relative_axis.set_ylabel("Relative mass change")
        axes[1].set_xlabel("Time")
        axes[1].set_title("Total mass and relative mass change")
        relative_axis.axhline(0, color="black", linewidth=0.8)

        mass_handles, mass_labels = axes[1].get_legend_handles_labels()
        relative_handles, relative_labels = relative_axis.get_legend_handles_labels()
        axes[1].legend(
            mass_handles + relative_handles,
            mass_labels + relative_labels,
            loc="best",
            ncol=2,
        )

        for axis in axes:
            axis.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"lumped_mass_matrix_cfl_{cfl}_cvel_comparison.png",
            dpi=300,
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
