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


def main(h: float = 0.1, CFL_list: tuple[float] = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]):
    output_dir = Path("project/part2/task1/plots/unstable")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create mesh
    N = round(4 / h)  # 4 is the domain width in both directions

    domain = get_mesh(N)
    p, _, t = dolfinx_to_pet(domain)
    geometry = precompute_triangle_geometry(p, t)

    M = mass_assembler_2d(p, t)
    mass_solve = linalg.factorized(M.tocsc())

    h_K = compute_element_diameters(p, t)
    epsilon_K = 0.5 * h_K
    viscosity_matrix = stiffness_assembler_2d_vec(p, t, epsilon_K)

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
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        relative_axis = axes[1].twinx()
        axes[0].plot(times, u_min, color=color, label=f"CFL = {cfl}")
        axes[0].plot(times, u_max, color=color, linestyle="--")

        axes[1].plot(times, mass, color=color, label="Total mass")

        relative_mass_change = np.abs(np.asarray(mass) - mass[0]) / abs(mass[0])
        relative_axis.plot(
            times,
            relative_mass_change,
            color=color,
            linestyle=":",
            label="Relative mass change",
        )

        axes[0].set_ylabel("Solution extrema")
        axes[0].set_title(f"Unstable Galerkin: solution extrema (CFL = {cfl})")
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
        )

        for axis in axes:
            axis.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"unstable_galerkin_cfl_{cfl}_comparison.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
