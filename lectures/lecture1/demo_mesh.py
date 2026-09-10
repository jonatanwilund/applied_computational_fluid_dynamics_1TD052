"""
demo_mesh.py

Small test of mesh.py.
"""

import matplotlib.pyplot as plt
from dolfinx import mesh
from mpi4py import MPI

from common.mesh import dolfinx_to_pet, plot_mesh


def main():
    # For the teaching [p,e,t] representation we deliberately use a serial mesh.
    msh = mesh.create_unit_square(
        MPI.COMM_SELF,
        4,
        4,
        cell_type=mesh.CellType.triangle,
    )

    p, e, t = dolfinx_to_pet(msh)

    print("p.shape =", p.shape)
    print("e.shape =", e.shape)
    print("t.shape =", t.shape)

    print("\nFirst few nodes:")
    print(p[:, :5])

    print("\nFirst few boundary edges:")
    print(e[:, :5])

    print("\nFirst few triangles:")
    print(t[:, :5])

    plot_mesh(
        p,
        e,
        t,
        show_node_numbers=True,
        show_element_numbers=False,
    )

    plt.savefig("demo_mesh.png", dpi=300)


if __name__ == "__main__":
    main()
