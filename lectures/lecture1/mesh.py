"""
mesh.py

Teaching utilities for converting a 2D triangular DOLFINx mesh into the
Larson-Bengzon style arrays

    p : node coordinates
    e : boundary-edge connectivity
    t : triangle connectivity

The convention used here is Pythonic:
    p.shape == (2, np)
    e.shape == (2, ne)
    t.shape == (3, nt)

and all indices are zero-based.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from dolfinx import mesh as dmesh


def _signed_triangle_area(p: np.ndarray, tri: np.ndarray) -> float:
    """Return the signed area of one triangle."""
    i, j, k = tri
    x1, y1 = p[:, i]
    x2, y2 = p[:, j]
    x3, y3 = p[:, k]

    return 0.5 * (
        (x2 - x1) * (y3 - y1)
        - (x3 - x1) * (y2 - y1)
    )


def dolfinx_to_pet(msh):
    """
    Convert a serial 2D triangular DOLFINx mesh to Larson-Bengzon style
    point-edge-triangle arrays.

    Parameters
    ----------
    msh
        A 2D triangular DOLFINx mesh.

    Returns
    -------
    p : numpy.ndarray
        Shape (2, np). Column i contains the coordinates of vertex i.

    e : numpy.ndarray
        Shape (2, ne). Column j contains the two vertices of boundary edge j.

    t : numpy.ndarray
        Shape (3, nt). Column K contains the three vertices of triangle K,
        ordered counterclockwise.

    Notes
    -----
    This routine is intentionally designed for teaching and assumes a serial
    mesh, e.g. a mesh created with MPI.COMM_SELF or run with one MPI rank.

    Python zero-based indexing is used.
    """
    tdim = msh.topology.dim

    if tdim != 2:
        raise ValueError("dolfinx_to_pet currently supports only 2D meshes.")

    # ------------------------------------------------------------
    # p: coordinates of mesh vertices
    # ------------------------------------------------------------
    #
    # For first-order triangular geometry, the geometry nodes are the vertices.
    # We retain only x and y coordinates.
    p = np.asarray(msh.geometry.x[:, :2], dtype=float).T.copy()

    # ------------------------------------------------------------
    # t: cell -> vertex connectivity
    # ------------------------------------------------------------
    msh.topology.create_connectivity(tdim, 0)
    cell_to_vertex = msh.topology.connectivity(tdim, 0)

    num_cells = msh.topology.index_map(tdim).size_local
    t = np.empty((3, num_cells), dtype=np.int32)

    for K in range(num_cells):
        vertices = cell_to_vertex.links(K)

        if len(vertices) != 3:
            raise ValueError(
                "dolfinx_to_pet requires a triangular mesh."
            )

        t[:, K] = vertices

    # Larson-Bengzon use counterclockwise triangle ordering.
    for K in range(num_cells):
        if _signed_triangle_area(p, t[:, K]) < 0.0:
            t[1, K], t[2, K] = t[2, K], t[1, K]

    # ------------------------------------------------------------
    # e: exterior facet -> vertex connectivity
    # ------------------------------------------------------------
    fdim = tdim - 1

    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(fdim, 0)

    boundary_facets = dmesh.exterior_facet_indices(msh.topology)
    facet_to_vertex = msh.topology.connectivity(fdim, 0)

    e = np.empty((2, len(boundary_facets)), dtype=np.int32)

    for j, facet in enumerate(boundary_facets):
        vertices = facet_to_vertex.links(facet)

        if len(vertices) != 2:
            raise ValueError(
                "A boundary edge of a triangular 2D mesh must have two vertices."
            )

        e[:, j] = vertices

    return p, e, t


def plot_mesh(
    p: np.ndarray,
    e: np.ndarray,
    t: np.ndarray,
    *,
    show_node_numbers: bool = False,
    show_element_numbers: bool = False,
    ax=None,
):
    """
    Plot a 2D triangular mesh in a style similar to MATLAB's pdemesh.

    Parameters
    ----------
    p, e, t
        Larson-Bengzon style mesh arrays.

    show_node_numbers
        If True, annotate vertex numbers.

    show_element_numbers
        If True, annotate triangle numbers.

    ax
        Optional matplotlib axis.

    Returns
    -------
    ax
        The matplotlib axis.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Draw all triangle edges.
    ax.triplot(p[0], p[1], t.T, linewidth=0.8)

    # Draw the exterior boundary a little thicker.
    for edge in e.T:
        i, j = edge
        ax.plot(
            [p[0, i], p[0, j]],
            [p[1, i], p[1, j]],
            linewidth=1.8,
        )

    if show_node_numbers:
        for i in range(p.shape[1]):
            ax.text(
                p[0, i],
                p[1, i],
                f" {i}",
                fontsize=9,
                ha="left",
                va="bottom",
            )

    if show_element_numbers:
        for K in range(t.shape[1]):
            vertices = t[:, K]
            xc = np.mean(p[0, vertices])
            yc = np.mean(p[1, vertices])
            ax.text(
                xc,
                yc,
                str(K),
                fontsize=8,
                ha="center",
                va="center",
            )

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Triangular mesh")

    return ax
