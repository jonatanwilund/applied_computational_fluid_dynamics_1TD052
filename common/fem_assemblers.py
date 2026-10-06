import numpy as np
from scipy.sparse import coo_matrix, csr_array


def hat_gradients(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    detJ = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])

    if abs(detJ) < 1.0e-15:
        raise ValueError("Degenerate triangle.")

    area = 0.5 * abs(detJ)

    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]], dtype=float) / detJ

    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]], dtype=float) / detJ

    return area, b, c


def _assemble_sparse(npnt, t, local_matrix) -> csr_array:
    nt = t.shape[1]

    rows = []
    cols = []
    vals = []

    for K in range(nt):
        loc2glb = t[:, K]
        AK = local_matrix(K, loc2glb)

        for i in range(3):
            for j in range(3):
                rows.append(loc2glb[i])
                cols.append(loc2glb[j])
                vals.append(AK[i, j])

    return coo_matrix((vals, (rows, cols)), shape=(npnt, npnt)).tocsr()


def mass_assembler_2d(p, t):
    """
    Assemble M_ij = int phi_j phi_i dx.

    Local matrix:
        M_K = |K|/12 [[2,1,1],
                      [1,2,1],
                      [1,1,2]]
    """
    npnt = p.shape[1]

    M0 = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 12.0

    def local_matrix(K, loc2glb):
        x = p[0, loc2glb]
        y = p[1, loc2glb]
        area, _, _ = hat_gradients(x, y)
        return area * M0

    return _assemble_sparse(npnt, t, local_matrix)


def load_assembler_2d(p, t, f):
    """
    Assemble b_i = int f phi_i dx using the corner quadrature rule
    from Larson-Bengzon.
    """
    npnt = p.shape[1]
    nt = t.shape[1]

    b_global = np.zeros(npnt)

    for K in range(nt):
        loc2glb = t[:, K]

        x = p[0, loc2glb]
        y = p[1, loc2glb]

        area, _, _ = hat_gradients(x, y)

        bK = (
            area
            / 3.0
            * np.array([f(x[0], y[0]), f(x[1], y[1]), f(x[2], y[2])], dtype=float)
        )

        b_global[loc2glb] += bK

    return b_global


def stiffness_assembler_2d(p, t, a=lambda x, y: 1.0):
    """
    Assemble A_ij = int a grad(phi_j).grad(phi_i) dx.

    The coefficient a is evaluated at the triangle centroid,
    as in Larson-Bengzon.
    """
    npnt = p.shape[1]

    def local_matrix(K, loc2glb):
        x = p[0, loc2glb]
        y = p[1, loc2glb]

        area, b, c = hat_gradients(x, y)

        xc = np.mean(x)
        yc = np.mean(y)
        abar = float(a(xc, yc))

        return abar * area * (np.outer(b, b) + np.outer(c, c))

    return _assemble_sparse(npnt, t, local_matrix)


def stiffness_assembler_2d_vec(p, t, a):
    """
    a is a 1D array of length nt, containing the value of a for each element K
    """
    npnt = p.shape[1]

    def local_matrix(K, loc2glb):
        x = p[0, loc2glb]
        y = p[1, loc2glb]

        area, b, c = hat_gradients(x, y)

        return a[K] * area * (np.outer(b, b) + np.outer(c, c))

    return _assemble_sparse(npnt, t, local_matrix)


def precompute_triangle_geometry(p, t):
    """Precompute areas and basis-function gradients for all triangles."""
    x = p[0, t]
    y = p[1, t]

    detJ = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    if np.any(np.abs(detJ) < 1.0e-15):
        raise ValueError("Degenerate triangle.")

    area = 0.5 * np.abs(detJ)
    b = np.stack((y[1] - y[2], y[2] - y[0], y[0] - y[1])) / detJ
    c = np.stack((x[2] - x[1], x[0] - x[2], x[1] - x[0])) / detJ
    return area, b, c


def convection_assembler_2d(p, t, bx, by):
    """
    Assemble

        C_ij = int (beta . grad(phi_j)) phi_i dx,

    following Larson-Bengzon.

    bx and by are nodal arrays.
    """
    npnt = p.shape[1]

    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)

    if bx.shape != (npnt,) or by.shape != (npnt,):
        raise ValueError("bx and by must have shape (p.shape[1],).")

    def local_matrix(K, loc2glb):
        x = p[0, loc2glb]
        y = p[1, loc2glb]

        area, b, c = hat_gradients(x, y)

        bxmid = np.mean(bx[loc2glb])
        bymid = np.mean(by[loc2glb])

        beta_grad_phi = bxmid * b + bymid * c

        return area / 3.0 * np.outer(np.ones(3), beta_grad_phi)

    return _assemble_sparse(npnt, t, local_matrix)


def nonlinear_domain_residual_assembler_2d(p, t, U, geometry=None):
    """
    Assemble the domain integral part of the nonlinear residual vector:
    C_i = int_Omega f(u_h) . grad(phi_i) dx

    The flux is evaluated at the triangle centroid.
    """
    npnt = p.shape[1]
    if geometry is None:
        geometry = precompute_triangle_geometry(p, t)

    area, b, c = geometry
    u_c = np.mean(U[t], axis=0)
    fx = np.sin(u_c)
    fy = np.cos(u_c)
    local_residual = area * (fx * b + fy * c)

    C_global = np.zeros(npnt)
    np.add.at(C_global, t, local_residual)
    return C_global


def nonlinear_divergence_residual_assembler_2d(p, t, U, geometry=None):
    """Assemble the positive P1 centroid-quadrature load for div(f(U))."""
    npnt = p.shape[1]
    if geometry is None:
        geometry = precompute_triangle_geometry(p, t)

    area, b, c = geometry
    u_c = np.mean(U[t], axis=0)
    du_dx = np.sum(U[t] * b, axis=0)
    du_dy = np.sum(U[t] * c, axis=0)
    divergence = np.cos(u_c) * du_dx - np.sin(u_c) * du_dy
    local_load = np.broadcast_to(area * divergence / 3.0, t.shape)

    divergence_load = np.zeros(npnt)
    np.add.at(divergence_load, t, local_load)
    return divergence_load


def nonlinear_boundary_residual_assembler_2d(p, e, U):
    """
    Assemble int_{dOmega} (f(u_h) . n) phi_i ds using a vectorized midpoint rule.

    Parameters:
      p: Node coordinate array, shape (2, npnt)
      boundary_edges: Boundary edge connectivity, shape (2, N_bound)
      U: Current solution vector, shape (npnt,)
    """
    npnt = p.shape[1]
    G_bound = np.zeros(npnt)

    # Extract node indices for all boundary edges
    n1 = e[0, :]
    n2 = e[1, :]

    # Extract coordinates
    x1, y1 = p[0, n1], p[1, n1]
    x2, y2 = p[0, n2], p[1, n2]

    # Edge vectors and lengths
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)

    # Outward unit normals (assumes counter-clockwise boundary traversal)
    nx = dy / L
    ny = -dx / L

    # Evaluate solution at edge midpoints
    u_m = 0.5 * (U[n1] + U[n2])

    # Evaluate nonlinear flux f(u) = (sin u, cos u) at midpoints
    fx = np.sin(u_m)
    fy = np.cos(u_m)

    # Compute dot product with normal and scale by quadrature weights (L * 0.5)
    flux_n = fx * nx + fy * ny
    val = 0.5 * L * flux_n

    # Scatter contributions back to the global nodes
    np.add.at(G_bound, n1, val)
    np.add.at(G_bound, n2, val)

    return G_bound
