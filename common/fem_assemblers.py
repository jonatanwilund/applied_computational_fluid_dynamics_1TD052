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


def convection_assembler_2d_vec(p, t, bx, by, geometry=None):
    """
    Vectorized assembly of

        C_ij = int (beta . grad(phi_j)) phi_i dx.

    bx and by are nodal arrays. If geometry is provided, it must
    be the tuple returned by the function "precompute_triangle_geometry". 
    """
    npnt = p.shape[1]

    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)

    if bx.shape != (npnt,) or by.shape != (npnt,):
        raise ValueError("bx and by must have shape (p.shape[1],).")

    if geometry is None:
        geometry = precompute_triangle_geometry(p, t)

    area, b, c = geometry
    bxmid = np.mean(bx[t], axis=0)
    bymid = np.mean(by[t], axis=0)
    beta_grad_phi = bxmid[None, :] * b + bymid[None, :] * c

    triangles = t.T
    rows = np.repeat(triangles, 3, axis=1).ravel()
    cols = np.tile(triangles, (1, 3)).ravel()
    values = (area[:, None] / 3.0 * np.tile(beta_grad_phi.T, (1, 3))).ravel()

    return coo_matrix((values, (rows, cols)), shape=(npnt, npnt)).tocsr()
