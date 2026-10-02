from scipy.sparse import csr_array, dia_array, diags_array


def lumped_mass_matrix(M: csr_array) -> dia_array:
    """
    Compute the lumped mass matrix from the consistent mass matrix M.
    The diagonal entries of the lumped mass matrix are the column sums of M.

    Parameters:
      M: Consistent mass matrix (sparse), shape (npnt, npnt)
    Returns:
        Lumped mass matrix (sparse diagonal), shape (npnt, npnt)
    """
    return diags_array(M.sum(axis=1).A1)  # Use .A1 to convert to 1D array