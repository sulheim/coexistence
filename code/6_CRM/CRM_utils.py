import numpy as np
import scipy.stats as st

"""
Utility functions for Consumer Resource Model (CRM) simulations.

These functions include:
- `numerical_error`: Checks if the maximum value in the last row of the abundance matrix exceeds a threshold.
- `has_converged`: Checks if the relative change in the last two rows of the abundance matrix is below a threshold.
- `richness`: Counts the number of species with abundance above a specified minimum value.
- `make_D`: Creates a random transfer matrix D for species and consumer-resource interactions.

Author: Snorre Sulheim
Date: August 2025
email: snorre.sulheim@unil.ch
"""
DEFAULT_D_dist = st.uniform(0, 1)

def numerical_error(N):
    return N[-1, :].max() > 1e4

def has_converged(N, tol = 1e-6):
    # return np.abs((N[-1,:]-N[-10, :])/N[-1, :]).max() < tol
    return np.abs(N[-1,:]-N[-2, :]).max() < tol

def richness(N, min_value = 1e-4):
    return np.sum(N[-1, :]>min_value)

def make_D(n_species, n_cs, D_dist = None):
    """
    Create a random D matrix with the specified number of species and CS.
    D is a 3D array where D[i, j, k] represents the transfer rate from species i in CS j to CS k.
    """
    if n_cs == 1:
        D = np.zeros((n_species, n_cs, n_cs))
        for i in range(n_species):
            D[i, 0, 0] = 1
    else:
        # Generate a random transfer matrix
        # Ensure that the diagonal elements are zero (no self-transfer)
        # and that each row sums to 1.
        if not D_dist:
            D_dist = DEFAULT_D_dist
        D = D_dist.rvs(size=(n_species, n_cs, n_cs))
        for i in range(n_cs):
            D[:, i, i] = 0  # No self-transfer
        for i in range(n_species):
            D[i, :, :] /= D[i, :, :].sum(axis=0)  # Normalize each species' transfers
        D = np.transpose(D, (0, 2, 1))  # Transpose to match the expected shape (N_species, N_cs, N_cs)
    return D

def make_C(n_species, n_cs, C_dist, C_sparsity, g_yield = 0.1, min_ratio=2, 
           initial_cs_index = 0, dilution_rate = 0.1):
    def fun():
        C = C_dist.rvs(size=(n_species, n_cs))
        C0 = (np.random.uniform(0,1, size = (n_species, n_cs))>C_sparsity).astype(int)
        C = C * C0
        return C

    C = fun()
    while C[:, initial_cs_index].max()*g_yield < min_ratio * dilution_rate:
        C = fun()
    return C