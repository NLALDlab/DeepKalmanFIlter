'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/TVDifferentiate2.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

def TVDifferentiate2(h, alpha, B, D, AtA, Atf, uEst, maxit):
    """
    Computes u = f' using TV normalization. f must be a column vector.
    """
    epsilon = 1e-8

    for _ in range(maxit):
        # Compute the L matrix
        DuEst = D @ uEst
        L = alpha * h * D.T @ np.diag(1. / (np.sqrt((0.5 * DuEst) ** 2 + epsilon))) @ D
        
        # Compute the Hessian and gradient
        H = L + AtA
        g = -(AtA @ uEst - Atf + L @ uEst)
        
        # Update uEst
        uEst += np.linalg.solve(H, g)
    
    # Return denoised f
    fEst = B @ uEst
    return uEst, fEst