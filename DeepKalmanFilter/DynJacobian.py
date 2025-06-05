'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/DynJacobian.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

def DynJacobian(F, x, p, u, s, Fxpu, Layer, N, P, NetParameters):
    """
    Computes the Jacobian matrix for F with respect to the p variables at the point (x, p, u).
    Fxpu = F(x, p, u) is given as an input for efficiency since it was already computed.

    Parameters:
        F (function): Function to compute the Jacobian of.
        x (numpy.ndarray): State vector.
        p (numpy.ndarray): Parameter vector.
        u (numpy.ndarray): Input vector.
        s (numpy.ndarray): Sparse matrix (or other needed matrices).
        Fxpu (numpy.ndarray): Function value at (x, p, u).
        Layer (int): Current layer index.
        N (int): Dimension of the state vector.
        P (int): Dimension of the parameter vector.
        NetParameters (dict): Dictionary containing network parameters.

    Returns:
        DynJac (numpy.ndarray): Jacobian matrix of F with respect to p.
    """
    FiniteDifferences = NetParameters['FiniteDifferences']
    h = NetParameters['FiniteDifferencesSkip']
    
    DynJac = np.zeros((N, P))
    
    # Cycle over columns of Jacobian
    for ColInd in range(P):
        # Increment in ColInd-th cardinal direction
        Increment = np.zeros((P,1))
        Increment[ColInd] = h
        
        if FiniteDifferences == 'Forward':
            DynJac[:,ColInd:ColInd+1] = (F(x, p + Increment, u, s, Layer, NetParameters)[0] - Fxpu) / h
        
        elif FiniteDifferences == 'Backward':
            DynJac[:,ColInd:ColInd+1] = (Fxpu - F(x, p - Increment, u, s, Layer, NetParameters)[0]) / h
        
        elif FiniteDifferences == 'Central':
            DynJac[:,ColInd:ColInd+1] = (F(x, p + Increment, u, s, Layer, NetParameters)[0] - F(x, p - Increment, u, s, Layer, NetParameters)[0]) / (2 * h)
    
    return DynJac
