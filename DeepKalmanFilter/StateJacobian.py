'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/StateJacobian.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

def StateJacobian(F, x, p, u, s, Fxpu, Layer, N, NetParameters):
    """
    Computes the Jacobian matrix for F with respect to the x variables at the point (x, p, u).
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
        NetParameters (dict): Dictionary containing network parameters.
    
    Returns:
        StateJac (numpy.ndarray): Jacobian matrix of F with respect to x.
    """
    FiniteDifferences = NetParameters['FiniteDifferences']
    h = NetParameters['FiniteDifferencesSkip']
    
    StateJac = np.atleast_2d(np.zeros((N, N)))
    
    # Cycle over columns of Jacobian
    for ColInd in range(N):
        # Increment in ColInd-th cardinal direction
        Increment = np.zeros((N,1))
        Increment[ColInd] = h
        
        if FiniteDifferences == 'Forward':
            StateJac[:,ColInd:ColInd+1] = (F(x + Increment, p, u, s, Layer, NetParameters)[0] - Fxpu) / h
        
        elif FiniteDifferences == 'Backward':
            StateJac[:,ColInd:ColInd+1] = (Fxpu - F(x - Increment, p, u, s, Layer, NetParameters)[0]) / h
        
        elif FiniteDifferences == 'Central':
            StateJac[:,ColInd:ColInd+1] = (F(x + Increment, p, u, s, Layer, NetParameters)[0] - F(x - Increment, p, u, s, Layer, NetParameters)[0]) / (2 * h)
        #endif
    #endfor
    return StateJac
