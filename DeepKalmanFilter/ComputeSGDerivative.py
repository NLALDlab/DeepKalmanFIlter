'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/ComputeSGDerivative.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

def ComputeSGDerivative(X, StencilA0, StencilA1, WinLen, TimeStep):
    """
    Computes the matrix X' = XPrime and XSmooth using SG filter.
    """
    # Variables 
    HalfWinLen = (WinLen - 1) // 2

    XPrime = np.zeros_like(X)
    XSmooth = np.zeros_like(X)

    States = X.shape[0]

    for State in range(States):
        # Select current state time series
        CurrState = X[State, :]

        # Extend it for polynomial fits at the boundaries
        CurrStateExt = np.concatenate([
            -np.flip(CurrState[1:HalfWinLen+1]) + 2*CurrState[0],
            CurrState,
            -np.flip(CurrState[-HalfWinLen-1:-1]) + 2*CurrState[-1]
        ])
        #print("CurrStateExt = ",CurrStateExt)
        #print("StencilA1 = ", StencilA1)
        # Compute coefficients of fitted polynomials (only first 2 are needed)
        A0 = np.convolve(CurrStateExt, StencilA0, mode='valid')
        A1 = np.convolve(CurrStateExt, StencilA1, mode='valid')

        XPrime[State,:] = A1 / TimeStep
        XSmooth[State,:] = A0

    return XPrime, XSmooth
