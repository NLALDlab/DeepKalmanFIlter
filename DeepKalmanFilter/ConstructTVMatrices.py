'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/ConstructTVMatrices.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

def ConstructTVMatrices(N, SamplingTimes):
    """
    Construct matrices used during TV regularization.
    """
    # Variables
    TimeStep = SamplingTimes[0]  # Assuming constant sampling

    # Set up matrix D
    vD = np.zeros(N-1)
    vD[-1] = 1
    MD = np.diag(-np.ones(N-1)) + np.diag(np.ones(N-2), 1)
    D = (2 / TimeStep) * np.column_stack((MD, vD))

    # Set up matrix A
    vA = np.ones(N-1)
    vA[0] = 3 / 4
    CoeffVecA = [1/4, 7/4] + [2] * (N-3)
    MA = np.zeros((N-1, N-1))
    for DiagInd in range(N-1):
        if DiagInd < len(CoeffVecA):
            MA += np.diag(CoeffVecA[DiagInd] * np.ones(N-1-DiagInd), -DiagInd)
    A = (TimeStep / 2) * np.column_stack((vA, MA))
    AtA = A.T @ A

    # Set up matrix B
    vB = np.ones(N-1)
    CoeffVecB = [1] + [2] * (N-2)
    MB = np.zeros((N-1, N-1))
    for DiagInd in range(N-1):
        if DiagInd < len(CoeffVecB):
            MB += np.diag(CoeffVecB[DiagInd] * np.ones(N-1-DiagInd), -DiagInd)
    B = (TimeStep / 2) * np.column_stack((vB, MB))

    return A, D, AtA, B
