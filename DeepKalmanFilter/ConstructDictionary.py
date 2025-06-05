'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/ConstructDictionary.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

def ConstructDictionary(X, NetParameters):
    """
    Constructs the dictionary for model discovery.
    """

    # Variables
    AllowedDictionaryBlocks = NetParameters['AllowedDictionaryBlocks']
    DictionaryBlocks = NetParameters['DictionaryBlocks']

    StateDimension, TimeInd = np.shape(X)
    Phi = np.empty((TimeInd, 0))  # Initialize an empty array

    if 'Constant' in DictionaryBlocks:
        ConstantBlock = np.ones((TimeInd, 1))
        Phi = np.hstack((Phi, ConstantBlock))

    if 'Linear' in DictionaryBlocks:
        LinearBlock = X.T
        Phi = np.hstack((Phi, LinearBlock))

    if 'Quadratic' in DictionaryBlocks:
        Counter = 0
        QuadraticBlock = np.zeros((TimeInd, AllowedDictionaryBlocks['Quadratic']))
        
        for i in range(StateDimension):
            for j in range(i, StateDimension):
                QuadraticBlock[:, Counter] = X[i, :] * X[j, :]
                Counter += 1

        Phi = np.hstack((Phi, QuadraticBlock))

    if 'Cubic' in DictionaryBlocks:
        Counter = 0
        CubicBlock = np.zeros((TimeInd, AllowedDictionaryBlocks['Cubic']))

        for i in range(StateDimension):
            for j in range(i, StateDimension):
                for k in range(j, StateDimension):
                    CubicBlock[:, Counter] = X[i, :] * X[j, :] * X[k, :]
                    Counter += 1

        Phi = np.hstack((Phi, CubicBlock))

    return Phi
