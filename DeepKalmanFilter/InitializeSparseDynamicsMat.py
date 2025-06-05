'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/InitializeSparseDynamicsMat.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np
from scipy.sparse import csr_matrix

def InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment):
    """
    Initializes the sparse dynamics matrix.
    """
    if Experiment == '4':
        SparseDynMat = np.zeros((DictionaryDimension, StateDimension)).astype('float64')
    elif Experiment == '7':
        SparseDynMat = np.zeros((DictionaryDimension, StateDimension)).astype('float64')
    elif Experiment == '8':
        SparseDynMat = np.zeros((DictionaryDimension, StateDimension)).astype('float64')
    else:
        SparseDynMat = np.zeros((DictionaryDimension, StateDimension)).astype('float64')
    return SparseDynMat