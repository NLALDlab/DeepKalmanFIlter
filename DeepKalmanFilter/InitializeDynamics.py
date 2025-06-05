'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/InitializeDynamics.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

def InitializeDynamics(HiddenDynamicsDimension, Model, Experiment):
    """
    Initializes the dynamics parameters.
    """
    if Experiment == '4':
        Dynamic = np.zeros((HiddenDynamicsDimension, 1))
    elif Experiment[0:4] == '3MKC':
        Dynamic = np.zeros((HiddenDynamicsDimension, 1))
    elif Experiment == '8':
        Dynamic = np.ones((HiddenDynamicsDimension, 1))
    else:
        Dynamic = np.zeros((HiddenDynamicsDimension, 1))
    #endif
    return Dynamic
