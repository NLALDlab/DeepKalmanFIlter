'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/DynamicEquation.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

from DeepKalmanFilter.ConstructDictionary import *

def DynamicEquation(z, x, p, u, s, M, K, d, Ts, NetParameters):
    """
    Encodes the (implicit) equation for the dynamics to be solved forward in time.
    """
    # Variables
    Experiment = NetParameters['Experiment']

    z = np.atleast_2d(z).T

    # Dictionary for model discovery
    Phi = ConstructDictionary(z, NetParameters)
    
    if Experiment == '1':
        F = M @ (z - x) - Ts * (K @ z + d + s.T @ Phi.T)

    elif Experiment == '2':
        F = M @ (z - x) - Ts * (K @ z + d + s.T @ Phi.T)

    elif Experiment == '3':
        F = M @ (z - x) - Ts * (K @ z + np.array([0, -z[0]*z[2], z[0]*z[1]]) + d + s.T @ Phi.T)

    elif Experiment == '4':
        F = ( M + np.diag([np.squeeze(p)] + [0] * (2)) ) @ (z - x) - Ts * (K @ z + np.array( [[0], [-z[0,0]*z[2,0]], [z[0,0]*z[1,0]]] ) + d + s.T @ Phi.T)

    elif Experiment == '5':
        K[1, 0] = K[1, 0] + p
        F = M @ (z - x) - Ts * (K @ z + np.array([0, -z[0]*z[2], z[0]*z[1]]) + d + s.T @ Phi.T)

    elif Experiment == '6':
        F = M @ (z - x) - Ts * (K @ z + d + s.T @ Phi.T)

    elif Experiment == '7':
        #print("u = ",np.squeeze(u))
        #print("d = ",np.squeeze(d))
        #print("p = ",np.squeeze(p)," , M[0,0] = ",M[0,0])
        Mest = ( M + np.diag([np.squeeze(p),0,0,0,0,0]) )
        F = Mest @ (z - x) - Ts * (K @ z + u + d + s.T @ Phi.T)

    elif Experiment == '8':
        #print("u = ",np.squeeze(u))
        #print("d = ",np.squeeze(d))
        #print("p = ",np.squeeze(p)," , M[0,0] = ",M[0,0])
        Mest = M.copy()
        F = Mest @ (z - x) - Ts * (K @ z + u + d + s.T @ Phi.T)

    return np.squeeze(F)
