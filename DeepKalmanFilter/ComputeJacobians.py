'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/ComputeJacobians.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

from DeepKalmanFilter.DynJacobian import *
from DeepKalmanFilter.StateJacobian import *

def ComputeJacobians(F, States, Dyn, Inputs, SparseMat, Dynamic, FStateDynInputs, NetParameters):
    """
    Computes the Jacobians of F at the different layers of the net. StateJacobians & DynJacobians are lists of size (1, NetParameters['Layers']) where
    StateJacobians[0] = [] since it is not used during backpropagation.
    """
    # Variables
    Experiment = NetParameters['Experiment']
    Layers = NetParameters['Layers']
    Jacobians = NetParameters['Jacobians']
    N = NetParameters['StateDimension']
    P = NetParameters['HiddenDynamicsDimension']
    xrMask = NetParameters['Model']['xrMask']

    # Setup output
    StateJacobians = [None] * Layers
    DynJacobians = [None] * Layers

    if Jacobians == 'Approximated':
        # Approximate Jacobians with finite differences
        if Experiment[0:4]=='DLTI' or Experiment[0:4]=='3MKC' or Experiment[0:4]=='L63_' or Experiment[0:4]=='HEV_' \
                                   or Experiment[0:4]=='Roes' or Experiment[0:4]=='HSE_':
            for Layer in range(1, Layers):
                StateJacobians[Layer] = StateJacobian(F, States[Layer], Dyn, Inputs[:,Layer:Layer+1], SparseMat, FStateDynInputs[Layer], Layer, N, NetParameters)
            #endfor            
            for Layer in range(Layers):
                DynJacobians[Layer] = DynJacobian(F, States[Layer], Dyn, Inputs[:,Layer:Layer+1], SparseMat, FStateDynInputs[Layer], Layer, N, P[Dynamic-1], NetParameters)
            #endfor
        #endif
    elif Jacobians == 'Algebraic':
        # Set Jacobians to their exact algebraic representation, when possible
        if Experiment[0:4]=='DLTI' or Experiment[0:4]=='3MKC' or Experiment[0:4]=='L63_' or Experiment[0:4]=='HEV_' \
                                   or Experiment[0:4]=='Roes' or Experiment[0:4]=='HSE_':
            for Layer in range(1, Layers):
                # Uncomment and define StateJacobianAlgebraic function when available
                # StateJacobians[Layer] = StateJacobianAlgebraic(F, States[Layer], Dyn, Inputs[Layer], SparseMat, FStateDynInputs[Layer], Layer, N, NetParameters)
                pass
            #endfor
            for Layer in range(Layers):
                # Uncomment and define DynJacobianAlgebraic function when available
                # DynJacobians[Layer] = DynJacobianAlgebraic(F, States[Layer], Dyn, Inputs[Layer], SparseMat, FStateDynInputs[Layer], Layer, N, P[Dynamic], NetParameters)
                pass
            #endfor
        #endif
    #endif
    return StateJacobians, DynJacobians
