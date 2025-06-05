'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/ConstructTensorizedGains.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

def ConstructTensorizedGains(NetWeights, NetParameters):
    """
    Inserts the Kalman gains into a 3-D tensor.
    """
    SharedWeights = NetParameters['SharedWeights']
    Layers = NetParameters['Layers']
    StateDimension = NetParameters['StateDimension']
    ObservationDimension = NetParameters['ObservationDimension']

    TensorizedGains = np.zeros((StateDimension, ObservationDimension, Layers))

    if SharedWeights == 'No':
        # Assemble tensor
        for Layer in range(Layers):
            TensorizedGains[:,:,Layer] = NetWeights[Layer]
    
    if SharedWeights == 'Yes':
        # Do nothing
        pass
    
    return TensorizedGains
