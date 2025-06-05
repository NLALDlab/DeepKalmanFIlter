'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/UpdateWeights.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

def UpdateWeights(NetWeights, Grads, Moment1, Moment2, Dynamic, Iterate, GainMask, NetParameters):
    """
    Updates the network's weights using the specified optimizer.

    Parameters:
        NetWeights (list of numpy.ndarray): List containing the network weights.
        Grads (list of numpy.ndarray): List containing the gradients for each weight matrix.
        Moment1 (list of numpy.ndarray): List containing the first moment estimates for Adam optimizer.
        Moment2 (list of numpy.ndarray): List containing the second moment estimates for Adam optimizer.
        Dynamic (int): Index for dynamic parameters.
        Iterate (int): Current iteration number.
        GainMask (numpy.ndarray): Mask for gain updates.
        NetParameters (dict): Dictionary containing network parameters including optimizer settings.

    Returns:
        NetWeights (list of numpy.ndarray): Updated network weights.
        Moment1 (list of numpy.ndarray): Updated first moment estimates.
        Moment2 (list of numpy.ndarray): Updated second moment estimates.
    """
    Layers = NetParameters['Layers']
    SharedWeights = NetParameters['SharedWeights']
    ProjectDynamics = NetParameters['ProjectDynamics']
    GainLearningRate = NetParameters['GainLearningRate']
    DynamicsLearningRate = NetParameters['DynamicsLearningRate']
    Optimizer = NetParameters['Optimizer']
    Epsilon = NetParameters['AdamEpsilon']
    
    if Optimizer == 'SGD':
        # No modification needed for SGD, use Grads as-is.
        pass

    elif Optimizer == 'Adam':
        Beta1 = NetParameters['BetaMoment1']
        Beta2 = NetParameters['BetaMoment2']
        
        if SharedWeights == 'No':
            for Layer in range(Layers + 1):
                if Layer < Layers:
                    # Kalman Gains
                    Moment1[Layer] = Beta1*Moment1[Layer] + (1 - Beta1)*Grads[Layer]
                    Moment2[Layer] = Beta2*Moment2[Layer]+ (1 - Beta2)*(Grads[Layer] ** 2)

                    Moment1Hat = Moment1[Layer] / (1 - Beta1 ** Iterate)
                    Moment2Hat = Moment2[Layer] / (1 - Beta2 ** Iterate)
                    Grads[Layer] = Moment1Hat / (np.sqrt(Moment2Hat) + Epsilon)
                else:
                    # Dynamic        
                    Moment1[Layer][Dynamic-1] = Beta1 * Moment1[Layer][Dynamic-1] + (1 - Beta1) * Grads[Layer][Dynamic-1]
                    Moment2[Layer][Dynamic-1] = Beta2 * Moment2[Layer][Dynamic-1] + (1 - Beta2) * (Grads[Layer][Dynamic-1] ** 2)
    
                    Moment1Hat = Moment1[Layer][Dynamic-1] / (1 - Beta1 ** Iterate)
                    Moment2Hat = Moment2[Layer][Dynamic-1] / (1 - Beta2 ** Iterate)

                    Grads[Layer][Dynamic-1] = Moment1Hat / (np.sqrt(Moment2Hat) + Epsilon)
                #endif
            #endfor
        #endif
    #endif
    # Update weights
    if SharedWeights == 'No':
        for Layer in range(Layers + 1):
            if Layer < Layers:
                # Kalman Gains
                #print("GainLearningRate = ",GainLearningRate)
                #print("GainMask = ",GainMask)
                NetWeights[Layer] -= GainLearningRate * GainMask * Grads[Layer]
            elif NetParameters['HiddenDynamicsDimension'][0] > 0:
                # Dynamics
                #print("len(Grads[Layer]) = ",len(Grads[Layer])," , Dynamic = ",Dynamic)
                #print("DynamicsLearningRate = ",DynamicsLearningRate," , Grads[Layer=",Layer,"][Dynamic-1=",Dynamic-1,"] = ",Grads[Layer][Dynamic-1])
                NetWeights[Layer][Dynamic-1] -= DynamicsLearningRate * Grads[Layer][Dynamic-1]
                #print("NetWeights[Layer][Dynamic-1] = ",NetWeights[Layer][Dynamic-1])

                if ProjectDynamics == 'Yes':
                    # Project dynamics vector
                    NetWeights[Layer][Dynamic-1] = np.abs(NetWeights[Layer][Dynamic-1])
                #endif
            #endif
        #endfor
    #endif

    return NetWeights, Moment1, Moment2
