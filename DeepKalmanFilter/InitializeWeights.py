'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/InitializeWeights.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

from DeepKalmanFilter.InitializeSparseDynamicsMat import *
from DeepKalmanFilter.InitializeDynamics import *

def InitializeWeights(NetParameters):
    """
    Initializes the net's weights with Gaussian noise of mean NetParameters.InitializationMean 
    and sigma NetParameters.InitializationSigma.
    """
    # Variables
    Experiment = NetParameters['Experiment']
    Layers = NetParameters['Layers']
    SharedWeights = NetParameters['SharedWeights']
    Initialization = NetParameters['Initialization']
    StateDimension = NetParameters['StateDimension']
    ObservationDimension = NetParameters['ObservationDimension']
    HiddenDynamicsNumber = NetParameters['HiddenDynamicsNumber']
    HiddenDynamicsDimension = NetParameters['HiddenDynamicsDimension']
    if NetParameters['ActivateModelDiscovery'] == 'Yes':
        DictionaryDimension = NetParameters['DictionaryDimension']
    #endif
    
    C = NetParameters['C']
    Model = NetParameters['Model']
    
    if SharedWeights == 'No':
        NetWeights = [None] * (Layers + 1)

        if Initialization == 'Deterministic':
            # Deterministic initialization
            if Experiment[0:4]=='DLTI' or Experiment[0:4]=='3MKC' or Experiment[0:4]=='L63_' or Experiment[0:4]=='HEV_' \
                                       or Experiment[0:4]=='Roes' or Experiment[0:4]=='HSE_':
                P = Model['PInit']
                A = Model['AInit']
                if Experiment[0:4]=='Roes':
                    Q = 0. #Model['QInit']
                else:
                    Q = Model['QInit']
                #endif
                InvR = Model['invRInit']

                InvP = np.linalg.inv(A @ P @ A.T + Q) + C.T @ InvR @ C
                KFGain = np.linalg.inv(InvP) @ (C.T @ InvR)

                for Layer in range(Layers):
                    NetWeights[Layer] = np.copy(KFGain)
                #endfor
            #endif
            NetWeights[Layers] = [None] * (HiddenDynamicsNumber + 1)
            for Dyn in range(HiddenDynamicsNumber):
                NetWeights[Layers][Dyn] = InitializeDynamics(HiddenDynamicsDimension[Dyn], Model, Experiment)
            #endfor
            if NetParameters['ActivateModelDiscovery'] == 'Yes':
                NetWeights[Layers][HiddenDynamicsNumber] = InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment)
            #endif
        elif Initialization == 'DeterministicComplete':
            # DeterministcComplete initialization
            if Experiment[0:4]=='DLTI' or Experiment[0:4]=='3MKC' or Experiment[0:4]=='L63_' or Experiment[0:4]=='HEV_' \
                                       or Experiment[0:4]=='Roes' or Experiment[0:4]=='HSE_':
                P = Model['PInit']
                A = Model['AInit']
                Q = Model['Q_KF']
                InvR = Model['invRInit']

                for Layer in range(Layers):
                    InvP = np.linalg.inv(A @ P @ A.T + Q) + C.T @ InvR @ C
                    P = np.linalg.inv(InvP)
                    KFGain = np.linalg.inv(InvP) @ (C.T @ InvR)
                    NetWeights[Layer] = np.copy(KFGain)
                #endfor
            NetWeights[Layers] = [None] * (HiddenDynamicsNumber + 1)
            for Dyn in range(HiddenDynamicsNumber):
                NetWeights[Layers][Dyn] = InitializeDynamics(HiddenDynamicsDimension[Dyn], Model, Experiment)
            #endfor
            if NetParameters['ActivateModelDiscovery'] == 'Yes':
                NetWeights[Layers][HiddenDynamicsNumber] = InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment)
            #endif
        elif Initialization == 'Random':
            #Random initialization
            Mean = NetParameters['InitializationMean']
            Sigma = NetParameters['InitializationSigma']

            for Layer in range(Layers):
                NetWeights[Layer] = np.random.normal(Mean, Sigma, (ObservationDimension,ObservationDimension))
            #endfor
            NetWeights[Layers] = [None] * (HiddenDynamicsNumber + 1)
            for Dyn in range(HiddenDynamicsNumber):
                NetWeights[Layers][Dyn] = np.random.normal(Mean, Sigma, (HiddenDynamicsDimension[Dyn],1))
            #endfor
            if NetParameters['ActivateModelDiscovery'] == 'Yes':
                NetWeights[Layers][HiddenDynamicsNumber] = InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment)
            #endif
        #endif
    elif SharedWeights == 'Yes':
        NetWeights = [None] * (2)

        if (Initialization == 'Deterministic') or (Initialization == 'DeterministicComplete'):
            if Experiment[0:4]=='DLTI' or Experiment[0:4]=='3MKC' or Experiment[0:4]=='L63_' or Experiment[0:4]=='HEV_' \
                                       or Experiment[0:4]=='Roes' or Experiment[0:4]=='HSE_':
                P = Model['PInit']
                A = Model['AInit']
                Q = Model['QInit']
                InvR = Model['invRInit']

                InvP = np.linalg.inv(A @ P @ A.T + Q) + C.T @ InvR @ C
                KFGain = np.linalg.inv(InvP) @ (C.T @ InvR)

                NetWeights[0] = np.copy(KFGain)
            #endif
            NetWeights[1] = [None] * (HiddenDynamicsNumber + 1)
            for Dyn in range(HiddenDynamicsNumber):
                NetWeights[1][Dyn] = InitializeDynamics(HiddenDynamicsDimension[Dyn], Model, Experiment)
            #endfor
            if NetParameters['ActivateModelDiscovery'] == 'Yes':
                NetWeights[1][HiddenDynamicsNumber] = InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment)
            #endif
        elif Initialization == 'Random':
            #Random initialization
            Mean = NetParameters['InitializationMean']
            Sigma = NetParameters['InitializationSigma']

            NetWeights[0] = np.random.normal(Mean, Sigma, (ObservationDimension,ObservationDimension))

            NetWeights[1] = [None] * (HiddenDynamicsNumber + 1)
            for Dyn in range(HiddenDynamicsNumber):
                NetWeights[1][Dyn] = np.random.normal(Mean, Sigma, (HiddenDynamicsDimension[Dyn],1))
            #ednfor
            if NetParameters['ActivateModelDiscovery'] == 'Yes':
                NetWeights[1][HiddenDynamicsNumber] = InitializeSparseDynamicsMat(DictionaryDimension, StateDimension, Model, Experiment)
            #endif
        #endif
    return NetWeights
