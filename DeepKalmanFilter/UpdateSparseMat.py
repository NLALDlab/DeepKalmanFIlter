'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/UpdateSparseMat.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np
from scipy.linalg import svd
from scipy.optimize import nnls

from DeepKalmanFilter.ComputeSGDerivative import *
from DeepKalmanFilter.ComputeTVDerivatives import *
from DeepKalmanFilter.ConstructDictionary import *
from DeepKalmanFilter.OMP import *

def UpdateSparseMat(NetWeights, States, ModelDiscoverySupport, Dynamic, NetParameters):
    """
    Updates the sparse matrix for model discovery.

    Parameters:
        NetWeights (list of np.ndarray): List containing the network weights.
        States (list of np.ndarray): List containing the states.
        ModelDiscoverySupport (np.ndarray): Matrix indicating model discovery support.
        Dynamic (int): Index for dynamic parameters.
        NetParameters (dict): Dictionary containing network parameters.

    Returns:
        SparseMat (np.ndarray): Updated sparse matrix.
    """
    Experiment = NetParameters['Experiment']
    Layers = NetParameters['Layers']
    StateDimension = NetParameters['StateDimension']
    Model = NetParameters['Model']
    FirstStateOffset = NetParameters['ModelDiscoveryFirstState']
    ModelDiscoveryMethod = NetParameters['ModelDiscoveryMethod']
    ModelDiscoverySmoothing = NetParameters['ModelDiscoverySmoothing']
    SharedWeights = NetParameters['SharedWeights']

    # Assemble X matrix & create weight vector
    X = np.zeros((StateDimension, Layers-FirstStateOffset))
    WeightVec = np.ones((Layers-FirstStateOffset,1))
    for Layer in range(Layers-FirstStateOffset):
        X[:,Layer:Layer+1] = States[Layer+FirstStateOffset]

        if SharedWeights == 'Yes':
            Indx = 0
        else:
            Indx = Layer+FirstStateOffset
        Sigma = svd(NetWeights[Indx], full_matrices=False)[1][0]
        # WeightVec[Layer] = 1/Sigma

    # Compute X' matrix and possibly a smoothed version of X
    if ModelDiscoverySmoothing == 'TV':
        A = NetParameters['A']
        D = NetParameters['D']
        AtA = NetParameters['AtA']
        B = NetParameters['B']
        XPrimeTarget, XTarget = ComputeTVDerivative(X, A, D, AtA, B, Model['SamplingTimes'][0])
    
    elif ModelDiscoverySmoothing == 'TVMixed':
        A = NetParameters['A']
        D = NetParameters['D']
        AtA = NetParameters['AtA']
        B = NetParameters['B']
        XTarget = ComputeTVDerivative(X, A, D, AtA, B, Model['SamplingTimes'][0])[1]
        XPrimeTarget = (XTarget[:,2:] - XTarget[:,:-2]) / (2 * Model['SamplingTimes'][0])
        XTarget = XTarget[:,1:-1]
        WeightVec = WeightVec[1:-1]

    elif ModelDiscoverySmoothing == 'SG':
        StencilA0 = NetParameters['StencilA0']
        StencilA1 = NetParameters['StencilA1']
        WinLen = NetParameters['WinLen']
        XPrimeTarget, XTarget = ComputeSGDerivative(X, StencilA0, StencilA1, WinLen, Model['SamplingTimes'][0])

    elif ModelDiscoverySmoothing == 'SGMixed1':
        StencilA0 = NetParameters['StencilA0']
        StencilA1 = NetParameters['StencilA1']
        WinLen = NetParameters['WinLen']
        XTarget = ComputeSGDerivative(X, StencilA0, StencilA1, WinLen, Model['SamplingTimes'][0])[1]
        XPrimeTarget, XTarget = ComputeSGDerivative(XTarget, StencilA0, StencilA1, WinLen, Model['SamplingTimes'][0])

    elif ModelDiscoverySmoothing == 'SGMixed2':
        StencilA0 = NetParameters['StencilA0']
        StencilA1 = NetParameters['StencilA1']
        WinLen = NetParameters['WinLen']
        XTarget = ComputeSGDerivative(X, StencilA0, StencilA1, WinLen, Model['SamplingTimes'][0])[1]
        XPrimeTarget = (XTarget[:,2:] - XTarget[:,:-2]) / (2 * Model['SamplingTimes'][0])
        XTarget = XTarget[:,1:-1]
        WeightVec = WeightVec[1:-1]

    elif ModelDiscoverySmoothing == 'No':
        XPrimeTarget = (X[:,2:] - X[:,:-2]) / (2 * Model['SamplingTimes'][0])
        XTarget = X[:,1:-1]
        WeightVec = WeightVec[1:-1]

    ColNum = np.shape(XPrimeTarget)[1]

    # Assemble target matrix
    if Experiment == '1':
        Target = (Model['M'] @ XPrimeTarget - Model['K'] @ XTarget - Model['D'][:, :ColNum]).T
    elif Experiment == '2':
        Target = (Model['M'] @ XPrimeTarget - Model['K'] @ XTarget - Model['D'][:, :ColNum]).T
    elif Experiment == '3':
        Target = (Model['M'] @ XPrimeTarget - Model['K'] @ XTarget - np.vstack([np.zeros((1, ColNum)), -XTarget[0, :] * XTarget[2, :], XTarget[0, :] * XTarget[1, :]]) - Model['D'][:, :ColNum]).T
    elif Experiment == 'L63_3' or Experiment == 'L63_4':
        Target = ( ( Model['M'] + np.diag([np.squeeze(NetWeights[-1][Dynamic-1])] + [0]*(StateDimension-1)) )@XPrimeTarget - Model['K']@XTarget - np.vstack([np.zeros((1, ColNum)), -XTarget[0,:]*XTarget[2,:], XTarget[0,:]*XTarget[1,:]]) - Model['D'][:,:ColNum] ).T
    elif Experiment == '5':
        Target = (Model['M'] @ XPrimeTarget - (Model['K'] + np.array([[0, 0, 0],[NetWeights[-1][Dynamic-1], 0, 0], [0, 0, 0]])) @ XTarget - np.vstack([np.zeros((1, ColNum)), -XTarget[0,:] * XTarget[2,:], XTarget[0,:] * XTarget[1,:]]) - Model['D'][:,:ColNum]).T
    elif Experiment == '6':
        Target = (Model['M'] @ XPrimeTarget - Model['K'] @ XTarget - Model['D'][:, :ColNum]).T
    elif Experiment == '7':
        Target = (Model['M'] @ XPrimeTarget - Model['K'] @ XTarget - Model['D'][:, :ColNum]).T
    elif Experiment == '3MKC2':
        Target = (Model['M'] @ XPrimeTarget - Model['K'] @ XTarget - Model['D'][:, :ColNum]).T
    #endif

    # Apply weights
    Target = Target*WeightVec[:ColNum]

    # Construct dictionary & normalize it
    Phi = WeightVec[:ColNum]*ConstructDictionary(XTarget, NetParameters)
    Norms = np.linalg.norm(Phi, axis=0)
    Phi = Phi / Norms

    # Update the sparse matrix using the chosen method
    SparseMat = np.zeros_like(NetWeights[-1][-1])
    
    if ModelDiscoveryMethod == 'OMP':
        OMPSparsity = NetParameters['OMPSparsity']
        ModelDiscoveryRelativeThreshold = NetParameters['ModelDiscoveryRelativeThreshold']
        
        for State in range(StateDimension):
            if np.any(ModelDiscoverySupport[:,State]):
                Temp = OMP(OMPSparsity, Target[:,State:State+1], Phi[:,ModelDiscoverySupport[:,State]])
                
                # Pruning
                RelRes = np.linalg.norm( Phi[:,ModelDiscoverySupport[:,State]]@Temp - Target[:,State] ) / np.linalg.norm(Target[:,State])
                ContinuePruning = True

                while ContinuePruning:
                    Support = np.nonzero(Temp)[0]
                    PruneRelRes = np.inf
                    
                    for SuppIndx in Support:
                        TempVal = Temp.copy()
                        TempVal[SuppIndx] = 0
                        RelResVal = np.linalg.norm(Phi[:,ModelDiscoverySupport[:,State]] @ TempVal - Target[:,State]) / np.linalg.norm(Target[:,State])
                        
                        if RelResVal < PruneRelRes:
                            PruneRelRes = RelResVal
                            PruneIndx = SuppIndx
                    
                    if PruneRelRes < (1 + ModelDiscoveryRelativeThreshold) * RelRes:
                        Temp[PruneIndx] = 0
                    else:
                        ContinuePruning = False

                SparseMat[ModelDiscoverySupport[:, State], State] = Temp

    elif ModelDiscoveryMethod == 'LH':
        ModelDiscoveryRelativeThreshold = NetParameters['ModelDiscoveryRelativeThreshold']
        
        for State in range(StateDimension):
            if np.any(ModelDiscoverySupport[:, State]):
                TempPos = nnls(Phi[:,ModelDiscoverySupport[:,State]], Target[:,State])[0]
                TempNeg = nnls(-Phi[:,ModelDiscoverySupport[:,State]], Target[:,State])[0]
                Temp = TempPos - TempNeg
                
                # Pruning
                RelRes = np.linalg.norm(Phi[:,ModelDiscoverySupport[:,State]] @ Temp - Target[:,State]) / np.linalg.norm(Target[:,State])
                ContinuePruning = True

                while ContinuePruning:
                    Support = np.nonzero(Temp)[0]
                    PruneRelRes = np.inf
                    
                    for SuppIndx in Support:
                        TempVal = Temp.copy()
                        TempVal[SuppIndx] = 0
                        RelResVal = np.linalg.norm(Phi[:,ModelDiscoverySupport[:,State]] @ TempVal - Target[:,State]) / np.linalg.norm(Target[:,State])
                        
                        if RelResVal < PruneRelRes:
                            PruneRelRes = RelResVal
                            PruneIndx = SuppIndx
                    
                    if PruneRelRes < (1 + ModelDiscoveryRelativeThreshold) * RelRes:
                        Temp[PruneIndx] = 0
                    else:
                        ContinuePruning = False

                SparseMat[ModelDiscoverySupport[:,State], State] = Temp

    # Recover sparse coefficients
    SparseMat = SparseMat / Norms[:,np.newaxis]
    
    return SparseMat
