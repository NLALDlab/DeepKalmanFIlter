'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/BackPropagateOutput.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

def BackPropagateOutput(StateTrue, Dynamic, States, MeasurementMinusCStates, GainMeasurementMinusCFs, MeasurementMinusCFs, TensorizedGains, MeasurementWeightMatsSym, PredictorWeightMatsSym, Grads, StateJacobians, DynJacobians, NetWeights, NetParameters):
    """
    Computes the gradients of the loss function with respect to the parameters.
    The loss function is:
    
    (Penalty0/2)*||States[Layers+1] - StateTrue||^2 + 
    sum_{Layer=1,...,Layers}(Penalty1/2)*( MeasurementMinusCStates[Layer].T )*MeasurementWeightMats[Layer]*( MeasurementMinusCStates[Layer] ) + 
    sum_{Layer=1,...,Layers}(Penalty2/2)*( GainMeasurementMinusCFs[Layer].T )*PredictorWeightMats[Layer]*( GainMeasurementMinusCFs[Layer] ) +
    (Penalty3/2)*||L*TensorizedGains||^2
    """

    Layers = NetParameters['Layers']
    C = NetParameters['C']
    LtL = NetParameters['LtL']
    StateDimension = NetParameters['StateDimension']
    SharedWeights = NetParameters['SharedWeights']
    BackPropagation = NetParameters['BackPropagation']
    Penalty0 = NetParameters['Penalty0']
    Penalty1 = NetParameters['Penalty1']
    Penalty2 = NetParameters['Penalty2']
    Penalty3 = NetParameters['Penalty3']
    xrMask = NetParameters['Model']['xrMask']
    HiddenDynamicsDimension = NetParameters['HiddenDynamicsDimension'][0]

    GradsStateEps = [None]
    GradsStateF = [None] * Layers
    GradsStateG = [None] * Layers

    # Loop backward over the layers
    for Layer in range(Layers - 1, -1, -1):
        if SharedWeights == 'Yes':
            Indx = 0
        else:
            Indx = Layer
        #endif
        # Common matrix components
        CommonMat = -NetWeights[Indx] @ C
        print("Layer = ",Layer," , type(CommonMat) = ",type(CommonMat)," , type(StateJacobians[Layer]) = ",type(StateJacobians[Layer]))
        if Layer > 0:
            print("CommonMat.shape = ",CommonMat.shape," , StateJacobians[Layer].shape = ",StateJacobians[Layer].shape)
            CommonMatState = CommonMat @ StateJacobians[Layer]
        #endif
        CommonMatDyn = CommonMat @ DynJacobians[Layer]
        CommonMat = np.eye(StateDimension) + CommonMat

        # Dynamics gradient matrix
        DynMat = (CommonMat @ DynJacobians[Layer]).T

        # Gradient update matrix at current layer
        if Layer > 0:
            UpdateMat = (CommonMat @ StateJacobians[Layer]).T

        # Gradient of H with respect to NetWeights[Indx]
        Grads[Indx] += Penalty3 * np.tensordot(TensorizedGains, LtL[Layer,:], axes=([2], [0]))

        if BackPropagation == 'Complete':
            if Layer == Layers - 1:
                # Gradient of Eps with respect to state at last layer
                GradsStateEps[0] = Penalty0 * (States[-1] - StateTrue[xrMask])
            
            # Gradient of Eps with respect to NetWeights[Indx]
            Grads[Indx] += GradsStateEps[0] @ MeasurementMinusCFs[Layer].T
            # Gradient of Eps with respect to NetWeights[-1][Dynamic]
            Grads[-1][Dynamic-1] += DynMat @ GradsStateEps[0]

            # Gradient of F^Layer
            GradsStateF[Layer] = -Penalty1[Layer] * (C.T @ MeasurementWeightMatsSym[Layer] @ MeasurementMinusCStates[Layer])
            Grads[Indx] += GradsStateF[Layer] @ MeasurementMinusCFs[Layer].T
            Grads[-1][Dynamic-1] += DynMat @ GradsStateF[Layer]

            # Gradient of G^Layer
            GradsStateG[Layer] = Penalty2[Layer] * PredictorWeightMatsSym[Layer] @ GainMeasurementMinusCFs[Layer]
            Grads[Indx] += GradsStateG[Layer] @ MeasurementMinusCFs[Layer].T
            Grads[-1][Dynamic-1] += CommonMatDyn.T @ GradsStateG[Layer]

            if Layer > 0:
                # Update gradients
                GradsStateEps[0] = UpdateMat @ GradsStateEps[0]
                GradsStateF[Layer] = UpdateMat @ GradsStateF[Layer]
                GradsStateG[Layer] = CommonMatState.T @ GradsStateG[Layer]

            # Loop over past layers for gradient accumulation
            for PastLayer in range(Layer + 1, Layers):
                Grads[Indx] += GradsStateF[PastLayer] @ MeasurementMinusCFs[Layer].T
                Grads[-1][Dynamic-1] += DynMat @ GradsStateF[PastLayer]
                Grads[Indx] += GradsStateG[PastLayer] @ MeasurementMinusCFs[Layer].T
                Grads[-1][Dynamic-1] += DynMat @ GradsStateG[PastLayer]

                if Layer > 0:
                    GradsStateF[PastLayer] = UpdateMat @ GradsStateF[PastLayer]
                    GradsStateG[PastLayer] = UpdateMat @ GradsStateG[PastLayer]

        elif BackPropagation == 'Truncated':
            #print("Grads[",Indx,"].shape = ",Grads[Indx].shape)
            if Layer == Layers - 1:
                GradsStateEps[0] = Penalty0 * (States[-1] - StateTrue[xrMask])
                Grads[Indx] += GradsStateEps[0] @ MeasurementMinusCFs[-1].T
            #endif
            if HiddenDynamicsDimension > 0: Grads[-1][Dynamic-1] += DynMat @ GradsStateEps[0]
            GradsStateF[Layer] = -Penalty1[Layer] * (C.T @ MeasurementWeightMatsSym[Layer] @ MeasurementMinusCStates[Layer])
            Grads[Indx] += GradsStateF[Layer] @ MeasurementMinusCFs[Layer].T
            if HiddenDynamicsDimension > 0: Grads[-1][Dynamic-1] += DynMat @ GradsStateF[Layer]

            GradsStateG[Layer] = Penalty2[Layer] * PredictorWeightMatsSym[Layer] @ GainMeasurementMinusCFs[Layer]
            Grads[Indx] += GradsStateG[Layer] @ MeasurementMinusCFs[Layer].T
            if HiddenDynamicsDimension > 0: Grads[-1][Dynamic-1] += CommonMatDyn.T @ GradsStateG[Layer]

            if Layer > 0:
                # Update gradients with respect to the states
                GradsStateEps[0] = UpdateMat @ GradsStateEps[0]
                GradsStateF[Layer] = UpdateMat @ GradsStateF[Layer]
                GradsStateG[Layer] = CommonMatState.T @ GradsStateG[Layer]
            #endif
            # Loop over past layers and update gradients
            for PastLayer in range(Layer + 1, Layers):
                if HiddenDynamicsDimension > 0: Grads[-1][Dynamic-1] += DynMat @ GradsStateF[PastLayer]
                if HiddenDynamicsDimension > 0: Grads[-1][Dynamic-1] += DynMat @ GradsStateG[PastLayer]

                if Layer > 0:
                    GradsStateF[PastLayer] = UpdateMat @ GradsStateF[PastLayer]
                    GradsStateG[PastLayer] = UpdateMat @ GradsStateG[PastLayer]
                #endif
            #endfor
        #endif
        #print("end: Grads[",Indx,"].shape = ",Grads[Indx].shape)
    #endfor
    return Grads
