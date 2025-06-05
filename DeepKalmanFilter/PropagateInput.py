'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/PropagateInput.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np
from matplotlib import pyplot as plt
import scipy

def PropagateInput(Inputs, Measurements, FirstState, Dynamic, F, NetWeights, NetParameters):
    """
    Propagates the Inputs vector (u) and Measurements vector (y) through the network. 
    They are lists of size (1, NetParameters['Layers']).
    F is the 'VARMION' function block. The output is the States vector (x),
    a list of size (1, NetParameters['Layers'] + 1). States[0] is given as an input. 
    Additional outputs MeasurementMinusCFs, GainMeasurementMinusCFs, MeasurementMinusCStates, 
    and FStateDynInputs are saved for later efficiency during backpropagation.
    """

    Layers = NetParameters['Layers']
    C = NetParameters['C']
    ny = NetParameters['C'].shape[0]
    SharedWeights = NetParameters['SharedWeights']
    xrMask = NetParameters['Model']['xrMask']
    Model = NetParameters['Model']
    if NetParameters['Experiment'][0:4] =='HSE_':
        internal_mesh_nodes = Model['internal_mesh_nodes']
        internal_mesh_nodes_slice0 = Model['internal_mesh_nodes_slice0']
        islice0_in_internal_mesh_nodes = Model['islice0_in_internal_mesh_nodes']
        map_gp2meshnode = Model['map_gp2meshnode']
    #endif
    
    # Setup output
    States = [None] * (Layers + 1)
    MeasurementMinusCStates = [None] * Layers
    GainMeasurementMinusCFs = [None] * Layers

    MeasurementMinusCFs = [None] * Layers
    FStateDynInputs = [None] * Layers

    # Initialize the first state
    States[0] = FirstState[xrMask]

    # Propagate through layers
    for Layer in range(Layers):
        if SharedWeights == 'Yes':
            Indx = 0 
        else:
            Indx = Layer
        #endif
        # Compute FStateDynInput using the provided function F
        if NetParameters['Experiment'][0:4] =='HSE_' and NetParameters['estimate_forcing_term']:
            max_principle_ok = False
            while not max_principle_ok:
                #print("max_principle: Layer = ",Layer)
                tmpprederr  = Measurements[:,Layer:Layer+1] - C@States[Layer]
                
                if 0:
                    Ct, Theta, Fterm, h, G = Model['HeatParameters']                    
                    tmpLayer = Layer; 
                    #if Layer < (Layers-1): tmpLayer = Layer+1
                    #print("nnz Fterm = ",len(np.where(Fterm[tmpLayer,:] > 0.95)[0]))
                    NetParameters['f_est'][map_gp2meshnode,Layer] = Fterm[tmpLayer,:]
                    #print("nnz f_est = ",len(np.where(NetParameters['f_est'][:,Layer] > 0.95)[0]))
                elif NetParameters['zero_est']:
                    NetParameters['f_est'][internal_mesh_nodes,Layer] = np.zeros(len(internal_mesh_nodes))
                else:
                    if 0:
                        islot = NetParameters['islot']
                        vim = NetParameters['ff_gain'] * np.linalg.pinv(Model['Mir'][ny*islot:ny*(islot+1),:]) @ tmpprederr
                        if Layer == 0:
                            NetParameters['f_est'][internal_mesh_nodes,Layer] = np.squeeze(vim)
                        else:
                            if 0:
                                NetParameters['f_est'][internal_mesh_nodes,Layer] = NetParameters['f_est'][internal_mesh_nodes,Layer-1] + np.squeeze(vim)
                            else:
                                NetParameters['f_est'][internal_mesh_nodes,Layer] = np.squeeze(vim)
                                #NetParameters['f_est'][96,Layer] = 1.0
                            #endif
                        #endif
                    else:
                        islot = Layer #NetParameters['islot']
                        limit_to_slice0 = False
                        if limit_to_slice0:
                            tmpMope = Model['Mir'][ny*islot:ny*(islot+1),islice0_in_internal_mesh_nodes]
                        else:
                            tmpMope = Model['Mir'][ny*islot:ny*(islot+1),:]
                        #endif
                        tmprhs = tmpprederr.copy()
                        if NetParameters['do_NNLS'] == 0:
                            #tmpsol = np.linalg.solve(tmpMope, tmprhs)
                            tmpsol = NetParameters['ff_gain'] * np.linalg.pinv(tmpMope) @ tmprhs
                            tmpI = np.where(tmpsol < 0)[0]
                            tmpsol[tmpI] = 0.0
                            tmpI = np.where(tmpsol > 1.0)[0]
                            #tmpsol[tmpI] = 1.0
                        else:
                            if 1:
                                tmpsol = scipy.optimize.nnls(tmpMope, np.squeeze(tmprhs), maxiter=None)[0]
                            else:
                                tmpsol = scipy.optimize.nnls(np.hstack([tmpMope, -tmpMope]), np.squeeze(tmprhs), maxiter=None)[0]
                            #endif
                        #endif
                        if (NetParameters['f_est'][internal_mesh_nodes_slice0,Layer] == 0).all():
                            if limit_to_slice0:
                                NetParameters['f_est'][internal_mesh_nodes_slice0,Layer] = np.squeeze(tmpsol)
                            else:
                                NetParameters['f_est'][internal_mesh_nodes,Layer] = np.squeeze(tmpsol)
                            #endif
                            print("STIMO LA FORZANTE! Layer = ",Layer); plt.pause(0.3)
                        #endif
                    #endif
                #endif

                FStateDynInput,NetWeights[-1][Dynamic-1] = F(States[Layer], NetWeights[-1][Dynamic-1], Inputs[:,Layer:Layer+1], NetWeights[-1][-1], Layer, NetParameters)
                #print("Layer+1=",Layer+1," , FStateDynInput = ",FStateDynInput)

                # Calculate MeasurementMinusCF and GainMeasurementMinusCF
                MeasurementMinusCF = Measurements[:,Layer+1:Layer+2] - C @ FStateDynInput
            
                max_principle_ok = True
            #endwhile
        else:
            FStateDynInput,NetWeights[-1][Dynamic-1] = F(States[Layer], NetWeights[-1][Dynamic-1], Inputs[:,Layer:Layer+1], NetWeights[-1][-1], Layer, NetParameters)
            #print("Layer+1=",Layer+1," , FStateDynInput = ",FStateDynInput)

            # Calculate MeasurementMinusCF and GainMeasurementMinusCF
            MeasurementMinusCF = Measurements[:,Layer+1:Layer+2] - C @ FStateDynInput
        #endif
        #print("NetWeights[",Indx,"].shape = ",NetWeights[Indx].shape)
        GainMeasurementMinusCF = NetWeights[Indx]@MeasurementMinusCF

        # Save outputs
        States[Layer + 1] = FStateDynInput + GainMeasurementMinusCF
        MeasurementMinusCStates[Layer] = Measurements[:,Layer+1:Layer+2] - C@States[Layer+1]
        GainMeasurementMinusCFs[Layer] = GainMeasurementMinusCF

        MeasurementMinusCFs[Layer] = MeasurementMinusCF
        FStateDynInputs[Layer] = FStateDynInput

    return States, MeasurementMinusCStates, GainMeasurementMinusCFs, MeasurementMinusCFs, FStateDynInputs
