'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/ComputeWeightMats.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np
import scipy.io as sio

def ComputeWeightMats(Directory,NetParameters):
    """
    Computes the weight matrices used for residue scaling.

    Parameters:
        NetParameters (dict): Dictionary containing network parameters.

    Returns:
        MeasurementWeightMats (list of numpy.ndarray): Measurement weight matrices for each layer.
        PredictorWeightMats (list of numpy.ndarray): Predictor weight matrices for each layer.
        MeasurementWeightMatsSym (list of numpy.ndarray, optional): Symmetric measurement weight matrices.
        PredictorWeightMatsSym (list of numpy.ndarray, optional): Symmetric predictor weight matrices.
    """
    Layers = NetParameters['Layers']
    StateDimension = NetParameters['StateDimension']
    ObservationDimension = NetParameters['ObservationDimension']
    WeightMats = NetParameters['WeightMats']

    MeasurementWeightMats = [None] * Layers
    PredictorWeightMats = [None] * Layers

    # Cleaner propagation for test phase
    MeasurementWeightMatsSym = [None] * Layers
    PredictorWeightMatsSym = [None] * Layers

    # Compute weight matrices
    if WeightMats == 'Identity':
        for Layer in range(Layers):
            MeasurementWeightMats[Layer] = np.eye(ObservationDimension)
            PredictorWeightMats[Layer] = np.eye(StateDimension)

            MeasurementWeightMatsSym[Layer] = np.eye(ObservationDimension)
            PredictorWeightMatsSym[Layer] = np.eye(StateDimension)

    elif WeightMats == 'Input':
        Experiment = NetParameters['Experiment']

        MeasurementWeightMats = sio.loadmat(Directory+f'MeasurementWeightMatsExp{Experiment}.mat',squeeze_me = False)['MeasurementWeightMats']
        PredictorWeightMats = sio.loadmat(Directory+f'PredictorWeightMatsExp{Experiment}.mat',squeeze_me = True)['PredictorWeightMats']

        for Layer in range(Layers):
            MeasurementWeightMatsSym[Layer] = 0.5 * (MeasurementWeightMats[Layer] + MeasurementWeightMats[Layer].T)
            PredictorWeightMatsSym[Layer] = 0.5 * (PredictorWeightMats[Layer] + PredictorWeightMats[Layer].T)

    return MeasurementWeightMats, PredictorWeightMats, MeasurementWeightMatsSym, PredictorWeightMatsSym
