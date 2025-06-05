'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/ComputePeriodogramResidues.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

from DeepKalmanFilter.TestBartlett import *

def ComputePeriodogramResidue(MeasurementMinusCStates, MeasurementMinusCFs):
    """
    Computes the periodogram residues for both correctors and predictors at every layer.
    """
    Layers = len(MeasurementMinusCStates)
    ObservationDimension = len(MeasurementMinusCStates[0])

    CorrectorResidues = np.zeros((ObservationDimension, Layers))
    PredictorResidues = np.zeros((ObservationDimension, Layers))

    for Layer in range(Layers):
        CorrectorResidues[:,Layer:Layer+1] = MeasurementMinusCStates[Layer]
        PredictorResidues[:,Layer:Layer+1] = MeasurementMinusCFs[Layer]

    CorrectorPeriodogramResidues = np.zeros(ObservationDimension)
    PredictorPeriodogramResidues = np.zeros(ObservationDimension)

    for ObservedState in range(ObservationDimension):
        CorrectorPeriodogram = TestBartlett(CorrectorResidues[ObservedState,:])[0]
        PredictorPeriodogram = TestBartlett(PredictorResidues[ObservedState,:])[0]

        CorrectorPeriodogramResidues[ObservedState] = np.linalg.norm(CorrectorPeriodogram - np.linspace(0, 1, len(CorrectorPeriodogram)))
        PredictorPeriodogramResidues[ObservedState] = np.linalg.norm(PredictorPeriodogram - np.linspace(0, 1, len(PredictorPeriodogram)))

    PeriodogramResidues = np.concatenate([CorrectorPeriodogramResidues, PredictorPeriodogramResidues])
    return PeriodogramResidues
