'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/ComputeTVDerivatives.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np

from DeepKalmanFilter.TVDifferentiate2 import *
from DeepKalmanFilter.TestBartlett import *

def ComputeTVDerivative(X, A, D, AtA, B, TimeStep):
    """
    Computes the matrix X' = XPrime and XSmooth using TV regularizer.
    """
    # Variables
    AlphaNum = 15
    AlphaMin = 1e-10
    AlphaMax = 0.5
    AlphaInterval = np.logspace(np.log10(AlphaMin), np.log10(AlphaMax), AlphaNum)
    SearchMaxIt = 100
    FinalMaxIt = 1000

    XPrime = np.zeros_like(X)
    XSmooth = np.zeros_like(X) if X.shape[1] > 1 else None

    States = X.shape[0]

    for State in range(States):
        # Select current state time series
        CurrState = X[State, :]

        # Normalize it
        CurrStateNorm = (CurrState[:-1] + CurrState[1:]) / 2
        Offset2 = CurrStateNorm[0]
        CurrStateNorm -= Offset2

        AtCurrState = A.T @ CurrStateNorm
        uInit = (1 / TimeStep) * np.concatenate(([0], np.diff(CurrStateNorm), [0]))

        # Estimate best alpha based on whiteness
        PeriodogramResidueOpt = np.inf

        for Alpha in AlphaInterval:
            # Estimate denoised state time series for current alpha value
            CurrStateEst = TVDifferentiate2(TimeStep, Alpha, B, D, AtA, AtCurrState, uInit, SearchMaxIt)[1]
            CurrStateEst += Offset2

            # Compute cumulative periodogram of the error & residue wrt white noise
            CumulativePeriodogram = TestBartlett(CurrState[1:] - CurrStateEst)[0]
            PeriodogramResidue = np.linalg.norm(CumulativePeriodogram - np.linspace(0, 1, len(CumulativePeriodogram)))

            # If current value of alpha creates a whiter residue, swap it
            if PeriodogramResidue < PeriodogramResidueOpt:
                PeriodogramResidueOpt = PeriodogramResidue
                AlphaOpt = Alpha

        # Estimate derivative for optimal alpha value
        CurrStateDerivativeEst, CurrStateEst = TVDifferentiate2(TimeStep, AlphaOpt, B, D, AtA, AtCurrState, uInit, FinalMaxIt)
        XPrime[State, :] = CurrStateDerivativeEst
        if XSmooth is not None:
            XSmooth[State, :] = np.concatenate(([0], CurrStateEst)) + Offset2

    return XPrime, XSmooth
