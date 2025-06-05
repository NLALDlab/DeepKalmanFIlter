"""
Create the training set for DKF examples with DLTI systems

Chinellato, E., Marcuzzi, F.: State, parameters and hidden dynamics estimation with
the Deep Kalman Filter: regularization strategies. Journal of Computational Science 87,
102569 (2025). https://doi.org/https://doi.org/10.1007/978-3-031-63775-9

© 2025 Erik Chinellato, Fabio Marcuzzi - https://github.com/NLALDlab/DeepKalmanFilter
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import scipy.io as sio
import pickle

WorkingDirectory = './'

def create(Experiment='DLTI1',TBegin=0,TEnd=0.128,dt=0.001,sigma_Q=2.e-2,sigma_R=3.5e-2,sigma_P=1.e-1,load_0=1.e9):
    # Experiment:
    # "DLTI1": a DLTI system optimal for KF
    # "DLTI2": a weakly-observable DLTI system not-so-bad for KF
    # "DLTI3": a weakly-observable DLTI system bad for KF

    #Model
    if Experiment == 'DLTI1': #
        nx = 2
        A = np.array([[1.01,-0.002],[0,1.005]])
        B = np.zeros((nx,1))
        C = np.eye(nx) #Observation matrix
    elif Experiment == 'DLTI2': #
        nx = 3
        B = np.zeros((nx,1))
        C = np.eye(nx) #Observation matrix
    elif Experiment == 'DLTI3': #
        nx = 3
        B = np.zeros((nx,1))
        C = np.eye(nx) #Observation matrix
    #endif
    ny = np.shape(C)[0]
    # Observability Grammian:
    O = C.copy()
    for k in range(nx-1):
        O = np.vstack((O, C@A)); 
    #endfor    
    print("Observability Grammian = ",O.T@O)
    
    #Stochastic matrices
    Q = (sigma_Q**2)*np.eye(nx)
    R = (sigma_R**2)*np.eye(ny)
    P0 = (sigma_P**2)*np.eye(nx)

    N = np.int64(np.floor((TEnd-TBegin)/dt))
    tEuler = np.arange(TBegin,TEnd+dt,dt)

    yBegin = np.ones((nx,1));
    yEuler = yBegin.copy()
    ySolver = np.zeros((len(tEuler), len(yBegin)))   # array for solution
    ySolver[0, :] = np.squeeze(yEuler)
    uSolver = np.zeros((len(yEuler), len(tEuler)))

    ActivateModelNoise = 1
    ActivateMeasNoise = 1
    ActivateFirstStateNoise = 1
    ModelNoise = np.sqrt(Q)@np.random.normal(0,1,(nx,N+1))
    MeasNoise = np.sqrt(R)@np.random.normal(0,1,(ny,N+1))
    FirstStateNoise = np.sqrt(P0)@np.random.normal(0,1,(nx,1))

    # inputs:
    load = np.zeros(N); load[0] = load_0
    u = np.zeros((1,N))
    u[0,:] = load

    for k in range(N):
        u_k = np.atleast_2d(u[:,[k]]); #print("u_k.shape = ",u_k.shape)
        uSolver[:,[k]] = B@u_k; #print("uSolver[:,[",k,"]] = ",uSolver[:,[k]])
        yEuler = A@yEuler + uSolver[:,[k]]
        #print("yEuler = ",yEuler)
        #print("yEuler.shape = ",yEuler.shape)
        yEuler = yEuler + ActivateModelNoise*ModelNoise[:,[k]];
        #print("noise: yEuler.shape = ",yEuler.shape)
        ySolver[k+1, :] = np.squeeze(yEuler)
    #endfor

    plt.figure(1)
    plt.plot(tEuler, ySolver)
    plt.show()

    #Set up training set
    Meas = C@ySolver.T + ActivateMeasNoise*MeasNoise
    plt.figure(2)
    plt.plot(tEuler, Meas.T)
    plt.title('Noisy measurements')
    plt.show()

    #Create dataset with solver data
    TrainingInstances = 1
    TrainingSet = [None] * (6)

    for i in range(6):
        TrainingSet[i] = [None] * (TrainingInstances)
    #endfor
    for i in range(TrainingInstances):
        TrainingSet[0][i] = uSolver #np.zeros((1,np.shape(ySolver)[0])) #Not used
        TrainingSet[1][i] = Meas #This should depend on i
        TrainingSet[2][i] = ySolver[0:1,:].T + ActivateFirstStateNoise*FirstStateNoise #This should depend on i
        TrainingSet[3][i] = ySolver.T #This should depend on i
        TrainingSet[4][i] = ySolver[N:N+1,:].T #This should depend on i
        TrainingSet[5][i] = 1
    #endfor

    #Save data
    sio.savemat(WorkingDirectory+'Experiment.mat', {'Experiment': Experiment})
    sio.savemat(WorkingDirectory+'LayersExp'+str(Experiment)+'.mat', {'Layers': N})

    with open(WorkingDirectory+'LatestTrainingSetExp'+str(Experiment)+'.mat', 'wb') as handle:
                pickle.dump(TrainingSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

    Model = {}
    nxm = nx
    Model['xrMask'] = range(nx)
    Cm = C.copy()
    sio.savemat(WorkingDirectory+'CExp'+str(Experiment)+'.mat', {'C': Cm})
    Model['QInit'] = Q
    Model['RInit'] = np.atleast_2d(R)
    Model['invRInit'] = np.atleast_2d(np.linalg.inv(R))
    Model['PInit'] = P0
    Model['AInit'] = A
    Model['B'] = B
    Model['M'] = []
    Model['Mtrue'] = []
    Model['K'] = []
    Model['D'] = np.tile(np.zeros((nxm,1)),(1,N)); #np.tile(KnownOffset,(1,N))
    Model['SamplingTimes'] = dt*np.ones((N,1))
    with open(WorkingDirectory+'ModelExp'+str(Experiment)+'.mat', 'wb') as handle:
                pickle.dump(Model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    HiddenDynDim = 1
    sio.savemat(WorkingDirectory+'HiddenDynDimExp'+str(Experiment)+'.mat', {'HiddenDynDim': HiddenDynDim})

    APAQInv = [None]*N
    RInv = [None]*N
    P = Model['PInit']
    A = Model['AInit']
    Q = Model['QInit']
    InvR = Model['invRInit'];

    for Layer in range(N):
        #print("Layer = ",Layer)
        RInv[Layer] = InvR
        APAQInv[Layer] = np.linalg.inv(A@P@A.T + Q)
        #print("APAQInv[Layer].shape = ",APAQInv[Layer].shape)
        InvP = APAQInv[Layer] + Cm.T@InvR@Cm
        P = np.linalg.inv(InvP)    
    #endfor
    sio.savemat(WorkingDirectory+'PredictorWeightMatsExp'+str(Experiment)+'.mat', {'PredictorWeightMats': APAQInv})
    sio.savemat(WorkingDirectory+'MeasurementWeightMatsExp'+str(Experiment)+'.mat', {'MeasurementWeightMats': RInv})
    return

