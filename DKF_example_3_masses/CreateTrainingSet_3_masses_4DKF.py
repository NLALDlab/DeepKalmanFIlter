"""
Create the training set for DKF examples with 3-dof mass-spring-damper model

...
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import scipy.io as sio
import pickle
from sistema_meccanico_3gdl import *

WorkingDirectory = './'

def create(Experiment='3MKC1',TBegin=0,TEnd=0.128,dt=0.001,sigma_Q=1.e-6,sigma_R=1.e-6,sigma_P=1.e-6,load_0=1.e9,M1=None,K1=None,C1=None,epsilon=1.0,deltaX1=None,deltaX2=None,deltaX3=None,config_attuatori=1,config_sensori=4):
    # Experiment:
    # "3MKC1": linear model, known with parameter mismatch, measured variables: 
    # "3MKC2": linear model, only 1-dof known and no model error estimation or Markovian closure, measured variables: 
    # "3MKC3": linear model, only 1-dof known and augmented model error estimation (non-Markovian closure), measured variables: 
    # "3MKC4": linear model, only 1-dof known and memory model error estimation (non-Markovian closure), measured variables: 
    # "3MKCx": nonlinear model
    
    #Model
    if Experiment == '3MKC1' or Experiment == '3MKC2' or Experiment == '3MKC3': #
        nx = 3*2
        if M1 == None:
            Ac,Bc,Cc,Dc,MassMat,LinMat = build_sistema_meccanico_3gdl(K1_stimato=K1,K2_stimato=K1,K3_stimato=K1,C1_stimato=C1,C2_stimato=C1,C3_stimato=C1,deltaX1_imposto=deltaX1,deltaX2_imposto=deltaX2,deltaX3_imposto=deltaX3,config_attuatori=config_attuatori,config_sensori=config_sensori)
        else:
            Ac,Bc,Cc,Dc,MassMat,LinMat = build_sistema_meccanico_3gdl(M1_stimato=M1,M2_stimato=epsilon*M1,M3_stimato=epsilon*M1,K1_stimato=K1,K2_stimato=K1,K3_stimato=K1,C1_stimato=C1,C2_stimato=C1,C3_stimato=C1,deltaX1_imposto=deltaX1,deltaX2_imposto=deltaX2,deltaX3_imposto=deltaX3,config_attuatori=config_attuatori,config_sensori=config_sensori)
        #endif
        C = Cc.copy() #Observation matrix
    #endif
    ny = np.shape(C)[0]
    
    #Stochastic matrices
    Q = (sigma_Q**2)*np.eye(nx)
    R = (sigma_R**2)*np.eye(ny)
    P0 = (sigma_P**2)*np.eye(nx)
    
    N = np.int64(np.floor((TEnd-TBegin)/dt))
    tEuler = np.arange(TBegin,TEnd+dt,dt)

    yBegin = get_initial_state() 
    yEuler = yBegin.copy()
    ySolver = np.zeros((len(tEuler), len(yBegin)))   # array for solution
    ySolver[0, :] = np.squeeze(yEuler)
    uSolver = np.zeros((len(yEuler), len(tEuler)))
    u1d = np.zeros((2, len(tEuler)))

    ActivateModelNoise = 1
    ActivateMeasNoise = 1
    ActivateFirstStateNoise = 1
    ModelNoise = np.sqrt(Q)@np.random.normal(0,1,(np.shape(C)[1],N+1))
    MeasNoise = np.sqrt(R)@np.random.normal(0,1,(np.shape(C)[0],N+1))
    FirstStateNoise = np.sqrt(P0)@np.random.normal(0,1,(np.shape(C)[1],1))

    # inputs:
    load = np.zeros(N); load[0] = load_0
    u = np.zeros((2,N))
    u[0,:] = load
    u[1,:] = np.ones(N)

    #Implicit Euler + discrete model noise
    if Experiment == '3MKC1' or Experiment == '3MKC2' or Experiment == '3MKC3': # linear model
        Mdiscr = np.eye(nx) - dt*np.linalg.inv(MassMat)@LinMat;
        AInit = np.linalg.inv(Mdiscr);
        A = AInit.copy()
        if Experiment == '3MKC2':
            #print("LinMat = ",LinMat)
            #print("yBegin = ",yBegin)
            tmpBc = np.array([[1./MassMat[0,0],   LinMat[0,4]*yBegin[3,0]/MassMat[0,0]], 
                              [0.,                 0.                  ]])
        #endif
        #print("Bc = ",Bc)
        #print("tmpBc = ",tmpBc)
        # Observability Grammian:
        O = C.copy()
        for k in range(nx-1):
            O = np.vstack((O, C@A)); 
        #endfor    
        print("Observability Grammian = ",O.T@O)
        for k in range(N):
            u_k = np.atleast_2d(u[:,[k]]); #print("u_k.shape = ",u_k.shape)
            if Experiment == '3MKC2':
                u1d[:,[k]] = dt*tmpBc@u_k; #print("u1d[:,[",k,"]] = ",u1d[:,[k]])
            #endif
            uSolver[:,[k]] = dt*Bc@u_k; #print("uSolver[:,[",k,"]] = ",uSolver[:,[k]])
            yEuler = np.linalg.solve(Mdiscr, yEuler + uSolver[:,[k]])
            #print("yEuler = ",yEuler)
            #print("yEuler.shape = ",yEuler.shape)
            yEuler = yEuler + ActivateModelNoise*ModelNoise[:,[k]];
            #print("noise: yEuler.shape = ",yEuler.shape)
            ySolver[k+1, :] = np.squeeze(yEuler)
        #endfor
    elif Experiment == '3MKCx': # nonlinear model
        Minv = np.linalg.inv(MassMat)
        for k in range(N):
            u_k = np.atleast_2d(u[:,[k]]); #print("u_k.shape = ",u_k.shape)
            uSolver[:,[k]] = dt*Bc@u_k; #print("uSolver[:,[",k,"]] = ",uSolver[:,[k]])
            if 1:
                K1 = 18.0e7 # + 1.e9*yEuler[3]			# N/m
                K2 = 6.0e7 #+ 1.e10*yEuler[4]			# N/m
                K3 = 1.2e8 + 1.e12*yEuler[5]			# N/m
                C1 = 5.0e4			# N*s/m
                C2 = 4.6e4			# N*s/m
                C3 = 2.4e5			# N*s/m
            else:
                K1 = 18.0e7/(1-np.min([0.9999,1e2*abs(yEuler[3]-0.060)]))			# N/m
                K2 = 6.0e7/(1-np.min([0.9999,1e2*abs(yEuler[4]-0.055)]))			# N/m
                K3 = 1.2e8/(1-np.min([0.9999,1e2*abs(yEuler[5]-0.050)]))			# N/m
                C1 = 5.0e4			# N*s/m
                C2 = 4.6e4			# N*s/m
                C3 = 2.4e5			# N*s/m
            #endif
            tmpLinMat = np.array([[-C1,    C1,      0.,  -K1, K1, 0.], \
                       [C1, -(C1+C2), C2,  K1, -(K1+K2),     K2       ], \
                       [   0.,      C2,     -(C2+C3),     0.,       K2,     -(K2+K3)   ], \
                       [   1.,        0.,          0.,          0.,         0.,           0.       ], \
                       [   0.,        1.,          0.,          0.,         0.,           0.       ], \
                       [   0.,        0.,          1.,          0.,         0.,           0.       ]],dtype=float);
            Mdiscr = np.eye(nx) - dt*Minv@tmpLinMat;
            yEuler = np.linalg.solve(Mdiscr, yEuler + uSolver[:,[k]])
            #print("yEuler = ",yEuler)
            #print("yEuler.shape = ",yEuler.shape)
            yEuler = yEuler + ActivateModelNoise*ModelNoise[:,[k]];
            #print("noise: yEuler.shape = ",yEuler.shape)
            ySolver[k+1, :] = np.squeeze(yEuler)
        #endfor
        AInit = np.linalg.inv(Mdiscr); # TODO: COSì NON HA SENSO!
    #endif

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
        if Experiment == '3MKC2':
            tmpnm = int(nx/2)
            TrainingSet[0][i] = u1d # np.vstack((np.sum(uSolver[0:tmpnm,:],axis=0,keepdims=True),np.sum(uSolver[tmpnm:,:],axis=0,keepdims=True))) 
            TrainingSet[2][i] = ySolver[0:1,:].T + ActivateFirstStateNoise*FirstStateNoise[0:1,:] #This should depend on i
        else:
            TrainingSet[0][i] = uSolver 
            TrainingSet[2][i] = ySolver[0:1,:].T + ActivateFirstStateNoise*FirstStateNoise #This should depend on i
        #endif
        TrainingSet[1][i] = Meas #This should depend on i
        TrainingSet[3][i] = ySolver.T #This should depend on i
        TrainingSet[4][i] = ySolver[N:N+1,:].T #This should depend on i
        TrainingSet[5][i] = 1
    #endfor

    #Save data
    sio.savemat(WorkingDirectory+'Experiment.mat', {'Experiment': Experiment})
    sio.savemat(WorkingDirectory+'LayersExp'+Experiment+'.mat', {'Layers': N})

    with open(WorkingDirectory+'LatestTrainingSetExp'+Experiment+'.mat', 'wb') as handle:
                pickle.dump(TrainingSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

    Model = {}
    if Experiment == '3MKC1': # full-known model with parameter mismatch
        nxm = nx
        Model['xrMask'] = range(nx)
        Cm = C.copy()
        sio.savemat(WorkingDirectory+'CExp'+Experiment+'.mat', {'C': Cm})
        Model['QInit'] = Q
        Model['RInit'] = np.atleast_2d(R)
        Model['invRInit'] = np.atleast_2d(np.linalg.inv(R))
        Model['PInit'] = P0
        Model['AInit'] = AInit
        HiddenDynDim = 3
        Model['M'] = np.diag([0.7,1,1,1,1,1])@MassMat
        Model['Mtrue'] = MassMat
        Model['K'] = LinMat
        Model['Ktrue'] = LinMat
    elif Experiment == '3MKC2': # 1-dof model, no model error estimation
        nxm = 2*1
        Model['xrMask'] = [0,3]
        Cm = np.atleast_2d(C[:,Model['xrMask']]); #print("Cm.shape = ",Cm.shape)
        sio.savemat(WorkingDirectory+'CExp'+Experiment+'.mat', {'C': Cm})
        Qm = Q[Model['xrMask'],:]; Qm = Qm[:,Model['xrMask']]
        Model['QInit'] = Qm
        Model['RInit'] = R
        Model['invRInit'] = np.atleast_2d(np.linalg.inv(R))
        P0m = P0[Model['xrMask'],:]; P0m = P0m[:,Model['xrMask']]
        Model['PInit'] = P0m
        print("MassMat = ",MassMat)
        MassMatm = MassMat[Model['xrMask'],:]; MassMatm = MassMatm[:,Model['xrMask']]; print("MassMatm = ",MassMatm)
        HiddenDynDim = 3
        Model['M'] = np.diag([1,1])@MassMatm
        Model['Mtrue'] = MassMat
        print("LinMat = ",LinMat)
        LinMatm = LinMat[Model['xrMask'],:]; LinMatm = LinMatm[:,Model['xrMask']]; print("LinMatm = ",LinMatm)
        Model['K'] = LinMatm
        Model['Ktrue'] = LinMat
        Mdiscrm = np.eye(nxm) - dt*np.linalg.inv(MassMatm)@LinMatm;
        AInitm = np.linalg.inv(Mdiscrm);
        Model['AInit'] = AInitm
    elif Experiment == '3MKC3': # 1-dof model, aumented
        nxm = nx
        Model['xrMask'] = range(nx)
        Cm = C.copy()
        sio.savemat(WorkingDirectory+'CExp'+Experiment+'.mat', {'C': Cm})
        Model['QInit'] = Q
        Model['RInit'] = np.atleast_2d(R)
        Model['invRInit'] = np.atleast_2d(np.linalg.inv(R))
        Model['PInit'] = P0
        Model['AInit'] = AInit
        HiddenDynDim = 9
        Model['M'] = np.diag([1,1./MassMat[1,1],1./MassMat[2,2],1,1,1])@MassMat; print("Model['M'] = ",Model['M'])
        #Model['M'] = np.diag([1,1,1,1,1,1])@MassMat; print("Model['M'] = ",Model['M'])
        Model['Mtrue'] = MassMat
        LinMatm = LinMat.copy(); #LinMatm[1,1] = -LinMatm[1,0]; LinMatm[1,2] = 1e-6; LinMatm[1,4] = -LinMatm[1,3]; LinMatm[1,5] = 1e-6;
        #LinMatm[2,1] = 1e-6; LinMatm[2,2] = 1e-6; LinMatm[2,4] = 1e-6; LinMatm[2,5] = 1e-6;
        Model['K'] = LinMatm; print("Model['K'] = ",Model['K'])
        Model['Ktrue'] = LinMat
    elif Experiment == '3MKCx': # linear known model (to be used when the real model is nonlinear)
        nxm = nx
        Model['xrMask'] = range(nx)
        Cm = C.copy()
        sio.savemat(WorkingDirectory+'CExp'+Experiment+'.mat', {'C': Cm})
        Model['QInit'] = Q
        Model['RInit'] = np.atleast_2d(R)
        Model['invRInit'] = np.atleast_2d(np.linalg.inv(R))
        Model['PInit'] = P0
        Model['AInit'] = AInit
        HiddenDynDim = 0
        Model['M'] = MassMat
        Model['Mtrue'] = MassMat
        Model['K'] = LinMat
        Model['Ktrue'] = LinMat
    #endif
    Model['D'] = np.tile(np.zeros((nxm,1)),(1,N)); #np.tile(KnownOffset,(1,N))
    Model['SamplingTimes'] = dt*np.ones((N,1))
    with open(WorkingDirectory+'ModelExp'+Experiment+'.mat', 'wb') as handle:
                pickle.dump(Model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sio.savemat(WorkingDirectory+'HiddenDynDimExp'+Experiment+'.mat', {'HiddenDynDim': HiddenDynDim})

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
    sio.savemat(WorkingDirectory+'PredictorWeightMatsExp'+Experiment+'.mat', {'PredictorWeightMats': APAQInv})
    sio.savemat(WorkingDirectory+'MeasurementWeightMatsExp'+Experiment+'.mat', {'MeasurementWeightMats': RInv})
    return

