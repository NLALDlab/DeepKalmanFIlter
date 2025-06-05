"""
Create the training set for DKF examples with Lorenz63 system

Chinellato, E., Marcuzzi, F.: State, parameters and hidden dynamics estimation with
the Deep Kalman Filter: regularization strategies. Journal of Computational Science 87,
102569 (2025). https://doi.org/https://doi.org/10.1007/978-3-031-63775-9
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import scipy.io as sio
import pickle

WorkingDirectory = './'

y0 = np.array([[0],[0],[0]])

g_Experiment = None

def vdp1(t, y):
    #print("vdp1: t = ",t)
    Experiment = g_Experiment; dt = g_dt; a = g_a; b = g_b; c = g_c; beta = g_beta
    if Experiment == 'Roes1' or Experiment == 'Roes2': #
        #print("t = ",t)
        tmpn = ActivateModelNoise*ModelNoise[:,[np.int64(np.floor(t/dt))]]
        tmpf = np.array([[-y[1]-y[2]], [y[0]+a*y[1]], [-c*y[2]+beta*y[0]*y[2]]]);
        yNew = np.squeeze( np.linalg.inv(MassMat)@(tmpf + Offset + tmpn ) )
        #print("yNew = ",yNew," , tmpf = ",tmpf," , tmpn = ",tmpn)
    #endif
    return yNew


def create(Experiment='Roes1',TBegin=0,TEnd=20.0,dt=0.1,a=0,b=0,c=0,beta=0,yBegin=y0,sigma_Q=1.*10**-1.5,sigma_R=1.*10**-1.5,sigma_P=1.*10**-1.5,load_0=1.e9):
    # Experiment:
    # "Roes1": true model, all state measured
    # "Roes2": true model, only first state variable measured
    global g_Experiment, ActivateModelNoise, ModelNoise, g_dt, g_a, g_b, g_c, g_beta, MassMat, Offset
    g_Experiment = Experiment; g_dt = dt; g_a = a; g_b = b; g_c = c; g_beta = beta
    
    #Model
    nx = 3
    if Experiment == 'Roes1': #
        C = np.eye(nx) #Observation matrix
    elif Experiment == 'Roes2': #
        C = np.eye(nx); C = np.atleast_2d(C[0,:]) #Observation matrix
    #endif
    #print("C.shape = ",C.shape)
    ny = np.shape(C)[0]
    
    Ktot = np.int64(np.floor((TEnd-TBegin)/dt))
    tEuler = np.arange(TBegin,TEnd+dt,dt)

    yEuler = yBegin
    YEuler = yEuler

    #Model
    Offset = np.array([[0],[0],[b]])
    MassMat = np.diag([1.0,1.0,1.0])
    LinMat = np.array([[0,-1,-1],[1,a,0],[beta,0,beta-c]]) #

    KnownOffset = np.array([[0],[0],[0]])

    #Stochastic matrices
    Q = (sigma_Q**2)*np.eye(nx)
    R = (sigma_R**2)*np.eye(ny)
    P0 = (sigma_P**2)*np.eye(nx)
    AInitExplEuler = (np.eye(nx)+dt*np.linalg.inv(MassMat)@LinMat)
    AInit = np.linalg.inv( np.eye(nx)-dt*np.linalg.inv(MassMat)@LinMat ) 

    # Observability Grammian:
    A = AInitExplEuler.copy()
    #A = AInit.copy()
    O = C.copy()
    for k in range(nx-1):
        O = np.vstack((O, C@A)); 
    #endfor    
    print("Observability Grammian = ",O.T@O)
    
    ActivateModelNoise = 1
    ActivateMeasNoise = 1
    ActivateFirstStateNoise = 1
    ModelNoise = np.sqrt(Q)@np.random.normal(0,1,(nx,Ktot+100)) # the integrator may require times beyond TEnd
    MeasNoise = np.sqrt(R)@np.random.normal(0,1,(ny,Ktot+1))
    FirstStateNoise = np.sqrt(P0)@np.random.normal(0,1,(nx,1))

    #Explicit Euler
    yEuler = np.zeros((len(yBegin), len(tEuler)))   # array for solution
    yEuler[:,0] = np.squeeze(yBegin)
    for i in range(1, tEuler.size):
        yEuler[:,[i]] = AInitExplEuler@yEuler[:,[i-1]] + ActivateModelNoise*ModelNoise[:,i:i+1]
    #endfor
    
    #Nonlinear Solver
    ySolver = np.zeros((len(yBegin), len(tEuler)))   # array for solution
    ySolver[:,0] = np.squeeze(yBegin)
    r = integrate.ode(vdp1).set_integrator("lsoda",atol=1.e-5)  # choice of method
    r.set_initial_value(np.squeeze(yBegin), TBegin)   # initial values
    for i in range(1, tEuler.size):
       tmp = r.integrate(tEuler[i]) # get one more value, add it to the array
       ySolver[:, i] = tmp
       if not r.successful():
           raise RuntimeError("Could not integrate")
       #endif
    #endfor
    # Plot
    ax = plt.figure(1).add_subplot(1,2,1,projection='3d')
    ax.plot(*ySolver, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Roessler model - 'lsoda'")
    ax = plt.figure(1).add_subplot(1,2,2,projection='3d')
    ax.plot(*yEuler, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Roessler model - Explicit Euler")
    plt.show()

    #Set up training set
    Meas = C@ySolver + ActivateMeasNoise*MeasNoise
    if 0:
        plt.figure(2)
        plt.plot(tEuler, Meas.T)
        plt.show()
        plt.title('Noisy measurements')
    #endif
    
    #Create dataset with solver data
    TrainingInstances = 1
    TrainingSet = [None] * (6)

    for i in range(6):
        TrainingSet[i] = [None] * (TrainingInstances)
    #endfor
    for i in range(TrainingInstances):
        TrainingSet[0][i] = np.zeros((1,Ktot)) #Not used
        TrainingSet[1][i] = Meas #This should depend on i
        TrainingSet[2][i] = ySolver[:,0:1] + ActivateFirstStateNoise*FirstStateNoise #This should depend on i
        TrainingSet[3][i] = ySolver #This should depend on i
        TrainingSet[4][i] = ySolver[:,Ktot:Ktot+1] #This should depend on i
        TrainingSet[5][i] = 1
    #endfor
    
    #Save data
    sio.savemat(WorkingDirectory+'Experiment.mat', {'Experiment': Experiment})
    sio.savemat(WorkingDirectory+'LayersExp'+Experiment+'.mat', {'Layers': Ktot})
    sio.savemat(WorkingDirectory+'CExp'+Experiment+'.mat', {'C': C})

    with open(WorkingDirectory+'LatestTrainingSetExp'+Experiment+'.mat', 'wb') as handle:
                pickle.dump(TrainingSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

    Model = {}
    Model['xrMask'] = range(nx)
    Model['QInit'] = Q
    # for KF and DKF gains initialization, here we add the model noise due to the Exlicit Euler discretization error
    Model['Q_KF'] = Model['QInit'] + (1.*10**-1)**2 * np.eye(nx)
    Model['RInit'] = R
    Model['invRInit'] = np.linalg.inv(R)
    Model['PInit'] = P0
    Model['AInit'] = AInit
    Model['AInitExplEuler'] = AInitExplEuler
    if 1: #Experiment == 'Roes1': #
        Model['M'] = MassMat
    #endif
    Model['K'] = LinMat
    Model['D'] = np.tile(KnownOffset,(1,Ktot))
    Model['SamplingTimes'] = dt*np.ones((Ktot,1))
    Model['nlp'] = vdp1
    with open(WorkingDirectory+'ModelExp'+Experiment+'.mat', 'wb') as handle:
                pickle.dump(Model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    HiddenDynDim = 1
    sio.savemat(WorkingDirectory+'HiddenDynDimExp'+Experiment+'.mat', {'HiddenDynDim': HiddenDynDim})

    APAQInv = [None]*Ktot
    RInv = [None]*Ktot
    P = Model['PInit']
    A = Model['AInit']
    Q = Model['Q_KF']
    InvR = Model['invRInit']

    for Layer in range(Ktot):
        RInv[Layer] = InvR
        APAQInv[Layer] = np.linalg.inv(A@P@A.T + Q)
        InvP = APAQInv[Layer] + C.T@InvR@C
        P = np.linalg.inv(InvP)    
    #endfor
    sio.savemat(WorkingDirectory+'PredictorWeightMatsExp'+Experiment+'.mat', {'PredictorWeightMats': APAQInv})
    sio.savemat(WorkingDirectory+'MeasurementWeightMatsExp'+Experiment+'.mat', {'MeasurementWeightMats': RInv})
    return


