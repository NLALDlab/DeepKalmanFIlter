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

y0 = np.array([[-5.6866],[-8.4929],[17.8452]])

def create(Experiment='L63_1',TBegin=0,TEnd=0.2,dt=0.001,sigma=10,ro=28,beta=8./3,epsilon=0.01,yBegin=y0,sigma_Q=1.*10**-1.5,sigma_R=1.*10**-1.5,sigma_P=1.*10**-1.5,load_0=1.e9):
    # Experiment:
    # "L63_1": true model, all observable
    # "L63_2": true model, only first state variable observed
    # "L63_3": true model plus cubic model error term, all observable
    # "L63_4": true model plus cubic model error term and a parameter mismatch, all observable

    #Model
    nx = 3
    if Experiment == 'L63_1' or Experiment == 'L63_3' or Experiment == 'L63_4': #
        C = np.eye(nx) #Observation matrix
    elif Experiment == 'L63_2': #
        C = np.eye(nx); C = np.atleast_2d(C[0,:]) #Observation matrix
    #endif
    #print("C.shape = ",C.shape)
    ny = np.shape(C)[0]
    
    Ktot = np.int64(np.floor((TEnd-TBegin)/dt))
    tEuler = np.arange(TBegin,TEnd+dt,dt)

    yEuler = yBegin
    YEuler = yEuler

    #Model
    Offset = np.array([[0],[0],[0]])
    NonLinCoeff = np.array([[epsilon],[1],[1]])
    MassMat = np.diag([0.1,0.1,0.1])
    LinMat = np.array([[-sigma,sigma,0],[ro,-1,0],[0,0,-beta]]) # A COSA SERVE PER Lorenz63 che è nonlineare?!?

    KnownOffset = np.array([[0],[0],[0]])

    #Stochastic matrices
    Q = (sigma_Q**2)*np.eye(nx)
    R = (sigma_R**2)*np.eye(ny)
    P0 = (sigma_P**2)*np.eye(nx)
    AInitExplEuler = (np.eye(nx)+dt*np.linalg.inv(MassMat)@LinMat)
    AInit = np.linalg.inv(np.eye(nx)-dt*np.linalg.inv(MassMat)@LinMat) 

    ActivateModelNoise = 1
    ActivateMeasNoise = 1
    ActivateFirstStateNoise = 1
    ModelNoise = np.sqrt(Q)@np.random.normal(0,1,(nx,Ktot+1))
    MeasNoise = np.sqrt(R)@np.random.normal(0,1,(ny,Ktot+1))
    FirstStateNoise = np.sqrt(P0)@np.random.normal(0,1,(nx,1))

    #Solver
    def vdp1(t, y):
        if Experiment == 'L63_1' or Experiment == 'L63_2': #
            yNew = np.squeeze( np.linalg.inv(MassMat)@( np.array([[-sigma*y[0]+sigma*y[1]], [ro*y[0]-y[1]-y[0]*y[2]], [-beta*y[2]+y[0]*y[1]]]) + Offset + ActivateModelNoise*ModelNoise[:,np.int64(np.floor(t/dt)):1+np.int64(np.floor(t/dt))] ) )
        elif Experiment == 'L63_3' or Experiment == 'L63_4': #
            yNew = np.squeeze( np.linalg.inv(MassMat)@( np.array([[-sigma*y[0]+sigma*y[1]+epsilon*y[0]**3], [ro*y[0]-y[1]-y[0]*y[2]], [-beta*y[2]+y[0]*y[1]]]) + Offset + ActivateModelNoise*ModelNoise[:,np.int64(np.floor(t/dt)):1+np.int64(np.floor(t/dt))] ) )
        return yNew
    ySolver = np.zeros((len(tEuler), len(yBegin)))   # array for solution
    ySolver[0, :] = np.squeeze(yBegin)
    r = integrate.ode(vdp1).set_integrator("dopri5")  # choice of method
    r.set_initial_value(np.squeeze(yBegin), TBegin)   # initial values
    for i in range(1, tEuler.size):
       ySolver[i, :] = r.integrate(tEuler[i]) # get one more value, add it to the array
       if not r.successful():
           raise RuntimeError("Could not integrate")
    plt.figure(1)
    plt.plot(tEuler, ySolver)
    plt.show()

    #Set up training set
    Meas = C@ySolver.T + ActivateMeasNoise*MeasNoise
    plt.figure(2)
    plt.plot(tEuler, Meas.T)
    plt.show()
    plt.title('Noisy measurements')

    #Create dataset with solver data
    TrainingInstances = 1
    TrainingSet = [None] * (6)

    for i in range(6):
        TrainingSet[i] = [None] * (TrainingInstances)
    #endfor
    for i in range(TrainingInstances):
        TrainingSet[0][i] = np.zeros((1,ySolver.shape[0])) #Not used
        TrainingSet[1][i] = Meas #This should depend on i
        TrainingSet[2][i] = ySolver[0:1,:].T + ActivateFirstStateNoise*FirstStateNoise #This should depend on i
        TrainingSet[3][i] = ySolver.T #This should depend on i
        TrainingSet[4][i] = ySolver[Ktot:Ktot+1,:].T #This should depend on i
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
    Model['RInit'] = R
    Model['invRInit'] = np.linalg.inv(R)
    Model['PInit'] = P0
    Model['AInit'] = AInit
    if Experiment == 'L63_4': #
        Model['M'] = np.diag([5,1,1])@MassMat
    else:
        Model['M'] = MassMat
    #endif
    Model['K'] = LinMat
    Model['D'] = np.tile(KnownOffset,(1,Ktot))
    Model['SamplingTimes'] = dt*np.ones((Ktot,1))
    with open(WorkingDirectory+'ModelExp'+Experiment+'.mat', 'wb') as handle:
                pickle.dump(Model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    HiddenDynDim = 1
    sio.savemat(WorkingDirectory+'HiddenDynDimExp'+Experiment+'.mat', {'HiddenDynDim': HiddenDynDim})

    APAQInv = [None]*Ktot
    RInv = [None]*Ktot
    P = Model['PInit']
    A = Model['AInit']
    Q = Model['QInit']
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


