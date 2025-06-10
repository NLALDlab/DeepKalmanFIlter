'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/F.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Marco Dell'Orto, Fabio Marcuzzi**
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.optimize import fsolve

from DeepKalmanFilter.DynamicEquation import *
from DeepKalmanFilter.VarMiONPredictor import *

def F(x, p, u, s, Layer, NetParameters):
    """
    Computes the state prediction xp.
    """
    # Variables
    Model = NetParameters['Model']

    #print("u.shape = ",u.shape," , x.shape = ",x.shape)
    if 0:
        # Equation for the dynamics
        def DynEq(z):
            return DynamicEquation(z, x, p, u, s, Model['M'], Model['K'], Model['D'][:,Layer:Layer+1], Model['SamplingTimes'][Layer], NetParameters)

        # Solve for the state prediction
        options = {'xtol': 1e-8, 'maxfev': 1000000}
        xp = np.atleast_2d( fsolve(DynEq, x, xtol=options['xtol'], maxfev=options['maxfev']) ).T
    elif NetParameters['Experiment'][0:4] == 'DLTI':
        xp = Model['AInit']@x + u
    else:
        nx = len(x); #print("nx = ",nx)
        dt = Model['SamplingTimes'][Layer]
        #print("p = ",p)
        #print("np.squeeze(p) = ",np.squeeze(p))
        #plt.pause(5.01)
        p = np.squeeze(p)
        if np.ndim(np.squeeze(p)) == 0: p = np.array([p]) 
        if NetParameters['Experiment'] == '3MKC1' or NetParameters['Experiment'] == '3MKC2' or NetParameters['Experiment'] == '3MKC3':
            Mest = Model['M'].copy()
            if (Mest[0,0] + p[0]) < 0.0: p[0] = -Mest[0,0] + 1.0
            tmpv = np.zeros(nx); tmpv[0] = p[0]; 
            if NetParameters['Experiment'] == '3MKC3':
                if (Mest[1,1] + p[3]) < 0.0: p[3] = -Mest[1,1] + 1.0
                if (Mest[2,2] + p[6]) < 0.0: p[6] = -Mest[2,2] + 1.0
                tmpv[1] = p[3]; tmpv[2] = p[6]
            #endif
            Mest = Mest + np.diag(tmpv); #print("Mest = ",Mest)
            Kest = Model['K'].copy();
            if NetParameters['Experiment'] == '3MKC1':
                Kest[0,3] = Kest[0,3] - p[1]; Kest[0,4] = Kest[0,4] + p[1];   
                Kest[1,3] = Kest[1,3] + p[1]; Kest[1,4] = Kest[1,4] - p[1]; 
                Kest[0,0] = Kest[0,0] - p[2]; Kest[0,1] = Kest[0,1] + p[2];   
                Kest[1,0] = Kest[1,0] + p[2]; Kest[1,1] = Kest[1,1] - p[2]; 
            elif NetParameters['Experiment'] == '3MKC2':
                if (-Kest[0,1] + p[1]) < 0.0: p[1] = Kest[0,1] + 1.e-6 
                Kest[0,1] = Kest[0,1] - p[1];
                if (-Kest[0,0] + p[2]) < 0.0: p[2] = Kest[0,0] + 1.e-6 
                Kest[0,0] = Kest[0,0] - p[2];
            elif NetParameters['Experiment'] == '3MKC3':
                if 0: # we know the structure
                    ### K1
                    if (-Kest[0,3] + p[1]) < 0.0: p[1] = Kest[0,3] + 1.e-6 
                    if (Kest[0,4] + p[1]) < 0.0: p[1] = -Kest[0,4] + 1.e-6 
                    if (-Kest[1,4] + (p[1]+p[4])) < 0.0: p[1] = Kest[1,4] - p[4] + 1.e-6 
                    Kest[0,3] = Kest[0,3] - p[1]; Kest[0,4] = Kest[0,4] + p[1]; Kest[1,3] = Kest[1,3] + p[1]; 
                    ### C1
                    if (-Kest[0,0] + p[2]) < 0.0: p[2] = Kest[0,0] + 1.e-6 
                    if (Kest[0,1] + p[2]) < 0.0: p[2] = -Kest[0,1] + 1.e-6 
                    if (-Kest[1,1] + (p[2]+p[5])) < 0.0: p[2] = Kest[1,1] - p[5] + 1.e-6 
                    Kest[0,0] = Kest[0,0] - p[2]; Kest[0,1] = Kest[0,1] + p[2]; Kest[1,0] = Kest[1,0] + p[2];   
                    ### K2
                    if (Kest[1,5] + p[4]) < 0.0: p[4] = -Kest[1,5] + 1.e-6 
                    if (Kest[2,4] + p[4]) < 0.0: p[4] = -Kest[2,4] + 1.e-6 
                    # not needed because p[1] modified with the same rule: if (-Kest[1,4] + (p[1]+p[4])) < 0.0: p[4] = Kest[1,4] - p[1] + 1.e-6 
                    Kest[1,4] = Kest[1,4] - (p[1]+p[4]); Kest[1,5] = Kest[1,5] + p[4]; Kest[2,4] = Kest[2,4] + p[4]; 
                    ### C2
                    if (Kest[1,2] + p[5]) < 0.0: p[5] = -Kest[1,2] + 1.e-6 
                    if (Kest[2,1] + p[5]) < 0.0: p[5] = -Kest[2,1] + 1.e-6 
                    # not needed because p[2] modified with the same rule: if (-Kest[1,1] + (p[2]+p[5])) < 0.0: p[5] = Kest[1,1] - p[2] + 1.e-6 
                    Kest[1,1] = Kest[1,1] - (p[2]+p[5]); Kest[1,2] = Kest[1,2] + p[5]; Kest[2,1] = Kest[2,1] + p[5]; 
                    ### K3
                    if (-Kest[2,5] + (p[4]+p[7])) < 0.0: p[7] = Kest[2,5] - p[4] + 1.e-6 
                    Kest[2,5] = Kest[2,5] - (p[4]+p[7]);
                    ### C3
                    if (-Kest[2,2] + (p[5]+p[8])) < 0.0: p[8] = Kest[2,2] - p[5] + 1.e-6 
                    Kest[2,2] = Kest[2,2] - (p[5]+p[8]);
                else:
                    Mest[0,1] = Mest[0,1] + p[1]; Mest[0,2] = Mest[0,2] + p[2];
                    Mest[1,0] = Mest[1,0] + p[4]; Mest[1,2] = Mest[1,2] + p[5];
                    Mest[2,0] = Mest[2,0] + p[7]; Mest[2,1] = Mest[2,1] + p[8];
                #endif
            #endif
        else:
            Mest = Model['M'].copy()
            Kest = Model['K'].copy()
        #endif
        #print("u = ",u)
        if NetParameters['Experiment'] == 'Roes1' or NetParameters['Experiment'] == 'Roes2':
            if 1:
                #print("np.squeeze(x) = ",np.squeeze(x)); plt.pause(0.5)
                r = integrate.ode(Model['nlp']).set_integrator("lsoda",atol=1.e-2)  # choice of method
                r.set_initial_value(np.squeeze(x), 0)   # initial values
                xp = r.integrate(dt) # get one more value, add it to the array
                if not r.successful():
                    raise RuntimeError("Could not integrate")
                #endif
                xp = np.atleast_2d(xp).T
            else:
                # Explicit Euler
                M1 = np.eye(nx) + dt*np.linalg.inv(Mest)@Kest;
                xp = M1@x + u
            #endif
        else:
            # Implicit Euler
            M1 = np.eye(nx) - dt*np.linalg.inv(Mest)@Kest;
            xp = np.linalg.solve(M1, x + u)
        #endif
        #print("xp = ",xp)
    #endif
    return xp,np.atleast_2d(p).T

def F_VarMiON(x, u, p, s, Layer, NetParameters):
    """
    Computes the state prediction xp.
    """
    # Variables
    Model = NetParameters['Model']
    if NetParameters['Experiment'][0:4] =='HSE_':
        map_gp2meshnode = Model['map_gp2meshnode']
    #endif
    
    if 0 and Layer > 0: Layer = Layer - 1  # F: aggiunta del 20250527
        
    if 1:
        #print("F_VarMiON: Layer = ",Layer)
        #print(f"{x.shape = }")
        #print(f"{u.shape = }")
        #print(f"{Layer = }")
        #print()
        pass
    #endif

    xrMask = Model['xrMask']
    Ct, Theta, F, h, G = Model['HeatParameters']

    if 0:
        pass
        #print(f"{Ct.shape = }")
        #print(f"{Theta.shape = }")
        #print(f"{F.shape = }")
        #print(f"{h = }")
        #print(f"{G.shape = }")
    #endif
    
    varmion = HeatEquationVarMiONRobin()
    weights_path = NetParameters['VarMiON_weights_path']
    varmion.load_state_dict(torch.load(weights_path, weights_only = True, map_location=torch.device('cpu')))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #if Layer == 0: print(f"{device = }")
    varmion.to(device)
    ####input_tx, input_c, input_theta, input_f, input_h, input_g, input_u0 = inputs

    dt = Model['SamplingTimes'][Layer]
    t = float(dt*(Layer+1))

    #print(f"{Layer = }")
    #print(f"{type(Layer) = }")

    nx = len(x); #print("nx = ",nx)

    varmion.eval()
    with torch.no_grad():
        # modify shapes, da togliere eventualmente
        n=10
        v = np.linspace(0,1,n)
        X_p = torch.tensor([[t, v[i], v[j]] for j in range(n) for i in range(n)]).view(1,1,-1,3).to(device)
        X_c = torch.tensor(Ct).view(1,-1).to(device)
        X_theta = torch.tensor(Theta).view(1,-1).to(device)
        if NetParameters['Experiment'][0:4]=='HSE_' and NetParameters['estimate_forcing_term']:
            if 0: #NOT EXISTS! NetParameters['est_f_with_p_and_backprop']:
                X_f = torch.tensor(p).unsqueeze(0).to(device)
            elif NetParameters['est_f_with_f_est_and_maxpr']:
                #X_f = torch.tensor(F[[Layer],:]).unsqueeze(0).to(device); 
                #print("F[[",Layer,"],:] = ",F[[Layer],:])
                X_f = torch.tensor(np.atleast_2d(NetParameters['f_est'][map_gp2meshnode,Layer])).unsqueeze(0).to(device)
                #print("NetParameters['f_est'][map_gp2meshnode,",Layer,"] = ",NetParameters['f_est'][map_gp2meshnode,Layer])
                #print("pausa ..."); #plt.pause(2.0)
            #endif
        else:
            X_f = torch.tensor(F[[Layer],:]).unsqueeze(0).to(device)
        #endif
        X_h = torch.tensor(h).view(1).to(device)
        X_g = torch.tensor(G[[Layer],:]).unsqueeze(0).to(device)
        X_u0 = torch.tensor(x.T).unsqueeze(0).to(device)

        X = (X_p.float(), X_c, X_theta, X_f, X_h, X_g, X_u0.float())

        if 0:
            for i, elem in enumerate(X):
                print(i, "\t", elem.shape)
                print(elem.dtype)
            #endfor
        #endif
        temperature_prediction = varmion(X).cpu().numpy()
        #print(f"{temperature_prediction.shape = }")
    #endwith
      
    xp = temperature_prediction.reshape(-1,1)

    return xp,np.atleast_2d(p).T


