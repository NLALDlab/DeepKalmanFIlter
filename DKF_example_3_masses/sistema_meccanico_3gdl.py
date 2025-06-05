import numpy as np
from  sistemi_DLTI import *

#======================================= PARAMETRI DEL SISTEMA
M1 = 27.0 			# Kg
print("M1 = ",M1)
M2 = 120.0			# Kg
print("M2 = ",M2)
M3 = 1.1246e4		# Kg
print("M3 = ",M3)
K1 = 18.0e7			# N/m
print("K1 = ",K1)
K2 = 6.0e7			# N/m
print("K2 = ",K2)
K3 = 1.2e8			# N/m
print("K3 = ",K3)
C1 = 5.0e4			# N*s/m
print("C1 = ",C1)
C2 = 4.6e4			# N*s/m
print("C2 = ",C2)
C3 = 2.4e5			# N*s/m
print("C3 = ",C3)
deltaX1 = 0.005		# m
deltaX2 = 0.005		# m
deltaX3 = 0.05		# m
print('coordinate di partenza: y1=%.2f, y2=%.2f, y3=%.2f' % (deltaX3+deltaX2+deltaX1, deltaX3+deltaX2, deltaX3))

def build_sistema_meccanico_3gdl(M1_stimato=None,M2_stimato=None,M3_stimato=None,K1_stimato=None,K2_stimato=None,K3_stimato=None,C1_stimato=None,C2_stimato=None,C3_stimato=None,deltaX1_imposto=None,deltaX2_imposto=None,deltaX3_imposto=None,config_attuatori=1,config_sensori=4):
    global M1, M2, M3, K1, K2, K3, C1, C2, C3, deltaX1, deltaX2, deltaX3
    if M1_stimato is not None:
      M1 = M1_stimato;
    #endif
    if M2_stimato is not None:
      M2 = M2_stimato;
    #endif
    if M3_stimato is not None:
      M3 = M3_stimato;
    #endif
    if K1_stimato is not None:
      K1 = K1_stimato;
    #endif
    if K2_stimato is not None:
      K2 = K2_stimato;
    #endif
    if K3_stimato is not None:
      K3 = K3_stimato;
    #endif
    if C1_stimato is not None:
      C1 = C1_stimato;
    #endif
    if C2_stimato is not None:
      C2 = C2_stimato;
    #endif
    if C3_stimato is not None:
      C3 = C3_stimato;
    #endif
    if deltaX1_imposto is not None:
      deltaX1 = deltaX1_imposto;
    #endif
    if deltaX2_imposto is not None:
      deltaX2 = deltaX2_imposto;
    #endif
    if deltaX3_imposto is not None:
      deltaX3 = deltaX3_imposto;
    #endif
    MassMat = np.diag([M1,M2,M3,1,1,1])
    LinMat = np.array([[-C1,    C1,      0.,  -K1, K1, 0.], \
                   [C1, -(C1+C2), C2,  K1, -(K1+K2),     K2       ], \
                   [   0.,      C2,     -(C2+C3),     0.,       K2,     -(K2+K3)   ], \
                   [   1.,        0.,          0.,          0.,         0.,           0.       ], \
                   [   0.,        1.,          0.,          0.,         0.,           0.       ], \
                   [   0.,        0.,          1.,          0.,         0.,           0.       ]]);
    A = np.linalg.inv(MassMat)@LinMat

    if config_attuatori == 1:
      Bf = [1/M1, 0, 0];
    elif config_attuatori == 2:
      Bf = [0, 1/M2, 0];
    elif config_attuatori == 3:
      Bf = [0, 0, 1/M3];
    #endif
    # Caso con 1 ingresso esterno:
    B = np.array([[Bf[0],   K1*deltaX1/M1                ], \
                   [Bf[1],   (K2*deltaX2-K1*deltaX1)/M2   ], \
                   [Bf[2],   (K3*deltaX3-K2*deltaX2)/M3   ], \
                   [0.,                 0.                ], \
                   [0.,                 0.                ], \
                   [0.,                 0.                ]])
    
    if config_sensori == 1:
      C = np.array([[0., 0., 0., 1., 0., 0.],[0., 0., 0., 0., 1., 0.],[0., 0., 0., 0., 0., 1.]]); D = np.array([[0., 0.],[0., 0.],[0., 0.]])
    elif config_sensori == 2:
      C = np.array([[0., 0., 0., 1., 0., 0.],[0., 0., 0., 0., 1., 0.]]);  D = np.array([[0., 0.],[0., 0.]])
    elif config_sensori == 3:
      C = np.array([[0., 0., 0., 1., 0., -1.],[0., 0., 0., 0., 1., -1.]]);  D = np.array([[0., 0.],[0., 0.]])
    elif config_sensori == 4:
      C = np.array([[0., 0., 0., 1., 0., 0.]]);  D = np.array([[0., 0.]])
    elif config_sensori == 5:
      C = np.array([[0., 0., 0., 0., 1., 0.]]);  D = np.array([[0., 0.]])
    elif config_sensori == 6:
      C = np.array([[0., 0., 0., 0., 0., 1.]]);  D = np.array([[0., 0.]])
    elif config_sensori == 7:
      C = np.array([[1., 0., 0., 0., 0., 0.]]);  D = np.array([[0., 0.]])
    #endif
    return A, B, C, D, MassMat, LinMat

def get_initial_state():
    x0 = np.atleast_2d(np.array([0., 0., 0., deltaX1+deltaX2+deltaX3, deltaX2+deltaX3, deltaX3])).T
    return x0

def simula_sistema_meccanico_3gdl(A, B, C, D, load, Ts):
    N = load.shape[load.ndim-1]
    # stato iniziale
    x0 = get_initial_state()
    # formo il segnale di ingresso 
    u = np.zeros([2,N])
    u[0,:] = load
    u[1,:] = np.ones(N)
    u = np.array(u)
    #
    y,X_hist,Ad = simula_DLTI_StateSpace_continuo(A,B,C,D,u,x0,Ts)
    response = y - C@x0
    return response, X_hist
