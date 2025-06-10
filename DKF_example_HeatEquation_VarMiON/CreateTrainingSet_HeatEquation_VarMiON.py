"""
Create the training set for DKF examples with a VarMiOn surrogate of the Heat Equation

[lavoro M2P 2023]


**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Marco Dell'Orto**
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import scipy.io as sio
import pickle

import mpi4py.MPI as MPI
import dolfinx
import ufl
from dolfinx import mesh, fem
import dolfinx.fem.petsc, petsc4py
from ufl import ds, dx, grad, inner


WorkingDirectory = './'

def create(Experiment='HEV_1',TBegin=0,TEnd=1.,dt=0.1,sigma=10,ro=28,beta=8./3,epsilon=0.01,yBegin=[],sigma_Q=5.*10**-10,sigma_R=5.*10**-10,sigma_P=1.*10**-10):
    # Experiment:
    # "HEV_1": true model, all observable

    TBegin = 0
    TEnd = 1
    dt = 0.1
    N = np.int64(np.floor((TEnd-TBegin)/dt))
    tEuler = np.arange(TBegin,TEnd+dt,dt)
    ntimes = len(tEuler)

    # grid
    n = 10
    v = np.linspace(0,1,n)
    grid_pts = np.array([[v[i], v[j], 0] for j in range(n) for i in range(n)]) # shape (n^2)x3
    #print(f"{grid_pts.shape = }")

    param_pts = grid_pts
    T_pts = grid_pts

    # param boundary points
    m = n
    w = np.linspace(0,1,m)
    bottom = np.column_stack((w, np.zeros(m)))
    top = np.column_stack((w, np.ones(m)))
    left = np.column_stack((np.zeros(m-2), w[1:-1]))
    right = np.column_stack((np.ones(m-2), w[1:-1]))
    param_bry_points = np.vstack((bottom, top, left, right))
    param_bry_pts = np.column_stack((param_bry_points, np.zeros(param_bry_points.shape[0])))
    #print(f"{param_bry_pts.shape = }")

    nx = len(T_pts) # u x_1,...,x_100
    ninput = len(param_pts) + len(param_bry_pts) # f on the grid and g on the bry
    #print(f"{nx = }")
    #print(f"{ninput = }")

    ySolver = np.zeros((len(tEuler), nx))  
    uSolver = np.zeros((nx, len(tEuler)))

    C = np.zeros((4*(n-1), nx)) # 36x100
    C[:n-1, :n-1] = np.eye(n-1)
    C[-n+1:, -n+1:] = np.eye(n-1) #???
    Clist = [n*i-j for i in range(1,n) for j in [0,1]] # 9,10, 19,20,...,89,90
    for i, index in enumerate(Clist):
        C[i + n-1, index] = 1
    #endfor
    ny = np.shape(C)[0]

    #Stochastic matrices
    Q = (sigma_Q**2)*np.eye(nx)
    R = (sigma_R**2)*np.eye(ny)
    P0 = (sigma_P**2)*np.eye(nx)

    ActivateModelNoise = 1
    ActivateMeasNoise = 1
    ActivateFirstStateNoise = 1
    ModelNoise = np.sqrt(Q)@np.random.normal(0,1,(nx,N+1))
    MeasNoise = np.sqrt(R)@np.random.normal(0,1,(ny,N+1))
    FirstStateNoise = np.sqrt(P0)@np.random.normal(0,1,(nx,1))

    # define mesh and function space
    ncells = n-1
    domain = mesh.create_rectangle(comm=MPI.COMM_WORLD, points=[[.0, .0], [1., 1.]], n=[ncells, ncells], cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    #print(domain.geometry.x)

    # if P_i is the i-th points in grid_pts and Q_i is the i-th point in domain.geometry.x, 
    # then P_[i] = Q_phi[i]
    phi = np.zeros(len(grid_pts), dtype = int)
    for i, point in enumerate(grid_pts):
        phi[i] = int(np.where(np.all(point == domain.geometry.x, axis=1) == True)[0][0])
    #endfor

    # if P_i is the i-th points in param_bry_pts and Q_i is the i-th point in domain.geometry.x, 
    # then P_[i] = Q_psi[i]
    psi = np.zeros(len(param_bry_pts), dtype = int)
    for i, point in enumerate(param_bry_pts):
        psi[i] = int(np.where(np.all(point == domain.geometry.x, axis=1) == True)[0][0])
    #endfor

    # sample data
    num_points = len(domain.geometry.x)
    tdim = domain.topology.dim
    fdim = tdim - 1

    length_scale_u0 = 0.2
    length_scale_f = 0.2
    length_scale_theta = 0.4
    length_scale_c = 0.4
    length_scale_g = 0.4

    distances_matrix = np.sqrt(np.sum(
                (domain.geometry.x[:, 0:tdim][:, np.newaxis, :] - domain.geometry.x[:, 0:tdim][np.newaxis, :, :]) ** 2,
                axis=2))

    covariance_matrix_u0 = np.exp(-distances_matrix ** 2 / (2 * length_scale_u0 ** 2))
    covariance_matrix_f = np.exp(-distances_matrix ** 2 / (2 * length_scale_f ** 2))
    covariance_matrix_theta = np.exp(-distances_matrix ** 2 / (2 * length_scale_theta ** 2))
    covariance_matrix_c = np.exp(-distances_matrix ** 2 / (2 * length_scale_c ** 2))
    covariance_matrix_g = np.exp(-distances_matrix ** 2 / (2 * length_scale_g ** 2))

    f_vec = [dolfinx.fem.Function(V) for _ in range(ntimes)] 
    f = dolfinx.fem.Function(V)
    f.name = 'f'

    theta = dolfinx.fem.Function(V)
    theta.name = 'theta'

    c = dolfinx.fem.Function(V)
    c.name = 'c'

    g_vec = [dolfinx.fem.Function(V) for _ in range(ntimes)]
    g = dolfinx.fem.Function(V)
    g.name = 'g'  

    solution_vec = [dolfinx.fem.Function(V) for _ in range(ntimes)]

    if 1: # sample data from GRFs
        theta_np = np.random.multivariate_normal(mean=np.zeros(num_points), cov=covariance_matrix_theta)
        c_np = np.random.multivariate_normal(mean=np.zeros(num_points), cov=covariance_matrix_c)         
        u0_np = np.random.multivariate_normal(mean=np.zeros(num_points), cov=covariance_matrix_u0)

        theta.x.array[:] = (theta_np - np.min(theta_np)) / (np.max(theta_np) - np.min(theta_np)) * 0.97 + 0.02
        c.x.array[:] = (c_np - np.min(c_np)) / (np.max(c_np) - np.min(c_np)) * 0.97 + 0.02
        solution_vec[0].x.array[:] = (u0_np - np.min(u0_np)) / (np.max(u0_np) - np.min(u0_np)) * 0.97 + 0.02

        h = np.float32(np.random.uniform(0.02, 0.99))

        for i in range(1, ntimes):
            f_np = np.random.multivariate_normal(mean=np.zeros(num_points), cov=covariance_matrix_f)
            f_vec[i].x.array[:] = (f_np - np.min(f_np)) / (np.max(f_np) - np.min(f_np)) * 0.97 + 0.02 

            g_np = np.random.multivariate_normal(mean=np.zeros(num_points), cov=covariance_matrix_g)
            g_vec[i].x.array[:] = ((g_np - np.min(g_np)) / (np.max(g_np) - np.min(g_np)) * 0.97 + 0.02)*1.

        #endfor
    else: # take constant data
        theta.x.array[:] = np.ones(num_points)
        c.x.array[:] = np.ones(num_points)
        solution_vec[0].x.array[:] = np.ones(num_points)*1
        solution_vec[0].x.array[0] = 10
        h = np.float32(0.5)

        for i in range(1, ntimes):
            f_vec[i].x.array[:] = np.ones(num_points)*0
            g_vec[i].x.array[:] = np.ones(num_points)*0
        #endfor
    #endif




    # evaluate param data
    # Find cells on which to evaluate param_pts and save them in cells_param_pts
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, param_pts)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, potential_colliding_cells, param_pts)
    cells_param_pts = []
    for i in range(len(param_pts)):
        cells_param_pts.append(colliding_cells.links(i)[0])
    #endfor

    # Find cells on which to evaluate param_bry_pts and save them in cells_param_bry_pts
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, param_bry_pts)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, potential_colliding_cells, param_bry_pts)
    cells_param_bry_pts = []
    for i in range(len(param_bry_pts)):
        cells_param_bry_pts.append(colliding_cells.links(i)[0])
    #endfor

    # Find cells on which to evaluate T_pts and save them in cells_T_pts
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, T_pts)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, potential_colliding_cells, T_pts)
    cells_T_pts = []
    for i in range(len(T_pts)):
        cells_T_pts.append(colliding_cells.links(i)[0])
    #endfor

    Ct = c.eval(x = param_pts, cells = cells_param_pts).astype(np.float32).flatten()
    Theta = theta.eval(x = param_pts, cells = cells_param_pts).astype(np.float32).flatten()

    F = np.zeros((ntimes-1, num_points),dtype=np.float32)
    G = np.zeros((ntimes-1, len(param_bry_pts)),dtype=np.float32)
    U0 = np.zeros(num_points,dtype=np.float32)

    for j in range(ntimes-1):
        F[j] = f_vec[j+1].eval(x = param_pts, cells = cells_param_pts).flatten()
        G[j] = g_vec[j+1].eval(x = param_bry_pts, cells = cells_param_bry_pts).flatten()
    #endfor

    U0 = solution_vec[0].eval(x = T_pts, cells = cells_T_pts).flatten()

    plt.imshow(U0.reshape(10,10), cmap="hot")
    plt.colorbar()


    # find boundary dof
    domain.topology.create_connectivity(fdim, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    degrees_of_freedom = dolfinx.fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)
    #boundary_conditions = dolfinx.fem.dirichletbc(value=petsc4py.PETSc.ScalarType(0), dofs=degrees_of_freedom, V=V)


    # compute matrices

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)


    #M1_form = c * u * v * dx + dt * theta * inner(grad(u), grad(v)) * dx + dt * h * u * v * ds
    #M1_mat = fem.petsc.assemble_matrix(dolfinx.fem.form(M1_form))
    #M1_mat.assemble()
    #M1_mat.convert("dense")
    #M1 = M1_mat.getDenseArray()
    #print(f"{M1.shape = }")


    ### K1(C)
    K1_form = c * u * v * dx
    K1_mat = fem.petsc.assemble_matrix(dolfinx.fem.form(K1_form))
    K1_mat.assemble()
    K1_mat.convert("dense")
    K1 = K1_mat.getDenseArray()
    #print(f"{K1.shape = }")

    ## K2(theta)
    K2_form = theta * inner(grad(u), grad(v)) * dx
    K2_mat = fem.petsc.assemble_matrix(dolfinx.fem.form(K2_form))
    K2_mat.assemble()
    K2_mat.convert("dense")
    K2 = K2_mat.getDenseArray()
    #print(f"{K2.shape = }")

    ## K3(h)
    K3_form = h * u * v * ds
    K3_mat = fem.petsc.assemble_matrix(dolfinx.fem.form(K3_form))
    K3_mat.assemble()
    K3_mat.convert("dense")
    K3 = K3_mat.getDenseArray()
    #print(f"{K3.shape = }")


    M1 = K1 + dt*K2 + dt*K3

    print("M1 nan: ", np.isnan(M1).any())
    print("M1 inf: ", np.isinf(M1).any()) 
    S1 = np.linalg.svd(M1, compute_uv=False)
    print(f"{np.min(S1) = }")

    B2_form = u * v * dx
    B2_mat = fem.petsc.assemble_matrix(dolfinx.fem.form(B2_form))
    B2_mat.assemble()
    B2_mat.convert("dense")
    B2 = B2_mat.getDenseArray()

    B1 = K1
    #B3 = K3[:, degrees_of_freedom]


    print(f"{B1.shape = }")
    print(f"{B2.shape = }")
    #print(f"{B3.shape = }")
    #print(f"{B23.shape = }")



    # we need to change the ordering of the rows and columns of the matrices to coincide with the one of grid_pts

    B3 = K3[phi,:][:, psi]
    K1 = K1[phi,:][:,phi]
    K2 = K2[phi,:][:,phi]
    K3 = K3[phi,:][:,phi]
    M1 = M1[phi,:][:,phi]
    B1 = K1 #B1[phi,:][:,phi]
    B2 = B2[phi,:][:,phi]

    B23 = np.hstack([B2,B3])



    yEuler = U0.reshape(-1,1)
    ySolver[0, :] = np.squeeze(yEuler)

    print(f"{ActivateModelNoise = }")
    print(f"{ActivateMeasNoise = }")
    print(f"{ActivateFirstStateNoise = }")

    for k in range(N):
        #print(f"{k = }")    
        u_k = np.concatenate((F[k], G[k])).reshape(-1,1)    
        uSolver[:,[k]] = dt*B23@u_k

        yEuler = np.linalg.solve(M1, B1@yEuler + uSolver[:,[k]])
        #print(yEuler.T)

        yEuler = yEuler + ActivateModelNoise*ModelNoise[:,[k]];
        ySolver[k+1, :] = np.squeeze(yEuler)
    #endfor

    AInit = np.eye(nx) # ?

    #Set up training set
    Meas = C@ySolver.T + ActivateMeasNoise*MeasNoise
    plt.figure(2)
    plt.plot(tEuler, Meas.T)
    plt.title('Noisy measurements')
    #plt.legend()
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
        TrainingSet[5][i] = 1 # ?
        #TrainingSet[6][i] = (Ct, Theta, F.T, h, G.T) # qui?


    #endfor
    #print(TrainingSet[5][0])


    #Save data
    sio.savemat(WorkingDirectory+'Experiment.mat', {'Experiment': Experiment})
    sio.savemat(WorkingDirectory+'LayersExp'+Experiment+'.mat', {'Layers': N})

    with open(WorkingDirectory+'LatestTrainingSetExp'+Experiment+'.mat', 'wb') as handle:
                pickle.dump(TrainingSet, handle, protocol=pickle.HIGHEST_PROTOCOL)


    Model = {}

    if 1:
        nxm = nx
        Model['xrMask'] = range(nx)
        Cm = C.copy()
        sio.savemat(WorkingDirectory+'CExp'+Experiment+'.mat', {'C': Cm})
        Model['QInit'] = Q
        Model['RInit'] = np.atleast_2d(R)
        Model['invRInit'] = np.atleast_2d(np.linalg.inv(R))
        print(f"{R.shape = }")
        Model['PInit'] = P0
        Model['AInit'] = AInit
        HiddenDynDim = 0
        Model['M'] = M1 #np.eye(nx) #MassMat ?
        Model['Mtrue'] = M1 #MassMat
        Model['K'] = B1 #LinMat
    Model['D'] = np.tile(np.zeros((nxm,1)),(1,N)); #np.tile(KnownOffset,(1,N))
    Model['SamplingTimes'] = dt*np.ones((N,1))
    Model['HeatParameters'] = (Ct, Theta, F, h, G)
    Model['StateMats_M1B1'] = (M1, B1)

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

