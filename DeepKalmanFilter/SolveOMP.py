'''
https://github.com/NLALDlab/DeepKalmanFilter/DeepKalmanFilter/SolveOMP.py

E. Chinellato and F. Marcuzzi: State, parameters and hidden dynamics estimation with the Deep Kalman Filter: Regularization strategies. *Journal of Computational Science* **87** (2025), Article number 102569.
10.1016/j.jocs.2025.102569

**SPDX-License-Identifier: GPL-3.0**  
**Copyright (c) 2025 NLALDlab**  
**Authors: Erik Chinellato, Fabio Marcuzzi**
'''
import numpy as np
from scipy.linalg import cho_solve, cho_factor

def SolveOMP(A, y, N, maxIters=None, lambdaStop=0, solFreq=0, verbose=0, OptTol=1e-5):
    # SolveOMP: Orthogonal Matching Pursuit
    # Usage
    #    [sols, iters, activationHist] = SolveOMP(A, y, N, maxIters, lambdaStop, solFreq, verbose, OptTol)
    # Input
    #    A           Either an explicit nxN matrix, with rank(A) = min(N,n) 
    #                by assumption, or a string containing the name of a 
    #                function implementing an implicit matrix (see below for 
    #                details on the format of the function).
    #    y           vector of length n.
    #    N           length of solution vector. 
    #    maxIters    maximum number of iterations to perform. If not
    #                specified, runs to stopping condition (default)
    #    lambdaStop  If specified, the algorithm stops when the last coefficient 
    #                entered has residual correlation <= lambdaStop. 
    #    solFreq     if =0 returns only the final solution, if >0, returns an 
    #                array of solutions, one every solFreq iterations (default 0). 
    #    verbose     1 to print out detailed progress at each iteration, 0 for
    #                no output (default)
    #    OptTol      Error tolerance, default 1e-5
    # Outputs
    #    sols            solution(s) of OMP
    #    iters           number of iterations performed
    #    activationHist  Array of indices showing elements entering  
    #                    the solution set
    # Note: The translation assumes that the A matrix is either an explicit matrix or an implicit operator implemented as a callable function. In the latter case, you need to define the function A(mode, m, n, x, I, dim) that takes inputs mode (1 or 2), m, n, x, I, and dim, and returns y = A(:,I) @ x if mode = 1 or y = A(:,I).T @ x if mode = 2. You'll need to implement this function separately based on your requirements.
    
    if maxIters is None:
        maxIters = len(y)
    #endif
    if np.ndim(y) == 1:
        y = np.atleast_2d(y).T
    #endif
    explicitA = not (isinstance(A, str) or callable(A)); 
    if verbose: print("explicitA = ",explicitA)
    n = len(y)
    
    # Parameters for linsolve function
    # Global variables for linsolve function
    opts = {'UT': True}
    opts_tr = {'UT': True, 'TRANSA': True}
    
    # Initialize
    x = np.zeros((N,1))
    k = 1
    R_I = np.empty((0, 0))
    activeSet = np.empty(0, dtype=int)
    sols = []
    res = y.copy()
    normy = np.linalg.norm(y)
    resnorm = normy
    done = False
    fatal_error = False
    
    while not done and not fatal_error:
        if explicitA:
            corr = A.T @ res
        else:
            corr = A(2, n, N, res, activeSet, N)  # = A'*y
        #endif
        maxcorr = np.max(np.abs(corr))
        if verbose: print("maxcorr = ",maxcorr)
        newIndex = np.argmax(np.abs(corr))
        #print("newIndex = ",newIndex)
        # Update Cholesky factorization of A_I
        #print("activeSet.shape: ",activeSet.shape)
        R_I, flag = updateChol(R_I, n, N, A, explicitA, activeSet, newIndex)
        #print("R_I.shape: ",R_I.shape)
        #print("R_I: \\",R_I)
        if flag == 1:
            print('ERROR: Collinear vector!')
            fatal_error = True
            break
        #endif
        activeSet = np.append(activeSet, newIndex)
        if verbose: print("activeSet: ",activeSet)
        
        dx = np.zeros((N,1))
        z = np.linalg.solve(R_I.T, corr[activeSet]) #cho_solve((R_I.T, True), corr[activeSet]) #, **opts_tr)
        #print("z solve = ",np.linalg.solve(R_I.T, corr[activeSet]))
        #print("z: ",z)
        dx[activeSet] = np.linalg.solve(R_I, z) # cho_solve((R_I, False), z) #, **opts)
        #print("dx[activeSet]: ",dx[activeSet])
        x[activeSet] += dx[activeSet]
        
        if explicitA:
            res = y - A[:, activeSet] @ x[activeSet]
        else:
            Ax = A(1, n, N, x, activeSet, N)
            res = y - Ax
        #endif
        resnorm = np.linalg.norm(res)
        if verbose: print("resnorm=",resnorm)
        if (resnorm <= OptTol * normy) or ((lambdaStop > 0) and (maxcorr <= lambdaStop)):
            print("done: resnorm=",resnorm," <= OptTol=",OptTol," * normy=",normy," OR ((lambdaStop=",lambdaStop," > 0) and (maxcorr=",maxcorr," <= lambdaStop))")
            done = True
        #endif
        if verbose:
            print(f'Iteration {k}: Adding variable {newIndex}')
        #endif
        k += 1
        if k >= maxIters:
            print("k >= maxIters!")
            done = True
        #endif
        if done or ((solFreq > 0) and (k % solFreq == 0)):
            sols.append(x.copy())
        #endif
    if fatal_error:
        sols.append(np.zeros(N))
    #endif
    iters = k
    activationHist = activeSet
    
    return x, iters, activationHist


def updateChol(R, n, N, A, explicitA, activeSet, newIndex):
    # updateChol: Updates the Cholesky factor R of the matrix 
    # A(:,activeSet)'*A(:,activeSet) by adding A(:,newIndex)
    # If the candidate column is in the span of the existing 
    # active set, R is not updated, and flag is set to 1.
    # NB: the Cholesky factor is upper triangular.
    
    flag = 0
    machPrec = 1e-15 # era 1e-5;
    
    if explicitA:
        newVec = A[:, newIndex]
    else:
        e = np.zeros(N)
        e[newIndex] = 1
        newVec = A(1, n, N, e, activeSet, N)
    #endif
    #print("newVec = ",newVec)
    #print("len(activeSet) = ",len(activeSet))
    if len(activeSet) == 0:
        R = np.atleast_2d(np.sqrt(np.sum(newVec ** 2)))
    else:
        if explicitA:
            #print("R = ",R)
            #print("newVec.shape: ",newVec.shape)
            #print("A[:, [newIndex]].shape: ",A[:, [newIndex]].shape)
            p = np.linalg.solve(R.T, A[:, activeSet].T @ A[:, [newIndex]]) #cho_solve((R, False), A[:, activeSet].T @ A[:, newIndex]) #, **opts_tr)
        else:
            AnewVec = A(2, n, len(activeSet), newVec, activeSet, N)
            p = cho_solve((R, False), AnewVec) #, **opts_tr)
        #endif
        #print("p = ",p)
        q = np.sum(newVec ** 2) - np.sum(p ** 2)
        #print("q = ",q)
        if q <= machPrec:  # Collinear vector
            flag = 1
        else:
            #q = np.atleast_2d(q); p = np.atleast_2d(p); 
            #print("R.shape = ",R.shape," , p.shape = ",p.shape," , q.shape = ",q.shape)
            # Assuming R and p are numpy arrays and q is a scalar value

            # Get the dimensions of R
            rows, cols = R.shape

            # Create a row of zeros with the same number of columns as R
            zeros_row = np.zeros((1, cols))

            # Calculate the square root of q
            sqrt_q = np.sqrt(q)

            # Concatenate R, p, zeros_row, and sqrt_q
            R = np.hstack((R, p.reshape(cols, 1)))
            #print("R.shape = ",R.shape)
            tmp = np.concatenate((zeros_row, sqrt_q.reshape(1, 1)), axis=1)
            #print("tmp.shape = ",tmp.shape)
            R = np.vstack((R, tmp))
        #endif
    #endif
    return R, flag

