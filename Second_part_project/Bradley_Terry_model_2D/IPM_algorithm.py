from First_part_project.Bradley_Terry_Model_2D.fonctions import inv_logit,logit_pi_ij 
from math import factorial, comb
import numpy as np
from scipy import minimize

# Objective function: Log-likelihood + penalty terms
def objective(params,N):
    lam = params[:-3].reshape(n, 2)  # Extract lambda
    a = params[-3:]                 # Extract a

    nij = N
    mij  = nij + nij.T
    likelihood = 0
    n = len(N)
    for i in range(n):
        for j in range(i + 1, n):
            pi_ij = inv_logit(logit_pi_ij(nij, i, j, lam))
            likelihood += np.log(factorial(mij)/(factorial(mij-nij)*factorial(nij)))+ nij[i, j] * np.log(pi_ij) + (mij[i, j] - nij[i, j]) * np.log(1 - pi_ij)
    # Penalty term
    penalty = a[0] * np.sum(lam[:, 0]) + a[1] * np.sum(lam[:, 1]) + a[2] * np.sum(lam[:, 0] * lam[:, 1])
    return -(likelihood + penalty)

# Equality constraint: phi(lambda) = 0
def eq_constraint(params,N):
    n = len(N)
    lam = params[:-3].reshape(n, 2)
    phi = np.array([
        np.sum(lam[:, 0]),                # Sum of lambda_i,1
        np.sum(lam[:, 1]),                # Sum of lambda_i,2
        np.sum(lam[:, 0] * lam[:, 1])     # Sum of lambda_i,1 * lambda_i,2
    ])
    return phi

def IPM_algorithm(N,a0,lam0,method) : 
    initial_guess = np.concatenate([lam0, a0])
    constraints = {'type': 'eq', 'fun': eq_constraint}

    # Optimization
    result = minimize(objective, initial_guess,args = (N,), method=method, constraints=constraints) ## add jacobian and hess
    n=len(N)
    # Results
    if result.success:
        optimized_params = result.x
        optimal_lambda = optimized_params[:-3].reshape(n, 2)
        optimal_a = optimized_params[-3:]
        print("Optimal lambda (10 x 2):\n", optimal_lambda)
        print("Optimal a (3 values):\n", optimal_a)
        print("Maximum log-likelihood:", -result.fun)
    else:
        print("Optimization failed:", result.message)
    
    return result
    


