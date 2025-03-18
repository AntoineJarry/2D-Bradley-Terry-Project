import sys
import os
from scipy.special import factorial

# Récupérer le chemin absolu du dossier racine du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Ajouter ce chemin à sys.path
sys.path.append(project_root)

import First_part_project.Bradley_Terry_Model_2D.fonctions as fonctions
import First_part_project.Bradley_Terry_Model_2D.NR_algorithm.starting_point as starting_point
from math import  comb
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt  


# Objective function: Log-likelihood + penalty terms
def objective(params,N):
    n = len(N)
    lam = params[:-3].reshape(n, 2)  # Extract lambda
    a = params[-3:]                 # Extract a

    nij = N
    mij  = nij + nij.T
    likelihood = 0
    for i in range(n):
        for j in range(i + 1, n):
            pi_ij = fonctions.inv_logit(fonctions.logit_pi_ij(nij, i, j, lam))
            likelihood += nij[i, j] * np.log(pi_ij) + (mij[i, j] - nij[i, j]) * np.log(1 - pi_ij)
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

def IPM_algorithm(N,a0,lam0,method): 
    initial_guess = np.concatenate([lam0.flatten(), a0.flatten()])
    constraints = {'type': 'eq', 'fun': eq_constraint,'args': (N,)}

    # Optimization
    result = minimize(fun=objective, x0= initial_guess,args = (N,), method=method, constraints=constraints) ## add jacobian and hess
    
    return result

def IPM_multistart(N, method, num_starts=10):
    best_lambda_0 = None
    best_V = -np.inf  # On cherche à maximiser la log-vraisemblance
    a_0 = np.zeros((3, 1))  # Initialisation de a_0

    for _ in range(num_starts):
        lambda_candidate = np.random.uniform(0, 1, size=(2*len(N), 1))  # tester np.random.normal ?
        res = IPM_algorithm(N, a0=a_0, lam0=lambda_candidate, method=method)  # Exécuter IPM
        
        if res.success:  # Vérifier que l'optimisation a convergé
            log_likelihood = -res.fun  # Extraire la log-vraisemblance
            
            if log_likelihood > best_V:  # Met à jour si meilleur score
                #best_lambda_0 = lambda_candidate
                best_V = log_likelihood
                best_res = res

    return best_V, best_res