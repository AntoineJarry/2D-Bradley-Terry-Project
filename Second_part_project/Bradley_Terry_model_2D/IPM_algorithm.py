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

def IPM_algorithm(N,a0,lam0,method) : 
    initial_guess = np.concatenate([lam0.flatten(), a0.flatten()])
    constraints = {'type': 'eq', 'fun': eq_constraint,'args': (N,)}

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

# Tests (à conserver pour le moment, à supprimer à la fin)
"""
N = np.array([
  [0, 39, 64, 40, 61, 76, 46],
  [61, 0, 65, 59, 55, 85, 60],
  [36, 35, 0, 31, 25, 41, 35],
  [60, 41, 69, 0, 41, 80, 28],
  [39, 45, 75, 59, 0, 71, 37],
  [24, 15, 59, 20, 29, 0, 18],
  [54, 40, 65, 72, 63, 82, 0]])

lambda_0 = starting_point.starting_point(N, False, True)
a_0 = np.zeros((3,1))

res = IPM_algorithm(N,a_0,lambda_0,'trust-constr')

# Extraire les paramètres optimaux depuis `res`
n = len(N)  # Taille de la matrice N
optimized_params = res.x  # Prendre les valeurs optimisées
optimal_lambda = optimized_params[:-3].reshape(n, 2)  # Reshape en (n,2)

labels = ['1', '2', '3', "4", "5", '6', '7']
# Annotating points with labels
for i, label in enumerate(labels):
    plt.text(optimal_lambda[:, 1][i] + 0.02, -optimal_lambda[:, 0][i] + 0.02, label, fontsize=12)

# Tracer optimal_lambda[:, 0] contre optimal_lambda[:, 1]
plt.scatter(optimal_lambda[:, 1], -optimal_lambda[:, 0], color='b', label=labels)
plt.xlabel("Lambda 1")
plt.ylabel("Lambda 2")
plt.title("Optimized Lambda Values")
plt.legend()
plt.grid(True)
plt.show()
"""