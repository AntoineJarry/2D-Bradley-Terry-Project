import sys
import os

# Récupérer le chemin absolu du dossier racine du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Ajouter ce chemin à sys.path
sys.path.append(project_root)

from First_part_project.Bradley_Terry_Model_2D.fonctions import d1_L_etoile,d_phi,second_derivative_L_star, phi
from First_part_project.Bradley_Terry_Model_2D.NR_algorithm.starting_point import starting_point

import numpy as np
from scipy import optimize

def fun(x,N):
    n=len(N)
    a_current = x[2*n:,].reshape(-1,1)
    lamb_current = x[:2*n,].reshape(-1,1)
    optim = d1_L_etoile(N,lamb_current)+d_phi(lamb_current)@a_current
    optim_bis = phi(a_current)
    print(optim_bis)
    print(optim)
    # Crucial change: Concatenate the flattened arrays
    return np.concatenate((optim.flatten(), optim_bis.flatten()))


def jac(x,N):
  n=len(N)
  param_current = x
  lamb_current = param_current[0:2*n].reshape(-1,1)
  matrice = np.block([[np.zeros((n,n)),np.eye(n)],
                       [np.eye(n),np.zeros((n,n))]])
  a = second_derivative_L_star(N,lamb_current)+param_current[-1,]*matrice
  b =  d_phi(lamb_current)
  c =  np.transpose(d_phi(lamb_current))
  d = np.zeros((3,3))
  return np.block([[a, b], [c, d]])

def NR_root(N) :
    n=len(N) 
    initial_0 = np.random.uniform(-5, 5, size=(2*n)).reshape(-1,1)  # Initial guess for lambda exploded
    vecteur_depart = starting_point(N,False,True)
    zeros = np.zeros(3).reshape(-1,1)
    initial = np.vstack([vecteur_depart, zeros])
    sol = optimize.root(fun, initial ,args = (N,), method='krylov', jac=jac, options={'xtol': 1e-9, 'ftol': 1e-9, 'maxiter': 500}) ## better krylov
    print(sol.status)
    print(sol.message)
    return sol.x 
    

"""croquettes_dog = ['SPF2', 'SPF4', 'BENCH4', 'SPF1', 'SPF3', 'BENCH1', 'BENCH2', 'BENCH3']
mat_comp_dog_opt3 = np.array([
    [ 0, 28, 24, 33, 21, 11, 20, 27],
    [10,  0, 12, 18, 20,  6, 18, 11],
    [16, 21,  0, 16, 17, 12,  8, 17],
    [ 2, 11, 20,  0,  9,  5, 18, 17],
    [15, 16, 18, 15,  0,  8, 15, 20],
    [23, 31, 23, 31, 33,  0, 16, 24],
    [17, 19, 27, 23, 17, 24,  0, 13],
    [14, 26, 23, 19, 16, 11, 18,  0]
])

NR_root(mat_comp_dog_opt3)"""