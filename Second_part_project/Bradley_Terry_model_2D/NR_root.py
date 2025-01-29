from First_part_project.Bradley_Terry_Model_2D.fonctions import d1_L_etoile,d_phi,second_derivative_L_star, phi
from First_part_project.Bradley_Terry_Model_2D.NR_algorithm import starting_point

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

def NR_root(initial,N) :
    n=len(N) 
    initial_0 = np.random.uniform(-5, 5, size=(2*n)).reshape(-1,1)  # Initial guess for lambda exploded
    vecteur_depart = starting_point(N)
    zeros = np.zeros(3).reshape(-1,1)
    initial = np.vstack([vecteur_depart, zeros])
    sol = optimize.root(fun, initial ,args = (N,), method='krylov', jac=jac, options={'xtol': 1e-9, 'ftol': 1e-9, 'maxiter': 500}) ## better krylov
    return sol.x