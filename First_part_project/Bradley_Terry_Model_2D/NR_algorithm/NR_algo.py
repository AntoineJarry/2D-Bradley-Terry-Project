from math import log
import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt

import First_part_project.Bradley_Terry_Model_2D.fonctions as fonctions

def newton_raphson(mat_N,lamb_current, a_current, tol=1e-6, max_iter=250):
    """
    Implémentation de l'algorithme de Newton-Raphson page 251

    Paramètres:
    - lamb_current, a_current: où on a des vecteurs colonnes comme dans le papier  estimation initiale (point de départ)
    - tol : tolérance pour la convergence (critère d'arrêt)
    - max_iter : nombre maximum d'itérations

    Retourne:
    - vecteur colonne de 2n+3 paramètres estimés du vecteur colonne lamb et a
    """
    n=len(mat_N) # recup le nombre de ligne
    param_current=np.vstack((lamb_current,a_current)) # empile sur un seul vecteur colonne de taille en tout 2n+3
    matrice = np.block([[np.zeros((n,n)),np.eye(n)],
                       [np.eye(n),np.zeros((n,n))]]) # matrice taille 2n carré

    count=0
    for i in range(max_iter):
      count = i
      inverse_part = np.linalg.inv(np.block([[fonctions.second_derivative_L_star(mat_N,lamb_current)+param_current[-1,]*matrice, fonctions.d_phi(lamb_current)],
                                                             [np.transpose(fonctions.d_phi(lamb_current)), np.zeros((3,3))]])) ## matrice de taille 2n+3 carré
      param_new = param_current - inverse_part@np.block([[fonctions.d1_L_etoile(mat_N,lamb_current)+fonctions.d_phi(lamb_current)@a_current],
                                                                                                 [fonctions.phi(lamb_current)]]) ## vecteur colonne de 2n +3 lignes

      # Afficher les détails à chaque étape
      # print(f"Iteration {i+1}: param_current = {param_current}, param_new = {param_new}")
      # print("param_new: ",param_new)
      # Vérifier la convergence (arrêt si la différence est petite)
      if np.all(abs(param_new - param_current) < tol):
          print(f"Converged after {i+1} iterations")
          mat_cov_var = fonctions.extract_submatrix(np.linalg.inv(np.block([[-fonctions.second_derivative_L_star(mat_N,param_new[0:2*n])-param_new[-1,]*matrice, fonctions.d_phi(param_new[0:2*n])],
                                                             [np.transpose(fonctions.d_phi(param_new[0:2*n])), np.zeros((3,3))]])), n) # moins en haut à gauche propt silvey
          return param_new , mat_cov_var
      mat_cov_var = fonctions.extract_submatrix(np.linalg.inv(np.block([[-fonctions.second_derivative_L_star(mat_N,param_new[0:2*n])-param_new[-1,]*matrice, fonctions.d_phi(param_new[0:2*n])],
                                                             [np.transpose(fonctions.d_phi(param_new[0:2*n])), np.zeros((3,3))]])), n) # moins en haut à gauche propt silvey
      param_current = param_new
      lamb_current= param_current[:2*n]
      a_current = param_current[-3:]

    print("Reached maximum iterations without convergence.")
    return param_current, mat_cov_var