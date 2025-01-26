from math import log
import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt

def logit(p):
    # On s'assure que p est entre 0 et 1
    if p <= 0 or p >= 1:
        raise ValueError("p doit être compris entre 0 et 1 (e)")

    return np.log(p / (1 - p))

def matrice_Q(mat_N) :
  n = len(mat_N)
  Q = np.zeros((n,n))

  for i in range(n):
    for j in range(n):
      if i != j :
        nij=mat_N[i,j]
        mij=mat_N[i,j]+mat_N[j,i]
        Q[i,j] = logit(nij/mij)**2
      else :
        Q[i,j] = 0

  return Q

def Q_c(Q):
  """Applique le centrage en ligne et colonne"""
  n = Q.shape[0]
  In = np.eye(n)
  Jn = np.ones((n, n))

  return (In - (1/n) * Jn) @ Q @ (In - (1/n) * Jn)


# Fonction pour normaliser les vecteurs colonnes
def normaliser_vecteurs_propres(vecteurs_propres):
    # Normaliser chaque vecteur colonne
    for i in range(vecteurs_propres.shape[1]):
        vecteur = vecteurs_propres[:, i]
        norme = np.linalg.norm(vecteur)  # Calcul de la norme du vecteur
        vecteurs_propres[:, i] = vecteur / norme  # Normalisation du vecteur
    return vecteurs_propres

def deux_plus_grandes_valeurs_propres(valeurs_propres, vecteurs_propres):
    # Filtrer les valeurs propres strictement positives et leurs indices
    indices_positifs = np.where(valeurs_propres > 0)[0]
    valeurs_propres_positives = valeurs_propres[indices_positifs]
    vecteurs_propres_positifs = vecteurs_propres[:, indices_positifs]

    # Trier les valeurs propres positives par ordre décroissant et garder les deux plus grandes
    indices_triees = np.argsort(valeurs_propres_positives)[::-1]  # Tri décroissant
    indices_top2 = indices_triees[:2]  # Garder les deux premières (les plus grandes)

    # Garder les deux plus grandes valeurs propres et leurs vecteurs associés
    top2_valeurs_propres = valeurs_propres_positives[indices_top2]
    top2_vecteurs_propres = vecteurs_propres_positifs[:, indices_top2]

    # Calculer le produit des racines des deux plus grandes valeurs propres avec leurs vecteurs
    racines_valeurs_propres = np.sqrt(top2_valeurs_propres)  # Racine carrée des deux plus grandes valeurs propres
    vecteurs_resultants = top2_vecteurs_propres * racines_valeurs_propres  # Multiplication vecteurs * racines
    # Empiler les deux vecteurs propres pour former un seul vecteur colonne
    vecteur_final = np.hstack((vecteurs_resultants[:,0],vecteurs_resultants[:,1])) 

    return vecteur_final.reshape(-1,1)

def starting_point(mat_N):
  Q = matrice_Q(mat_N)
  Qc= Q_c(Q)
  valeurs_propres, vecteurs_propres = np.linalg.eig((-1/2)*Qc)
  vect_prop_norm = normaliser_vecteurs_propres(vecteurs_propres)
  return deux_plus_grandes_valeurs_propres(valeurs_propres,vect_prop_norm) ## retourne vecteur colonne lambda 0