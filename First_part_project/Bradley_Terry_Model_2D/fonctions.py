from math import log
import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt

def estim_sigma(mat_N,i,j):
  'estimation de sigma comme défini p250'
  if(mat_N[i,j]>=mat_N[j,i]):
    return 1
  else :
    return -1
  
# Fonction qui défini l'inverse du logit (sigmoid)
def inv_logit(x):
    """
    Calcule la fonction sigmoïde pour une entrée x.

    Paramètre:
    - x : correspondra à logit(pi) dans notre cas ici défini page 248

    Retourne:
    - La valeur de la fonction sigmoïde pour x
    """
    return 1 / (1 + np.exp(-x))

# Fonction logit pi defini p248
def logit_pi_ij(mat_N,i, j, lamb):
    """
    Calcule logit(π_ij) pour deux objets i et j. défini page 248

    Paramètres :
    - mat_N : tableau de comparaison (np.array matrice)
    - i,j : index pour n parmis dans la coordonnée lambda 1 et lambda 2 (sur lesquels on va boucler)
    - lamb : matrice n lignes et 2 colonnes (1ère colonne lambda 1 et 2ème lambda 2)

    Retourne :
    - valeur dans R
    """
    diff_1 = lamb[i, 0] - lamb[j, 0]
    diff_2 = lamb[i, 1] - lamb[j, 1]
    distance = np.sqrt(diff_1**2 + diff_2**2)
    logit_value = estim_sigma(mat_N,i,j) * distance
    return logit_value

def d1_logit_ij(mat_N, i, j, k, r, lamb):
    """
    Calcul de la dérivée première du logit(pij) par rapport à lambda_k défini p258 appendix A

    Paramètres :
    - k : index allant de 0 à n-1 selon le type choisi
    - r : index soit 0 ou 1 pour soit lambda 1 ou lambda 2
    - mat_N : tableau de comparaison (np.array matrice)
    - i,j : index pour n parmis dans la coordonnée lambda 1 et lambda 2 (sur lesquels on va boucler)
    - lamb : matrice n lignes et 2 colonnes (1ère colonne lambda 1 et 2ème lambda 2)

    Retourne :
    - valeur dans R
    """
    n=len(mat_N)
    if k == i or k == j:
        delta_r = lamb[i,r] - lamb[j,r]
        denom = np.sqrt((lamb[i,0] - lamb[j,0])**2 + (lamb[i,1] - lamb[j,1])**2)
        if k == i:
            return estim_sigma(mat_N,i,j)*(delta_r / denom)
        elif k == j:
            return -estim_sigma(mat_N,i,j)*(delta_r / denom)
    return 0

def d1_L_etoile(mat_N,lamb):

    """
    Calcul de la dérivée première de L* par rapport à lambda_k défini p258 appendix A

    Paramètres :
    - mat_N : tableau de comparaison (np.array matrice)
    - lamb : vecteur colonne avec lambda 1 de 0 à n-1 et lambda 2 de n à 2n-1 en index

    Retourne :
    - vecteur colonne de taille 2n des dérivées de L star en fonction des différents types de lambdas et pour lambda 1 et 2
    """
    n_lignes, n_colonnes = mat_N.shape
    n_total = len(mat_N) ## récupère n (par exemple le nombre de type différent de cornflakes comparés)

    # Initialiser la matrice des dérivées avec des zéros
    L_star_derivatives = np.zeros((n_total, 2)) # pour stocker toutes nos valeurs
    lamb = np.hstack((lamb[0:n_total,], lamb[n_total:2*n_total,])) # retransforme en une matrice 2 colonnes  pour l'indexation plus clair
    # Pour chaque k et pour r = 0, 1
    for k in range(n_total):
        for r in [0, 1]:
            L_star = 0
            for i in range(n_total):
                for j in range(i + 1, n_total):
                    nij = mat_N[i,j]
                    mij = mat_N[i,j]+mat_N[j,i] ## le nombre de comparaisons total effectuées
                    pij = inv_logit(logit_pi_ij(mat_N,i, j, lamb))
                    d1_logit = d1_logit_ij(mat_N,i, j, k, r, lamb)
                    L_star += (nij - mij*pij) * d1_logit
            L_star_derivatives[k,r] = L_star
    a = np.vstack((L_star_derivatives[:,0].reshape(-1, 1),L_star_derivatives[:,1].reshape(-1, 1)))
    return a

def u_r_s(i, j, lamb, r, s): # r et s qui valent 1 ou 2 et i,j vont de 0 à n-1
    """
    Calcule la valeur de u_{r,s}(i,j) selon les conditions données p259 appendix A

    Paramètres :
    - r,s : index soit 1 ou 2 pour soit lambda 1 ou lambda 2
    - i,j : index pour n parmis dans la coordonnée lambda 1 et lambda 2 (sur lesquels on va boucler)
    - lamb : matrice n lignes et 2 colonnes (1ère colonne lambda 1 et 2ème lambda 2)

    Retourne :
    - valeur dans R
    """

    diff_1 = lamb[i, 0] - lamb[j, 0]
    diff_2 = lamb[i, 1] - lamb[j, 1]

    if r == s == 1:
        return diff_2 ** 2
    elif r == s == 2:
        return diff_1 ** 2
    else:
        return -diff_1 * diff_2

def d2_logit_ij(mat_N, i, j, k, r,s,l,lamb):
    """
    Calcule la valeur des dérivées secondes de logit(pij) comme défini p259.


    Paramètres :
    - k,l : index allant de 0 à n-1 selon le type choisi
    - r,s : index soit 1 ou 2 pour soit lambda 1 ou lambda 2
    - mat_N : tableau de comparaison (np.array matrice)
    - i,j : index pour n parmis dans la coordonnée lambda 1 et lambda 2 (sur lesquels on va boucler)
    - lamb : matrice n lignes et 2 colonnes (1ère colonne lambda 1 et 2ème lambda 2)

    Retourne:
    - valeur dans R
    """
    distance = (lamb[i, 0] - lamb[j, 0]) ** 2 + (lamb[i, 1] - lamb[j, 1]) ** 2 # la partie dénominateur du calcul

    if k == i and l == j:
      return -estim_sigma(mat_N, i, j) * u_r_s(i, j, lamb, r + 1, s + 1) / (distance ** (3 / 2))
    elif k == j and l == i:
      return -estim_sigma(mat_N, i, j) * u_r_s(i, j, lamb, r + 1, s + 1) / (distance ** (3 / 2))
    elif l == k== i or l==k==j :
      return estim_sigma(mat_N, i, j)* u_r_s(i, j, lamb, r + 1, s + 1) / (distance ** (3 / 2))
    else :
      return 0

def second_derivative_L_star(mat_N,lamb):
    """
    Calcule la matrice des dérivées secondes de L* comme défini p259.

    Paramètres:
    - mat_N : tableau de comparaison (np.array matrice)
    - lamb : vecteur colonne avec lambda 1 de 0 à n-1 et lambda 2 de n à 2n-1 en index

    Retourne:
    - hessian : matrice de taille 2n carré des dérivées secondes de L* par rapport à toutes les combinaisons possibles de lambda
    """

    n_total = mat_N.shape[0]  # Nombre total de lignes dans mat_N
    hessian = np.zeros((n_total * 2, n_total * 2))  # Matrice de Hessian pour stocker nos dérivées secondes
    lamb = np.atleast_2d(lamb).T if lamb.ndim == 1 else lamb
    lamb = np.hstack((lamb[0:n_total,:], lamb[n_total:2*n_total,:]))  # retransforme en une matrice 2 colonnes  pour l'indexation plus clair

    for k in range(n_total):
                for l in range(n_total):
                    for r in [0, 1]:
                        for s in [0, 1]:
                          hess = 0
                          for i in range(n_total):
                              for j in range(i + 1, n_total):
                                  nij = mat_N[i, j]
                                  mij = mat_N[i, j] + mat_N[j, i]
                                  pi_ij = inv_logit(logit_pi_ij(mat_N,i, j, lamb))
                                  hess += -mij*pi_ij*(1-pi_ij)*d1_logit_ij(mat_N, i, j, k, r, lamb)*d1_logit_ij(mat_N, i, j, l, s, lamb)+ (nij-mij*pi_ij)*d2_logit_ij(mat_N, i, j, k, r,s,l,lamb)
                          if s==1 and r==1 :
                            hessian[n_total+k, n_total+l ] = hess
                          elif s== 1 and r==0 :
                            hessian[k,n_total+l] = hess
                          elif s== 0 and r== 1 :
                            hessian[n_total+k,l]= hess
                          else :
                            hessian[k,l]=hess
    return hessian

def phi(lamb):
  """
  Calcul phi défini p250

  Paramètres :
  -  lamb : vecteur colonne avec lambda 1 de 0 à n-1 et lambda 2 de n à 2n-1 en index

  Retourne  :
  - Vecteur colonne de 3 lignes
  """
  n=int(len(lamb)/2)
  return (np.array([[np.sum(lamb[0:n,])],
                            [np.sum(lamb[n:2*n, ])],
                            [np.sum(lamb[0:n,] * lamb[n:2*n,])]]))

def d_phi(lamb):
    """
  Calcul dérivée de phi défini p251

  Paramètres :
  -  lamb : vecteur colonne avec lambda 1 de 0 à n-1 et lambda 2 de n à 2n-1 en index

  Retourne  :
  - matrice 3 colonnes et 2n lignes
  """
    n = int(len(lamb)/2)

    # Créer des vecteurs colonnes
    vec1 = np.ones((n, 1))         # Vecteur colonne de 1 (1_n)
    vec0 = np.zeros((n, 1))        # Vecteur colonne de 0 (0_n)
    vec_lamb1 = lamb[0:n,]  # Vecteur colonne pour lamb[:, 1]
    vec_lamb2 = lamb[n:2*n,] # Vecteur colonne pour lamb[:, 0]
    vec_lamb2 = vec_lamb2.reshape(-1, 1)
    vec_lamb1 = vec_lamb1.reshape(-1, 1)
    #print(f"vec0 shape: {vec0.shape}, vec1 shape: {vec1.shape}, vec_lamb1 shape: {vec_lamb1.shape}, vec_lamb2 shape: {vec_lamb2.shape}")
    
    # Création des deux lignes avec np.hstack
    ligne1 = np.hstack((vec1, vec0, vec_lamb2))  # Première ligne
    ligne2 = np.hstack((vec0, vec1, vec_lamb1))  # Deuxième ligne

    # Empiler verticalement les deux lignes pour créer la matrice finale
    D_phi = np.vstack((ligne1, ligne2))
    return D_phi

def extract_submatrix(A, n, corner='top-left'): ## Pour récupérer la matrice de variance covariance
    """
    Extrait une sous-matrice carrée de taille 2n à partir d'une matrice de taille (2n+3) x (2n+3).

    Arguments:
    - A : np.ndarray
        La matrice carrée de taille (2n+3) x (2n+3).
    - n : int
        Taille de la sous-matrice (2n x 2n).
    - corner : str
        Coin de la sous-matrice à extraire. Peut être 'top-left' ou 'top-right'.

    Retourne :
    - submatrix : np.ndarray
        La sous-matrice extraite de taille 2n x 2n.
    """
    # Taille de la matrice d'entrée
    size = 2 * n + 3

    if corner == 'top-left':
        # Extraire la sous-matrice du coin haut-gauche
        submatrix = A[:2*n, :2*n]
    elif corner == 'top-right':
        # Extraire la sous-matrice du coin haut-droit
        submatrix = A[:2*n, -2*n:]
    else:
        raise ValueError("Le coin doit être 'top-left' ou 'top-right'")

    return submatrix