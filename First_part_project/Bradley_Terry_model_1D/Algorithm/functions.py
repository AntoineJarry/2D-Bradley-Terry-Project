import numpy as np


def bradley_terry_iterative(N, max_iter=100, tol=1e-6):
    """
    Calcule les scores Bradley-Terry avec l'algorithme itératif donné.

    Arguments :
    - N : Matrice de comparaisons par paires (n_ij = nombre de victoires de i contre j).
    - max_iter : Nombre maximum d'itérations.
    - tol : Tolérance pour la convergence.

    Retour :
    - lamda_i : Vecteur des scores Bradley-Terry normalisés.
    """

    n = N.shape[0]  # Nombre d'entités
    pi = np.ones(n)  # Initialisation des scores à 1 (ou tout autre valeur positive)

    for iteration in range(max_iter):
        pi_old = pi.copy()  # Conserver les anciennes valeurs pour vérifier la convergence

        for i in range(n):
            numer = 0  # Numérateur
            denom = 0  # Dénominateur
            for j in range(n):
                numer += (pi_old[j] * N[i, j]) / (pi_old[i] + pi_old[j])
                denom += N[j, i] / (pi_old[i] + pi_old[j])
            pi[i] = numer / denom

        # Vérifier la convergence
        if np.linalg.norm(pi - pi_old, ord=1) < tol:
            print(f"Convergence atteinte à l'itération {iteration + 1}")
            break

    lambda_i = np.log(pi)
    return lambda_i
