import numpy as np

def bradley_terry_mle(W, max_iter=100, tol=1e-6):
    n = W.shape[0]  # Nombre d'entités
    beta = np.zeros(n)  # Initialisation des scores
    for _ in range(max_iter):
        beta_old = beta.copy()
        for i in range(n):
            numer = np.sum(W[i, :] - (W[i, :] + W[:, i]) * np.exp(beta[i]) / (np.exp(beta[i]) + np.exp(beta)))
            beta[i] += 0.1 * numer  # Pas de mise à jour
        if np.linalg.norm(beta - beta_old) < tol:
            break
    return beta

def bradley_terry_mle_2(N, max_iter=100, tol=1e-6):
    n = N.shape[0]  # Nombre d'entités
    lambda_ = np.zeros(n)  # Initialisation des scores
    for _ in range(max_iter):
        lambda_old = lambda_.copy()
        for i in range(n):
            numer = np.sum(N[i, :] - (N[i, :] + N[:, i]) * np.exp(lambda_[i]) / (np.exp(lambda_[i]) + np.exp(lambda_)))
            lambda_[i] += 0.05 * numer  # Pas de mise à jour
        if np.linalg.norm(lambda_ - lambda_old) < tol:
            break
    return lambda_

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
                # if i != j:  # Pas de comparaison avec soi-même
                    numer += (pi_old[j] * N[i, j]) / (pi_old[i] + pi_old[j])
                    denom += N[j, i] / (pi_old[i] + pi_old[j])
            pi[i] = numer / denom

        # Vérifier la convergence
        if np.linalg.norm(pi - pi_old, ord=1) < tol:
            print(f"Convergence atteinte à l'itération {iteration + 1}")
            break

    lambda_i = np.log(pi)
    return lambda_i

def bradley_terry_with_confidence_intervals(N, max_iter=100, tol=1e-6, alpha=0.05):
    """
    Calcule les scores Bradley-Terry avec estimation des variances et intervalles de confiance.

    Arguments :
    - N : Matrice de comparaisons par paires (n_ij = nombre de victoires de i contre j).
    - max_iter : Nombre maximum d'itérations.
    - tol : Tolérance pour la convergence.
    - alpha : Niveau de confiance (par défaut 0.05 pour des IC à 95%).

    Retour :
    - lambda_i : Vecteur des scores Bradley-Terry normalisés (log(pi)).
    - variances : Vecteur des variances des lambda_i.
    - conf_intervals : Intervalles de confiance pour chaque lambda_i.
    """
    n = N.shape[0]  # Nombre d'entités
    pi = np.ones(n)  # Initialisation des scores à 1 (ou tout autre valeur positive)

    # Étape 1 : Calcul des scores avec l'algorithme itératif
    for iteration in range(max_iter):
        pi_old = pi.copy()

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

    # Étape 2 : Calcul de la matrice d'information de Fisher
    fisher_info = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                fisher_info[i, j] = -N[i, j] / ((pi[i] + pi[j]) ** 2)  # Terme hors diagonale
                fisher_info[i, i] += N[i, j] / ((pi[i] + pi[j]) ** 2)  # Contribution à la diagonale

    # Étape 3 : Calcul des variances (inverse de la matrice d'information)
    try:
        fisher_info_inv = np.linalg.inv(fisher_info)
        variances = np.diag(fisher_info_inv)
    except np.linalg.LinAlgError:
        print("La matrice d'information de Fisher n'est pas inversible.")
        return lambda_i, None, None

    # Étape 4 : Calcul des intervalles de confiance
    z = -np.percentile([-alpha / 2, alpha / 2], 50)  # Quantile de la loi normale
    conf_intervals = [(lambda_i[i] - z * np.sqrt(variances[i]),
                       lambda_i[i] + z * np.sqrt(variances[i])) for i in range(n)]

    return lambda_i, variances, conf_intervals

